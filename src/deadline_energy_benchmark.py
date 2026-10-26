from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))

from load_scenario import load_scenario

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
GRID_IMPORT_LIMIT_KW = 125.0
FULL_CHARGE_TOL_KWH = 0.05
TOL = 1e-9


def load_solar():
    path = DATA / "solar_profile_36h.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Solar profile not found: {path}. Run src/create_solar_profile.py first."
        )
    df = pd.read_csv(path)
    if "solar_power_kw" not in df.columns or len(df) != N_SLOTS:
        raise ValueError(
            "Expected 144 rows and a 'solar_power_kw' column in solar_profile_36h.csv."
        )
    solar = df["solar_power_kw"].to_numpy(float)
    if not np.isfinite(solar).all() or np.any(solar < 0):
        raise ValueError("Solar profile has invalid values.")
    return solar


def evaluate(schedule, scenario, solar):
    P = np.asarray(schedule, dtype=float)
    n_ev = len(scenario)
    if P.shape != (n_ev, N_SLOTS):
        raise ValueError(f"Expected shape {(n_ev, N_SLOTS)}, got {P.shape}")

    required = scenario["required_energy_kwh"].to_numpy(float)
    max_power = scenario["max_power_kw"].to_numpy(float)
    arrival = np.clip(scenario["arrival_slot"].to_numpy(int), 0, N_SLOTS)
    departure = np.clip(scenario["departure_slot"].to_numpy(int), 0, N_SLOTS)

    ev_energy = P.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - ev_energy, 0.0)
    ev_load = P.sum(axis=0)
    grid_import = np.maximum(ev_load - solar, 0.0)
    active = (P > TOL).sum(axis=0)

    time_violations = 0
    for i in range(n_ev):
        valid = np.zeros(N_SLOTS, dtype=bool)
        valid[arrival[i]:departure[i]] = True
        time_violations += int(((P[i] > TOL) & ~valid).sum())

    metrics = {
        "required_energy_kwh": float(required.sum()),
        "delivered_energy_kwh": float(ev_energy.sum()),
        "unmet_energy_kwh": float(unmet_by_ev.sum()),
        "demand_met_pct": float(100 * ev_energy.sum() / max(required.sum(), 1e-12)),
        "fully_charged_evs": int((unmet_by_ev <= FULL_CHARGE_TOL_KWH).sum()),
        "peak_ev_load_kw": float(ev_load.max()),
        "peak_grid_import_after_solar_kw": float(grid_import.max()),
        "solar_available_kwh": float(solar.sum() * STEP_HOURS),
        "solar_used_kwh": float(np.minimum(ev_load, solar).sum() * STEP_HOURS),
        "grid_import_limit_violations": int(
            (grid_import > GRID_IMPORT_LIMIT_KW + TOL).sum()
        ),
        "charger_limit_violations": int((active > N_CHARGERS).sum()),
        "individual_power_violations": int(
            (P > max_power[:, None] + TOL).sum()
        ),
        "time_window_violations": int(time_violations),
        "negative_power_violations": int((P < -TOL).sum()),
        "overcharge_violations": int(
            (ev_energy > required + FULL_CHARGE_TOL_KWH).sum()
        ),
    }
    checks = [
        "grid_import_limit_violations", "charger_limit_violations",
        "individual_power_violations", "time_window_violations",
        "negative_power_violations", "overcharge_violations",
    ]
    metrics["feasible"] = all(metrics[k] == 0 for k in checks)
    return metrics, ev_load, grid_import


def build_deadline_energy_first(scenario, solar, tie_break="earliest"):
    """
    Greedy deadline-aware benchmark.

    At each 15-minute slot:
      1. Find arrived EVs that have not departed and still need energy.
      2. Prioritize earliest departure (EDF).
      3. Use remaining charger slots and EV-specific power limits.
      4. Cap EV load at grid-import limit + solar power.
    This is a heuristic benchmark, not a globally optimal solver.
    """
    n_ev = len(scenario)
    required = scenario["required_energy_kwh"].to_numpy(float)
    max_power = scenario["max_power_kw"].to_numpy(float)
    arrival = np.clip(scenario["arrival_slot"].to_numpy(int), 0, N_SLOTS)
    departure = np.clip(scenario["departure_slot"].to_numpy(int), 0, N_SLOTS)

    P = np.zeros((n_ev, N_SLOTS), dtype=float)
    delivered = np.zeros(n_ev, dtype=float)

    for t in range(N_SLOTS):
        eligible = [
            i for i in range(n_ev)
            if arrival[i] <= t < departure[i] and required[i] - delivered[i] > 1e-8
        ]
        if tie_break == "largest_unmet":
            eligible.sort(key=lambda i: (
                departure[i], -(required[i] - delivered[i]), arrival[i], i
            ))
        elif tie_break == "highest_rate":
            eligible.sort(key=lambda i: (
                departure[i], -max_power[i], -(required[i] - delivered[i]), i
            ))
        else:
            eligible.sort(key=lambda i: (
                departure[i], arrival[i], -(required[i] - delivered[i]), i
            ))

        active_count = 0
        ev_load_room = GRID_IMPORT_LIMIT_KW + solar[t]
        for i in eligible:
            if active_count >= N_CHARGERS or ev_load_room <= 1e-9:
                break
            remaining = required[i] - delivered[i]
            power = min(max_power[i], remaining / STEP_HOURS, ev_load_room)
            if power > TOL:
                P[i, t] = power
                delivered[i] += power * STEP_HOURS
                ev_load_room -= power
                active_count += 1
    return P


def load_schedule(filename, n_ev):
    path = RESULTS / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = [f"slot_{t}" for t in range(N_SLOTS)]
    if len(df) != n_ev or not all(c in df.columns for c in cols):
        print(f"Skipping {filename}: schedule format mismatch.")
        return None
    return df[cols].to_numpy(float)


def save_schedule(schedule, filename, scenario):
    df = pd.DataFrame(schedule, columns=[f"slot_{t}" for t in range(N_SLOTS)])
    df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    path = RESULTS / filename
    df.to_csv(path, index=False)
    return path


def main():
    scenario = load_scenario()
    solar = load_solar()
    n_ev = len(scenario)
    print(f"Loaded {n_ev} EVs")
    print(f"Grid-import cap after solar: {GRID_IMPORT_LIMIT_KW:.1f} kW")
    print("Benchmark: earliest-departure-first, greedy power allocation")

    benchmark_specs = [
        ("EDF earliest departure", "earliest"),
        ("EDF largest unmet tie-break", "largest_unmet"),
        ("EDF highest power tie-break", "highest_rate"),
    ]
    schedules = {}
    rows = []

    for name, tie in benchmark_specs:
        P = build_deadline_energy_first(scenario, solar, tie)
        schedules[name] = P
        metrics, _, _ = evaluate(P, scenario, solar)
        metrics["method"] = name
        rows.append(metrics)
        safe = name.lower().replace(" ", "_").replace("-", "")
        path = save_schedule(P, f"edf_benchmark_{safe}_schedule.csv", scenario)
        print(f"Saved {name} schedule: {path}")

    # Load and re-evaluate existing schedules with the same evaluator.
    existing = [
        ("Original GA", "ga_best_schedule.csv"),
        ("Feasibility-first", "feasibility_baseline_schedule.csv"),
        ("PSO", "pso_best_schedule.csv"),
        ("Battery-aware GA v3", "ga_battery_aware_v3_schedule.csv"),
        ("Solar-aware baseline", "solar_aware_schedule.csv"),
        ("Peak-constrained energy-first GA", "ga_peak_constrained_best_schedule.csv"),
    ]
    for name, filename in existing:
        P = load_schedule(filename, n_ev)
        if P is None:
            continue
        metrics, _, _ = evaluate(P, scenario, solar)
        metrics["method"] = name
        rows.append(metrics)

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        ["feasible", "delivered_energy_kwh"], ascending=[False, False]
    )
    out_csv = RESULTS / "deadline_energy_benchmark_comparison.csv"
    comparison.to_csv(out_csv, index=False)

    # Plot delivered energy and unmet energy.
    plt.figure(figsize=(11, 5))
    sorted_rows = comparison.sort_values("delivered_energy_kwh", ascending=False)
    plt.bar(sorted_rows["method"], sorted_rows["delivered_energy_kwh"])
    plt.ylabel("Delivered energy (kWh)")
    plt.title("Deadline-Aware Benchmark: Delivered Energy")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    energy_plot = RESULTS / "deadline_energy_benchmark_delivered.png"
    plt.savefig(energy_plot, dpi=170)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(sorted_rows["method"], sorted_rows["peak_grid_import_after_solar_kw"])
    plt.axhline(GRID_IMPORT_LIMIT_KW, linestyle="--", label="125 kW grid-import limit")
    plt.ylabel("Peak grid import after solar (kW)")
    plt.title("Deadline-Aware Benchmark: Peak Grid Import")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    peak_plot = RESULTS / "deadline_energy_benchmark_peak_grid.png"
    plt.savefig(peak_plot, dpi=170)
    plt.close()

    cols = [
        "method", "delivered_energy_kwh", "unmet_energy_kwh", "demand_met_pct",
        "fully_charged_evs", "peak_ev_load_kw",
        "peak_grid_import_after_solar_kw", "solar_used_kwh",
        "grid_import_limit_violations", "charger_limit_violations", "feasible"
    ]
    print("\nDEADLINE-AWARE ENERGY BENCHMARK COMPARISON")
    print("=" * 115)
    print(comparison[cols].round(3).to_string(index=False))
    print(f"\nComparison saved: {out_csv}")
    print(f"Delivered-energy graph: {energy_plot}")
    print(f"Peak-grid-import graph: {peak_plot}")
    print("\nInterpretation: these EDF variants are greedy heuristics, not mathematical optima.")
    print("All schedules are evaluated using the same solar-adjusted grid-import definition.")


if __name__ == "__main__":
    main()
