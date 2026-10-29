from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Same-assumptions comparison + energy-first benchmark.

All schedules are evaluated using:
- 15-minute slots
- 20 simultaneous chargers
- per-EV maximum charging power
- arrival/departure windows
- 125 kW grid-import limit AFTER solar offset

The "Energy-max benchmark" is a deterministic earliest-departure-first
heuristic, NOT a mathematically proven global optimum.
"""

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144
MAX_CHARGERS = 20
GRID_LIMIT_KW = 125.0
TOL = 1e-6
ENERGY_TOL_KWH = 0.05

scenario_path = DATA / "ev_scenario.csv"
solar_path = DATA / "solar_profile_36h.csv"

scenario = pd.read_csv(scenario_path)
solar_df = pd.read_csv(solar_path)

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = np.clip(scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS)
departure = np.clip(scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS)
solar = solar_df["solar_power_kw"].to_numpy(dtype=float)
n_ev = len(scenario)

if len(solar) != N_SLOTS:
    raise ValueError(f"Expected {N_SLOTS} solar rows, found {len(solar)}")
if not np.isfinite(solar).all() or np.any(solar < 0):
    raise ValueError("Solar profile has invalid values")

# Synthetic illustrative tariff, consistent with earlier experiments.
hours = np.arange(N_SLOTS) * STEP_HOURS
hour_of_day = hours % 24.0
prices = np.select(
    [(hour_of_day < 6), ((hour_of_day >= 17) & (hour_of_day < 22))],
    [4.0, 10.0],
    default=6.0,
).astype(float)


def evaluate(schedule, method):
    P = np.asarray(schedule, dtype=float)
    if P.shape != (n_ev, N_SLOTS):
        raise ValueError(f"{method}: expected schedule shape {(n_ev, N_SLOTS)}, got {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError(f"{method}: schedule contains NaN/infinity")

    energy_by_ev = P.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - energy_by_ev, 0.0)
    ev_load = P.sum(axis=0)
    grid_import = np.maximum(ev_load - solar, 0.0)
    active = (P > TOL).sum(axis=0)
    solar_used = np.minimum(ev_load, solar)

    grid_v = int((grid_import > GRID_LIMIT_KW + TOL).sum())
    charger_v = int((active > MAX_CHARGERS).sum())
    power_v = int((P > max_power[:, None] + TOL).sum())
    time_v = 0
    for i in range(n_ev):
        valid = np.zeros(N_SLOTS, dtype=bool)
        valid[arrival[i]:departure[i]] = True
        time_v += int(((P[i] > TOL) & ~valid).sum())
    negative_v = int((P < -TOL).sum())
    overcharge_v = int((energy_by_ev > required + ENERGY_TOL_KWH).sum())

    # Grid import is used for cost, using the same synthetic tariff for all methods.
    cost_units = float(np.sum(grid_import * STEP_HOURS * prices))
    total_required = float(required.sum())
    delivered = float(energy_by_ev.sum())
    solar_available = float(solar.sum() * STEP_HOURS)
    solar_used_kwh = float(solar_used.sum() * STEP_HOURS)

    checks = [grid_v, charger_v, power_v, time_v, negative_v, overcharge_v]
    return {
        "method": method,
        "required_energy_kwh": total_required,
        "delivered_energy_kwh": delivered,
        "unmet_energy_kwh": float(unmet_by_ev.sum()),
        "energy_demand_met_pct": 100.0 * delivered / total_required if total_required else 100.0,
        "fully_charged_evs": int((unmet_by_ev <= ENERGY_TOL_KWH).sum()),
        "peak_ev_load_kw": float(ev_load.max()),
        "peak_grid_import_kw_after_solar": float(grid_import.max()),
        "solar_available_kwh": solar_available,
        "solar_used_kwh": solar_used_kwh,
        "solar_utilization_pct": 100.0 * solar_used_kwh / max(solar_available, 1e-12),
        "synthetic_grid_import_cost_units": cost_units,
        "grid_import_limit_violations": grid_v,
        "charger_limit_violations": charger_v,
        "individual_power_violations": power_v,
        "time_window_violations": time_v,
        "negative_power_violations": negative_v,
        "overcharge_violations": overcharge_v,
        "feasible": all(v == 0 for v in checks),
    }


def energy_first_benchmark():
    """
    Deterministic chronological EDF benchmark.
    Each slot is filled up to min(charger capacity, grid limit + solar).
    Among eligible EVs, earlier departures get power first.
    """
    schedule = np.zeros((n_ev, N_SLOTS), dtype=float)
    delivered = np.zeros(n_ev, dtype=float)
    duration = np.maximum(departure - arrival, 0)
    ev_order = sorted(
        range(n_ev),
        key=lambda i: (departure[i], duration[i], i),
    )

    for t in range(N_SLOTS):
        eligible = [
            i for i in ev_order
            if arrival[i] <= t < departure[i]
            and required[i] - delivered[i] > 1e-9
        ]
        grid_import_room = GRID_LIMIT_KW
        active_count = 0
        # The allowed total EV load is grid limit plus solar at this slot.
        ev_load_room = GRID_LIMIT_KW + solar[t]

        for i in eligible:
            if active_count >= MAX_CHARGERS or ev_load_room <= 1e-9:
                break
            remaining = required[i] - delivered[i]
            p = min(max_power[i], remaining / STEP_HOURS, ev_load_room)
            if p <= 1e-9:
                continue
            schedule[i, t] = p
            delivered[i] += p * STEP_HOURS
            ev_load_room -= p
            active_count += 1

    return schedule


def load_schedule(filename):
    path = RESULTS / filename
    if not path.exists():
        return None, f"not found: {filename}"
    df = pd.read_csv(path)
    slot_cols = [f"slot_{t}" for t in range(N_SLOTS)]
    missing = [c for c in slot_cols if c not in df.columns]
    if missing:
        return None, f"missing slot columns in {filename}"
    P = df[slot_cols].to_numpy(dtype=float)
    if P.shape[0] != n_ev:
        return None, f"EV row count mismatch in {filename}"
    return P, None


def save_schedule(schedule, filename):
    df = pd.DataFrame(schedule, columns=[f"slot_{t}" for t in range(N_SLOTS)])
    df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    path = RESULTS / filename
    df.to_csv(path, index=False)
    return path


def main():
    methods = [
        ("Uncontrolled", "uncontrolled_schedule.csv"),
        ("FCFS", "fcfs_schedule.csv"),
        ("Original GA", "ga_best_schedule.csv"),
        ("PSO", "pso_best_schedule.csv"),
        ("Feasibility-first", "feasibility_baseline_schedule.csv"),
        ("Cost-aware GA", "ga_cost_best_schedule.csv"),
        ("Solar-aware baseline", "solar_aware_schedule.csv"),
        ("Solar-aware GA", "ga_solar_best_schedule.csv"),
        ("Battery-aware GA v2", "ga_battery_aware_v2_schedule.csv"),
        ("Battery-aware GA v3 chronological", "ga_battery_aware_v3_schedule.csv"),
    ]

    records = []
    schedules = {}
    missing_files = []

    for method, filename in methods:
        P, error = load_schedule(filename)
        if error:
            # Some projects may have the non-latest cost-aware filename.
            if method == "Cost-aware GA":
                P, error = load_schedule("ga_cost_best_schedule.csv")
            if error:
                missing_files.append(f"{method}: {error}")
                continue
        schedules[method] = P
        records.append(evaluate(P, method))

    benchmark = energy_first_benchmark()
    save_schedule(benchmark, "energy_first_benchmark_schedule.csv")
    schedules["Energy-first EDF benchmark"] = benchmark
    records.append(evaluate(benchmark, "Energy-first EDF benchmark"))

    if not records:
        raise RuntimeError("No schedules were loaded or generated.")

    comparison = pd.DataFrame(records)
    comparison = comparison.sort_values(
        ["feasible", "unmet_energy_kwh"],
        ascending=[False, True],
    ).reset_index(drop=True)
    comparison_path = RESULTS / "same_assumptions_solar_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nSAME-ASSUMPTIONS COMPARISON")
    print("=" * 112)
    cols = [
        "method", "delivered_energy_kwh", "unmet_energy_kwh",
        "energy_demand_met_pct", "fully_charged_evs",
        "peak_ev_load_kw", "peak_grid_import_kw_after_solar",
        "synthetic_grid_import_cost_units", "solar_used_kwh", "feasible",
    ]
    print(comparison[cols].round(3).to_string(index=False))

    if missing_files:
        print("\nSCHEDULES NOT INCLUDED")
        for item in missing_files:
            print("-", item)

    # Plot delivered energy and unmet energy.
    plot_df = comparison.sort_values("delivered_energy_kwh", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["method"], plot_df["delivered_energy_kwh"])
    plt.ylabel("Delivered energy (kWh)")
    plt.title("Delivered Energy — Same Solar and Grid Assumptions")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    delivered_path = RESULTS / "same_assumptions_delivered_energy.png"
    plt.savefig(delivered_path, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(plot_df["method"], plot_df["unmet_energy_kwh"])
    plt.ylabel("Unmet energy (kWh)")
    plt.title("Unmet Energy — Same Solar and Grid Assumptions")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    unmet_path = RESULTS / "same_assumptions_unmet_energy.png"
    plt.savefig(unmet_path, dpi=180)
    plt.close()

    # Export slot-by-slot benchmark load profile.
    benchmark_metrics = evaluate(benchmark, "Energy-first EDF benchmark")
    profile = pd.DataFrame({
        "slot": np.arange(N_SLOTS),
        "hour_from_start": hours,
        "total_ev_load_kw": benchmark.sum(axis=0),
        "solar_available_kw": solar,
        "grid_import_after_solar_kw": np.maximum(benchmark.sum(axis=0) - solar, 0),
        "synthetic_tariff_units_per_kwh": prices,
    })
    profile_path = RESULTS / "energy_first_benchmark_load_profile.csv"
    profile.to_csv(profile_path, index=False)

    print("\nENERGY-FIRST EDF BENCHMARK")
    print("=" * 50)
    for key in [
        "required_energy_kwh", "delivered_energy_kwh", "unmet_energy_kwh",
        "energy_demand_met_pct", "fully_charged_evs", "peak_ev_load_kw",
        "peak_grid_import_kw_after_solar", "solar_available_kwh",
        "solar_used_kwh", "synthetic_grid_import_cost_units",
        "grid_import_limit_violations", "charger_limit_violations",
        "individual_power_violations", "time_window_violations", "feasible",
    ]:
        print(f"{key}: {benchmark_metrics[key]}")

    print("\nSaved comparison:", comparison_path)
    print("Saved benchmark schedule:", RESULTS / "energy_first_benchmark_schedule.csv")
    print("Saved benchmark profile:", profile_path)
    print("Saved plots:", delivered_path, "and", unmet_path)
    print("\nInterpretation note: the EDF benchmark is a heuristic, not a proven global optimum.")
    print("Solar and tariff data are synthetic; cost values are illustrative units.")


if __name__ == "__main__":
    main()
