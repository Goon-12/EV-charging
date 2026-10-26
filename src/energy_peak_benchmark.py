from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
STEP, SLOTS, CHARGERS, GRID = 0.25, 144, 20, 125.0
TOL, ENERGY_TOL = 1e-6, 0.05

scenario = pd.read_csv(DATA / "ev_scenario.csv")
solar_df = pd.read_csv(DATA / "solar_profile_36h.csv")
n = len(scenario)
solar = solar_df["solar_power_kw"].to_numpy(float)
required = scenario["required_energy_kwh"].to_numpy(float)
max_power = scenario["max_power_kw"].to_numpy(float)
arrival = np.clip(scenario["arrival_slot"].to_numpy(int), 0, SLOTS)
departure = np.clip(scenario["departure_slot"].to_numpy(int), 0, SLOTS)
hours = np.arange(SLOTS) * STEP

if len(solar) != SLOTS:
    raise ValueError(f"Expected {SLOTS} solar values, got {len(solar)}")


def evaluate(P, method, cap=GRID):
    P = np.asarray(P, float)
    if P.shape != (n, SLOTS) or not np.isfinite(P).all():
        raise ValueError(f"{method}: invalid schedule shape or values")
    e = P.sum(axis=1) * STEP
    unmet = np.maximum(required - e, 0)
    load = P.sum(axis=0)
    imp = np.maximum(load - solar, 0)
    active = (P > TOL).sum(axis=0)
    grid_v = int((imp > cap + TOL).sum())
    charger_v = int((active > CHARGERS).sum())
    power_v = int((P > max_power[:, None] + TOL).sum())
    negative_v = int((P < -TOL).sum())
    time_v = 0
    for i in range(n):
        valid = np.zeros(SLOTS, bool)
        valid[arrival[i]:departure[i]] = True
        time_v += int(((P[i] > TOL) & ~valid).sum())
    overcharge_v = int((e > required + ENERGY_TOL).sum())
    return {
        "method": method,
        "grid_cap_kw": cap,
        "required_energy_kwh": float(required.sum()),
        "delivered_energy_kwh": float(e.sum()),
        "unmet_energy_kwh": float(unmet.sum()),
        "energy_demand_met_pct": float(100 * e.sum() / required.sum()),
        "fully_charged_evs": int((unmet <= ENERGY_TOL).sum()),
        "peak_ev_load_kw": float(load.max()),
        "peak_grid_import_kw": float(imp.max()),
        "grid_limit_violations": grid_v,
        "charger_limit_violations": charger_v,
        "individual_power_violations": power_v,
        "negative_power_violations": negative_v,
        "time_window_violations": time_v,
        "overcharge_violations": overcharge_v,
        "feasible": all(v == 0 for v in [
            grid_v, charger_v, power_v, negative_v, time_v, overcharge_v
        ]),
    }


def edf_schedule(cap=GRID, reverse=False):
    P = np.zeros((n, SLOTS))
    delivered = np.zeros(n)
    duration = np.maximum(departure - arrival, 0)
    order = sorted(range(n), key=lambda i: (departure[i], duration[i], i))
    slot_order = range(SLOTS - 1, -1, -1) if reverse else range(SLOTS)
    for t in slot_order:
        eligible = [
            i for i in order
            if arrival[i] <= t < departure[i]
            and required[i] - delivered[i] > 1e-8
        ]
        room = cap + solar[t]
        active = 0
        for i in eligible:
            if active >= CHARGERS or room <= 1e-9:
                break
            p = min(max_power[i], (required[i] - delivered[i]) / STEP, room)
            if p > 1e-9:
                P[i, t] = p
                delivered[i] += p * STEP
                room -= p
                active += 1
    return P


def load_schedule(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = [f"slot_{t}" for t in range(SLOTS)]
    if len(df) != n or not all(c in df.columns for c in cols):
        print(f"Skipping malformed schedule: {filename}")
        return None
    return df[cols].to_numpy(float)


def save_schedule(P, filename):
    df = pd.DataFrame(P, columns=[f"slot_{t}" for t in range(SLOTS)])
    df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    path = RESULTS / filename
    df.to_csv(path, index=False)
    return path


def main():
    schedules = {}
    candidates = [
        ("Original GA", "ga_best_schedule.csv"),
        ("PSO", "pso_best_schedule.csv"),
        ("Feasibility-first", "feasibility_baseline_schedule.csv"),
        ("Battery-aware GA v3", "ga_battery_aware_v3_schedule.csv"),
        ("Solar-aware baseline", "solar_aware_schedule.csv"),
        ("Cost-aware GA", "ga_cost_best_schedule_latest.csv"),
    ]
    for name, filename in candidates:
        P = load_schedule(filename)
        if P is None and name == "Cost-aware GA":
            P = load_schedule("ga_cost_best_schedule.csv")
        if P is not None:
            schedules[name] = P

    for name, reverse, filename in [
        ("EDF chronological", False, "energy_first_edf_chronological_schedule.csv"),
        ("EDF reverse-time sensitivity", True, "energy_first_edf_reverse_schedule.csv"),
    ]:
        P = edf_schedule(GRID, reverse)
        schedules[name] = P
        save_schedule(P, filename)

    table = pd.DataFrame([evaluate(P, name) for name, P in schedules.items()])
    table = table.sort_values(["feasible", "delivered_energy_kwh"], ascending=[False, False])
    table.to_csv(RESULTS / "energy_peak_benchmark_comparison.csv", index=False)
    cols = [
        "method", "delivered_energy_kwh", "unmet_energy_kwh",
        "energy_demand_met_pct", "fully_charged_evs",
        "peak_ev_load_kw", "peak_grid_import_kw",
        "grid_limit_violations", "charger_limit_violations", "feasible",
    ]
    print("\nENERGY / PEAK BENCHMARK COMPARISON")
    print("=" * 108)
    print(table[cols].round(3).to_string(index=False))

    rows = []
    for cap in [75.0, 90.0, 100.0, 110.0, 125.0]:
        rows.append(evaluate(edf_schedule(cap), f"EDF cap {cap:.0f} kW", cap))
    sens = pd.DataFrame(rows)
    sens.to_csv(RESULTS / "grid_cap_sensitivity.csv", index=False)
    print("\nGRID-IMPORT CAP SENSITIVITY")
    print("=" * 82)
    print(sens[[
        "method", "delivered_energy_kwh", "unmet_energy_kwh",
        "fully_charged_evs", "peak_ev_load_kw", "peak_grid_import_kw", "feasible"
    ]].round(3).to_string(index=False))

    plt.figure(figsize=(8, 4.5))
    plt.plot(sens["grid_cap_kw"], sens["delivered_energy_kwh"], marker="o")
    plt.xlabel("Grid-import limit after solar (kW)")
    plt.ylabel("Delivered energy (kWh)")
    plt.title("Energy Delivery vs Grid-Import Limit")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "grid_cap_sensitivity.png", dpi=180)
    plt.close()

    print("\nSaved comparison: results/energy_peak_benchmark_comparison.csv")
    print("Saved schedules: results/energy_first_edf_chronological_schedule.csv")
    print("                results/energy_first_edf_reverse_schedule.csv")
    print("Saved sensitivity: results/grid_cap_sensitivity.csv")
    print("Saved plot: results/grid_cap_sensitivity.png")
    print("\nNotes: EDF is a heuristic, not a proven global optimum.")
    print("Solar profile and tariff are synthetic; common evaluation does not make original optimization objectives identical.")


if __name__ == "__main__":
    main()
