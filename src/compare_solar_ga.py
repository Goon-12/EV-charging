
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

scenario = pd.read_csv(DATA_DIR / "ev_scenario.csv")
solar = pd.read_csv(DATA_DIR / "solar_profile_36h.csv")

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
solar_kw = solar["solar_power_kw"].to_numpy(dtype=float)

STEP_HOURS = 0.25
N_SLOTS = 144
MAX_GRID_KW = 125.0
TOL = 1e-8

methods = {
    "Uncontrolled": "uncontrolled_schedule.csv",
    "FCFS": "fcfs_schedule.csv",
    "Original GA": "ga_best_schedule.csv",
    "PSO": "pso_best_schedule.csv",
    "Feasibility-first": "feasibility_baseline_schedule.csv",
    "Cost-aware GA": "ga_cost_best_schedule_latest.csv",
    "Solar-aware": "solar_aware_schedule.csv",
    "Solar-aware GA": "ga_solar_best_schedule.csv",
}

rows = []

for method, filename in methods.items():
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"Skipping {method}: {filename} not found")
        continue

    df = pd.read_csv(path)
    columns = [f"slot_{t}" for t in range(N_SLOTS)]

    if len(df) != len(scenario) or not all(c in df.columns for c in columns):
        print(f"Skipping {method}: incompatible schedule shape")
        continue

    schedule = df[columns].to_numpy(dtype=float)
    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)

    ev_load = schedule.sum(axis=0)
    solar_used = np.minimum(ev_load, solar_kw)
    grid_import = np.maximum(ev_load - solar_kw, 0.0)

    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)

    active = (schedule > TOL).sum(axis=0)
    grid_violations = int((grid_import > MAX_GRID_KW + TOL).sum())
    charger_violations = int((active > 20).sum())
    power_violations = int(
        (schedule > max_power[:, None] + TOL).sum()
    )

    time_violations = 0
    for i in range(len(scenario)):
        valid = np.zeros(N_SLOTS, dtype=bool)
        start = int(np.clip(arrival[i], 0, N_SLOTS))
        end = int(np.clip(departure[i], 0, N_SLOTS))
        valid[start:end] = True
        time_violations += int(
            ((schedule[i] > TOL) & ~valid).sum()
        )

    # Report the raw EV-load peak separately from grid import after solar.
    rows.append({
        "method": method,
        "required_kwh": required.sum(),
        "delivered_kwh": delivered_by_ev.sum(),
        "unmet_kwh": unmet_by_ev.sum(),
        "demand_met_pct": 100 * delivered_by_ev.sum() / required.sum(),
        "fully_charged_evs": int((unmet_by_ev <= 0.05).sum()),
        "peak_ev_load_kw": ev_load.max(),
        "peak_grid_import_kw_after_solar": grid_import.max(),
        "solar_used_kwh": solar_used.sum() * STEP_HOURS,
        "grid_limit_violations": grid_violations,
        "charger_limit_violations": charger_violations,
        "power_violations": power_violations,
        "time_window_violations": time_violations,
        "feasible": (
            grid_violations == 0
            and charger_violations == 0
            and power_violations == 0
            and time_violations == 0
        ),
    })

comparison = pd.DataFrame(rows).round(3)
output = RESULTS_DIR / "solar_algorithm_comparison.csv"
comparison.to_csv(output, index=False)

print("\nSOLAR EXPERIMENT: CONSISTENT SCHEDULE COMPARISON")
print("=" * 120)
print(comparison.to_string(index=False))
print(f"\nSaved: {output}")

# Plot the two different peak measures separately and label them clearly.
plt.figure(figsize=(12, 5))
plt.bar(comparison["method"], comparison["peak_ev_load_kw"])
plt.ylabel("Peak total EV charging load (kW)")
plt.title("Peak EV Charging Load by Schedule")
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "solar_comparison_ev_peak.png", dpi=200)
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(
    comparison["method"],
    comparison["peak_grid_import_kw_after_solar"],
)
plt.axhline(MAX_GRID_KW, linestyle="--", label="125 kW grid-import limit")
plt.ylabel("Peak grid import after solar (kW)")
plt.title("Estimated Grid Import After Solar Offset")
plt.xticks(rotation=25, ha="right")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "solar_comparison_grid_peak.png", dpi=200)
plt.close()

print("Graphs saved in results/.")
print(
    "\nNote: solar is applied to all schedules for the second peak metric. "
    "Only the solar-aware algorithms explicitly used solar during scheduling. "
    "This is a diagnostic comparison, not a claim of equal optimization conditions."
)
