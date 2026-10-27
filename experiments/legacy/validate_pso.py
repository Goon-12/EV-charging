# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from load_scenario import load_scenario

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0
TOL = 1e-6

scenario = load_scenario()
schedule_path = PROJECT_DIR / "results" / "pso_best_schedule.csv"

if not schedule_path.exists():
    raise FileNotFoundError(f"Schedule not found: {schedule_path}")

schedule_df = pd.read_csv(schedule_path)

slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]
required_columns = ["ev_id"] + slot_columns

missing_columns = [
    col for col in required_columns if col not in schedule_df.columns
]
if missing_columns:
    raise ValueError(f"Missing schedule columns: {missing_columns}")

if schedule_df["ev_id"].duplicated().any():
    raise ValueError("Duplicate EV IDs in PSO schedule.")

# Match schedules to the scenario by EV ID, not row position.
scenario = scenario.copy()
schedule_df["ev_id"] = schedule_df["ev_id"].astype(str)
scenario["ev_id"] = scenario["ev_id"].astype(str)

if set(schedule_df["ev_id"]) != set(scenario["ev_id"]):
    raise ValueError("PSO schedule EV IDs do not match the saved scenario.")

scenario = scenario.set_index("ev_id").loc[schedule_df["ev_id"]].reset_index()

schedule = schedule_df[slot_columns].to_numpy(dtype=float)

arrival = scenario["arrival_slot"].to_numpy(dtype=int)
departure = scenario["departure_slot"].to_numpy(dtype=int)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)

# Basic numeric checks.
if not np.isfinite(schedule).all():
    raise ValueError("Schedule contains NaN or infinite power values.")

negative_power = int(np.count_nonzero(schedule < -TOL))

# Energy delivered and remaining unmet energy per EV.
delivered_per_ev = np.maximum(schedule, 0).sum(axis=1) * STEP_HOURS
unmet_per_ev = np.maximum(required - delivered_per_ev, 0.0)

# Grid load and number of simultaneously active EVs in each slot.
grid_load = schedule.sum(axis=0)
active_ev_count = np.count_nonzero(schedule > TOL, axis=0)

grid_violations = int(np.count_nonzero(grid_load > MAX_GRID_KW + TOL))
charger_violations = int(np.count_nonzero(active_ev_count > N_CHARGERS))

# Check individual charging power limits.
individual_power_violations = int(
    np.count_nonzero(schedule > max_power[:, None] + TOL)
)

# Check charging outside each EV's arrival/departure window.
time_window_violations = 0
for i in range(len(scenario)):
    valid = np.zeros(N_SLOTS, dtype=bool)
    start = max(0, arrival[i])
    end = min(N_SLOTS, departure[i])
    valid[start:end] = True

    time_window_violations += int(
        np.count_nonzero((schedule[i] > TOL) & ~valid)
    )

total_required = float(required.sum())
total_delivered = float(delivered_per_ev.sum())
total_unmet = float(unmet_per_ev.sum())
fully_charged = int(np.count_nonzero(unmet_per_ev <= 0.05))

print("\nPSO SCHEDULE VALIDATION")
print("=" * 45)
print(f"EVs: {len(scenario)}")
print(f"Total required energy: {total_required:.3f} kWh")
print(f"Total delivered energy: {total_delivered:.3f} kWh")
print(f"Total unmet energy: {total_unmet:.3f} kWh")
print(f"Energy demand met: {100 * total_delivered / total_required:.2f}%")
print(f"Fully charged EVs (within 0.01 kWh): {fully_charged}")
print(f"Peak grid demand: {grid_load.max():.3f} kW")
print(f"Maximum simultaneous active EVs: {active_ev_count.max()}")
print(f"Negative power entries: {negative_power}")
print(f"Grid-limit violations: {grid_violations}")
print(f"Charger-limit violations: {charger_violations}")
print(f"Individual power violations: {individual_power_violations}")
print(f"Time-window violations: {time_window_violations}")

metrics = {
    "method": "PSO",
    "peak_grid_kw": float(grid_load.max()),
    "total_energy_kwh": total_delivered,
    "total_unmet_energy_kwh": total_unmet,
    "evs_fully_charged": fully_charged,
    "grid_limit_violations": grid_violations,
    "charger_limit_violations": charger_violations,
    "individual_power_violations": individual_power_violations,
    "time_window_violations": time_window_violations,
    "negative_power_entries": negative_power,
}

metrics_path = PROJECT_DIR / "results" / "pso_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

print("\nMetrics saved to:", metrics_path)

if any([
    negative_power,
    grid_violations,
    charger_violations,
    individual_power_violations,
    time_window_violations,
]):
    print("WARNING: One or more hard constraints failed.")
else:
    print("All checked hard constraints passed.")
