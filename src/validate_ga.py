
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from load_scenario import load_scenario
from evaluate_schedule import evaluate_schedule

RESULTS_DIR = PROJECT_DIR / "results"
SCHEDULE_FILE = RESULTS_DIR / "ga_best_schedule.csv"

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0

scenario = load_scenario()

if not SCHEDULE_FILE.exists():
    raise FileNotFoundError(f"GA schedule not found: {SCHEDULE_FILE}")

schedule_df = pd.read_csv(SCHEDULE_FILE)

expected_columns = [f"slot_{t}" for t in range(N_SLOTS)]
missing = [c for c in expected_columns if c not in schedule_df.columns]

if missing:
    raise ValueError(f"Missing schedule columns: {missing[:5]}")

if len(schedule_df) != len(scenario):
    raise ValueError("Schedule EV count does not match the scenario.")

if not np.array_equal(
    schedule_df["ev_id"].to_numpy(),
    scenario["ev_id"].to_numpy()
):
    raise ValueError("EV IDs/order do not match the saved scenario.")

schedule = schedule_df[expected_columns].to_numpy(dtype=float)

metrics = evaluate_schedule(
    schedule=schedule,
    required_energy_kwh=scenario["required_energy_kwh"].to_numpy(),
    step_hours=STEP_HOURS,
    max_grid_kw=MAX_GRID_KW,
    max_chargers=N_CHARGERS,
    max_power_kw=scenario["max_power_kw"].to_numpy(),
    arrival_slot=scenario["arrival_slot"].to_numpy(),
    departure_slot=scenario["departure_slot"].to_numpy(),
)

print("\nGA SCHEDULE VALIDATION")
print("-" * 45)

for name, value in metrics.items():
    print(f"{name}: {value}")

grid_load = schedule.sum(axis=0)
print("\nMaximum grid load (kW):", round(grid_load.max(), 3))
print("Total required energy (kWh):",
      round(scenario["required_energy_kwh"].sum(), 3))
print("Total delivered energy (kWh):",
      round(schedule.sum() * STEP_HOURS, 3))

if (
    metrics["grid_limit_violations"] == 0
    and metrics["charger_limit_violations"] == 0
    and metrics["individual_power_violations"] == 0
    and metrics["time_window_violations"] == 0
):
    print("\nAll checked hard constraints passed.")
else:
    print("\nWARNING: One or more hard constraints are violated.")
