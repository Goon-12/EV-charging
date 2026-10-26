
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from load_scenario import load_scenario
from evaluate_schedule import evaluate_schedule

STEP_HOURS = 0.25
MAX_GRID_KW = 125.0
N_CHARGERS = 20
N_SLOTS = 144

scenario = load_scenario()

schedule_path = RESULTS_DIR / "ga_cost_best_schedule.csv"
df = pd.read_csv(schedule_path)

slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]

if len(df) != len(scenario):
    raise ValueError("EV count does not match the scenario.")

if not np.array_equal(
    df["ev_id"].to_numpy(),
    scenario["ev_id"].to_numpy()
):
    raise ValueError("EV IDs/order do not match the scenario.")

schedule = df[slot_columns].to_numpy(dtype=float)

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

print("\nCOST-AWARE GA SCHEDULE VALIDATION")
print("=" * 50)

for name, value in metrics.items():
    print(f"{name}: {value}")

passed = (
    metrics["grid_limit_violations"] == 0
    and metrics["charger_limit_violations"] == 0
    and metrics["individual_power_violations"] == 0
    and metrics["time_window_violations"] == 0
)

print(
    "\nAll checked hard constraints passed."
    if passed
    else "\nWARNING: One or more hard constraints are violated."
)
