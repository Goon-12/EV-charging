
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
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0

scenario = load_scenario()

def validate_baseline(name, filename):
    path = RESULTS_DIR / filename
    df = pd.read_csv(path)

    slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]
    missing = [c for c in slot_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing columns: {missing[:5]}")

    if len(df) != len(scenario):
        raise ValueError(f"EV count mismatch in {filename}")

    if not np.array_equal(
        df["ev_id"].to_numpy(),
        scenario["ev_id"].to_numpy()
    ):
        raise ValueError(f"EV ID/order mismatch in {filename}")

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

    metrics["method"] = name
    return metrics


results = pd.DataFrame([
    validate_baseline("Uncontrolled", "uncontrolled_schedule.csv"),
    validate_baseline("FCFS", "fcfs_schedule.csv"),
])

columns = [
    "method",
    "peak_grid_kw",
    "total_energy_kwh",
    "total_unmet_energy_kwh",
    "evs_fully_charged",
    "grid_limit_violations",
    "charger_limit_violations",
    "individual_power_violations",
    "time_window_violations",
]

results = results[columns]
results.to_csv(RESULTS_DIR / "baseline_validation.csv", index=False)

print("\nBASELINE SCHEDULE VALIDATION")
print("=" * 75)
print(results.round(3).to_string(index=False))
print(f"\nValidation saved to: {RESULTS_DIR / 'baseline_validation.csv'}")
