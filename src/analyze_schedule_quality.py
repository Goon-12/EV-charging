
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_scenario import load_scenario

STEP_HOURS = 0.25
N_SLOTS = 144

scenario = load_scenario()

def analyze_schedule(filename, method):
    path = RESULTS_DIR / filename
    schedule_df = pd.read_csv(path)

    slot_columns = [f"slot_{i}" for i in range(N_SLOTS)]
    schedule = schedule_df[slot_columns].to_numpy(dtype=float)

    required = scenario["required_energy_kwh"].to_numpy(dtype=float)
    delivered = schedule.sum(axis=1) * STEP_HOURS
    unmet = np.maximum(required - delivered, 0.0)

    results = pd.DataFrame({
        "ev_id": scenario["ev_id"].to_numpy(),
        "required_kwh": required,
        "delivered_kwh": delivered,
        "unmet_kwh": unmet,
    })

    print(f"\n--- {method} ---")
    print("Total required energy:", round(required.sum(), 3))
    print("Total delivered energy:", round(delivered.sum(), 3))
    print("Total unmet energy:", round(unmet.sum(), 3))
    print("Fully charged EVs:", int((unmet <= 0.01).sum()))
    print("EVs receiving zero energy:", int((delivered <= 1e-9).sum()))

    print("\n10 EVs with the most unmet energy:")
    print(
        results.sort_values("unmet_kwh", ascending=False)
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    results.to_csv(
        RESULTS_DIR / f"{method.lower()}_ev_energy_audit.csv",
        index=False
    )

analyze_schedule("ga_best_schedule.csv", "GA")
analyze_schedule("pso_best_schedule.csv", "PSO")

print("\nDiagnostic reports saved in:", RESULTS_DIR)
