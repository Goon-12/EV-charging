
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCENARIO_FILE = PROJECT_DIR / "data" / "processed" / "ev_scenario.csv"

scenario = pd.read_csv(SCENARIO_FILE)

required_columns = [
    "ev_id",
    "arrival_hour",
    "departure_hour",
    "arrival_slot",
    "departure_slot",
    "required_energy_kwh",
    "max_power_kw",
]

missing = [c for c in required_columns if c not in scenario.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

print("SCENARIO VALIDATION")
print("-" * 40)
print("Number of EVs:", len(scenario))
print("Duplicate EV IDs:", scenario["ev_id"].duplicated().sum())
print("Missing values:\n", scenario[required_columns].isna().sum())
print("Invalid time windows:",
      (scenario["departure_hour"] <= scenario["arrival_hour"]).sum())
print("Invalid energy requirements:",
      (scenario["required_energy_kwh"] <= 0).sum())
print("Invalid power limits:",
      (scenario["max_power_kw"] <= 0).sum())

print("\nSummary statistics:")
print(
    scenario[
        ["required_energy_kwh", "max_power_kw"]
    ].describe().round(2)
)

# Check whether each EV could receive its required energy
# within its parking window at its individual power limit.
possible_energy = (
    (scenario["departure_hour"] - scenario["arrival_hour"])
    * scenario["max_power_kw"]
)

infeasible = (
    scenario["required_energy_kwh"] > possible_energy + 1e-9
)

print("\nEVs individually infeasible even without competition:",
      int(infeasible.sum()))

if len(scenario) != 100:
    print("\nWARNING: Expected 100 EVs.")

if missing:
    raise ValueError("Fix missing scenario columns before continuing.")

print("\nValidation completed.")
