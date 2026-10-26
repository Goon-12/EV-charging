
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from load_scenario import load_scenario

STEP_HOURS = 0.25
scenario = load_scenario().copy()

# Maximum energy each EV could receive within its available window,
# ignoring competition from other EVs and the shared grid limit.
window_slots = (
    scenario["departure_slot"] - scenario["arrival_slot"]
).clip(lower=0)

scenario["available_hours"] = window_slots * STEP_HOURS
scenario["max_possible_energy_kwh"] = (
    scenario["max_power_kw"] * scenario["available_hours"]
)

scenario["individually_feasible"] = (
    scenario["max_possible_energy_kwh"]
    >= scenario["required_energy_kwh"] - 1e-9
)

scenario["energy_shortfall_if_alone_kwh"] = np.maximum(
    scenario["required_energy_kwh"]
    - scenario["max_possible_energy_kwh"],
    0.0
)

print("\nSCENARIO FEASIBILITY AUDIT")
print("-" * 40)
print("Number of EVs:", len(scenario))
print("Total required energy (kWh):",
      round(scenario["required_energy_kwh"].sum(), 2))
print("Individually feasible EVs:",
      int(scenario["individually_feasible"].sum()))
print("Individually infeasible EVs:",
      int((~scenario["individually_feasible"]).sum()))
print("Total unavoidable individual shortfall (kWh):",
      round(scenario["energy_shortfall_if_alone_kwh"].sum(), 2))

print("\nIndividually infeasible EVs:")
print(
    scenario.loc[
        ~scenario["individually_feasible"],
        [
            "ev_id",
            "available_hours",
            "required_energy_kwh",
            "max_power_kw",
            "max_possible_energy_kwh",
            "energy_shortfall_if_alone_kwh",
        ],
    ].to_string(index=False)
)

output = PROJECT_DIR / "results" / "scenario_feasibility_audit.csv"
scenario.to_csv(output, index=False)
print("\nDetailed audit saved to:", output)
