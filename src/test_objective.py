
from pathlib import Path
import sys
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from evaluate_objective import evaluate_objective

# Two EVs and two 15-minute charging slots.
schedule = np.array([
    [4.0, 0.0],
    [0.0, 2.0],
])

required_energy = np.array([1.0, 0.5])
prices = np.array([5.0, 10.0])

result = evaluate_objective(
    schedule=schedule,
    required_energy_kwh=required_energy,
    price_per_kwh=prices,
    step_hours=0.25,
    max_grid_kw=125.0,
)

# Both EVs receive exactly their required energy.
# Slot 0 delivers 4 kW * 0.25 h = 1 kWh.
# Slot 1 delivers 2 kW * 0.25 h = 0.5 kWh.
expected_cost = 10.0
expected_unmet = 0.0
expected_peak = 4.0

print("OBJECTIVE FUNCTION TEST")
print("-" * 40)

for key, value in result.items():
    print(f"{key}: {value:.6f}")

assert np.isclose(
    result["electricity_cost_units"], expected_cost
), "Electricity cost calculation failed."

assert np.isclose(
    result["unmet_energy_kwh"], expected_unmet
), "Unmet energy calculation failed."

assert np.isclose(
    result["peak_grid_kw"], expected_peak
), "Peak demand calculation failed."

assert np.isfinite(result["objective"]), "Objective is not finite."

print("\nPASS: All objective calculations are correct.")
