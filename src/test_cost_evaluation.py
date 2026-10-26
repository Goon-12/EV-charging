
from pathlib import Path
import sys
import numpy as np

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from evaluate_cost import evaluate_cost

# Two EVs, each with two 15-minute time slots.
# Aggregate power:
# Slot 0: 4 kW + 0 kW = 4 kW
# Slot 1: 0 kW + 2 kW = 2 kW
schedule = np.array([
    [4.0, 0.0],
    [0.0, 2.0],
])

step_hours = 0.25

# Synthetic example tariff in currency units per kWh.
prices = np.array([5.0, 10.0])

# Expected energy:
# Slot 0: 4 kW * 0.25 h = 1 kWh
# Slot 1: 2 kW * 0.25 h = 0.5 kWh
#
# Expected cost:
# 1 * 5 + 0.5 * 10 = 10 currency units.
expected_cost = 10.0

actual_cost = evaluate_cost(
    schedule=schedule,
    price_per_kwh=prices,
    step_hours=step_hours,
)

print("COST EVALUATION TEST")
print("-" * 35)
print(f"Expected cost: {expected_cost:.2f}")
print(f"Calculated cost: {actual_cost:.2f}")

if np.isclose(actual_cost, expected_cost):
    print("PASS: Cost calculation is correct.")
else:
    raise AssertionError("Cost evaluation did not match expected result.")
