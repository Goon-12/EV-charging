
from pathlib import Path
import sys
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from load_scenario import load_scenario

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20

df = load_scenario()

arrival = df["arrival_slot"].to_numpy(dtype=int)
departure = df["departure_slot"].to_numpy(dtype=int)
power = df["max_power_kw"].to_numpy(dtype=float)
required = float(df["required_energy_kwh"].sum())

print("\nGRID CAPACITY SENSITIVITY")
print("-" * 60)
print(f"Total energy required: {required:.2f} kWh\n")
print(f"{'Grid limit (kW)':>16} {'Capacity bound (kWh)':>22} {'Coverage':>12}")

for grid_limit in [100, 125, 150, 175, 200]:
    capacity = 0.0

    for t in range(N_SLOTS):
        available = (arrival <= t) & (departure > t)
        available_power = np.sort(power[available])[::-1]
        charger_capacity = available_power[:N_CHARGERS].sum()

        capacity += min(charger_capacity, grid_limit) * STEP_HOURS

    coverage = min(capacity, required) / required * 100

    print(f"{grid_limit:16.0f} {capacity:22.2f} {coverage:11.1f}%")

print("\nThese are optimistic upper bounds, not achieved schedule results.")
