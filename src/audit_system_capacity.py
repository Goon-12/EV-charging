
from pathlib import Path
import sys
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from load_scenario import load_scenario

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0

df = load_scenario()

arrival = df["arrival_slot"].to_numpy(dtype=int)
departure = df["departure_slot"].to_numpy(dtype=int)
required = df["required_energy_kwh"].to_numpy(dtype=float)
power_limit = df["max_power_kw"].to_numpy(dtype=float)

# Theoretical upper bound: in each slot, select the 20 available EVs
# with the highest power limits, then apply the grid limit.
capacity_by_slot = np.zeros(N_SLOTS)

for t in range(N_SLOTS):
    available = (arrival <= t) & (departure > t)
    available_power = np.sort(power_limit[available])[::-1]
    charger_capacity = available_power[:N_CHARGERS].sum()

    capacity_by_slot[t] = min(charger_capacity, MAX_GRID_KW)


# Individual energy upper bound:
# each EV can charge only during its available time window.
available_hours = np.maximum(
    np.minimum(departure, N_SLOTS) - np.maximum(arrival, 0),
    0
) * STEP_HOURS

individual_capacity = power_limit * available_hours

print("\nINDIVIDUAL CAPACITY CHECK")
print("-" * 40)

print(
    "EVs whose individual capacity is below their request:",
    int((individual_capacity + 1e-9 < required).sum())
)

print(
    "Total individual capacity upper bound:",
    f"{individual_capacity.sum():.2f} kWh"
)

print(
    "Total energy requested:",
    f"{required.sum():.2f} kWh"
)

upper_bound_energy = capacity_by_slot.sum() * STEP_HOURS
total_required = required.sum()

print("\nSYSTEM CAPACITY AUDIT")
print("-" * 40)
print(f"Total energy required: {total_required:.2f} kWh")
print(f"Theoretical capacity upper bound: {upper_bound_energy:.2f} kWh")
print(
    "Minimum possible system-wide shortfall (lower bound): "
    f"{max(0.0, total_required - upper_bound_energy):.2f} kWh"
)
print(
    "Upper-bound energy coverage: "
    f"{100 * min(upper_bound_energy, total_required) / total_required:.1f}%"
)

print("\nNote: This is an optimistic upper bound, not a guaranteed achievable schedule.")
