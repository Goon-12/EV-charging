
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
power = df["max_power_kw"].to_numpy(dtype=float)

starts = np.clip(arrival, 0, N_SLOTS)
ends = np.clip(departure, 0, N_SLOTS)
available_slots = np.maximum(ends - starts, 0)

# Upper bound on each EV's energy within its own time window.
individual_capacity = available_slots * STEP_HOURS * power
individual_shortfall = np.maximum(required - individual_capacity, 0.0)

# For each slot, estimate the maximum power available from
# the 20 eligible EVs with the largest power limits.
capacity_by_slot = np.zeros(N_SLOTS)

for t in range(N_SLOTS):
    eligible = (starts <= t) & (ends > t)
    available_power = np.sort(power[eligible])[::-1]
    charger_capacity = available_power[:N_CHARGERS].sum()
    capacity_by_slot[t] = min(charger_capacity, MAX_GRID_KW)

total_required = required.sum()
total_individual_capacity = np.minimum(
    required, individual_capacity
).sum()
system_upper_bound = capacity_by_slot.sum() * STEP_HOURS

print("\nSCHEDULE CAPACITY AUDIT")
print("-" * 45)
print(f"EVs: {len(df)}")
print(f"Total required energy: {total_required:.3f} kWh")
print(
    "EVs individually unable to meet their request: "
    f"{int((individual_shortfall > 1e-8).sum())}"
)
print(
    "Individual-window energy upper bound: "
    f"{total_individual_capacity:.3f} kWh"
)
print(f"System slot-capacity upper bound: {system_upper_bound:.3f} kWh")
print(
    "Optimistic system shortfall lower bound: "
    f"{max(0.0, total_required - system_upper_bound):.3f} kWh"
)
print(
    "\nImportant: these are upper bounds, not proof that a schedule "
    "achieving them exists."
)
