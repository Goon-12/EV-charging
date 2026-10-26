
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

STEP = 0.25
SLOTS = 144
MAX_CHARGERS = 20
GRID_LIMIT = 125.0
TOL = 1e-6

scenario = pd.read_csv(DATA / "ev_scenario.csv")
solar = pd.read_csv(DATA / "solar_profile_36h.csv")
df = pd.read_csv(RESULTS / "ga_battery_aware_v2_schedule.csv")

cols = [f"slot_{t}" for t in range(SLOTS)]
P = df[cols].to_numpy(dtype=float)
solar_kw = solar["solar_power_kw"].to_numpy(dtype=float)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = scenario["arrival_slot"].to_numpy(dtype=int)
departure = scenario["departure_slot"].to_numpy(dtype=int)

assert P.shape == (len(scenario), SLOTS), "Schedule shape mismatch"
assert len(solar_kw) == SLOTS, "Solar length mismatch"
assert np.isfinite(P).all(), "Schedule has invalid values"
assert (P >= -TOL).all(), "Negative charging power found"

delivered = P.sum(axis=1) * STEP
unmet = np.maximum(required - delivered, 0)
ev_load = P.sum(axis=0)
grid_import = np.maximum(ev_load - solar_kw, 0)
active = (P > TOL).sum(axis=0)

grid_v = int((grid_import > GRID_LIMIT + TOL).sum())
charger_v = int((active > MAX_CHARGERS).sum())
power_v = int((P > max_power[:, None] + TOL).sum())

time_v = 0
for i in range(len(scenario)):
    valid = np.zeros(SLOTS, dtype=bool)
    start = int(np.clip(arrival[i], 0, SLOTS))
    end = int(np.clip(departure[i], 0, SLOTS))
    valid[start:end] = True
    time_v += int(((P[i] > TOL) & ~valid).sum())

# Detect if any EV receives more energy than requested.
overcharge_v = int(
    (delivered > required + 0.05).sum()
)

print("\nBATTERY-AWARE GA V2 VALIDATION")
print("=" * 48)
print(f"Required energy: {required.sum():.3f} kWh")
print(f"Delivered energy: {delivered.sum():.3f} kWh")
print(f"Unmet energy: {unmet.sum():.3f} kWh")
print(f"Demand met: {100 * delivered.sum() / required.sum():.2f}%")
print(f"Fully charged EVs: {int((unmet <= 0.05).sum())}")
print(f"Peak EV load: {ev_load.max():.3f} kW")
print(f"Peak grid import: {grid_import.max():.3f} kW")
print(f"Grid violations: {grid_v}")
print(f"Charger violations: {charger_v}")
print(f"Individual power violations: {power_v}")
print(f"Time-window violations: {time_v}")
print(f"EV overcharge violations: {overcharge_v}")

passed = all(v == 0 for v in [
    grid_v, charger_v, power_v, time_v, overcharge_v
])
print(f"All checked constraints passed: {passed}")

if not passed:
    raise SystemExit("Validation failed.")
