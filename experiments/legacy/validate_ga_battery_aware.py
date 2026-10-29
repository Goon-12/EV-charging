# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0
TOL = 1e-6

scenario = pd.read_csv(DATA / "ev_scenario.csv")
solar = pd.read_csv(DATA / "solar_profile_36h.csv")
schedule_df = pd.read_csv(
    RESULTS / "ga_battery_aware_best_schedule.csv"
)

columns = [f"slot_{t}" for t in range(N_SLOTS)]
assert len(schedule_df) == len(scenario), "EV count mismatch"
assert len(solar) == N_SLOTS, "Solar profile length mismatch"
assert all(c in schedule_df.columns for c in columns), (
    "Schedule is missing slot columns"
)

P = schedule_df[columns].to_numpy(dtype=float)
solar_kw = solar["solar_power_kw"].to_numpy(dtype=float)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = scenario["arrival_slot"].to_numpy(dtype=int)
departure = scenario["departure_slot"].to_numpy(dtype=int)

assert np.isfinite(P).all(), "Schedule contains NaN or infinity"

delivered = P.sum(axis=1) * STEP_HOURS
unmet = np.maximum(required - delivered, 0.0)
ev_load = P.sum(axis=0)

# Solar is assumed to offset EV load directly.
grid_import = np.maximum(ev_load - solar_kw, 0.0)
active = (P > TOL).sum(axis=0)

grid_violations = int((grid_import > MAX_GRID_KW + TOL).sum())
charger_violations = int((active > N_CHARGERS).sum())
power_violations = int(
    (P > max_power[:, None] + TOL).sum()
)

time_violations = 0
for i in range(len(scenario)):
    valid = np.zeros(N_SLOTS, dtype=bool)
    start = int(np.clip(arrival[i], 0, N_SLOTS))
    end = int(np.clip(departure[i], 0, N_SLOTS))
    valid[start:end] = True
    time_violations += int(
        ((P[i] > TOL) & ~valid).sum()
    )

print("\nBATTERY-AWARE GA: INDEPENDENT VALIDATION")
print("=" * 55)
print(f"EVs: {len(scenario)}")
print(f"Required energy: {required.sum():.3f} kWh")
print(f"Delivered energy: {delivered.sum():.3f} kWh")
print(f"Unmet energy: {unmet.sum():.3f} kWh")
print(f"Demand met: {100 * delivered.sum() / required.sum():.2f}%")
print(f"Fully charged EVs (0.05 kWh tolerance): "
      f"{int((unmet <= 0.05).sum())}")
print(f"Peak EV load: {ev_load.max():.3f} kW")
print(f"Peak grid import after solar: {grid_import.max():.3f} kW")
print(f"Grid violations: {grid_violations}")
print(f"Charger violations: {charger_violations}")
print(f"Individual power violations: {power_violations}")
print(f"Time-window violations: {time_violations}")

passed = all(v == 0 for v in [
    grid_violations,
    charger_violations,
    power_violations,
    time_violations,
])
print(f"All checked constraints passed: {passed}")

if not passed:
    raise SystemExit("Validation failed. Do not report this schedule as feasible.")
