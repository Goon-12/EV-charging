# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

STEP = 0.25
N_SLOTS = 144
MAX_CHARGERS = 20
GRID_LIMIT_KW = 125.0
TOL = 1e-6
ENERGY_TOL = 0.05

scenario = pd.read_csv(DATA / "ev_scenario.csv")
solar_df = pd.read_csv(DATA / "solar_profile_36h.csv")
schedule_df = pd.read_csv(RESULTS / "ga_battery_aware_v3_schedule.csv")

slot_cols = [f"slot_{t}" for t in range(N_SLOTS)]
if len(solar_df) != N_SLOTS:
    raise ValueError(f"Expected {N_SLOTS} solar rows, got {len(solar_df)}")
if len(schedule_df) != len(scenario):
    raise ValueError("Schedule EV count does not match scenario")
missing = [c for c in slot_cols if c not in schedule_df.columns]
if missing:
    raise ValueError(f"Missing schedule columns: {missing[:5]}")

P = schedule_df[slot_cols].to_numpy(dtype=float)
solar = solar_df["solar_power_kw"].to_numpy(dtype=float)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = np.clip(scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS)
departure = np.clip(scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS)

if not np.isfinite(P).all() or not np.isfinite(solar).all():
    raise ValueError("Schedule or solar profile contains NaN/infinity")

energy_by_ev = P.sum(axis=1) * STEP
unmet_by_ev = np.maximum(required - energy_by_ev, 0.0)
ev_load = P.sum(axis=0)
grid_import = np.maximum(ev_load - solar, 0.0)
active = (P > TOL).sum(axis=0)

negative_power_v = int((P < -TOL).sum())
grid_v = int((grid_import > GRID_LIMIT_KW + TOL).sum())
charger_v = int((active > MAX_CHARGERS).sum())
power_v = int((P > max_power[:, None] + TOL).sum())

time_v = 0
for i in range(len(scenario)):
    valid = np.zeros(N_SLOTS, dtype=bool)
    valid[arrival[i]:departure[i]] = True
    time_v += int(((P[i] > TOL) & ~valid).sum())

overcharge_v = int((energy_by_ev > required + ENERGY_TOL).sum())

checks = {
    "negative_power_violations": negative_power_v,
    "grid_import_limit_violations": grid_v,
    "charger_limit_violations": charger_v,
    "individual_power_violations": power_v,
    "time_window_violations": time_v,
    "overcharge_violations": overcharge_v,
}
feasible = all(v == 0 for v in checks.values())

total_required = float(required.sum())
total_delivered = float(energy_by_ev.sum())
total_unmet = float(unmet_by_ev.sum())
solar_available_kwh = float(solar.sum() * STEP)
solar_used_kwh = float(np.minimum(ev_load, solar).sum() * STEP)

print("\nBATTERY-AWARE GA V3 — INDEPENDENT VALIDATION")
print("=" * 58)
print(f"EV count: {len(scenario)}")
print(f"Required energy: {total_required:.3f} kWh")
print(f"Delivered energy: {total_delivered:.3f} kWh")
print(f"Unmet energy: {total_unmet:.3f} kWh")
print(f"Energy demand met: {100*total_delivered/total_required:.2f}%")
print(f"Fully charged EVs (<= {ENERGY_TOL:.2f} kWh unmet): "
      f"{int((unmet_by_ev <= ENERGY_TOL).sum())}")
print(f"Peak total EV load: {ev_load.max():.3f} kW")
print(f"Peak grid import after solar: {grid_import.max():.3f} kW")
print(f"Grid-import limit: {GRID_LIMIT_KW:.3f} kW")
print(f"Solar available: {solar_available_kwh:.3f} kWh")
print(f"Solar used: {solar_used_kwh:.3f} kWh")
print(f"Solar utilization: {100*solar_used_kwh/max(solar_available_kwh, 1e-12):.2f}%")

print("\nCONSTRAINT AUDIT")
for name, value in checks.items():
    print(f"{name}: {value}")
print(f"All checked constraints passed: {feasible}")

# Diagnose EVs whose individual requested energy exceeds the theoretical
# power-energy bound within their own parking windows.
window_slots = np.maximum(departure - arrival, 0)
individual_upper = np.minimum(
    required,
    max_power * window_slots * STEP,
)
individually_impossible = int(
    (max_power * window_slots * STEP < required - ENERGY_TOL).sum()
)
print("\nCAPACITY DIAGNOSTICS")
print(f"EVs whose individual window/power bound is below requested energy: "
      f"{individually_impossible}")
print(f"Sum of individual theoretical energy upper bounds: "
      f"{individual_upper.sum():.3f} kWh")
print("Note: this sum is an upper bound; it does not account for shared")
print("charger/grid contention and does not prove the bound is achievable.")

if not feasible:
    raise SystemExit("Validation failed; do not report v3 as feasible.")
