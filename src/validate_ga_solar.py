
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0
TOL = 1e-8

scenario = pd.read_csv(DATA_DIR / "ev_scenario.csv")
solar = pd.read_csv(DATA_DIR / "solar_profile_36h.csv")
schedule_df = pd.read_csv(RESULTS_DIR / "ga_solar_best_schedule.csv")

slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]

assert len(solar) == N_SLOTS, "Solar profile length mismatch."
assert len(schedule_df) == len(scenario), "EV count mismatch."
assert all(c in schedule_df.columns for c in slot_columns), (
    "Schedule is missing slot columns."
)

schedule = schedule_df[slot_columns].to_numpy(dtype=float)
solar_kw = solar["solar_power_kw"].to_numpy(dtype=float)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = scenario["arrival_slot"].to_numpy(dtype=int)
departure = scenario["departure_slot"].to_numpy(dtype=int)

delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)

ev_load_kw = schedule.sum(axis=0)
solar_used_kw = np.minimum(ev_load_kw, solar_kw)
grid_import_kw = np.maximum(ev_load_kw - solar_kw, 0.0)
active_count = (schedule > TOL).sum(axis=0)

grid_violations = int((grid_import_kw > MAX_GRID_KW + TOL).sum())
charger_violations = int((active_count > N_CHARGERS).sum())
power_violations = int(
    (schedule > max_power[:, None] + TOL).sum()
)

time_violations = 0
for i in range(len(scenario)):
    valid = np.zeros(N_SLOTS, dtype=bool)
    start = int(np.clip(arrival[i], 0, N_SLOTS))
    end = int(np.clip(departure[i], 0, N_SLOTS))
    valid[start:end] = True
    time_violations += int(
        ((schedule[i] > TOL) & ~valid).sum()
    )

total_required = float(required.sum())
total_delivered = float(delivered_by_ev.sum())
total_unmet = float(unmet_by_ev.sum())
solar_available_kwh = float(solar_kw.sum() * STEP_HOURS)
solar_used_kwh = float(solar_used_kw.sum() * STEP_HOURS)

print("\nINDEPENDENT SOLAR-AWARE GA VALIDATION")
print("=" * 55)
print(f"EVs: {len(scenario)}")
print(f"Required energy: {total_required:.3f} kWh")
print(f"Delivered energy: {total_delivered:.3f} kWh")
print(f"Unmet energy: {total_unmet:.3f} kWh")
print(f"Energy demand met: {100 * total_delivered / total_required:.2f}%")
print(f"Fully charged EVs (0.05 kWh tolerance): "
      f"{int((unmet_by_ev <= 0.05).sum())}")
print(f"Peak EV charging load: {ev_load_kw.max():.3f} kW")
print(f"Peak grid import after solar: {grid_import_kw.max():.3f} kW")
print(f"Solar used: {solar_used_kwh:.3f} kWh")
print(f"Solar utilization: "
      f"{100 * solar_used_kwh / solar_available_kwh:.2f}%")
print(f"Grid-limit violations: {grid_violations}")
print(f"Charger-limit violations: {charger_violations}")
print(f"Individual power violations: {power_violations}")
print(f"Time-window violations: {time_violations}")

passed = all(x == 0 for x in [
    grid_violations,
    charger_violations,
    power_violations,
    time_violations,
])
print(f"All checked hard constraints passed: {passed}")

if not passed:
    raise SystemExit("Validation failed: inspect the saved schedule.")
