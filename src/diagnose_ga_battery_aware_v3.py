
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

STEP = 0.25
SLOTS = 144

scenario = pd.read_csv(DATA / "ev_scenario.csv")
schedule_df = pd.read_csv(
    RESULTS / "ga_battery_aware_v3_schedule.csv"
)

cols = [f"slot_{t}" for t in range(SLOTS)]
P = schedule_df[cols].to_numpy(dtype=float)

required = scenario["required_energy_kwh"].to_numpy(float)
max_power = scenario["max_power_kw"].to_numpy(float)
arrival = scenario["arrival_slot"].to_numpy(int)
departure = scenario["departure_slot"].to_numpy(int)

delivered = P.sum(axis=1) * STEP
unmet = np.maximum(required - delivered, 0.0)
window_slots = np.maximum(departure - arrival, 0)

individual_upper = np.minimum(
    required,
    max_power * window_slots * STEP,
)

report = scenario.copy()
report["delivered_kwh"] = delivered
report["unmet_kwh"] = unmet
report["demand_met_pct"] = np.where(
    required > 0,
    100 * delivered / required,
    100.0,
)
report["individual_upper_bound_kwh"] = individual_upper
report["individual_shortfall_bound_kwh"] = np.maximum(
    required - individual_upper, 0.0
)
report["window_hours"] = window_slots * STEP

report = report.sort_values("unmet_kwh", ascending=False)

output = RESULTS / "ga_battery_aware_v3_ev_diagnostics.csv"
report.to_csv(output, index=False)

print("\nV3 EV-LEVEL DIAGNOSTICS")
print("=" * 60)
print("Total unmet energy:", round(unmet.sum(), 3), "kWh")
print("EVs with > 0.05 kWh unmet:", int((unmet > 0.05).sum()))
print("EVs individually unable to meet request:",
      int((individual_upper < required - 0.05).sum()))

print("\n10 EVs with the most unmet energy:")
columns = [
    "ev_id", "required_energy_kwh", "delivered_kwh",
    "unmet_kwh", "demand_met_pct", "window_hours",
    "individual_upper_bound_kwh",
]
print(report[columns].head(10).round(3).to_string(index=False))

print("\nSaved diagnostics to:", output)
