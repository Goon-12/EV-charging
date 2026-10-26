
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

scenario = pd.read_csv(ROOT / "data" / "processed" / "ev_scenario.csv")
schedule_df = pd.read_csv(ROOT / "results" / "pso_best_schedule.csv")

slots = [f"slot_{i}" for i in range(144)]

scenario["ev_id"] = scenario["ev_id"].astype(str)
schedule_df["ev_id"] = schedule_df["ev_id"].astype(str)

df = scenario.merge(
    schedule_df[["ev_id"] + slots],
    on="ev_id",
    how="inner",
    validate="one_to_one"
)

schedule = df[slots].to_numpy(dtype=float)
delivered = schedule.sum(axis=1) * 0.25
required = df["required_energy_kwh"].to_numpy(dtype=float)

overcharged = np.maximum(delivered - required, 0)
unmet = np.maximum(required - delivered, 0)

print("PSO ENERGY ACCOUNTING AUDIT")
print("-" * 40)
print(f"Required energy: {required.sum():.3f} kWh")
print(f"Delivered energy: {delivered.sum():.3f} kWh")
print(f"Unmet energy: {unmet.sum():.3f} kWh")
print(f"Energy delivered beyond requirements: {overcharged.sum():.3f} kWh")
print(f"EVs receiving excess energy: {(overcharged > 0.01).sum()}")

df["delivered_kwh"] = delivered
df["overcharged_kwh"] = overcharged
df["unmet_kwh"] = unmet

print("\nLargest overcharging cases:")
print(
    df.nlargest(10, "overcharged_kwh")[
        ["ev_id", "required_energy_kwh", "delivered_kwh",
         "overcharged_kwh", "unmet_kwh"]
    ].to_string(index=False)
)
