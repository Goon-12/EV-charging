
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

STEP_HOURS = 0.25
N_SLOTS = 144
MAX_GRID_KW = 125.0

scenario = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "ev_scenario.csv"
)

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
total_required = float(required.sum())

def tariff(slot_hours):
    hour = slot_hours % 24
    return np.select(
        [(hour < 6), ((hour >= 17) & (hour < 22))],
        [4.0, 10.0],
        default=6.0,
    )

prices = tariff(np.arange(N_SLOTS) * STEP_HOURS)

files = {
    "Original GA": "ga_best_schedule.csv",
    "Cost-aware GA": "ga_cost_best_schedule.csv",
}

rows = []

for method, filename in files.items():
    df = pd.read_csv(RESULTS_DIR / filename)
    columns = [f"slot_{t}" for t in range(N_SLOTS)]

    if len(df) != len(scenario):
        raise ValueError(f"EV count mismatch: {filename}")

    if not np.array_equal(
        df["ev_id"].to_numpy(),
        scenario["ev_id"].to_numpy(),
    ):
        raise ValueError(f"EV IDs/order mismatch: {filename}")

    schedule = df[columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(schedule)) or np.any(schedule < 0):
        raise ValueError(f"Invalid power values: {filename}")

    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    delivered = float(delivered_by_ev.sum())
    unmet = float(np.maximum(required - delivered_by_ev, 0).sum())

    load = schedule.sum(axis=0)
    cost = float(np.dot(load * STEP_HOURS, prices))

    rows.append({
        "method": method,
        "delivered_kwh": delivered,
        "unmet_kwh": unmet,
        "demand_met_pct": 100 * delivered / total_required,
        "fully_charged_evs": int(
            (required - delivered_by_ev <= 0.05).sum()
        ),
        "peak_grid_kw": float(load.max()),
        "simulated_cost_units": cost,
        "grid_limit_violations": int(
            (load > MAX_GRID_KW + 1e-9).sum()
        ),
    })

comparison = pd.DataFrame(rows).round(3)

output = RESULTS_DIR / "ga_version_comparison.csv"
comparison.to_csv(output, index=False)

print("\nORIGINAL GA VS COST-AWARE GA")
print("=" * 100)
print(comparison.to_string(index=False))
print(f"\nSaved to: {output}")
print("\nTariff is illustrative; cost values are simulated units.")
