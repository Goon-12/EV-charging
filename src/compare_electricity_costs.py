
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144
MAX_GRID_KW = 125.0

scenario = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "ev_scenario.csv"
)

required_by_ev = scenario["required_energy_kwh"].to_numpy(dtype=float)
total_required = float(required_by_ev.sum())

slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]

# Illustrative synthetic tariff; not actual electricity prices.
def illustrative_tariff(slot_hours):
    hour = slot_hours % 24
    return np.select(
        [
            (hour >= 0) & (hour < 6),
            (hour >= 17) & (hour < 22),
        ],
        [4.0, 10.0],
        default=6.0,
    ).astype(float)


slot_hours = np.arange(N_SLOTS) * STEP_HOURS
prices = illustrative_tariff(slot_hours)

schedule_files = {
    "Uncontrolled": "uncontrolled_schedule.csv",
    "FCFS": "fcfs_schedule.csv",
    "GA": "ga_best_schedule.csv",
    "PSO": "pso_best_schedule.csv",
    "Feasibility-first": "feasibility_baseline_schedule.csv",
}

rows = []

for method, filename in schedule_files.items():
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required schedule not found: {path}")

    df = pd.read_csv(path)

    if len(df) != len(scenario):
        raise ValueError(f"EV count mismatch in {filename}")

    if not np.array_equal(
        df["ev_id"].to_numpy(),
        scenario["ev_id"].to_numpy()
    ):
        raise ValueError(f"EV ID/order mismatch in {filename}")

    missing = [c for c in slot_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing schedule columns in {filename}: {missing[:5]}")

    schedule = df[slot_columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(schedule)) or np.any(schedule < 0):
        raise ValueError(f"Invalid power values in {filename}")

    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required_by_ev - delivered_by_ev, 0.0)

    delivered = float(delivered_by_ev.sum())
    unmet = float(unmet_by_ev.sum())
    grid_load = schedule.sum(axis=0)
    energy_by_slot = grid_load * STEP_HOURS

    cost = float(np.dot(energy_by_slot, prices))

    rows.append({
        "method": method,
        "required_energy_kwh": total_required,
        "delivered_energy_kwh": delivered,
        "unmet_energy_kwh": unmet,
        "energy_demand_met_pct": (
            100.0 * delivered / total_required
            if total_required > 0 else np.nan
        ),
        "fully_charged_evs": int((unmet_by_ev <= 0.05).sum()),
        "peak_grid_kw": float(grid_load.max()),
        "grid_limit_violations": int(
            (grid_load > MAX_GRID_KW + 1e-9).sum()
        ),
        "simulated_cost_units": cost,
        "average_cost_per_delivered_kwh": (
            cost / delivered if delivered > 0 else np.nan
        ),
    })

results = pd.DataFrame(rows)

output_csv = RESULTS_DIR / "electricity_cost_comparison.csv"
results.to_csv(output_csv, index=False)

print("\nCOST AND CHARGING PERFORMANCE COMPARISON")
print("=" * 115)
print(results.round(3).to_string(index=False))
print("\nTariff: illustrative synthetic time-of-use prices.")
print("Cost values are simulated units, not actual electricity bills.")
print(f"CSV saved to: {output_csv}")

# Plot simulated cost.
plt.figure(figsize=(10, 5))
plt.bar(results["method"], results["simulated_cost_units"])
plt.ylabel("Simulated cost (tariff units)")
plt.title("Electricity Cost Under an Illustrative Tariff")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

cost_plot = RESULTS_DIR / "electricity_cost_comparison.png"
plt.savefig(cost_plot, dpi=150)
plt.close()
print(f"Cost plot saved to: {cost_plot}")

# Plot energy demand met.
plt.figure(figsize=(10, 5))
plt.bar(results["method"], results["energy_demand_met_pct"])
plt.ylabel("Energy demand met (%)")
plt.title("Charging Demand Fulfilment by Method")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

coverage_plot = RESULTS_DIR / "energy_demand_coverage_comparison.png"
plt.savefig(coverage_plot, dpi=150)
plt.close()
print(f"Coverage plot saved to: {coverage_plot}")

