# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0
TOLERANCE_KWH = 0.05

SCENARIO_FILE = DATA_DIR / "ev_scenario.csv"
scenario = pd.read_csv(SCENARIO_FILE)

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = scenario["arrival_slot"].to_numpy(dtype=int)
departure = scenario["departure_slot"].to_numpy(dtype=int)

total_required = float(required.sum())
slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]

# Keep the same illustrative tariff for every schedule.
# These are synthetic tariff units, not actual electricity prices.
slot_hours = np.arange(N_SLOTS) * STEP_HOURS
hour_of_day = slot_hours % 24

prices = np.select(
    [
        hour_of_day < 6,
        (hour_of_day >= 17) & (hour_of_day < 22),
    ],
    [4.0, 10.0],
    default=6.0,
).astype(float)

schedule_files = {
    "Uncontrolled": "uncontrolled_schedule.csv",
    "FCFS": "fcfs_schedule.csv",
    "Original GA": "ga_best_schedule.csv",
    "PSO": "pso_best_schedule.csv",
    "Feasibility-first": "feasibility_baseline_schedule.csv",
    "Cost-aware GA": "ga_cost_best_schedule_latest.csv",
}


def evaluate_schedule_file(method, filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing schedule for {method}: {path}\n"
            "Check that the relevant schedule was saved."
        )

    df = pd.read_csv(path)

    if len(df) != len(scenario):
        raise ValueError(f"EV count mismatch in {filename}")

    if not np.array_equal(
        df["ev_id"].to_numpy(),
        scenario["ev_id"].to_numpy(),
    ):
        raise ValueError(f"EV IDs/order mismatch in {filename}")

    missing = [c for c in slot_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing slot columns in {filename}: {missing[:5]}"
        )

    schedule = df[slot_columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(schedule)) or np.any(schedule < -1e-9):
        raise ValueError(f"Invalid power values in {filename}")

    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    delivered = float(delivered_by_ev.sum())
    unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)
    unmet = float(unmet_by_ev.sum())

    grid_load = schedule.sum(axis=0)
    peak = float(grid_load.max())
    cost = float(np.dot(grid_load * STEP_HOURS, prices))

    grid_violations = int(
        (grid_load > MAX_GRID_KW + 1e-9).sum()
    )
    charger_violations = int(
        ((schedule > 1e-9).sum(axis=0) > N_CHARGERS).sum()
    )
    power_violations = int(
        (schedule > max_power[:, None] + 1e-9).sum()
    )

    time_violations = 0
    for i in range(len(scenario)):
        invalid_before = schedule[i, :max(0, arrival[i])]
        invalid_after = schedule[i, max(0, departure[i]):]
        time_violations += int((invalid_before > 1e-9).sum())
        time_violations += int((invalid_after > 1e-9).sum())

    violations = (
        grid_violations
        + charger_violations
        + power_violations
        + time_violations
    )

    return {
        "method": method,
        "required_energy_kwh": total_required,
        "delivered_energy_kwh": delivered,
        "unmet_energy_kwh": unmet,
        "energy_demand_met_pct": (
            100.0 * delivered / total_required
            if total_required > 0 else np.nan
        ),
        "fully_charged_evs": int(
            (unmet_by_ev <= TOLERANCE_KWH).sum()
        ),
        "peak_grid_kw": peak,
        "simulated_cost_units": cost,
        "average_cost_per_delivered_kwh": (
            cost / delivered if delivered > 0 else np.nan
        ),
        "grid_limit_violations": grid_violations,
        "charger_limit_violations": charger_violations,
        "individual_power_violations": power_violations,
        "time_window_violations": time_violations,
        "total_constraint_violations": violations,
        "feasible": violations == 0,
    }


results = pd.DataFrame([
    evaluate_schedule_file(method, filename)
    for method, filename in schedule_files.items()
])

results = results.round(3)

csv_path = RESULTS_DIR / "final_algorithm_comparison.csv"
results.to_csv(csv_path, index=False)

print("\nFINAL ALGORITHM COMPARISON")
print("=" * 120)
print(results.to_string(index=False))
print(f"\nComparison CSV saved to: {csv_path}")
print("\nTariff is illustrative; cost values are synthetic units.")
print("Feasibility is based on the four checked hard constraints.")

# Plot 1: peak demand
plt.figure(figsize=(11, 5))
plt.bar(results["method"], results["peak_grid_kw"])
plt.axhline(
    MAX_GRID_KW,
    linestyle="--",
    label=f"Grid limit ({MAX_GRID_KW:.0f} kW)",
)
plt.ylabel("Peak demand (kW)")
plt.title("Peak Grid Demand by Charging Method")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_peak_demand.png", dpi=200)
plt.close()

# Plot 2: unmet energy
plt.figure(figsize=(11, 5))
plt.bar(results["method"], results["unmet_energy_kwh"])
plt.ylabel("Unmet energy (kWh)")
plt.title("Unmet Charging Energy by Method")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_unmet_energy.png", dpi=200)
plt.close()

# Plot 3: illustrative electricity cost
plt.figure(figsize=(11, 5))
plt.bar(results["method"], results["simulated_cost_units"])
plt.ylabel("Simulated cost (tariff units)")
plt.title("Illustrative Electricity Cost by Method")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_electricity_cost.png", dpi=200)
plt.close()

# Plot 4: service vs peak demand
plt.figure(figsize=(9, 6))
for _, row in results.iterrows():
    plt.scatter(row["peak_grid_kw"], row["energy_demand_met_pct"], s=65)
    plt.annotate(
        row["method"],
        (row["peak_grid_kw"], row["energy_demand_met_pct"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

plt.xlabel("Peak grid demand (kW)")
plt.ylabel("Energy demand met (%)")
plt.title("Charging Service vs Peak Grid Demand")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "final_service_vs_peak.png", dpi=200)
plt.close()

print("\nGraphs saved:")
print(RESULTS_DIR / "final_peak_demand.png")
print(RESULTS_DIR / "final_unmet_energy.png")
print(RESULTS_DIR / "final_electricity_cost.png")
print(RESULTS_DIR / "final_service_vs_peak.png")
