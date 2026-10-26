
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
SCENARIO_PATH = PROJECT_DIR / "data" / "processed" / "ev_scenario.csv"

STEP_HOURS = 0.25
MAX_GRID_KW = 125.0
TOLERANCE_KWH = 0.05

scenario = pd.read_csv(SCENARIO_PATH)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)

def summarize_schedule(name, filepath):
    """Calculate comparison metrics directly from a saved schedule."""
    schedule_df = pd.read_csv(filepath)
    columns = [f"slot_{t}" for t in range(144)]

    missing = [c for c in columns if c not in schedule_df.columns]
    if missing:
        raise ValueError(f"{filepath.name} is missing columns: {missing[:5]}")

    if len(schedule_df) != len(scenario):
        raise ValueError(f"EV count mismatch in {filepath.name}")

    if not (schedule_df["ev_id"].to_numpy() == scenario["ev_id"].to_numpy()).all():
        raise ValueError(f"EV ID/order mismatch in {filepath.name}")

    power = schedule_df[columns].to_numpy(dtype=float)
    delivered_by_ev = power.sum(axis=1) * STEP_HOURS
    unmet_by_ev = (required - delivered_by_ev).clip(min=0)
    grid_load = power.sum(axis=0)
    active = (power > 1e-9).sum(axis=0)

    return {
        "method": name,
        "peak_grid_kw": float(grid_load.max()),
        "total_energy_kwh": float(delivered_by_ev.sum()),
        "total_unmet_energy_kwh": float(unmet_by_ev.sum()),
        "evs_fully_charged": int((unmet_by_ev <= TOLERANCE_KWH).sum()),
        "grid_limit_violations": int((grid_load > MAX_GRID_KW + 1e-9).sum()),
        "charger_limit_violations": int((active > 20).sum()),
    }


# Existing baseline results.
baseline_path = RESULTS_DIR / "baseline_metrics.csv"
baseline = pd.read_csv(baseline_path)

baseline_rows = baseline[
    baseline["method"].isin(["Uncontrolled", "FCFS"])
][[
    "method", "peak_grid_kw", "total_energy_kwh",
    "total_unmet_energy_kwh", "evs_fully_charged"
]].copy()

# Preserve unknown violation counts as missing rather than inventing values.
baseline_rows["grid_limit_violations"] = float("nan")
baseline_rows["charger_limit_violations"] = float("nan")

baseline_rows = baseline_rows.to_dict("records")

# Recalculate all optimized schedules from their saved CSV files.
schedule_rows = [
    summarize_schedule("GA", RESULTS_DIR / "ga_best_schedule.csv"),
    summarize_schedule("PSO", RESULTS_DIR / "pso_best_schedule.csv"),
    summarize_schedule(
        "Feasibility-first",
        RESULTS_DIR / "feasibility_baseline_schedule.csv"
    ),
]

comparison = pd.DataFrame(baseline_rows + schedule_rows)

comparison = comparison.round(3)

output_csv = RESULTS_DIR / "algorithm_comparison.csv"
comparison.to_csv(output_csv, index=False)

print("\nALGORITHM COMPARISON")
print("=" * 100)
print(comparison.to_string(index=False))
print(f"\nComparison CSV saved to: {output_csv}")

# Plot peak demand. The dashed line is the configured 125 kW limit.
plt.figure(figsize=(10, 5))
plt.bar(comparison["method"], comparison["peak_grid_kw"])
plt.axhline(
    MAX_GRID_KW,
    linestyle="--",
    label=f"{MAX_GRID_KW:.0f} kW grid limit"
)
plt.ylabel("Peak grid demand (kW)")
plt.title("Peak Grid Demand Comparison")
plt.xticks(rotation=15)
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plot_path = RESULTS_DIR / "algorithm_peak_comparison.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Comparison graph saved to: {plot_path}")

