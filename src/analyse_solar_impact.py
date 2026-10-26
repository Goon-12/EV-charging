
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144

scenario = pd.read_csv(DATA_DIR / "ev_scenario.csv")
solar_df = pd.read_csv(DATA_DIR / "solar_profile_36h.csv")

required_columns = {"solar_power_kw"}
if not required_columns.issubset(solar_df.columns):
    raise ValueError("Solar profile must contain solar_power_kw.")

solar_kw = solar_df["solar_power_kw"].to_numpy(dtype=float)

if len(solar_kw) != N_SLOTS:
    raise ValueError(f"Expected {N_SLOTS} solar slots, found {len(solar_kw)}.")

if not np.isfinite(solar_kw).all() or np.any(solar_kw < 0):
    raise ValueError("Solar profile contains invalid power values.")

slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]

schedule_files = {
    "Uncontrolled": "uncontrolled_schedule.csv",
    "FCFS": "fcfs_schedule.csv",
    "Original GA": "ga_best_schedule.csv",
    "PSO": "pso_best_schedule.csv",
    "Feasibility-first": "feasibility_baseline_schedule.csv",
    "Cost-aware GA": "ga_cost_best_schedule_latest.csv",
}

results = []
profiles = {}

for method, filename in schedule_files.items():
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Skipping {method}: missing {filename}")
        continue

    df = pd.read_csv(path)

    if len(df) != len(scenario):
        raise ValueError(f"EV count mismatch in {filename}.")

    missing = [c for c in slot_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing slot columns.")

    ev_power = df[slot_columns].to_numpy(dtype=float)

    if not np.isfinite(ev_power).all() or np.any(ev_power < -1e-9):
        raise ValueError(f"Invalid power values in {filename}.")

    # Aggregate EV charging demand in each slot.
    ev_load_kw = ev_power.sum(axis=0)

    # Solar is used locally before importing energy from the grid.
    solar_used_kw = np.minimum(ev_load_kw, solar_kw)
    grid_import_kw = np.maximum(ev_load_kw - solar_kw, 0.0)
    excess_solar_kw = np.maximum(solar_kw - ev_load_kw, 0.0)

    # Convert slot power to energy.
    ev_energy_kwh = float(ev_load_kw.sum() * STEP_HOURS)
    solar_available_kwh = float(solar_kw.sum() * STEP_HOURS)
    solar_used_kwh = float(solar_used_kw.sum() * STEP_HOURS)
    grid_import_kwh = float(grid_import_kw.sum() * STEP_HOURS)
    excess_solar_kwh = float(excess_solar_kw.sum() * STEP_HOURS)

    solar_fraction = (
        100.0 * solar_used_kwh / ev_energy_kwh
        if ev_energy_kwh > 0 else 0.0
    )
    solar_utilization = (
        100.0 * solar_used_kwh / solar_available_kwh
        if solar_available_kwh > 0 else 0.0
    )

    results.append({
        "method": method,
        "ev_energy_kwh": ev_energy_kwh,
        "solar_available_kwh": solar_available_kwh,
        "solar_used_kwh": solar_used_kwh,
        "grid_import_kwh": grid_import_kwh,
        "excess_solar_kwh": excess_solar_kwh,
        "solar_share_of_ev_energy_pct": solar_fraction,
        "solar_utilization_pct": solar_utilization,
        "peak_ev_load_kw": float(ev_load_kw.max()),
        "peak_grid_import_kw": float(grid_import_kw.max()),
    })

    profiles[method] = {
        "ev_load_kw": ev_load_kw,
        "solar_kw": solar_kw,
        "grid_import_kw": grid_import_kw,
    }

comparison = pd.DataFrame(results).round(3)
csv_path = RESULTS_DIR / "solar_impact_comparison.csv"
comparison.to_csv(csv_path, index=False)

print("\nSOLAR ENERGY-BALANCE COMPARISON")
print("=" * 105)
print(comparison.to_string(index=False))
print(f"\nCSV saved to: {csv_path}")

# Plot 1: grid-import peak for each schedule after solar offset.
plt.figure(figsize=(11, 5))
plt.bar(comparison["method"], comparison["peak_grid_import_kw"])
plt.axhline(125.0, linestyle="--", label="125 kW grid limit")
plt.ylabel("Peak grid import (kW)")
plt.title("Estimated Grid-Import Peak with Solar Offset")
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.25)
plt.legend()
plt.tight_layout()
peak_path = RESULTS_DIR / "solar_grid_import_peak.png"
plt.savefig(peak_path, dpi=200)
plt.close()

# Plot 2: show the solar offset on the cost-aware GA profile, if available.
chosen_method = (
    "Cost-aware GA" if "Cost-aware GA" in profiles
    else next(iter(profiles), None)
)

if chosen_method is not None:
    p = profiles[chosen_method]
    time_hours = np.arange(N_SLOTS) * STEP_HOURS

    plt.figure(figsize=(11, 5))
    plt.plot(time_hours, p["ev_load_kw"], label="EV charging demand")
    plt.plot(time_hours, p["solar_kw"], label="Solar available")
    plt.plot(time_hours, p["grid_import_kw"], label="Estimated grid import")
    plt.xlabel("Hours from simulation start")
    plt.ylabel("Power (kW)")
    plt.title(f"Solar Energy Balance: {chosen_method}")
    plt.xticks(np.arange(0, 37, 3))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    profile_path = RESULTS_DIR / "solar_energy_balance_profile.png"
    plt.savefig(profile_path, dpi=200)
    plt.close()
    print(f"Energy-balance graph saved to: {profile_path}")

print(f"Peak comparison graph saved to: {peak_path}")
print(
    "\nImportant: this is an energy-offset analysis of existing schedules, "
    "not yet solar-aware optimized charging. The schedules have not been "
    "re-optimized, and grid-limit feasibility must be checked again after "
    "solar is integrated into the optimizer."
)
