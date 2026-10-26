
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144
TOL = 1e-9

scenario = pd.read_csv(DATA_DIR / "ev_scenario.csv")

required_columns = {
    "ev_id",
    "required_energy_kwh",
    "max_power_kw",
}

if not required_columns.issubset(scenario.columns):
    raise ValueError("Scenario is missing required columns.")

max_power = scenario["max_power_kw"].to_numpy(dtype=float)

if not np.isfinite(max_power).all() or np.any(max_power <= 0):
    raise ValueError("Maximum EV power must be positive and finite.")


def battery_stress_proxy(schedule, max_power_kw, step_hours=0.25):
    """
    Calculate an illustrative high-power charging stress proxy.

    B = sum(E_it * (P_it / P_i_max)^2) / sum(E_it)

    This is a proxy, not a validated battery-aging or degradation model.
    """
    schedule = np.asarray(schedule, dtype=float)
    max_power_kw = np.asarray(max_power_kw, dtype=float)

    if schedule.ndim != 2:
        raise ValueError("Schedule must be a 2D array.")

    if schedule.shape[0] != len(max_power_kw):
        raise ValueError("EV count does not match power-limit array.")

    if not np.isfinite(schedule).all() or np.any(schedule < -TOL):
        raise ValueError("Schedule contains invalid or negative power.")

    if np.any(max_power_kw <= 0) or not np.isfinite(max_power_kw).all():
        raise ValueError("Maximum power must be positive and finite.")

    if step_hours <= 0:
        raise ValueError("step_hours must be positive.")

    relative_rate = np.clip(
        schedule / max_power_kw[:, None],
        0.0,
        1.0,
    )

    energy = schedule * step_hours
    total_energy = float(energy.sum())

    if total_energy <= TOL:
        return 0.0

    score = float(
        (energy * relative_rate**2).sum() / total_energy
    )

    return score


# Unit test 1: charging at 50% of maximum power gives B = 0.25.
test_schedule_half = np.array([[2.0, 2.0, 0.0, 0.0]])
test_max_power = np.array([4.0])
half_score = battery_stress_proxy(
    test_schedule_half, test_max_power, STEP_HOURS
)

# Unit test 2: charging at maximum power gives B = 1.0.
test_schedule_full = np.array([[4.0, 4.0, 0.0, 0.0]])
full_score = battery_stress_proxy(
    test_schedule_full, test_max_power, STEP_HOURS
)

assert np.isclose(half_score, 0.25), (
    f"Expected 0.25 for half-power charging, got {half_score}"
)
assert np.isclose(full_score, 1.0), (
    f"Expected 1.0 for maximum-power charging, got {full_score}"
)

print("\nBATTERY STRESS PROXY UNIT TEST")
print("=" * 50)
print(f"Half-power test: {half_score:.3f} (expected 0.250)")
print(f"Maximum-power test: {full_score:.3f} (expected 1.000)")
print("Both unit tests passed.")


# Analyze currently saved schedules without changing them.
schedule_files = {
    "Original GA": "ga_best_schedule.csv",
    "PSO": "pso_best_schedule.csv",
    "Feasibility-first": "feasibility_baseline_schedule.csv",
    "Cost-aware GA": "ga_cost_best_schedule_latest.csv",
    "Solar-aware": "solar_aware_schedule.csv",
    "Solar-aware GA": "ga_solar_best_schedule.csv",
}

rows = []

for method, filename in schedule_files.items():
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Skipping {method}: {filename} not found.")
        continue

    df = pd.read_csv(path)
    slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]

    if len(df) != len(scenario):
        print(f"Skipping {method}: EV count mismatch.")
        continue

    if not all(column in df.columns for column in slot_columns):
        print(f"Skipping {method}: slot columns are missing.")
        continue

    schedule = df[slot_columns].to_numpy(dtype=float)

    if np.any(schedule > max_power[:, None] + 1e-6):
        print(
            f"Warning: {method} exceeds an individual power limit; "
            "interpret its proxy score cautiously."
        )

    score = battery_stress_proxy(schedule, max_power, STEP_HOURS)
    delivered = float(schedule.sum() * STEP_HOURS)

    rows.append({
        "method": method,
        "delivered_energy_kwh": delivered,
        "high_power_stress_proxy": score,
        "interpretation": "Lower means less high-power charging under this proxy",
    })

results = pd.DataFrame(rows)

if not results.empty:
    results["delivered_energy_kwh"] = results[
        "delivered_energy_kwh"
    ].round(3)
    results["high_power_stress_proxy"] = results[
        "high_power_stress_proxy"
    ].round(4)

output_path = RESULTS_DIR / "battery_proxy_comparison.csv"
results.to_csv(output_path, index=False)

print("\nBATTERY PROXY COMPARISON")
print("=" * 80)
if results.empty:
    print("No compatible schedules were found.")
else:
    print(results.to_string(index=False))

print(f"\nResults saved to: {output_path}")
print(
    "\nImportant: this proxy measures relative charging power only. "
    "It does not estimate battery life, capacity fade, or actual degradation."
)
