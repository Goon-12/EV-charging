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

scenario = pd.read_csv(DATA_DIR / "ev_scenario.csv")
solar_df = pd.read_csv(DATA_DIR / "solar_profile_36h.csv")

if len(scenario) != 100:
    print(f"Note: scenario contains {len(scenario)} EVs, not 100.")

if len(solar_df) != N_SLOTS:
    raise ValueError(
        f"Expected {N_SLOTS} solar slots, found {len(solar_df)}."
    )

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = np.clip(
    scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS
)
departure = np.clip(
    scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS
)
solar_kw = solar_df["solar_power_kw"].to_numpy(dtype=float)

if not np.isfinite(solar_kw).all() or np.any(solar_kw < 0):
    raise ValueError("Solar profile contains invalid values.")

n_ev = len(scenario)
schedule = np.zeros((n_ev, N_SLOTS), dtype=float)
delivered = np.zeros(n_ev, dtype=float)

# Schedule chronologically. Within each slot, prioritize EVs that:
# 1. have less time remaining,
# 2. have more energy remaining,
# 3. have a stable lower EV index as a tie-breaker.
for t in range(N_SLOTS):
    eligible = [
        i for i in range(n_ev)
        if arrival[i] <= t < departure[i]
        and delivered[i] < required[i] - 1e-9
    ]

    eligible.sort(
        key=lambda i: (
            departure[i],
            -(required[i] - delivered[i]),
            i,
        )
    )

    active_count = 0
    grid_import_kw = 0.0
    solar_available_kw = solar_kw[t]

    for i in eligible:
        if active_count >= N_CHARGERS:
            break

        remaining_energy = required[i] - delivered[i]
        if remaining_energy <= 1e-9:
            continue

        # Maximum EV charging power allowed by the remaining energy.
        requested_kw = min(
            max_power[i],
            remaining_energy / STEP_HOURS,
        )

        # Local solar is used first. Grid supplies the remainder.
        solar_for_ev_kw = min(solar_available_kw, requested_kw)
        grid_needed_kw = requested_kw - solar_for_ev_kw
        grid_room_kw = MAX_GRID_KW - grid_import_kw

        if grid_needed_kw > grid_room_kw:
            # If grid capacity is insufficient, use remaining solar
            # plus the available grid capacity, without exceeding EV power.
            actual_kw = min(
                requested_kw,
                solar_available_kw + max(grid_room_kw, 0.0),
            )
        else:
            actual_kw = requested_kw

        if actual_kw <= 1e-9:
            continue

        # Solar allocation for this EV; grid makes up any difference.
        solar_used_now = min(solar_available_kw, actual_kw)
        grid_used_now = actual_kw - solar_used_now

        if grid_used_now > grid_room_kw + 1e-8:
            raise RuntimeError("Grid capacity accounting error.")

        schedule[i, t] = actual_kw
        delivered[i] += actual_kw * STEP_HOURS
        solar_available_kw -= solar_used_now
        grid_import_kw += grid_used_now
        active_count += 1

# Compute per-slot solar and grid energy accounting.
ev_load_kw = schedule.sum(axis=0)
solar_used_kw = np.minimum(ev_load_kw, solar_kw)
grid_import_kw = np.maximum(ev_load_kw - solar_used_kw, 0.0)
excess_solar_kw = np.maximum(solar_kw - ev_load_kw, 0.0)

unmet_by_ev = np.maximum(required - delivered, 0.0)
total_required = float(required.sum())
total_delivered = float(delivered.sum())
total_unmet = float(unmet_by_ev.sum())

solar_available_kwh = float(solar_kw.sum() * STEP_HOURS)
solar_used_kwh = float(solar_used_kw.sum() * STEP_HOURS)
grid_import_kwh = float(grid_import_kw.sum() * STEP_HOURS)
excess_solar_kwh = float(excess_solar_kw.sum() * STEP_HOURS)

# Save schedule.
schedule_df = pd.DataFrame(
    schedule,
    columns=[f"slot_{t}" for t in range(N_SLOTS)]
)
schedule_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())

schedule_path = RESULTS_DIR / "solar_aware_schedule.csv"
schedule_df.to_csv(schedule_path, index=False)

# Validate constraints against the actual grid import, not EV load alone.
active_per_slot = (schedule > 1e-9).sum(axis=0)
grid_violations = int((grid_import_kw > MAX_GRID_KW + 1e-8).sum())
charger_violations = int((active_per_slot > N_CHARGERS).sum())
power_violations = int(
    (schedule > max_power[:, None] + 1e-8).sum()
)

time_violations = 0
for i in range(n_ev):
    valid = np.zeros(N_SLOTS, dtype=bool)
    valid[arrival[i]:departure[i]] = True
    time_violations += int(
        ((schedule[i] > 1e-8) & ~valid).sum()
    )

feasible = (
    grid_violations == 0
    and charger_violations == 0
    and power_violations == 0
    and time_violations == 0
)

metrics = {
    "method": "Solar-aware feasibility-first",
    "required_energy_kwh": total_required,
    "delivered_energy_kwh": total_delivered,
    "unmet_energy_kwh": total_unmet,
    "energy_demand_met_pct": (
        100 * total_delivered / total_required
        if total_required else 0.0
    ),
    "fully_charged_evs": int(
        (unmet_by_ev <= TOLERANCE_KWH).sum()
    ),
    "solar_available_kwh": solar_available_kwh,
    "solar_used_kwh": solar_used_kwh,
    "solar_utilization_pct": (
        100 * solar_used_kwh / solar_available_kwh
        if solar_available_kwh else 0.0
    ),
    "grid_import_kwh": grid_import_kwh,
    "excess_solar_kwh": excess_solar_kwh,
    "peak_ev_load_kw": float(ev_load_kw.max()),
    "peak_grid_import_kw": float(grid_import_kw.max()),
    "grid_limit_violations": grid_violations,
    "charger_limit_violations": charger_violations,
    "individual_power_violations": power_violations,
    "time_window_violations": time_violations,
    "feasible": feasible,
}

metrics_path = RESULTS_DIR / "solar_aware_metrics.csv"
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

# Plot EV demand, solar availability, and grid imports.
time_hours = np.arange(N_SLOTS) * STEP_HOURS

plt.figure(figsize=(12, 5))
plt.plot(time_hours, ev_load_kw, label="EV charging demand")
plt.plot(time_hours, solar_kw, label="Solar available")
plt.plot(time_hours, grid_import_kw, label="Grid import")
plt.axhline(
    MAX_GRID_KW, linestyle="--", label="125 kW grid-import limit"
)
plt.xlabel("Hours from simulation start")
plt.ylabel("Power (kW)")
plt.title("Solar-Aware Feasibility-First Charging")
plt.xticks(np.arange(0, 37, 3))
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plot_path = RESULTS_DIR / "solar_aware_load_profile.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print("\nSOLAR-AWARE SCHEDULER RESULTS")
print("=" * 60)
for key, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")

print("\nValidation:")
print(f"Grid violations: {grid_violations}")
print(f"Charger violations: {charger_violations}")
print(f"Individual power violations: {power_violations}")
print(f"Time-window violations: {time_violations}")
print(f"All checked constraints passed: {feasible}")

print(f"\nSchedule saved to: {schedule_path}")
print(f"Metrics saved to: {metrics_path}")
print(f"Load profile saved to: {plot_path}")
print(
    "\nNote: this is a new greedy solar-aware baseline, not GA, PSO, "
    "or a globally optimal schedule."
)
