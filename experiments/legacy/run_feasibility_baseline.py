# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from load_scenario import load_scenario

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0
TOLERANCE_KWH = 0.05


def build_schedule(scenario):
    """Allocate charging to EVs with the tightest windows first."""
    n_ev = len(scenario)
    schedule = np.zeros((n_ev, N_SLOTS), dtype=float)

    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)
    required = scenario["required_energy_kwh"].to_numpy(dtype=float)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)

    start = np.clip(arrival, 0, N_SLOTS)
    end = np.clip(departure, 0, N_SLOTS)
    window_length = np.maximum(end - start, 0)

    # Estimate the time each EV needs at its maximum charging power.
    required_slots = np.ceil(
        required / np.maximum(max_power * STEP_HOURS, 1e-9)
    )

    # Prioritize EVs with the least scheduling flexibility.
    urgency = required_slots / np.maximum(window_length, 1)

    order = sorted(
        range(n_ev),
        key=lambda i: (
            -urgency[i],
            window_length[i],
            start[i],
            i,
        ),
    )

    for i in order:
        remaining = required[i]

        # Use available slots with the most remaining grid capacity.
        candidate_slots = list(range(start[i], end[i]))
        candidate_slots.sort(
            key=lambda t: (
                schedule[:, t].sum(),
                np.count_nonzero(schedule[:, t] > 1e-9),
                t,
            )
        )

        for t in candidate_slots:
            if remaining <= 1e-9:
                break

            active = np.count_nonzero(schedule[:, t] > 1e-9)
            if active >= N_CHARGERS:
                continue

            available_grid = MAX_GRID_KW - schedule[:, t].sum()
            if available_grid <= 1e-9:
                continue

            power = min(
                max_power[i],
                remaining / STEP_HOURS,
                available_grid,
            )

            if power > 1e-9:
                schedule[i, t] = power
                remaining -= power * STEP_HOURS

    return schedule


def main():
    scenario = load_scenario()
    schedule = build_schedule(scenario)

    required = scenario["required_energy_kwh"].to_numpy(dtype=float)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)
    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)

    delivered = schedule.sum(axis=1) * STEP_HOURS
    unmet = np.maximum(required - delivered, 0.0)
    grid_load = schedule.sum(axis=0)
    active = (schedule > 1e-9).sum(axis=0)

    power_violations = int(
        (schedule > max_power[:, None] + 1e-9).sum()
    )

    time_violations = 0
    for i in range(len(scenario)):
        for t in range(N_SLOTS):
            if schedule[i, t] > 1e-9:
                if not (arrival[i] <= t < departure[i]):
                    time_violations += 1

    grid_violations = int(
        (grid_load > MAX_GRID_KW + 1e-9).sum()
    )
    charger_violations = int(
        (active > N_CHARGERS).sum()
    )

    print("\nFEASIBILITY-FIRST BASELINE")
    print("=" * 42)
    print(f"EVs: {len(scenario)}")
    print(f"Required energy: {required.sum():.3f} kWh")
    print(f"Delivered energy: {delivered.sum():.3f} kWh")
    print(f"Unmet energy: {unmet.sum():.3f} kWh")
    print(
        "Fully charged EVs: "
        f"{int((unmet <= TOLERANCE_KWH).sum())}"
    )
    print(f"Energy demand met: {100 * delivered.sum() / required.sum():.2f}%")
    print(f"Peak grid demand: {grid_load.max():.3f} kW")
    print(f"Grid-limit violations: {grid_violations}")
    print(f"Charger-limit violations: {charger_violations}")
    print(f"Individual power violations: {power_violations}")
    print(f"Time-window violations: {time_violations}")

    output = pd.DataFrame(
        schedule,
        columns=[f"slot_{t}" for t in range(N_SLOTS)],
    )
    output.insert(0, "ev_id", scenario["ev_id"].to_numpy())

    schedule_file = RESULTS_DIR / "feasibility_baseline_schedule.csv"
    output.to_csv(schedule_file, index=False)

    metrics = pd.DataFrame([{
        "method": "Feasibility-first",
        "peak_grid_kw": grid_load.max(),
        "total_energy_kwh": delivered.sum(),
        "total_unmet_energy_kwh": unmet.sum(),
        "evs_fully_charged": int((unmet <= TOLERANCE_KWH).sum()),
        "grid_limit_violations": grid_violations,
        "charger_limit_violations": charger_violations,
        "individual_power_violations": power_violations,
        "time_window_violations": time_violations,
    }])

    metrics_file = RESULTS_DIR / "feasibility_baseline_metrics.csv"
    metrics.to_csv(metrics_file, index=False)

    print(f"\nSchedule saved to: {schedule_file}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
