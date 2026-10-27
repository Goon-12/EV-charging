"""
Build the frozen EV charging scenario and evaluate the conventional baselines.

    .venv/Scripts/python src/create_solar_profile.py     # once
    .venv/Scripts/python src/simulate_baselines.py

Writes:

* ``data/processed/ev_scenario.csv``  - THE scenario every algorithm is compared on
* ``results/uncontrolled_schedule.csv``, ``results/fcfs_schedule.csv``
* ``results/baseline_metrics.csv``, ``results/baseline_load_profiles.png``

The scenario is a build product. Run this once and freeze the CSV: comparing
algorithms on regenerated scenarios would make every difference meaningless.

Tier 0 changes
--------------
* The local ``run_fcfs`` implementation was deleted. It was NOT genuine FCFS and
  it used a different per-slot capacity rule (``max_grid_kw`` instead of
  ``max_grid_kw + solar``) from the shared repair routine, so the "FCFS"
  baseline and the optimizers were not solving the same problem. Baselines now
  come from ``src/baselines.py``.
* The local evaluator (``evaluate_schedule.py``) was replaced by the single
  shared one in ``src/multiobjective_core.py``; the old module is archived in
  ``experiments/legacy/``.
* Scenario generation is UNCHANGED, down to the order of the random draws, so
  ``ev_scenario.csv`` is byte-identical to the frozen input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from artifacts import RESULT_COLUMNS, metrics_row, save_load_profile_plot, save_schedule_csv
from baselines import all_baselines
from multiobjective_core import load_inputs


def build_scenario():
    """
    Sample the cleaned ACN sessions into a synthetic apartment scenario.

    Energy and power are inspired by measured ACN sessions; arrival and
    departure windows are SIMULATED assumptions, not observations.
    """
    if not cfg.CLEAN_SESSIONS.exists():
        raise FileNotFoundError(
            f"Cleaned session data not found: {cfg.CLEAN_SESSIONS}. "
            "Run src/create_clean_datasets.py first (requires the raw ACN dataset)."
        )

    rng = np.random.default_rng(cfg.SCENARIO_SEED)
    data = pd.read_csv(cfg.CLEAN_SESSIONS)

    usable = data[
        (data["energy_delivered_kwh_provisional"] > 0)
        & data["mean_positive_power_kw"].notna()
        & (data["mean_positive_power_kw"] > 0)
    ].copy()

    if len(usable) < cfg.N_EV:
        raise ValueError(
            f"Only {len(usable)} usable sessions; need at least {cfg.N_EV}."
        )

    sample = usable.sample(
        n=cfg.N_EV, replace=True, random_state=cfg.SCENARIO_SEED
    ).reset_index(drop=True)

    energy = np.clip(
        sample["energy_delivered_kwh_provisional"].to_numpy(),
        cfg.ENERGY_MIN_KWH, cfg.ENERGY_MAX_KWH,
    )
    power = np.clip(
        sample["mean_positive_power_kw"].to_numpy(),
        cfg.POWER_MIN_KW, cfg.POWER_MAX_KW,
    )

    # Draw order matters: it is what makes this file reproducible.
    arrival_hour = rng.uniform(cfg.ARRIVAL_HOUR_MIN, cfg.ARRIVAL_HOUR_MAX, cfg.N_EV)
    stay_hours = rng.uniform(cfg.STAY_HOURS_MIN, cfg.STAY_HOURS_MAX, cfg.N_EV)
    departure_hour = arrival_hour + stay_hours

    arrival_slot = np.floor(arrival_hour / cfg.STEP_HOURS).astype(int)
    departure_slot = np.ceil(departure_hour / cfg.STEP_HOURS).astype(int)
    departure_slot = np.clip(departure_slot, 1, cfg.N_SLOTS)

    return pd.DataFrame({
        "ev_id": np.arange(cfg.N_EV),
        "arrival_hour": arrival_hour,
        "departure_hour": departure_hour,
        "arrival_slot": arrival_slot,
        "departure_slot": departure_slot,
        "required_energy_kwh": energy,
        "max_power_kw": power,
    })


def main():
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario()
    scenario.to_csv(cfg.DEFAULT_SCENARIO, index=False)
    print(f"Scenario saved to: {cfg.DEFAULT_SCENARIO}")
    print(scenario.head().to_string(index=False))

    # The solar profile is generated independently by create_solar_profile.py.
    ev, solar = load_inputs(cfg.DEFAULT_SCENARIO, cfg.DEFAULT_SOLAR)

    schedules = all_baselines(ev, solar)
    rows = [metrics_row(s, ev, solar, name) for name, s in schedules.items()]

    save_schedule_csv(
        cfg.RESULTS_DIR / "uncontrolled_schedule.csv",
        schedules["Uncontrolled (infeasible reference)"], ev,
    )
    save_schedule_csv(
        cfg.RESULTS_DIR / "fcfs_schedule.csv",
        schedules["FCFS (chronological)"], ev,
    )
    save_load_profile_plot(
        cfg.RESULTS_DIR / "baseline_load_profiles.png",
        schedules["FCFS (chronological)"], solar,
        "Baseline FCFS load profile",
    )

    table = pd.DataFrame(rows)[RESULT_COLUMNS]
    table.to_csv(cfg.RESULTS_DIR / "baseline_metrics.csv", index=False)

    print("\nBASELINE COMPARISON (shared objective, repair and evaluator)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nMetrics saved to: {cfg.RESULTS_DIR / 'baseline_metrics.csv'}")
    print("Note: 'Uncontrolled' is deliberately infeasible and is a reference only.")


if __name__ == "__main__":
    main()
