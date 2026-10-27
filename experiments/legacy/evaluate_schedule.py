# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md

import numpy as np


def evaluate_schedule(
    schedule,
    required_energy_kwh,
    step_hours,
    max_grid_kw,
    max_chargers,
    max_power_kw,
    arrival_slot,
    departure_slot,
    tolerance_kwh=0.05,
):
    """
    Evaluate a charging schedule.

    schedule[i, t] = charging power for EV i in slot t, in kW.
    All energy values are measured in kWh.
    """
    schedule = np.asarray(schedule, dtype=float)
    required = np.asarray(required_energy_kwh, dtype=float)
    max_power = np.asarray(max_power_kw, dtype=float)
    arrival = np.asarray(arrival_slot, dtype=int)
    departure = np.asarray(departure_slot, dtype=int)

    n_ev, n_slots = schedule.shape

    if len(required) != n_ev:
        raise ValueError("Energy requirements do not match EV count.")

    if len(max_power) != n_ev:
        raise ValueError("Power limits do not match EV count.")

    if np.any(~np.isfinite(schedule)):
        raise ValueError("Schedule contains non-finite values.")

    if np.any(schedule < -1e-9):
        raise ValueError("Schedule contains negative power.")

    energy_by_ev = schedule.sum(axis=1) * step_hours
    grid_load = schedule.sum(axis=0)

    # Count EVs actively charging in each slot.
    active_count = (schedule > 1e-9).sum(axis=0)

    # Check individual power limits.
    individual_power_violations = int(
        (schedule > max_power[:, None] + 1e-9).sum()
    )

    # Check charging outside arrival/departure windows.
    time_violations = 0
    for i in range(n_ev):
        for t in range(n_slots):
            if schedule[i, t] > 1e-9:
                if not (arrival[i] <= t < departure[i]):
                    time_violations += 1

    unmet = np.maximum(required - energy_by_ev, 0.0)

    return {
        "peak_grid_kw": float(grid_load.max()),
        "total_energy_kwh": float(energy_by_ev.sum()),
        "total_unmet_energy_kwh": float(unmet.sum()),
        "evs_fully_charged": int((unmet <= tolerance_kwh).sum()),
        "grid_limit_violations": int(
            (grid_load > max_grid_kw + 1e-9).sum()
        ),
        "charger_limit_violations": int(
            (active_count > max_chargers).sum()
        ),
        "individual_power_violations": individual_power_violations,
        "time_window_violations": time_violations,
    }
