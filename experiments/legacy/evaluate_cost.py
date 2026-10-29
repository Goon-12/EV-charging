# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md

import numpy as np


def evaluate_cost(
    schedule,
    price_per_kwh,
    step_hours=0.25,
):
    """
    Calculate electricity cost for an EV charging schedule.

    Parameters
    ----------
    schedule : array, shape (n_ev, n_slots)
        Charging power in kW.
    price_per_kwh : array, shape (n_slots,)
        Electricity price for each time slot.
    step_hours : float
        Duration of each slot in hours.

    Returns
    -------
    float
        Total simulated electricity cost.
    """
    schedule = np.asarray(schedule, dtype=float)
    prices = np.asarray(price_per_kwh, dtype=float)

    if schedule.ndim != 2:
        raise ValueError("Schedule must be a 2D array.")

    if prices.ndim != 1 or len(prices) != schedule.shape[1]:
        raise ValueError("Price array must match the schedule time slots.")

    if not np.all(np.isfinite(schedule)):
        raise ValueError("Schedule contains non-finite values.")

    if not np.all(np.isfinite(prices)):
        raise ValueError("Price array contains non-finite values.")

    if np.any(schedule < 0):
        raise ValueError("Charging power cannot be negative.")

    if np.any(prices < 0):
        raise ValueError("Prices cannot be negative in this prototype.")

    if step_hours <= 0:
        raise ValueError("step_hours must be positive.")

    energy_by_slot = schedule.sum(axis=0) * step_hours
    return float(np.dot(energy_by_slot, prices))
