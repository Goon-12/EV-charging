
import numpy as np


def evaluate_objective(
    schedule,
    required_energy_kwh,
    price_per_kwh,
    step_hours=0.25,
    max_grid_kw=125.0,
    unmet_weight=10000.0,
    cost_weight=1.0,
    peak_weight=10.0,
):
    """
    Evaluate charging cost, unmet energy, and peak demand.

    Cost and peak terms are normalized for a 100-EV-scale
    simulation. Unmet energy receives a strong penalty.
    """
    schedule = np.asarray(schedule, dtype=float)
    required = np.asarray(required_energy_kwh, dtype=float)
    prices = np.asarray(price_per_kwh, dtype=float)

    if schedule.ndim != 2:
        raise ValueError("Schedule must be a 2D array.")

    n_ev, n_slots = schedule.shape

    if required.shape != (n_ev,):
        raise ValueError("Required energy must match the EV count.")

    if prices.shape != (n_slots,):
        raise ValueError("Tariff must match the number of slots.")

    if (
        not np.all(np.isfinite(schedule))
        or not np.all(np.isfinite(required))
        or not np.all(np.isfinite(prices))
    ):
        raise ValueError("Inputs must contain only finite values.")

    if np.any(schedule < 0) or np.any(required < 0) or np.any(prices < 0):
        raise ValueError("Power, requirements, and prices must be nonnegative.")

    if step_hours <= 0 or max_grid_kw <= 0:
        raise ValueError("Time step and grid limit must be positive.")

    delivered_by_ev = schedule.sum(axis=1) * step_hours
    unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)
    total_unmet = float(unmet_by_ev.sum())

    grid_load = schedule.sum(axis=0)
    peak_kw = float(grid_load.max())

    energy_by_slot = grid_load * step_hours
    electricity_cost = float(np.dot(energy_by_slot, prices))

    # Normalize using fixed reference values, not candidate-dependent values.
    normalized_cost = electricity_cost / (
        max_grid_kw * n_slots * step_hours * max(prices.max(), 1.0)
    )
    normalized_peak = peak_kw / max_grid_kw

    objective = (
        unmet_weight * total_unmet
        + cost_weight * normalized_cost
        + peak_weight * normalized_peak
    )

    return {
        "objective": float(objective),
        "unmet_energy_kwh": total_unmet,
        "electricity_cost_units": electricity_cost,
        "peak_grid_kw": peak_kw,
        "normalized_cost": normalized_cost,
        "normalized_peak": normalized_peak,
    }
