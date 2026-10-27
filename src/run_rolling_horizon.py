"""
Receding-horizon EV charging controller.

At every slot the controller plans over a look-ahead window, executes only the
first slot of that plan, observes the result, and replans. Only the current and
future slots are ever modified; already-executed decisions are frozen.

Tier 0 bug fixes
----------------
1. **Tariff clock.** The look-ahead window is a *slice* of the horizon, but the
   tariff used to be rebuilt with ``tariff_vector(width)``, i.e. every window
   believed it started at midnight and therefore applied the 4.0 off-peak rate.
   The controller was blind to the evening peak. Windows now carry their
   absolute ``slot_offset``.
2. **Urgency scale.** The deadline-urgency term inside the repair heuristic is
   normalised by the number of slots in the priority matrix. Inside a 24-slot
   window that compressed urgency towards zero for every EV departing after the
   window, so the controller deferred far-deadline EVs instead of charging them
   early. ``horizon_slots`` now reports the slots actually left in the
   simulation.
"""
from __future__ import annotations

import numpy as np

import config as cfg
from multiobjective_core import constraint_report
from multiobjective_optimizers import optimize_ga, optimize_pso


def optimize_rolling(
    ev,
    solar_kw,
    method="ga",
    window_slots=cfg.ROLLING_WINDOW_SLOTS,
    seed=cfg.ROLLING_SEED,
    population_size=cfg.ROLLING_GA_POPULATION_SIZE,
    generations=cfg.ROLLING_GA_GENERATIONS,
    particles=cfg.ROLLING_PSO_PARTICLES,
    iterations=cfg.ROLLING_PSO_ITERATIONS,
    check_feasibility=True,
):
    """
    Return the executed charging schedule for the whole horizon.

    Each replanning step runs a deliberately small optimization so that the
    total runtime stays manageable; the controller, not the optimizer, is what
    is being evaluated here.
    """
    n_ev, n_slots = len(ev), len(solar_kw)
    arr = ev["arrival_slot"].to_numpy(int)
    dep = ev["departure_slot"].to_numpy(int)
    original_req = ev["required_energy_kwh"].to_numpy(float)

    remaining = original_req.copy()
    final = np.zeros((n_ev, n_slots), dtype=float)

    for t in range(n_slots):
        end = min(n_slots, t + window_slots)
        width = end - t
        if width <= 0:
            break

        local = ev.copy()
        local["arrival_slot"] = np.clip(arr - t, 0, width)
        local["departure_slot"] = np.clip(dep - t, 0, width)
        # EVs whose departure has passed cannot receive any more energy.
        local_requirement = remaining.copy()
        local_requirement[dep <= t] = 0.0
        local["required_energy_kwh"] = local_requirement
        local_solar = np.asarray(solar_kw[t:end], dtype=float)

        # The window's first column is global slot t, so pass the offsets through.
        window_kwargs = dict(
            seed=seed + t,
            slot_offset=t,
            horizon_slots=n_slots - t,
        )
        if method.lower() == "pso":
            local_schedule, _ = optimize_pso(
                local, local_solar,
                particles=particles, iterations=iterations, **window_kwargs,
            )
        elif method.lower() == "ga":
            local_schedule, _ = optimize_ga(
                local, local_solar,
                population_size=population_size, generations=generations,
                **window_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown method: {method!r} (expected 'ga' or 'pso')"
            )

        executed = local_schedule[:, 0]
        final[:, t] = executed
        remaining = np.maximum(remaining - executed * cfg.STEP_HOURS, 0.0)

    if check_feasibility:
        report = constraint_report(final, ev, solar_kw)
        if report["total"] != 0:
            raise ValueError(
                "Rolling-horizon controller produced an infeasible trajectory. "
                f"This indicates a controller bug: {report}"
            )

    return final
