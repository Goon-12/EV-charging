"""
Equal-service comparison: judge every method on cost and peak *per delivered kWh*.

Why this module exists
----------------------
``results/final_unified/comparison_metrics.csv`` reports TOTAL tariff cost and
TOTAL peak demand. Both are volume metrics. This scenario is heavily congested -
100 EVs share 20 ports over one evening - so no method serves more than about 75%
of the demand and the methods span roughly 963-1134 kWh delivered. A method that
delivers less energy therefore looks cheaper and flatter for free. Ranking the
raw totals is not a comparison of scheduling ability at all; it is mostly a
comparison of how much energy each method happened to move.

The fix is to express cost and peak per delivered kWh:

    cost_per_delivered_kwh = grid_cost_units / delivered_energy_kwh
    peak_per_delivered_kwh = peak_ev_load_kw  / delivered_energy_kwh

Scaling every power in a schedule by a factor beta scales delivered energy and
peak by beta exactly, so both ratios are unchanged: the amount of service
cancels out. That is the sense in which these metrics judge a method "at equal
service". (The cancellation is exact for the peak. For cost it is exact on the
EV load; with solar present the grid import is ``max(load - solar, 0)``, which is
convex rather than linear, so a small residual volume effect remains.)

Two further pieces are needed to read the table honestly:

* **The service denominator.** ``service_pct_of_required`` can never reach 100%:
  2 of the 100 EVs cannot physically receive the energy they ask for inside their
  own connection window. ``service_pct_of_attainable`` divides by the energy the
  scenario actually permits, and ``unmet_congestion_kwh`` separates the
  unfixable shortfall from the shortfall a better scheduler could reduce.
* **Dominance.** Even at equal service, cost per kWh and peak per kWh trade off
  against each other (charge the cheap hours vs. spread the load). A method that
  is worse on all three of service, cost intensity and peak intensity can be
  eliminated outright, so it is flagged rather than left for a reader to average
  incomparable numbers.

What this does NOT do: cost per kWh is an intensity, not the cost of serving a
mandated number of kWh. The marginal kWh is harder to deliver than the first one,
because it belongs to EVs with tighter deadlines, so at equal efficiency a method
that delivers more energy still shows a slightly *higher* cost per kWh. A true
equal-service (epsilon-constraint) frontier requires re-running every method
against a swept service requirement; see docs/TIER1_CHANGES.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from multiobjective_core import evaluate_schedule, make_windows

# Canonical column order of the equal-service comparison table.
EQUAL_SERVICE_COLUMNS = [
    "method",
    "evaluations",
    "service_delivered_kwh",
    "service_pct_of_required",
    "service_pct_of_attainable",
    "fully_charged_evs",
    "unmet_energy_kwh",
    "unmet_congestion_kwh",
    "cost_units",
    "cost_per_delivered_kwh",
    "peak_ev_load_kw",
    "peak_per_delivered_kwh",
    "peak_to_average_ratio",
    "constraint_violations",
    "feasible",
    "pareto_status",
    "dominated_by",
]

# Dominance is decided on service (higher is better) plus the two intensity
# metrics (lower is better). Raw cost and raw peak are deliberately excluded:
# including them would reintroduce exactly the volume effect this table removes.
PARETO_KEYS = (
    ("service_pct_of_attainable", "max"),
    ("cost_per_delivered_kwh", "min"),
    ("peak_per_delivered_kwh", "min"),
)

# Relative slack that keeps exact ties from registering as strict improvements.
PARETO_RELATIVE_TOLERANCE = 1e-9


def per_ev_attainable_energy(ev, n_slots=None):
    """
    Energy each EV could receive if it had the whole site to itself.

    That is ``min(required, (departure - arrival) * max_power)``. The difference
    from ``required`` is demand the scenario makes physically impossible, which
    no scheduler can recover.
    """
    n_slots = cfg.N_SLOTS if n_slots is None else int(n_slots)
    arrival, departure = make_windows(ev, n_slots)
    required = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)
    hours = np.maximum(departure - arrival, 0) * cfg.STEP_HOURS
    return np.minimum(required, hours * pmax)


def service_accounting(ev, n_slots=None):
    """
    The scenario's service ceiling, independent of any scheduler.

    ``attainable + infeasible == required`` by construction, so every service
    percentage in this module is measured against a denominator that a perfect
    scheduler could actually reach.
    """
    n_slots = cfg.N_SLOTS if n_slots is None else int(n_slots)
    required = ev["required_energy_kwh"].to_numpy(float)
    attainable = per_ev_attainable_energy(ev, n_slots)
    infeasible = np.maximum(required - attainable, 0.0)
    return {
        "required_energy_kwh": float(required.sum()),
        "attainable_energy_kwh": float(attainable.sum()),
        "infeasible_energy_kwh": float(infeasible.sum()),
        "infeasible_evs": int(np.sum(infeasible > cfg.FULLY_CHARGED_TOLERANCE_KWH)),
    }


def equal_service_row(method, schedule, ev, solar, accounting=None, evaluations=0):
    """
    Build one equal-service row using the shared evaluator.

    ``accounting`` is the scenario ceiling from :func:`service_accounting`; it is
    passed in by :func:`equal_service_table` so it is computed once per run.
    """
    metrics = evaluate_schedule(schedule, ev, solar)
    accounting = service_accounting(ev, len(solar)) if accounting is None else accounting

    delivered = metrics["delivered_energy_kwh"]
    if delivered <= 0:
        raise ValueError(f"{method}: an empty schedule has no service to compare.")

    cost = metrics["grid_cost_units"]
    peak = metrics["peak_ev_load_kw"]
    horizon_hours = len(solar) * cfg.STEP_HOURS

    return {
        "method": method,
        "evaluations": int(evaluations),
        "service_delivered_kwh": delivered,
        "service_pct_of_required": 100.0 * delivered / metrics["required_energy_kwh"],
        "service_pct_of_attainable": 100.0 * delivered / accounting["attainable_energy_kwh"],
        "fully_charged_evs": metrics["fully_charged_evs"],
        "unmet_energy_kwh": metrics["unmet_energy_kwh"],
        # Shortfall beyond the physically impossible part, i.e. the part a better
        # scheduler could in principle close.
        "unmet_congestion_kwh": max(
            metrics["unmet_energy_kwh"] - accounting["infeasible_energy_kwh"], 0.0
        ),
        "cost_units": cost,
        "cost_per_delivered_kwh": cost / delivered,
        "peak_ev_load_kw": peak,
        "peak_per_delivered_kwh": peak / delivered,
        # Identical information to peak_per_delivered_kwh times the horizon; it is
        # kept because it is dimensionless and therefore easier to read.
        "peak_to_average_ratio": peak / (delivered / horizon_hours),
        "constraint_violations": int(metrics["constraint_violations"]),
        "feasible": bool(metrics["constraint_violations"] == 0),
    }


def flag_pareto(frame, keys=PARETO_KEYS, tolerance=PARETO_RELATIVE_TOLERANCE):
    """
    Add ``pareto_status`` and ``dominated_by`` columns to an equal-service table.

    A feasible row is *dominated* when another feasible row is at least as good
    on every ranking key and strictly better on at least one, in which case it
    cannot be preferred under any weighting of those keys and needs no trade-off
    argument.

    Infeasible rows are labelled ``infeasible reference`` and take no part in the
    comparison: the ``Uncontrolled`` policy delivers the most energy of any row
    precisely because it ignores the port and grid limits, so it would otherwise
    dominate every honest scheduler.
    """
    frame = frame.copy()
    directions = np.array([1.0 if direction == "max" else -1.0 for _, direction in keys])
    values = frame[[column for column, _ in keys]].to_numpy(float)
    feasible = frame["feasible"].to_numpy(bool)

    status, dominated_by = [], []
    for i in range(len(frame)):
        if not feasible[i]:
            status.append("infeasible reference")
            dominated_by.append("")
            continue
        if not np.isfinite(values[i]).all():
            status.append("incomplete")
            dominated_by.append("")
            continue

        # Work in "higher is better" space so one comparison covers both directions.
        mine = values[i] * directions
        slack = tolerance * np.maximum(np.abs(mine), 1.0)

        winners = []
        for j in range(len(frame)):
            if i == j or not feasible[j] or not np.isfinite(values[j]).all():
                continue
            theirs = values[j] * directions
            if np.all(theirs >= mine - slack) and np.any(theirs > mine + slack):
                winners.append(str(frame["method"].iloc[j]))

        status.append("dominated" if winners else "frontier")
        dominated_by.append("; ".join(sorted(winners)))

    frame["pareto_status"] = status
    frame["dominated_by"] = dominated_by
    return frame


def equal_service_table(schedules, ev, solar):
    """
    Build the whole equal-service table from ``{method: schedule}``.

    Returns a DataFrame in :data:`EQUAL_SERVICE_COLUMNS` order, Pareto-flagged.
    """
    accounting = service_accounting(ev, len(solar))
    rows = [
        equal_service_row(name, schedule, ev, solar, accounting)
        for name, schedule in schedules.items()
    ]
    return flag_pareto(pd.DataFrame(rows))[EQUAL_SERVICE_COLUMNS]


def frontier_rows(frame):
    """
    Feasible, non-dominated rows, cheapest service first.

    These are the only rows that can be ranked against each other without first
    choosing how to trade service against cost and peak.
    """
    frontier = frame[frame["pareto_status"] == "frontier"]
    return frontier.sort_values("cost_per_delivered_kwh").reset_index(drop=True)
