"""
Conventional charging baselines, all evaluated with the same core.

There are two deliberately different families here, and the distinction
matters for the research claim:

1. **Policy baselines** (``uncontrolled``, ``chronological_fcfs``,
   ``chronological_edf``) commit each slot as simulated time advances. They
   embody a real operating rule.
2. **Priority baselines** (``priority_fcfs``, ``priority_edf``) build a fixed
   ranking of EV-slot opportunities and hand it to the shared repair heuristic.
   They are a reasonable benchmark for a metaheuristic *search*, but they are
   not first-come-first-served in any operational sense.

Before Tier 0 the comparison table called ``priority_fcfs`` "FCFS", which made
it look as though the schedulers lost to a conventional queueing policy when in
fact they lost to another gainfully-constructed priority vector.

Chronological baselines use the same per-slot capacity rule as the repair
heuristic (``max_grid_kw + solar_kw[t]``) so that all rows of a comparison
table face an identical constraint set.

See docs/TIER0_CHANGES.md.
"""
from __future__ import annotations

import numpy as np

import config as cfg
from multiobjective_core import make_windows, repair_schedule


def uncontrolled(ev, solar_kw):
    """
    Every EV charges at its own maximum power from arrival until satisfied.

    Ignores the charger count and the grid limit, so the result is INFEASIBLE
    by construction. It exists only as an upper reference for delivered energy.
    """
    n_ev, n_slots = len(ev), len(solar_kw)
    a, d = make_windows(ev, n_slots)
    req = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)

    schedule = np.zeros((n_ev, n_slots), dtype=float)
    for i in range(n_ev):
        remaining = req[i]
        for t in range(a[i], d[i]):
            power = min(pmax[i], remaining / cfg.STEP_HOURS)
            schedule[i, t] = power
            remaining -= power * cfg.STEP_HOURS
            if remaining <= 1e-8:
                break
    return schedule


def _chronological(
    ev,
    solar_kw,
    policy_key,
    max_grid_kw=cfg.MAX_GRID_KW,
    n_chargers=cfg.N_CHARGERS,
):
    """
    Forward-simulate a port-allocation policy, one slot at a time.

    At each slot the eligible EVs (connected, still needing energy) are sorted
    by ``policy_key``; the first ``n_chargers`` take a port and charge at the
    highest power allowed jointly by their own limit, their remaining energy,
    and the remaining slot capacity.
    """
    n_ev, n_slots = len(ev), len(solar_kw)
    a, d = make_windows(ev, n_slots)
    req = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)
    solar = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)

    schedule = np.zeros((n_ev, n_slots), dtype=float)
    delivered = np.zeros(n_ev, dtype=float)
    capacity = np.maximum(max_grid_kw + solar, 0.0)

    for t in range(n_slots):
        eligible = [
            i for i in range(n_ev)
            if a[i] <= t < d[i] and req[i] - delivered[i] > 1e-8
        ]
        if not eligible:
            continue

        eligible.sort(key=policy_key)
        used = 0.0
        for i in eligible[:n_chargers]:
            room = capacity[t] - used
            if room <= 1e-8:
                break
            power = min(pmax[i], (req[i] - delivered[i]) / cfg.STEP_HOURS, room)
            if power <= 1e-8:
                continue
            schedule[i, t] = power
            delivered[i] += power * cfg.STEP_HOURS
            used += power

    return schedule


def chronological_fcfs(ev, solar_kw, max_grid_kw=cfg.MAX_GRID_KW,
                       n_chargers=cfg.N_CHARGERS):
    """
    Genuine first-come-first-served: whoever arrived first holds a port, and
    an EV keeps its port from slot to slot until it is full or departs.
    """
    n_slots = len(solar_kw)
    a, _ = make_windows(ev, n_slots)
    return _chronological(
        ev, solar_kw, lambda i: (a[i], i), max_grid_kw, n_chargers
    )


def chronological_edf(ev, solar_kw, max_grid_kw=cfg.MAX_GRID_KW,
                      n_chargers=cfg.N_CHARGERS):
    """
    Earliest-deadline-first: the EV that departs soonest takes the next port.
    Ties are broken by earlier arrival, then by EV index, for reproducibility.
    """
    n_slots = len(solar_kw)
    a, d = make_windows(ev, n_slots)
    return _chronological(
        ev, solar_kw, lambda i: (d[i], a[i], i), max_grid_kw, n_chargers
    )


def priority_fcfs(ev, solar_kw):
    """
    Priority-vector baseline (NOT operational FCFS): a constant priority per EV
    that decreases with arrival time, handed to the shared repair heuristic.
    This is what the pre-Tier-0 comparison table labelled "FCFS".
    """
    n_ev, n_slots = len(ev), len(solar_kw)
    arrival = ev["arrival_slot"].to_numpy(float)
    spread = max(arrival.max() - arrival.min(), 1.0)
    rank = (arrival.max() - arrival) / spread
    return repair_schedule(np.repeat(rank[:, None], n_slots, axis=1), ev, solar_kw)


def priority_edf(ev, solar_kw):
    """
    Priority-vector baseline (NOT operational EDF): priority rises as an EV's
    departure approaches, handed to the shared repair heuristic.
    This is what the pre-Tier-0 comparison table labelled "EDF".
    """
    n_ev, n_slots = len(ev), len(solar_kw)
    a, d = make_windows(ev, n_slots)
    urgency = np.zeros((n_ev, n_slots), dtype=float)
    for t in range(n_slots):
        eligible = (a <= t) & (d > t)
        urgency[eligible, t] = 1.0 / np.maximum(d[eligible] - t, 1)
    return repair_schedule(urgency, ev, solar_kw)


# Registry used by the experiment runner and the tests.
POLICIES = {
    "Uncontrolled (infeasible reference)": uncontrolled,
    "FCFS (chronological)": chronological_fcfs,
    "EDF (chronological)": chronological_edf,
    "FCFS-priority (repair-based)": priority_fcfs,
    "EDF-priority (repair-based)": priority_edf,
}


def all_baselines(ev, solar_kw):
    """Run every baseline and return ``{name: schedule}``."""
    return {name: fn(ev, solar_kw) for name, fn in POLICIES.items()}
