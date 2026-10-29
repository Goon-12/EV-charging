"""
Scenario integrity, replacing the print-only ``validate_scenarios.py``.

This module also locks in three structural findings from the review, all of
which are properties of the *scenario* rather than bugs in the code:

* 2 of the 100 EVs cannot be charged even when nothing else competes for a port,
  so full service is impossible for every method;
* the site offers 310 port-hours while EVs are connected against ~342.6
  port-hours of work at rated power, so the site is oversubscribed;
* the 125 kW grid limit is never approached, because peak load is set by 20
  ports times the fleet's average power (~85-99 kW), not by the grid.

Together these explain why "GA peaked higher than FCFS" is not the damning
result it first appears to be, and why every method lands in a narrow band.
"""
from __future__ import annotations

import numpy as np

import config as cfg
from load_scenario import REQUIRED_COLUMNS, load_scenario


def test_scenario_shape_and_columns(ev):
    assert len(ev) == cfg.N_EV
    assert set(REQUIRED_COLUMNS).issubset(ev.columns)


def test_no_missing_values_or_duplicate_ids(ev):
    assert not ev[REQUIRED_COLUMNS].isna().any().any()
    assert not ev["ev_id"].duplicated().any()


def test_windows_and_requests_are_valid(ev):
    assert (ev["departure_slot"] > ev["arrival_slot"]).all()
    assert (ev["required_energy_kwh"] > 0).all()
    assert (ev["max_power_kw"] > 0).all()
    assert (ev["arrival_slot"] >= 0).all()
    assert (ev["departure_slot"] <= cfg.N_SLOTS).all()


def test_scenario_contains_individually_infeasible_requests(ev):
    """
    Characterisation test: 2 of the 100 EVs cannot be charged even alone.

    Energy was clipped to [2, 30] kWh and power to [1, 7] kW independently of the
    charging window, so two EVs were sampled with a window too short to take
    their requested energy at their own rated power. Consequences that the
    write-up must state rather than bury:

    * ``demand_met_pct`` can never reach 100%;
    * ``fully_charged_evs`` is capped at 98;
    * the reported unmet energy includes ~2.3 kWh that no scheduler can ever fix.

    The exact count is pinned because the scenario is a frozen build product: if
    it changes, every comparison in ``results/`` is stale and must be redone.
    """
    window_hours = (ev["departure_slot"] - ev["arrival_slot"]) * cfg.STEP_HOURS
    achievable = window_hours * ev["max_power_kw"]
    shortfall = ev["required_energy_kwh"] - achievable
    infeasible = shortfall > 1e-9

    assert int(infeasible.sum()) == 2
    assert float(shortfall[infeasible].sum()) > 0


def test_scenario_is_port_hour_limited(ev):
    """
    Characterisation test: while EVs are connected there are fewer port-hours
    than the fleet needs to charge at its own rated power.

    Only 62 of the 144 slots have any EV connected (arrivals are simulated into
    the 17:00-21:00 window), giving 20 x 62 x 0.25 = 310 port-hours against
    ~342.6 port-hours of work. The site is therefore oversubscribed by roughly
    10%, which is why no method reaches full service. An earlier note quoted 268
    port-hours from results/capacity_feasibility_summary.csv; the conclusion is
    the same but 310 is the correctly derived figure (ports x slots-with-a-
    connected-EV).
    """
    connected = np.zeros(cfg.N_SLOTS, dtype=int)
    for arrival, departure in zip(ev["arrival_slot"], ev["departure_slot"]):
        connected[arrival:departure] += 1

    port_hours = cfg.N_CHARGERS * int((connected > 0).sum()) * cfg.STEP_HOURS
    needed_at_rated_power = float(
        (ev["required_energy_kwh"] / ev["max_power_kw"]).sum()
    )

    assert port_hours == 310.0
    assert needed_at_rated_power > port_hours


def test_grid_limit_is_not_the_binding_constraint(ev):
    """
    Characterisation test: the 125 kW grid cap cannot explain the results.

    The theoretical maximum is ``N_CHARGERS * max(max_power_kw)`` = 131.0 kW, so
    the cap is *structurally* reachable but never approached: the mean 20-port
    ceiling is only ~84.7 kW and every observed peak is 95-99 kW. The real
    binding constraint is the 20 ports against up to 100 simultaneously
    connected EVs, not the grid.

    If this ever fails, the scenario has gained real headroom, the peak-demand
    objective becomes meaningful, and the Tier 1 analysis must be revisited.
    """
    mean_ceiling = cfg.N_CHARGERS * float(ev["max_power_kw"].mean())
    assert mean_ceiling < cfg.MAX_GRID_KW
    assert cfg.N_CHARGERS * float(ev["max_power_kw"].max()) < 1.1 * cfg.MAX_GRID_KW


def test_loader_raises_on_a_broken_scenario(tmp_path, ev):
    """A scenario with a zero-length window must be rejected, not silently used."""
    import pandas as pd

    broken = ev.copy()
    broken.loc[0, "departure_slot"] = broken.loc[0, "arrival_slot"]
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)

    try:
        load_scenario(path)
    except ValueError:
        return
    raise AssertionError("load_scenario accepted a zero-length charging window")


def test_ev_connection_count_far_exceeds_the_charger_count(ev):
    """
    Up to five EVs compete for every port; on average about three do.

    This is why the repair order matters so much: whoever holds a port keeps it,
    so most connected EVs cannot charge in any given slot. The value is what
    makes the fixed heuristic terms inside the repair routine, rather than the
    optimizer's own genes, decide most of the outcome.
    """
    connected = np.zeros(cfg.N_SLOTS, dtype=int)
    for arrival, departure in zip(ev["arrival_slot"], ev["departure_slot"]):
        connected[arrival:departure] += 1

    assert connected.max() == len(ev)          # every EV is present at the peak
    assert connected.max() == 5 * cfg.N_CHARGERS
    assert connected[connected > 0].mean() > 2 * cfg.N_CHARGERS
