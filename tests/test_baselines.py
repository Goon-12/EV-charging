"""
Conventional baselines.

Replaces the print-only ``validate_baselines.py``. The behavioural test at the
bottom is the important one: it demonstrates on a two-EV case that
"FCFS (chronological)" and "FCFS-priority (repair-based)" are genuinely
different policies, which is why the pre-Tier-0 table could not claim that the
schedulers lost to first-come-first-served.
"""
from __future__ import annotations

import numpy as np
import pytest

import config as cfg
from baselines import POLICIES, all_baselines, chronological_edf, chronological_fcfs, priority_fcfs
from helpers import make_inputs
from multiobjective_core import constraint_report

FEASIBLE_POLICIES = [name for name in POLICIES if "Uncontrolled" not in name]


@pytest.fixture(scope="module")
def baseline_schedules(ev, solar):
    return all_baselines(ev, solar)


@pytest.mark.parametrize("name", FEASIBLE_POLICIES)
def test_feasible_baselines_obey_every_hard_constraint(
    name, baseline_schedules, ev, solar
):
    report = constraint_report(baseline_schedules[name], ev, solar)
    assert report["total"] == 0, report


def test_uncontrolled_is_infeasible_by_construction(baseline_schedules, ev, solar):
    report = constraint_report(
        baseline_schedules["Uncontrolled (infeasible reference)"], ev, solar
    )
    assert report["total"] > 0
    assert report["grid_violations"] > 0 or report["charger_violations"] > 0


@pytest.mark.parametrize("name", list(POLICIES))
def test_baselines_are_deterministic(name, baseline_schedules, ev, solar):
    assert np.array_equal(baseline_schedules[name], POLICIES[name](ev, solar))


@pytest.mark.parametrize("name", FEASIBLE_POLICIES)
def test_feasible_baselines_deliver_energy(name, baseline_schedules):
    assert baseline_schedules[name].sum() > 0


def test_chronological_fcfs_holds_the_port_for_the_earliest_arrival():
    """
    Two EVs, one port. The earlier arrival keeps the port and finishes;
    the later arrival is starved.

    EV0 arrives at slot 0 and departs at 4; EV1 arrives at slot 1 and departs at
    slot 3, so EV1 has the tighter deadline.
    """
    ev, solar = make_inputs(
        required=[4.0, 4.0], pmax=[4.0, 4.0],
        arrival=[0, 1], departure=[4, 3], n_slots=4,
    )
    fcfs = chronological_fcfs(ev, solar, max_grid_kw=1000.0, n_chargers=1)

    assert fcfs[0].sum() * cfg.STEP_HOURS == pytest.approx(4.0)
    assert fcfs[1].sum() == pytest.approx(0.0)


def test_chronological_edf_gives_the_port_to_the_tighter_deadline():
    """Same two EVs, but EDF serves the EV that departs first."""
    ev, solar = make_inputs(
        required=[4.0, 4.0], pmax=[4.0, 4.0],
        arrival=[0, 1], departure=[4, 3], n_slots=4,
    )
    edf = chronological_edf(ev, solar, max_grid_kw=1000.0, n_chargers=1)

    # Neither EV can finish: between them they hold the single port for 4 slots.
    assert edf[1].sum() * cfg.STEP_HOURS == pytest.approx(2.0)
    assert edf[0].sum() * cfg.STEP_HOURS == pytest.approx(2.0)


def test_priority_baselines_are_not_chronological_policies(ev, solar):
    """
    Regression test for the mislabelled baseline.

    If ``priority_fcfs`` were operationally FCFS, this would fail. It is a fixed
    priority vector fed through the repair heuristic, so on this scenario it
    produces a different schedule from a real chronological queue.
    """
    assert not np.array_equal(priority_fcfs(ev, solar), chronological_fcfs(ev, solar))
    assert "priority" in "FCFS-priority (repair-based)"
