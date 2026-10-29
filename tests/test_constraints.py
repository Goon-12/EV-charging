"""
Hard-constraint invariants of the shared repair routine.

This replaces ``test_repair_quality.py``, which printed diagnostics and always
exited 0, so a broken repair could never fail a run. The previous repair had a
final stage that *added* energy no candidate asked for; the assertions below are
what would have caught that.

Everything here is a property that must hold for ANY input, so the tests feed
both random and adversarial priority matrices through the real repair routine.
"""
from __future__ import annotations

import numpy as np
import pytest

import config as cfg
from helpers import make_inputs, random_priority
from multiobjective_core import (
    constraint_report,
    evaluate_schedule,
    make_windows,
    repair_schedule,
)


def assert_feasible(schedule, ev, solar):
    report = constraint_report(schedule, ev, solar)
    assert report["total"] == 0, report
    return report


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_priorities_always_repair_to_a_feasible_schedule(ev, solar, seed):
    """The core promise: repair output never violates a hard constraint."""
    rng = np.random.default_rng(seed)
    priority = random_priority(rng, len(ev), len(solar))
    assert_feasible(repair_schedule(priority, ev, solar), ev, solar)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_degenerate_priorities_are_feasible(ev, solar, value):
    """All-zero and all-one priorities are the worst cases for a greedy repair."""
    priority = np.full((len(ev), len(solar)), value)
    assert_feasible(repair_schedule(priority, ev, solar), ev, solar)


def test_single_ev_greedy_priority_is_feasible(ev, solar):
    priority = np.zeros((len(ev), len(solar)))
    priority[0, :] = 1.0
    assert_feasible(repair_schedule(priority, ev, solar), ev, solar)


def test_repair_never_delivers_more_than_requested(ev, solar):
    rng = np.random.default_rng(7)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    delivered = schedule.sum(axis=1) * cfg.STEP_HOURS
    assert (delivered <= ev["required_energy_kwh"].to_numpy() + 1e-6).all()


def test_repair_only_charges_inside_connection_windows(ev, solar):
    rng = np.random.default_rng(11)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    a, d = make_windows(ev, len(solar))

    for i in range(len(ev)):
        assert np.all(schedule[i, : a[i]] == 0)
        assert np.all(schedule[i, d[i]:] == 0)


def test_repair_respects_individual_power_limits(ev, solar):
    rng = np.random.default_rng(13)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    pmax = ev["max_power_kw"].to_numpy(float)
    assert (schedule <= pmax[:, None] + 1e-9).all()


def test_repair_respects_the_charger_count(ev, solar):
    rng = np.random.default_rng(17)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    active = (schedule > 1e-9).sum(axis=0)
    assert (active <= cfg.N_CHARGERS).all()


def test_repair_respects_the_grid_limit_after_solar(ev, solar):
    rng = np.random.default_rng(19)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    grid_import = np.maximum(schedule.sum(axis=0) - np.maximum(solar, 0), 0)
    assert (grid_import <= cfg.MAX_GRID_KW + 1e-6).all()


def test_repair_never_produces_negative_or_non_finite_power(ev, solar):
    rng = np.random.default_rng(23)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    assert np.isfinite(schedule).all()
    assert (schedule >= 0).all()


def test_evaluator_agrees_with_the_constraint_report(ev, solar):
    """The single reported total must equal the sum of the per-constraint counts."""
    rng = np.random.default_rng(29)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    report = constraint_report(schedule, ev, solar)
    metrics = evaluate_schedule(schedule, ev, solar)
    assert metrics["constraint_violations"] == report["total"]


def test_evaluator_detects_each_kind_of_violation(ev, solar):
    """A deliberately broken schedule must be reported as infeasible."""
    base = repair_schedule(
        random_priority(np.random.default_rng(31), len(ev), len(solar)), ev, solar
    )

    before = constraint_report(base, ev, solar)["total"]
    assert before == 0

    broken = base.copy()
    broken[0, 0] = 1e6
    report = constraint_report(broken, ev, solar)
    assert report["power_violations"] > 0
    assert report["total"] > 0


def test_repair_rejects_mismatched_dimensions(ev, solar):
    with pytest.raises(ValueError):
        repair_schedule(np.zeros((len(ev) + 1, len(solar))), ev, solar)


def test_fully_charged_count_counts_completed_evs(ev, solar):
    rng = np.random.default_rng(37)
    schedule = repair_schedule(random_priority(rng, len(ev), len(solar)), ev, solar)
    metrics = evaluate_schedule(schedule, ev, solar)
    delivered = schedule.sum(axis=1) * cfg.STEP_HOURS
    unmet = np.maximum(ev["required_energy_kwh"].to_numpy() - delivered, 0.0)
    assert metrics["fully_charged_evs"] == int((unmet <= 1e-3).sum())


def test_repair_writes_nothing_outside_the_horizon_for_empty_windows():
    """An EV whose departure equals its arrival must receive zero energy."""
    ev, solar = make_inputs(
        required=[2.0, 2.0], pmax=[3.0, 3.0],
        arrival=[0, 2], departure=[4, 2], n_slots=4,
    )
    schedule = repair_schedule(np.ones((2, 4)), ev, solar)
    assert schedule[1].sum() == 0
    assert_feasible(schedule, ev, solar)
