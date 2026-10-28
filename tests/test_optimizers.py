"""
GA and PSO contracts.

Small settings throughout: these tests check the *contract* (deterministic under
a seed, always feasible, reports its history), not solution quality. Solution
quality is what Tier 2 addresses, and it needs a benchmark rather than a unit
test.
"""
from __future__ import annotations

import numpy as np
import pytest

import config as cfg
from multiobjective_core import constraint_report
from multiobjective_optimizers import optimize_ga, optimize_pso
from run_final_experiments import count_ga_evaluations, count_pso_evaluations

# ``polish_evaluations=0`` keeps these contract tests fast; the memetic budget is
# exercised separately in test_genome.py and in the budget arithmetic below.
GA_TINY = dict(population_size=4, generations=2, seed=123, polish_evaluations=0)
PSO_TINY = dict(particles=4, iterations=2, seed=123, polish_evaluations=0)


def test_ga_is_deterministic_for_a_fixed_seed(ev, solar):
    first, score_a = optimize_ga(ev, solar, **GA_TINY)
    second, score_b = optimize_ga(ev, solar, **GA_TINY)

    assert np.array_equal(first, second)
    assert score_a == score_b


def test_pso_is_deterministic_for_a_fixed_seed(ev, solar):
    first, score_a = optimize_pso(ev, solar, **PSO_TINY)
    second, score_b = optimize_pso(ev, solar, **PSO_TINY)

    assert np.array_equal(first, second)
    assert score_a == score_b


def test_different_seeds_explore_different_schedules(ev, solar):
    """
    The random draw must actually drive the search.

    Tested with ``warm_start=False``: the warm-start genomes are seed-independent
    by construction and can dominate a tiny budget, so a seeded run can legally
    reach the same schedule from any seed. Without them, the population is pure
    random and two seeds must not collapse onto the same first search path.
    """
    first, _ = optimize_ga(
        ev, solar, population_size=4, generations=2, seed=1,
        polish_evaluations=0, warm_start=False,
    )
    second, _ = optimize_ga(
        ev, solar, population_size=4, generations=2, seed=2,
        polish_evaluations=0, warm_start=False,
    )

    assert not np.array_equal(first, second)


def test_ga_returns_a_feasible_schedule(ev, solar):
    schedule, _ = optimize_ga(ev, solar, **GA_TINY)
    assert constraint_report(schedule, ev, solar)["total"] == 0


def test_pso_returns_a_feasible_schedule(ev, solar):
    schedule, _ = optimize_pso(ev, solar, **PSO_TINY)
    assert constraint_report(schedule, ev, solar)["total"] == 0


def test_ga_records_one_history_entry_per_generation(ev, solar):
    history = []
    optimize_ga(
        ev, solar, population_size=4, generations=3, seed=7, history=history,
        polish_evaluations=0,
    )
    assert len(history) == 3
    assert all(np.isfinite(history))


def test_pso_records_one_history_entry_per_iteration(ev, solar):
    history = []
    optimize_pso(
        ev, solar, particles=3, iterations=3, seed=7, history=history,
        polish_evaluations=0,
    )
    assert len(history) == 3


def test_ga_history_is_non_increasing(ev, solar):
    """Best-so-far must never get worse: that would mean the elite was lost."""
    history = []
    optimize_ga(
        ev, solar, population_size=6, generations=5, seed=11, history=history,
        polish_evaluations=0,
    )

    for earlier, later in zip(history, history[1:]):
        assert later <= earlier + 1e-12


def test_ga_handles_zero_generations(ev, solar):
    """With no generations it must still return a decoded, feasible candidate."""
    schedule, _ = optimize_ga(
        ev, solar, population_size=3, generations=0, seed=5, polish_evaluations=0
    )

    assert schedule.shape == (len(ev), len(solar))
    assert constraint_report(schedule, ev, solar)["total"] == 0


def test_ga_population_one_is_feasible(ev, solar):
    """A degenerate population must not break the loop."""
    schedule, _ = optimize_ga(
        ev, solar, population_size=1, generations=2, seed=5, polish_evaluations=0
    )
    assert constraint_report(schedule, ev, solar)["total"] == 0


def test_evaluation_budget_arithmetic():
    """
    Elites keep their score, children are evaluated, polish is counted exactly.

    Guards the Tier 0 memory fix (elites are not re-evaluated) and the Tier 2
    claim that the reported budget is the real number of objective calls: the
    memetic pass returns the number of evaluations it used, so a run that claims
    10,000 evaluations performed 10,000 of them.
    """
    # 20 + 30 * (20 - 4) with the configured elite count of 4.
    assert count_ga_evaluations(population_size=20, generations=30) == 500
    assert count_ga_evaluations(population_size=8, generations=5, polish_evaluations=20) == 48
    assert count_ga_evaluations(population_size=4, generations=0) == 4
    assert count_pso_evaluations(particles=12, iterations=15) == 192
    assert count_pso_evaluations(particles=6, iterations=4, polish_evaluations=20) == 50


def test_ga_and_pso_are_never_better_than_perfect_service(ev, solar):
    """Sanity bound: no schedule can deliver more than the fleet requested."""
    total_required = float(ev["required_energy_kwh"].sum())
    for optimizer, kwargs in ((optimize_ga, GA_TINY), (optimize_pso, PSO_TINY)):
        schedule, _ = optimizer(ev, solar, **kwargs)
        assert schedule.sum() * cfg.STEP_HOURS <= total_required + 1e-6
