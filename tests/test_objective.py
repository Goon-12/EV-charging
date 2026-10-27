"""
Cost, peak and objective assertions.

Replaces ``test_objective.py`` and ``test_cost_evaluation.py``, which tested the
now-archived evaluators (``evaluate_objective.py``, ``evaluate_cost.py``). Those
two computed cost slightly differently from each other and from
``evaluate_schedule.py``, which is why the same schedule could be reported with
three different prices. There is now one evaluator, so there is one cost.
"""
from __future__ import annotations

import numpy as np
import pytest

import config as cfg
from helpers import make_inputs
from multiobjective_core import evaluate_schedule, objective, repair_schedule


def known_case(solar_kw=None):
    """
    Two EVs, two slots, both connected throughout.

    slot 0: 4 kW, slot 1: 2 kW, both 15 minutes and both off-peak (tariff 4.0).
    Energy: EV0 1.0 kWh, EV1 0.5 kWh - exactly their requests.
    """
    ev, solar = make_inputs(
        required=[1.0, 0.5], pmax=[4.0, 2.0],
        arrival=[0, 0], departure=[2, 2], n_slots=2, solar_kw=solar_kw,
    )
    schedule = np.array([[4.0, 0.0], [0.0, 2.0]])
    return ev, solar, schedule


def test_cost_peak_and_energy_on_a_known_schedule():
    ev, solar, schedule = known_case()
    metrics = evaluate_schedule(schedule, ev, solar)

    # (4 kW + 2 kW) * 0.25 h * 4.0 units/kWh
    assert metrics["grid_cost_units"] == pytest.approx(6.0)
    assert metrics["peak_ev_load_kw"] == pytest.approx(4.0)
    assert metrics["delivered_energy_kwh"] == pytest.approx(1.5)
    assert metrics["unmet_energy_kwh"] == pytest.approx(0.0)
    assert metrics["demand_met_pct"] == pytest.approx(100.0)
    assert metrics["fully_charged_evs"] == 2
    assert metrics["constraint_violations"] == 0


def test_both_evs_charge_at_their_rated_power():
    """Both EVs are at 100% of rated power, so the stress proxy must be 1.0."""
    ev, solar, schedule = known_case()
    assert evaluate_schedule(schedule, ev, solar)["battery_stress_proxy"] == pytest.approx(1.0)


def test_solar_offsets_grid_import_and_cost():
    """Solar covers slot 0 entirely, so only slot 1 is billed."""
    ev, solar, schedule = known_case(solar_kw=[8.0, 0.0])
    metrics = evaluate_schedule(schedule, ev, solar)

    assert metrics["grid_cost_units"] == pytest.approx(2.0)
    assert metrics["solar_used_kwh"] == pytest.approx(1.0)
    assert metrics["peak_grid_import_kw"] == pytest.approx(2.0)


def test_tariff_is_applied_by_time_of_day():
    """The same energy in the evening peak must cost more than overnight."""
    ev, solar = make_inputs(
        required=[4.0, 4.0], pmax=[4.0, 4.0],
        arrival=[0, 68], departure=[4, 72], n_slots=cfg.N_SLOTS,
    )
    overnight = np.zeros((2, cfg.N_SLOTS))
    overnight[0, 0:4] = 4.0
    peak = np.zeros((2, cfg.N_SLOTS))
    peak[1, 68:72] = 4.0

    cost_overnight = evaluate_schedule(overnight, ev, solar)["grid_cost_units"]
    cost_peak = evaluate_schedule(peak, ev, solar)["grid_cost_units"]

    assert cost_overnight > 0
    assert cost_peak > cost_overnight
    assert cost_peak / cost_overnight == pytest.approx(cfg.TARIFF_PEAK / cfg.TARIFF_OFFPEAK)


def test_empty_schedule_delivers_nothing_and_is_scored_badly(ev, solar):
    empty = np.zeros((len(ev), len(solar)))
    metrics = evaluate_schedule(empty, ev, solar)

    assert metrics["delivered_energy_kwh"] == 0.0
    assert metrics["peak_ev_load_kw"] == 0.0
    assert metrics["unmet_energy_kwh"] == pytest.approx(
        ev["required_energy_kwh"].sum()
    )

    rng = np.random.default_rng(3)
    feasible = repair_schedule(rng.random((len(ev), len(solar))), ev, solar)
    assert objective(empty, ev, solar) > objective(feasible, ev, solar)


def test_infeasibility_penalty_dominates_the_soft_terms():
    """A constraint-violating schedule must never out-score a feasible one."""
    ev, solar = make_inputs(
        required=[1.0], pmax=[1.0], arrival=[0], departure=[4], n_slots=4
    )
    feasible = np.array([[1.0, 0.0, 0.0, 0.0]])
    infeasible = np.array([[10.0, 10.0, 10.0, 10.0]])  # exceeds power and energy

    assert evaluate_schedule(feasible, ev, solar)["constraint_violations"] == 0
    assert evaluate_schedule(infeasible, ev, solar)["constraint_violations"] > 0
    assert objective(infeasible, ev, solar) >= cfg.INFEASIBILITY_PENALTY
    assert objective(feasible, ev, solar) < objective(infeasible, ev, solar)


def test_objective_is_deterministic_and_finite(ev, solar):
    rng = np.random.default_rng(5)
    schedule = repair_schedule(rng.random((len(ev), len(solar))), ev, solar)
    first = objective(schedule, ev, solar)
    second = objective(schedule, ev, solar)

    assert np.isfinite(first)
    assert isinstance(first, float)
    assert first == second


def test_objective_prints_nothing(ev, solar, capsys):
    """
    Regression test: the objective used to print three diagnostic lines per
    call, which meant a 560-evaluation GA emitted ~1700 lines of console noise.
    """
    rng = np.random.default_rng(9)
    schedule = repair_schedule(rng.random((len(ev), len(solar))), ev, solar)
    objective(schedule, ev, solar)
    objective(schedule, ev, solar)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
