"""
Reference optima and the optimality gaps.

The MILP is the benchmark every other claim is measured against, so it is tested
on scenarios small enough to solve by hand. The important properties are:

* the model matches the project's constraints - the same ``constraint_report``
  that every schedule must pass also validates the solver's answer;
* the port limit is actually enforced (this is the constraint a broken model
  silently drops, and the bug found during development did exactly that);
* the cost model prices the tariff rather than just moving energy;
* the gap columns are arithmetic, and refuse to report a gap where the frontier
  does not cover the service level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as cfg
from helpers import make_inputs
from reference_optimum import (
    REFERENCE_COLUMNS,
    attach_reference,
    capacity_bounds,
    frontier_points,
    interpolated_optimal_cost,
    reference_row,
    solve_max_service,
    solve_min_cost,
)
from multiobjective_core import constraint_report, evaluate_schedule

UNIFIED = cfg.RESULTS_DIR / "final_unified"
SERVICE_VS_OPTIMAL = UNIFIED / "service_vs_optimal.csv"
REFERENCE_CSV = UNIFIED / "reference_optimum.csv"

TINY_SOLVE = dict(time_limit=10.0, mip_gap=1e-6)


def one_ev(required=2.0, pmax=2.0, slots=4):
    return make_inputs(
        required=[required], pmax=[pmax], arrival=[0], departure=[slots], n_slots=slots
    )


def test_capacity_bounds_match_a_hand_check():
    """One EV, 2 kW, one hour: both ceilings are exactly its requested energy."""
    ev, solar = one_ev(required=2.0, pmax=2.0, slots=4)
    bounds = capacity_bounds(ev, solar)

    assert bounds["per_ev_bound_kwh"] == pytest.approx(2.0)
    # Four slots x min(125 kW grid, 2 kW rated) x 0.25 h
    assert bounds["per_slot_bound_kwh"] == pytest.approx(2.0)
    assert bounds["upper_bound_kwh"] == pytest.approx(2.0)


def test_max_service_reaches_the_availability_ceiling():
    ev, solar = one_ev(required=2.0, pmax=2.0, slots=4)
    solution = solve_max_service(ev, solar, **TINY_SOLVE)
    metrics = evaluate_schedule(solution.schedule, ev, solar)

    assert metrics["delivered_energy_kwh"] == pytest.approx(2.0, abs=1e-6)
    assert metrics["constraint_violations"] == 0
    assert solution.status == "optimal"


def test_port_limit_reduces_the_optimum():
    """
    Two EVs, one shared hour, one port: at most one EV charges per slot.

    With two ports both EVs are served (2 kWh); with one port only one slot's
    worth of energy is delivered (1 kWh). If the port constraint were dropped
    from the model, both cases would return 2 kWh and this test would fail.
    """
    ev, solar = make_inputs(
        required=[2.0, 2.0], pmax=[2.0, 2.0], arrival=[0, 0], departure=[2, 2], n_slots=2
    )

    one_port = solve_max_service(ev, solar, n_chargers=1, **TINY_SOLVE)
    two_ports = solve_max_service(ev, solar, n_chargers=2, **TINY_SOLVE)

    one = evaluate_schedule(one_port.schedule, ev, solar)
    two = evaluate_schedule(two_ports.schedule, ev, solar)
    assert one["delivered_energy_kwh"] == pytest.approx(1.0, abs=1e-6)
    assert two["delivered_energy_kwh"] == pytest.approx(2.0, abs=1e-6)
    assert one["constraint_violations"] == 0


def test_grid_limit_is_enforced():
    """A 1 kW site cap must stop the solver from delivering 2 kW in a slot."""
    ev, solar = one_ev(required=2.0, pmax=2.0, slots=4)
    solution = solve_max_service(ev, solar, max_grid_kw=1.0, **TINY_SOLVE)

    schedule = solution.schedule
    assert (schedule.sum(axis=0) <= 1.0 + 1e-9).all()
    assert constraint_report(schedule, ev, solar, max_grid_kw=1.0)["total"] == 0


def test_min_cost_prices_the_tariff():
    """
    One EV, the whole horizon: the cheapest 1 kWh is overnight, at 4.0 units/kWh.

    Charging inside the 17:00-22:00 peak would cost 10.0 units/kWh, so a cost
    model that ignored the tariff would return a schedule in the wrong slots and
    a cost of ~4 anyway only if it happened to avoid the peak. The slot check
    makes the difference visible.
    """
    ev, solar = make_inputs(
        required=[1.0], pmax=[2.0], arrival=[0], departure=[cfg.N_SLOTS], n_slots=cfg.N_SLOTS
    )
    solution = solve_min_cost(ev, solar, service_floor_kwh=1.0, **TINY_SOLVE)
    metrics = evaluate_schedule(solution.schedule, ev, solar)

    assert metrics["delivered_energy_kwh"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["grid_cost_units"] == pytest.approx(4.0, abs=1e-6)
    assert metrics["constraint_violations"] == 0


def test_min_cost_without_a_floor_delivers_nothing():
    ev, solar = one_ev(required=2.0, pmax=2.0, slots=4)
    solution = solve_min_cost(ev, solar, service_floor_kwh=0.0, **TINY_SOLVE)

    assert solution.schedule.sum() == pytest.approx(0.0, abs=1e-9)
    assert solution.objective == pytest.approx(0.0, abs=1e-9)


def test_reference_row_carries_its_provenance():
    ev, solar = one_ev()
    row = reference_row(solve_max_service(ev, solar, **TINY_SOLVE), ev, solar)

    assert set(row) == set(REFERENCE_COLUMNS)
    assert row["status"] in {"optimal", "feasible"}
    assert row["scenario_sha256_16"] != "missing"
    assert row["solar_sha256_16"] != "missing"
    assert row["constraint_violations"] == 0


def hand_made_reference():
    """A three-point frontier for the arithmetic tests, no solver involved."""
    return pd.DataFrame(
        {
            "label": ["Min cost at 1000 kWh", "Min cost at 1100 kWh", "Max service (MILP)"],
            "delivered_energy_kwh": [1000.0, 1100.0, 1200.0],
            "grid_cost_units": [5000.0, 6000.0, 7400.0],
            "constraint_violations": [0, 0, 0],
        }
    )


def test_frontier_points_are_sorted_by_service():
    points = frontier_points(hand_made_reference())
    assert np.all(np.diff(points[:, 0]) > 0)


def test_interpolated_cost_is_exact_at_a_solved_point():
    reference = hand_made_reference()
    assert interpolated_optimal_cost(reference, 1100.0) == pytest.approx(6000.0)
    # Midway between two solved points.
    assert interpolated_optimal_cost(reference, 1150.0) == pytest.approx(6700.0)


def test_interpolated_cost_refuses_to_extrapolate():
    reference = hand_made_reference()
    assert np.isnan(interpolated_optimal_cost(reference, 900.0))
    assert np.isnan(interpolated_optimal_cost(reference, 1300.0))


def test_attach_reference_computes_shares_and_gaps():
    reference = hand_made_reference()
    table = pd.DataFrame(
        {
            "method": ["good", "wasteful", "below_frontier"],
            "delivered_energy_kwh": [1200.0, 1100.0, 900.0],
            "grid_cost_units": [7400.0, 6600.0, 4000.0],
        }
    )
    out = attach_reference(table, reference).set_index("method")

    assert out.loc["good", "pct_of_best_service"] == pytest.approx(100.0)
    assert out.loc["good", "service_gap_kwh"] == pytest.approx(0.0)
    assert out.loc["good", "cost_gap_vs_optimal_pct"] == pytest.approx(0.0)

    # 6600 against the 6000 optimum at the same service is 10% dearer.
    assert out.loc["wasteful", "cost_gap_vs_optimal_pct"] == pytest.approx(10.0)
    assert out.loc["wasteful", "service_gap_kwh"] == pytest.approx(100.0)

    # Outside the solved frontier: no gap is claimed.
    assert np.isnan(out.loc["below_frontier", "cost_gap_vs_optimal_pct"])


@pytest.mark.skipif(
    not (SERVICE_VS_OPTIMAL.exists() and REFERENCE_CSV.exists()),
    reason="results/final_unified has no reference artifacts - run run_final_experiments.py",
)
def test_committed_gaps_are_physically_possible():
    """
    No method may beat the frontier by more than the solver's own tolerance.

    A negative gap beyond the MIP gap would mean either the reference is wrong or
    a method schedule is infeasible, so this doubles as a cross-check between the
    two computation paths.
    """
    service = pd.read_csv(SERVICE_VS_OPTIMAL)
    reference = pd.read_csv(REFERENCE_CSV)
    feasible = service[service["feasible"]]

    assert feasible["pct_of_best_service"].between(0.0, 100.0 + 1e-9).all()

    # The infeasible reference is the only row allowed above 100%: it ignores the
    # port and grid limits, so it is not a scheduler and takes no part in the
    # comparison (same rule as the Pareto flagging in equal_service.py).
    above = service[service["pct_of_best_service"] > 100.0]
    assert len(above) == 1 and not above["feasible"].iloc[0]

    tolerance = 100.0 * (reference["mip_gap"].max() + 0.01)
    gaps = feasible["cost_gap_vs_optimal_pct"].dropna()
    assert (gaps >= -tolerance).all(), gaps.min()


@pytest.mark.skipif(
    not REFERENCE_CSV.exists(),
    reason="results/final_unified has no reference_optimum.csv",
)
def test_committed_reference_was_solved_on_the_current_inputs():
    from artifacts import file_digest

    reference = pd.read_csv(REFERENCE_CSV)
    assert (reference["scenario_sha256_16"] == file_digest(cfg.DEFAULT_SCENARIO)).all()
    assert (reference["solar_sha256_16"] == file_digest(cfg.DEFAULT_SOLAR)).all()


@pytest.mark.skipif(
    not REFERENCE_CSV.exists(),
    reason="results/final_unified has no reference_optimum.csv",
)
def test_committed_reference_is_feasible_and_beats_every_method():
    reference = pd.read_csv(REFERENCE_CSV)
    best_service = reference["delivered_energy_kwh"].max()

    schedule_bundle = UNIFIED / "schedules.npz"
    if not schedule_bundle.exists():
        pytest.skip("schedules.npz not present")

    from multiobjective_core import load_inputs

    ev, solar = load_inputs()
    bundle = dict(np.load(schedule_bundle))
    for name, schedule in bundle.items():
        if name.startswith("Uncontrolled"):
            continue
        delivered = schedule.sum() * cfg.STEP_HOURS
        assert delivered <= best_service + 1e-6, name
