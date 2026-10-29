"""
Equal-service comparison: cost and peak per delivered kWh.

Two things are checked here. First, the arithmetic properties that make these
metrics an equal-service comparison at all - the service ceiling partitions the
demand, the percentages use the attainable denominator, and the intensities are
invariant to a uniform change in service volume. Second, the dominance verdicts:
infeasible rows are excluded, exact ties are not dominance, and every
``dominated_by`` claim is re-verified semantically rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as cfg
from baselines import chronological_edf, uncontrolled
from equal_service import (
    EQUAL_SERVICE_COLUMNS,
    equal_service_row,
    equal_service_table,
    flag_pareto,
    frontier_rows,
    per_ev_attainable_energy,
    service_accounting,
)
from helpers import make_inputs, random_priority
from multiobjective_core import evaluate_schedule, repair_schedule

# Dominance keys as (column, 1 for higher-is-better / -1 for lower-is-better).
KEYS = (
    ("service_pct_of_attainable", 1),
    ("cost_per_delivered_kwh", -1),
    ("peak_per_delivered_kwh", -1),
)

UNIFIED = cfg.RESULTS_DIR / "final_unified"
SERVICE_TABLE = UNIFIED / "equal_service_comparison.csv"
COMPARISON = UNIFIED / "comparison_metrics.csv"
BUNDLE = UNIFIED / "schedules.npz"

artifact = pytest.mark.skipif(
    not (SERVICE_TABLE.exists() and COMPARISON.exists() and BUNDLE.exists()),
    reason="results/final_unified is missing - run: python src/run_final_experiments.py",
)


# --------------------------------------------------------------------------
# Service accounting
# --------------------------------------------------------------------------
def test_attainable_and_infeasible_partition_the_demand(ev, solar):
    """Every kWh requested is either physically reachable or provably not."""
    acc = service_accounting(ev, len(solar))
    assert acc["attainable_energy_kwh"] + acc["infeasible_energy_kwh"] == pytest.approx(
        acc["required_energy_kwh"], abs=1e-9
    )
    assert acc["infeasible_evs"] == 2
    assert acc["infeasible_energy_kwh"] == pytest.approx(2.284, abs=1e-3)


def test_attainable_energy_is_the_per_ev_ceiling(ev, solar):
    attainable = per_ev_attainable_energy(ev, len(solar))
    required = ev["required_energy_kwh"].to_numpy(float)
    assert (attainable <= required + 1e-9).all()
    assert (attainable >= 0).all()
    assert attainable.sum() == pytest.approx(
        service_accounting(ev, len(solar))["attainable_energy_kwh"], abs=1e-9
    )


def test_the_infeasible_reference_reaches_the_service_ceiling(ev, solar):
    """
    Ties the accounting to an independent construction.

    ``uncontrolled`` charges each EV at its own maximum power for as long as it
    is connected, which is exactly the per-EV ceiling - so it must deliver the
    attainable energy and no more. If the two definitions ever drift apart, one
    of them is wrong.
    """
    delivered = evaluate_schedule(uncontrolled(ev, solar), ev, solar)[
        "delivered_energy_kwh"
    ]
    assert delivered == pytest.approx(
        service_accounting(ev, len(solar))["attainable_energy_kwh"], abs=1e-6
    )


def test_service_percentages_use_the_attainable_denominator(ev, solar):
    row = equal_service_row("EDF (chronological)", chronological_edf(ev, solar), ev, solar)
    assert row["service_pct_of_attainable"] == pytest.approx(75.02, abs=0.05)
    assert row["service_pct_of_required"] < row["service_pct_of_attainable"] < 100.0


def test_unmet_energy_splits_into_infeasible_and_congestion(ev, solar):
    row = equal_service_row("EDF (chronological)", chronological_edf(ev, solar), ev, solar)
    acc = service_accounting(ev, len(solar))
    assert row["unmet_congestion_kwh"] == pytest.approx(
        row["unmet_energy_kwh"] - acc["infeasible_energy_kwh"], abs=1e-9
    )
    # The physically impossible part is a tiny fraction: the shortfall is
    # overwhelmingly congestion, which is what a scheduler can actually improve.
    assert row["unmet_congestion_kwh"] > 100 * acc["infeasible_energy_kwh"]


# --------------------------------------------------------------------------
# The intensity metrics
# --------------------------------------------------------------------------
def test_intensity_metrics_are_ratios_of_the_evaluated_totals(ev, solar):
    schedule = chronological_edf(ev, solar)
    row = equal_service_row("EDF (chronological)", schedule, ev, solar)
    metrics = evaluate_schedule(schedule, ev, solar)

    assert row["cost_per_delivered_kwh"] == pytest.approx(
        metrics["grid_cost_units"] / metrics["delivered_energy_kwh"], rel=1e-12
    )
    assert row["peak_per_delivered_kwh"] == pytest.approx(
        metrics["peak_ev_load_kw"] / metrics["delivered_energy_kwh"], rel=1e-12
    )
    # Documented identity: the two peak metrics are the same number, one scaled
    # by the horizon. They are reported together because both readings are useful.
    assert row["peak_to_average_ratio"] == pytest.approx(
        row["peak_per_delivered_kwh"] * len(solar) * cfg.STEP_HOURS, rel=1e-12
    )


def test_intensity_is_invariant_to_uniform_scaling_without_solar():
    """
    The property that makes these metrics an equal-service comparison.

    Delivering half as much energy by halving every power leaves cost per
    delivered kWh and peak per delivered kWh exactly unchanged, so a method is
    not rewarded for simply moving less energy.

    The check is run without solar because cost is only exactly linear in the
    schedule when grid import equals EV load. With solar present the import is
    ``max(load - solar, 0)``, which is convex, so a small residual volume effect
    survives - see the module docstring in equal_service.py.
    """
    ev, solar = make_inputs(
        required=[6.0, 6.0], pmax=[3.0, 3.0], arrival=[0, 0], departure=[4, 3],
        n_slots=4,
    )
    full = repair_schedule(random_priority(np.random.default_rng(0), 2, 4), ev, solar)
    half = 0.5 * full

    full_row = equal_service_row("full service", full, ev, solar)
    half_row = equal_service_row("half service", half, ev, solar)

    assert full_row["service_delivered_kwh"] > 0
    assert half_row["service_delivered_kwh"] == pytest.approx(
        0.5 * full_row["service_delivered_kwh"], rel=1e-12
    )
    assert half_row["cost_per_delivered_kwh"] == pytest.approx(
        full_row["cost_per_delivered_kwh"], rel=1e-12
    )
    assert half_row["peak_per_delivered_kwh"] == pytest.approx(
        full_row["peak_per_delivered_kwh"], rel=1e-12
    )


def test_an_empty_schedule_is_refused(ev, solar):
    """A method that delivers nothing has no service to divide by."""
    with pytest.raises(ValueError, match="no service to compare"):
        equal_service_row("empty", np.zeros((len(ev), len(solar))), ev, solar)


# --------------------------------------------------------------------------
# Dominance
# --------------------------------------------------------------------------
def _synthetic(methods):
    return pd.DataFrame(
        [
            {
                "method": name,
                "feasible": True,
                "service_pct_of_attainable": service,
                "cost_per_delivered_kwh": cost,
                "peak_per_delivered_kwh": peak,
            }
            for name, service, cost, peak in methods
        ]
    )


def test_pareto_flags_a_strictly_worse_method():
    flagged = flag_pareto(_synthetic([
        ("good", 75.0, 6.0, 0.08),
        ("worse", 70.0, 6.5, 0.09),
        ("tradeoff", 78.0, 6.9, 0.07),
    ]))
    by_method = flagged.set_index("method")

    assert by_method.loc["good", "pareto_status"] == "frontier"
    assert by_method.loc["worse", "pareto_status"] == "dominated"
    assert by_method.loc["worse", "dominated_by"] == "good"
    # More service but worse on both intensities is a genuine trade-off.
    assert by_method.loc["tradeoff", "pareto_status"] == "frontier"
    assert frontier_rows(flagged)["method"].tolist() == ["good", "tradeoff"]


def test_exact_ties_are_not_dominance():
    """Two identical methods must not eliminate each other."""
    flagged = flag_pareto(_synthetic([("a", 75.0, 6.0, 0.08), ("b", 75.0, 6.0, 0.08)]))
    assert (flagged["pareto_status"] == "frontier").all()


def test_a_single_worse_key_is_enough_to_be_dominated():
    flagged = flag_pareto(_synthetic([
        ("good", 75.0, 6.0, 0.08),
        ("equal_but_peakier", 75.0, 6.0, 0.09),
    ]))
    assert flagged.set_index("method").loc["equal_but_peakier", "pareto_status"] == "dominated"


def test_infeasible_rows_are_labeled_and_never_dominate():
    """
    The infeasible reference delivers the most energy of any row, so if it were
    allowed into the ranking it would eliminate every honest scheduler.
    """
    frame = pd.DataFrame([
        {
            "method": "honest", "feasible": True,
            "service_pct_of_attainable": 75.0,
            "cost_per_delivered_kwh": 6.0, "peak_per_delivered_kwh": 0.08,
        },
        {
            "method": "cheater", "feasible": False,
            "service_pct_of_attainable": 100.0,
            "cost_per_delivered_kwh": 20.0, "peak_per_delivered_kwh": 0.20,
        },
    ])
    by_method = flag_pareto(frame).set_index("method")
    assert by_method.loc["cheater", "pareto_status"] == "infeasible reference"
    assert by_method.loc["honest", "pareto_status"] == "frontier"
    assert by_method.loc["honest", "dominated_by"] == ""


# --------------------------------------------------------------------------
# The committed artifact
# --------------------------------------------------------------------------
@artifact
def test_committed_table_has_the_canonical_columns():
    frame = pd.read_csv(SERVICE_TABLE)
    assert list(frame.columns) == EQUAL_SERVICE_COLUMNS


@artifact
def test_committed_table_covers_the_same_methods_as_the_volume_table():
    service = pd.read_csv(SERVICE_TABLE)
    comparison = pd.read_csv(COMPARISON)
    assert sorted(service["method"]) == sorted(comparison["method"])


@artifact
def test_committed_intensities_agree_with_the_volume_table():
    service = pd.read_csv(SERVICE_TABLE).set_index("method")
    comparison = pd.read_csv(COMPARISON).set_index("method")
    for name in service.index:
        assert service.loc[name, "service_delivered_kwh"] == pytest.approx(
            comparison.loc[name, "delivered_energy_kwh"], rel=1e-12
        )
        assert service.loc[name, "cost_per_delivered_kwh"] == pytest.approx(
            comparison.loc[name, "cost_per_delivered_kwh"], rel=1e-12
        )


@artifact
def test_no_feasible_method_over_reaches_the_service_ceiling():
    frame = pd.read_csv(SERVICE_TABLE)
    feasible = frame[frame["feasible"]]
    assert (feasible["service_pct_of_attainable"] <= 100.0 + 1e-9).all()
    assert (feasible["service_pct_of_attainable"] > 60.0).all()


@artifact
def test_frontier_and_dominated_rows_are_consistently_annotated():
    frame = pd.read_csv(SERVICE_TABLE)
    frontier = frame[frame["pareto_status"] == "frontier"]
    dominated = frame[frame["pareto_status"] == "dominated"]

    assert len(frontier) > 1  # a trade-off exists; this is not a single winner
    assert dominated["dominated_by"].notna().all()
    assert frontier["dominated_by"].isna().all()
    # No method may dominate itself. Compare parsed names, not substrings: "PSO"
    # is contained in "PSO rolling horizon", so a substring test would pass
    # vacuously for the wrong reason.
    assert all(
        name not in [entry for entry in str(dominated_by).split("; ") if entry]
        for name, dominated_by in zip(dominated["method"], dominated["dominated_by"])
    )


@artifact
def test_the_infeasible_reference_never_appears_as_a_dominator():
    frame = pd.read_csv(SERVICE_TABLE)
    claimed = frame["dominated_by"].dropna().astype(str)
    assert not claimed.str.contains("Uncontrolled").any()


@artifact
def test_every_dominance_verdict_holds_under_recomputation(ev, solar):
    """
    Re-derive the table from the shipped schedules and verify each verdict
    semantically: a dominator must be at least as good on all three keys and
    strictly better on one, and a frontier row must have no dominator at all.
    """
    bundle = dict(np.load(BUNDLE))
    table = equal_service_table(bundle, ev, solar).set_index("method")
    feasible = table[table["feasible"]]

    def worse_than(candidate, other):
        """True when ``other`` is at least as good everywhere and better once."""
        at_least_as_good = all(
            sign * other[column] >= sign * candidate[column]
            - 1e-9 * max(abs(sign * candidate[column]), 1.0)
            for column, sign in KEYS
        )
        strictly_better = any(
            sign * other[column] > sign * candidate[column]
            + 1e-9 * max(abs(sign * candidate[column]), 1.0)
            for column, sign in KEYS
        )
        return at_least_as_good and strictly_better

    for name, row in feasible.iterrows():
        dominators = [d for d in str(row["dominated_by"]).split("; ") if d]
        recomputed = [
            other_name
            for other_name, other in feasible.iterrows()
            if other_name != name and worse_than(row, other)
        ]
        assert sorted(dominators) == sorted(recomputed), name
