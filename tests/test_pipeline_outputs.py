"""
Provenance and consistency of the committed results.

This module exists because of a concrete failure found during the Tier 0 review:
the committed ``results/final_unified/comparison_metrics.csv`` was a ``--quick``
smoke test. Nothing in the file said so, and its GA delivered 1089 kWh, which
looks like a plausible headline number next to the full run's 1116 kWh. The fix
was a run manifest; these tests are what keep the fix honest.

These tests read artifacts, so they skip if the pipeline has not been run yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config as cfg
from artifacts import RESULT_COLUMNS, load_schedule_csv, metrics_row
from multiobjective_core import constraint_report
from run_final_experiments import count_ga_evaluations, file_digest

UNIFIED = cfg.RESULTS_DIR / "final_unified"
COMPARISON = UNIFIED / "comparison_metrics.csv"
MANIFEST = UNIFIED / "run_manifest.csv"
BUNDLE = UNIFIED / "schedules.npz"

pytestmark = pytest.mark.skipif(
    not (COMPARISON.exists() and MANIFEST.exists() and BUNDLE.exists()),
    reason="results/final_unified is missing - run: python src/run_final_experiments.py",
)


@pytest.fixture(scope="module")
def comparison():
    return pd.read_csv(COMPARISON)


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(MANIFEST).iloc[0]


@pytest.fixture(scope="module")
def bundle():
    return dict(np.load(BUNDLE))


def test_committed_results_are_a_full_run(manifest):
    """The headline table must not be a smoke test."""
    assert manifest["mode"] == "full"
    assert bool(manifest["rolling_horizon"]) in (True, False)  # recorded either way


def test_manifest_hashes_match_the_current_inputs(manifest):
    """If the scenario changes, the committed numbers are stale and must be regenerated."""
    assert manifest["scenario_sha256_16"] == file_digest(cfg.DEFAULT_SCENARIO)
    assert manifest["solar_sha256_16"] == file_digest(cfg.DEFAULT_SOLAR)


def test_manifest_describes_the_loaded_scenario(manifest, ev, solar):
    assert manifest["n_ev"] == len(ev)
    assert manifest["n_slots"] == len(solar)
    assert manifest["n_chargers"] == cfg.N_CHARGERS
    assert manifest["max_grid_kw"] == cfg.MAX_GRID_KW


def test_comparison_table_has_the_canonical_columns(comparison):
    assert list(comparison.columns) == RESULT_COLUMNS


def test_every_feasible_row_really_is_feasible(comparison):
    feasible = comparison[~comparison["method"].str.startswith("Uncontrolled")]
    assert (feasible["constraint_violations"] == 0).all()
    assert feasible["feasible"].all()


def test_the_infeasible_reference_is_the_only_infeasible_row(comparison):
    infeasible = comparison[~comparison["feasible"]]
    assert len(infeasible) == 1
    assert infeasible["method"].iloc[0].startswith("Uncontrolled")


def test_feasible_rows_respect_the_grid_limit(comparison):
    feasible = comparison[comparison["feasible"]]
    assert (feasible["peak_ev_load_kw"] <= cfg.MAX_GRID_KW + 1e-6).all()


def test_no_method_approaches_the_grid_limit(comparison):
    """
    The 125 kW cap is never near-binding: peak EV load sits around 95-99 kW.

    Peak is set by 20 ports times the fleet's average power, not by the grid, so
    the peak-demand objective cannot buy anything by throttling below the cap.
    See ``test_grid_limit_is_not_the_binding_constraint`` in test_scenario.py.
    """
    feasible = comparison[comparison["feasible"]]
    assert feasible["peak_ev_load_kw"].max() <= 0.85 * cfg.MAX_GRID_KW


def test_demand_is_structurally_underserved(comparison):
    """
    No method gets close to complete service, and none is near-zero either.

    Two EVs cannot be charged at all (see test_scenario.py) and all 100 compete
    for 20 ports each evening, so every method delivers roughly 64-75% of the
    requested energy. This is the context required to read the cost column: a
    method that delivers less energy will always look cheaper.
    """
    feasible = comparison[comparison["feasible"]]
    assert (feasible["demand_met_pct"] < 100.0).all()
    assert (feasible["demand_met_pct"] > 60.0).all()


def test_methods_all_land_within_a_narrow_band(comparison):
    """
    Delivered energy spans only ~11 percentage points across every method.

    This is the single most important caveat on the headline comparison: the
    scenario is so congested that the achievable spread is small, so an
    advantage in delivered energy alone is weak evidence of a better scheduler.
    """
    feasible = comparison[comparison["feasible"]]
    spread = feasible["demand_met_pct"].max() - feasible["demand_met_pct"].min()
    assert spread < 15.0


def test_ga_and_pso_used_the_configured_full_budget(comparison):
    """Pins the provenance: these evaluation counts only come from the full settings."""
    by_method = comparison.set_index("method")["evaluations"]
    assert by_method["GA"] == count_ga_evaluations(
        cfg.GA_POPULATION_SIZE, cfg.GA_GENERATIONS
    )
    assert by_method["PSO"] == cfg.PSO_PARTICLES * (1 + cfg.PSO_ITERATIONS)


def test_baselines_perform_no_evaluations(comparison):
    baselines = comparison[comparison["method"].str.contains("FCFS|EDF|Uncontrolled")]
    assert (baselines["evaluations"] == 0).all()


def test_saved_schedules_obey_the_hard_constraints(bundle, ev, solar):
    for name, schedule in bundle.items():
        if name.startswith("Uncontrolled"):
            continue
        assert constraint_report(schedule, ev, solar)["total"] == 0, name


def test_saved_metrics_agree_with_the_saved_schedules(bundle, comparison, ev, solar):
    """The table must be reproducible from the schedules it shipped with."""
    indexed = comparison.set_index("method")

    for name, schedule in bundle.items():
        row = indexed.loc[name]
        recomputed = metrics_row(schedule, ev, solar, name)
        assert recomputed["delivered_energy_kwh"] == pytest.approx(
            row["delivered_energy_kwh"], abs=1e-6
        )
        assert recomputed["grid_cost_units"] == pytest.approx(
            row["grid_cost_units"], abs=1e-6
        )
        assert recomputed["constraint_violations"] == row["constraint_violations"]


def test_per_algorithm_schedule_files_match_the_bundle(bundle, ev):
    for name in ("GA", "PSO"):
        path = cfg.RESULTS_DIR / f"{name.lower()}_best_schedule.csv"
        if not path.exists():
            pytest.skip(f"{path.name} not present")
        assert np.allclose(load_schedule_csv(path, ev), bundle[name])


def test_convergence_files_are_not_empty():
    for stem in ("ga", "pso"):
        path = cfg.RESULTS_DIR / f"{stem}_convergence.csv"
        if not path.exists():
            pytest.skip(f"{path.name} not present")
        frame = pd.read_csv(path)
        assert len(frame) > 1
        assert np.isfinite(frame.iloc[:, 1]).all()


def test_no_stale_latest_duplicates_survive_in_results():
    """
    Regression: ``*_latest.csv`` files were written by one script and read by
    another, so the two could silently disagree.
    """
    stale = sorted(p.name for p in cfg.RESULTS_DIR.glob("*_latest*"))
    assert stale == []
