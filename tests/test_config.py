"""The single source of constants must stay self-consistent."""
from __future__ import annotations

import config as cfg


def test_slot_count_derives_from_horizon():
    assert cfg.N_SLOTS == int(round(cfg.HORIZON_HOURS / cfg.STEP_HOURS))
    assert cfg.N_SLOTS == 144


def test_objective_weights_are_a_valid_scalarisation():
    weights = cfg.OBJECTIVE_WEIGHTS
    assert set(weights) == {"unmet", "peak", "cost", "battery", "solar"}
    assert all(value >= 0 for value in weights.values())
    # The five terms form a convex weighted sum (the solar term is subtracted).
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_tariff_boundaries_are_ordered_and_inside_a_day():
    assert 0 <= cfg.OFFPEAK_END_HOUR < cfg.PEAK_START_HOUR < cfg.PEAK_END_HOUR <= 24
    assert cfg.TARIFF_OFFPEAK < cfg.TARIFF_MID < cfg.TARIFF_PEAK
    assert cfg.MAX_TARIFF == cfg.TARIFF_PEAK


def test_paths_point_inside_the_project():
    for path in (cfg.DATA_DIR, cfg.RESULTS_DIR, cfg.DEFAULT_SCENARIO, cfg.DEFAULT_SOLAR):
        assert str(path).startswith(str(cfg.PROJECT_DIR))


def test_full_run_budget_is_of_order_ten_thousand_evaluations():
    """
    Guards the Tier 2 fix for Tier 0 item 11.

    Tier 0/1 searched with 560 evaluations and Tier 0 said the budget needed
    10^4-10^5. Both optimizers now report an exact count in that range, memetic
    polish included, so a silent return to a smoke-test budget fails here.
    """
    from run_final_experiments import count_ga_evaluations, count_pso_evaluations

    ga = count_ga_evaluations(
        cfg.GA_POPULATION_SIZE,
        cfg.GA_GENERATIONS,
        polish_evaluations=cfg.GA_POLISH_EVALUATIONS,
    )
    pso = count_pso_evaluations(
        cfg.PSO_PARTICLES,
        cfg.PSO_ITERATIONS,
        polish_evaluations=cfg.PSO_POLISH_EVALUATIONS,
    )
    assert ga >= 10_000
    assert pso >= 10_000
    assert ga <= 100_000 and pso <= 100_000
