"""
The Tier 2 genome and its decoder.

Two properties matter and are tested separately:

* **Feasibility is structural.** Every genome decodes to a schedule that respects
  every hard constraint, including the degenerate ones (all-zero genome, EVs with
  an empty window, an EV whose rated power is zero). If that ever stops being
  true, the objective silently starts trading violations against service.
* **A gene means something.** Changing one gene must move the schedule in a
  predictable direction - that is the property the old 14,400-gene priority
  matrix lacked, and the reason the search could not make progress.
"""
from __future__ import annotations

import numpy as np
import pytest

import config as cfg
from genome import (
    GENE_BOUNDS,
    N_GENES,
    clip_genome,
    decode_genome,
    empty_genome,
    genome_from_unit,
    genome_to_unit,
    heuristic_genome,
    local_search,
    random_genome,
)
from helpers import make_inputs
from multiobjective_core import constraint_report, evaluate_schedule, objective


def assert_feasible(schedule, ev, solar, label="", **limits):
    report = constraint_report(schedule, ev, solar, **limits)
    assert report["total"] == 0, f"{label}: {report}"


@pytest.mark.parametrize("kind", ["power", "deadline", "need", "arrival"])
def test_warm_starts_are_feasible_and_deliver_energy(kind, ev, solar):
    schedule = decode_genome(heuristic_genome(ev, kind), ev, solar)
    assert_feasible(schedule, ev, solar, kind)
    assert schedule.sum() > 0


def test_warm_starts_stay_inside_their_gene_bounds(ev):
    for kind in ("power", "deadline", "need", "arrival"):
        genome = heuristic_genome(ev, kind)
        assert genome.shape == (len(ev), N_GENES)
        assert (genome >= GENE_BOUNDS[:, 0] - 1e-12).all()
        assert (genome <= GENE_BOUNDS[:, 1] + 1e-12).all()


def test_unknown_warm_start_is_rejected(ev):
    with pytest.raises(ValueError):
        heuristic_genome(ev, "crystal-ball")


def test_random_genomes_decode_feasibly_everywhere(ev, solar):
    rng = np.random.default_rng(11)
    for draw in range(8):
        schedule = decode_genome(random_genome(rng, len(ev)), ev, solar)
        assert_feasible(schedule, ev, solar, f"draw {draw}")


def test_all_zero_genome_is_empty_but_feasible(ev, solar):
    """
    The zero genome requests zero power, so it delivers nothing - but it is still
    a valid schedule. The decoder must never produce an infeasible one, including
    for the degenerate genome; a test that only covers warm starts would miss it.
    """
    schedule = decode_genome(empty_genome(len(ev)), ev, solar)
    assert_feasible(schedule, ev, solar)
    assert schedule.sum() == 0


def test_degenerate_evs_are_handled():
    """An EV with no window, no energy or no power must not break the decode."""
    ev, solar = make_inputs(
        required=[2.0, 0.0, 1.0],
        pmax=[2.0, 1.0, 0.0],
        arrival=[0, 0, 1],
        departure=[2, 2, 1],   # third EV: arrival == departure
        n_slots=4,
    )
    rng = np.random.default_rng(2)
    schedule = decode_genome(random_genome(rng, len(ev)), ev, solar)
    assert_feasible(schedule, ev, solar)
    assert np.all(schedule[2] == 0)


def test_shape_mismatch_is_rejected(ev, solar):
    with pytest.raises(ValueError):
        decode_genome(np.zeros((len(ev), N_GENES + 1)), ev, solar)


def test_ports_and_grid_limits_hold_on_a_tight_site():
    """Five EVs, two ports, one small grid cap - the decode must not oversubscribe."""
    ev, solar = make_inputs(
        required=[4.0] * 5,
        pmax=[2.0] * 5,
        arrival=[0] * 5,
        departure=[4] * 5,
        n_slots=4,
    )
    genome = heuristic_genome(ev, "power")
    schedule = decode_genome(genome, ev, solar, max_grid_kw=3.0, n_chargers=2)
    assert_feasible(schedule, ev, solar, max_grid_kw=3.0, n_chargers=2)

    assert (schedule > 1e-9).sum(axis=0).max() <= 2
    assert (schedule.sum(axis=0) <= 3.0 + 1e-9).all()


def test_power_gene_scales_the_delivered_power():
    """``power`` is a rate fraction: 0.5 must deliver exactly half the power."""
    ev, solar = make_inputs(
        required=[8.0], pmax=[4.0], arrival=[0], departure=[8], n_slots=8
    )
    full = heuristic_genome(ev, "power")
    half = full.copy()
    half[:, 3] = 0.5

    full_energy = decode_genome(full, ev, solar).sum() * cfg.STEP_HOURS
    half_energy = decode_genome(half, ev, solar).sum() * cfg.STEP_HOURS
    assert half_energy == pytest.approx(0.5 * full_energy)


def test_rank_gene_decides_who_gets_the_scarce_port():
    """
    With one port, the higher-ranked EV takes the port first.

    EV0 wants the whole horizon (8 kWh at 2 kW over 2 h), so whichever EV ranks
    first is served first and the other gets only the leftovers.
    """
    ev, solar = make_inputs(
        required=[8.0, 2.0], pmax=[2.0, 2.0], arrival=[0, 0], departure=[8, 8], n_slots=8
    )
    genome = heuristic_genome(ev, "power")
    genome[:, 1] = 0.0   # urgency off, so the ranking decides
    genome[:, 0] = [1.0, 0.0]

    dominant_first = decode_genome(genome, ev, solar, n_chargers=1).sum(axis=1)
    assert dominant_first[0] > 0
    assert dominant_first[1] == 0

    genome[:, 0] = [0.0, 1.0]
    other_first = decode_genome(genome, ev, solar, n_chargers=1).sum(axis=1)
    assert other_first[1] > 0
    assert other_first[0] < dominant_first[0]


def test_tariff_gene_defers_charging_out_of_the_peak():
    """
    A high tariff weight must move charging away from the evening peak.

    Two slots, one cheap and one expensive; the EV is connected for both.
    """
    ev, solar = make_inputs(
        required=[2.0], pmax=[2.0], arrival=[0], departure=[cfg.N_SLOTS], n_slots=cfg.N_SLOTS
    )
    genome = heuristic_genome(ev, "power")
    genome[0, 0] = 0.0        # no base priority, so the tariff decides
    genome[0, 1] = 0.0        # no urgency pressure
    genome[0, 2] = 3.0        # maximum tariff sensitivity
    genome[0, 3] = 1.0

    schedule = decode_genome(genome, ev, solar)
    peak_slots = slice(68, 88)   # 17:00-22:00
    assert schedule[0, peak_slots].sum() == 0
    assert schedule.sum() * cfg.STEP_HOURS == pytest.approx(2.0)


def test_decode_is_deterministic(ev, solar):
    rng = np.random.default_rng(4)
    genome = random_genome(rng, len(ev))
    first = decode_genome(genome, ev, solar)
    second = decode_genome(genome, ev, solar)
    assert np.array_equal(first, second)


def test_unit_roundtrip_and_clipping():
    rng = np.random.default_rng(8)
    genome = rng.normal(size=(7, N_GENES)) * 5.0   # deliberately out of bounds
    clipped = clip_genome(genome)
    assert (clipped >= GENE_BOUNDS[:, 0]).all()
    assert (clipped <= GENE_BOUNDS[:, 1]).all()

    unit = genome_to_unit(clipped)
    assert (unit >= -1e-12).all() and (unit <= 1 + 1e-12).all()
    assert np.allclose(genome_from_unit(unit), clipped)


def test_local_search_only_accepts_improvements_and_counts_exactly(ev, solar):
    rng = np.random.default_rng(12)
    start = heuristic_genome(ev, "deadline")

    calls = []

    def evaluate(genome):
        schedule = decode_genome(genome, ev, solar)
        calls.append(1)
        return objective(schedule, ev, solar)

    start_score = evaluate(start)
    calls.clear()

    best, best_score, used = local_search(
        start, evaluate, rng, evaluations=25, initial_score=start_score
    )
    assert used == 25
    assert len(calls) == 25
    assert best_score <= start_score + 1e-12
    assert best.shape == start.shape


def test_local_search_with_no_budget_is_a_no_op(ev, solar):
    start = heuristic_genome(ev, "need")
    best, score, used = local_search(
        start,
        lambda g: objective(decode_genome(g, ev, solar), ev, solar),
        np.random.default_rng(0),
        evaluations=0,
    )
    assert used == 0
    assert np.array_equal(best, start)
    assert np.isfinite(score)


def test_decoded_schedules_never_exceed_the_request(ev, solar):
    """No genome may add unrequested energy (the legacy "fill capacity" bug)."""
    rng = np.random.default_rng(21)
    required = ev["required_energy_kwh"].to_numpy(float)
    for _ in range(5):
        schedule = decode_genome(random_genome(rng, len(ev)), ev, solar)
        delivered = schedule.sum(axis=1) * cfg.STEP_HOURS
        assert (delivered <= required + 1e-6).all()


def test_warm_start_ordering_beats_the_other_warm_starts_on_service(ev, solar):
    """
    The power-aware warm start should serve more energy than arrival order.

    This is a property of the scenario (ports are the scarce resource, so
    charging high-power EVs first wastes fewer port-hours), not of the code. It
    is asserted because it is the intuition behind the genome's ``rank`` gene,
    and because a silent inversion would mean the decoder ignores its ranking.
    """
    def service(kind):
        schedule = decode_genome(heuristic_genome(ev, kind), ev, solar)
        return evaluate_schedule(schedule, ev, solar)["delivered_energy_kwh"]

    assert service("power") > service("arrival")
