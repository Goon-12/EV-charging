"""
GA and PSO implementations that share one genome, one decoder and one objective.

Tier 2 replaced the search representation. The optimizers used to search a
``(n_ev, n_slots)`` priority matrix - 14,400 genes for a 560-evaluation budget -
through a repair heuristic that supplied most of its own ranking score from fixed
weights. Both are gone: ``optimize_ga`` and ``optimize_pso`` now search the
400-gene per-EV genome in :mod:`genome` and decode it with the feasible slot-major
walk in :func:`genome.decode_genome`. Consequences worth knowing:

* **A small step is now a small change.** One gene is one decision about one EV,
  so a mutation or a velocity component perturbs the schedule locally. That is
  what makes the PSO update meaningful instead of decorative (Tier 0 item 9).
* **Feasibility is structural.** No repair stage, no post-hoc "fill remaining
  capacity" pass, no penalty that can be traded against service.
* **Both are memetic.** After the search each optimizer spends a fixed,
  exactly-counted budget of hill-climbing moves on its best genome
  (:func:`genome.local_search`), so the reported evaluation count is the real
  number of objective calls and not an estimate.

Budgets are set in :mod:`config` and are of order 10^4 evaluations, which is the
fix for Tier 0 item 11 ("560 evaluations cannot search 14,400 genes"). The
evaluation counts are exact: ``count_ga_evaluations`` in
``run_final_experiments`` mirrors the arithmetic below.

Memory note (Tier 0): the old GA memoised every evaluation with a 115 KB
chromosome as the dict key and the 115 KB schedule as the value, so one run grew
to hundreds of megabytes. The genome is 3.2 KB, elites keep their score, and
nothing is cached. The schedule is decoded once, at the end.
"""
from __future__ import annotations

import numpy as np

import config as cfg
from genome import (
    GENOME_COLUMNS,
    GENE_BOUNDS,
    N_GENES,
    clip_genome,
    decode_genome,
    genome_from_unit,
    heuristic_genome,
    local_search,
    random_genome,
)
from multiobjective_core import objective

# Warm starts used to seed the initial population. Every optimizer gets the same
# four, so neither algorithm starts from a better place than the other.
WARM_START_KINDS = ("power", "deadline", "need", "arrival")


def _seed_population(rng, ev, population_size, warm_start=True):
    """Warm starts first, then uniform random genomes to fill the population.

    With ``warm_start=False`` the population is entirely random. That is only
    used by the tests (and by anyone who wants a from-scratch run), because the
    warm starts are seed-independent by construction and therefore hide the
    effect of the seed.
    """
    population = []
    if warm_start:
        population = [heuristic_genome(ev, kind) for kind in WARM_START_KINDS]
    while len(population) < population_size:
        population.append(random_genome(rng, len(ev)))
    return population[:population_size]


def optimize_ga(
    ev,
    solar_kw,
    population_size=cfg.GA_POPULATION_SIZE,
    generations=cfg.GA_GENERATIONS,
    seed=cfg.GA_SEED,
    weights=None,
    elite_count=cfg.GA_ELITE_COUNT,
    mutation_rate=cfg.GA_MUTATION_RATE,
    tournament_size=3,
    history=None,
    slot_offset=0,
    horizon_slots=None,
    polish_evaluations=cfg.GA_POLISH_EVALUATIONS,
    warm_start=True,
):
    """
    Run a real-coded GA over the per-EV genome.

    Returns ``(best_schedule, best_score)``. If ``history`` is a list, the best
    objective value of every generation is appended to it (for convergence
    plots and audits).

    ``slot_offset`` is forwarded to the decoder so the rolling-horizon
    controller sees the real time of day. ``horizon_slots`` is accepted for
    signature compatibility with the controller but is *not* used: deadline
    urgency is measured against each EV's own departure, so no horizon-wide
    rescaling is needed (the Tier 0 bug was caused by exactly that rescaling).

    ``polish_evaluations`` is the memetic budget spent hill-climbing the best
    genome after the last generation; set it to 0 for a purely evolutionary run.
    """
    rng = np.random.default_rng(seed)
    n_ev = len(ev)

    def decode(genome):
        return decode_genome(genome, ev, solar_kw, slot_offset=slot_offset)

    def assess(genome):
        """Score the genome. The schedule is discarded; only the best is decoded."""
        return float(objective(decode(genome), ev, solar_kw, weights))

    population = _seed_population(rng, ev, population_size, warm_start)
    scored = [(assess(g), g) for g in population]

    for _ in range(generations):
        scored.sort(key=lambda entry: entry[0])
        if history is not None:
            history.append(float(scored[0][0]))

        # Elites keep their already-computed score, so they are not re-evaluated.
        n_elite = min(elite_count, len(scored))
        elite = list(scored[:n_elite])
        children = []

        while len(elite) + len(children) < population_size:
            parents = []
            for _ in range(2):
                picks = rng.integers(0, len(scored), tournament_size)
                # Smallest score wins; ties broken by the later draw, which keeps
                # the original selection behaviour (see docs/TIER0_CHANGES.md).
                best = picks[0]
                for candidate in picks[1:]:
                    if scored[candidate][0] <= scored[best][0]:
                        best = candidate
                parents.append(scored[best][1])

            p1, p2 = parents
            mask = rng.random((n_ev, N_GENES)) < 0.5
            child = np.where(mask, p1, p2)

            mutate = rng.random((n_ev, N_GENES)) < mutation_rate
            if mutate.any():
                lo = GENE_BOUNDS[:, 0]
                span = GENE_BOUNDS[:, 1] - GENE_BOUNDS[:, 0]
                fresh = lo + rng.random((n_ev, N_GENES)) * span
                child = np.where(mutate, fresh, child)
            children.append(child)

        scored = elite + [(assess(c), c) for c in children]

    scored.sort(key=lambda entry: entry[0])
    best_score, best_genome = scored[0]

    if polish_evaluations and polish_evaluations > 0:
        best_genome, best_score, _ = local_search(
            best_genome,
            evaluate=assess,
            rng=rng,
            evaluations=polish_evaluations,
            initial_score=best_score,
            n_ev=n_ev,
        )

    return decode(best_genome), float(best_score)


def _clip_unit(positions):
    """Keep PSO positions inside the unit box."""
    return np.clip(positions, 0.0, 1.0)


def optimize_pso(
    ev,
    solar_kw,
    particles=cfg.PSO_PARTICLES,
    iterations=cfg.PSO_ITERATIONS,
    seed=cfg.PSO_SEED,
    weights=None,
    inertia=0.72,
    cognitive=1.49,
    social=1.49,
    max_velocity=0.2,
    history=None,
    slot_offset=0,
    horizon_slots=None,
    polish_evaluations=cfg.PSO_POLISH_EVALUATIONS,
    warm_start=True,
):
    """
    Run a random-key PSO over the same genome and return the best schedule.

    Positions live in the unit box and are mapped to the genome by
    :func:`genome.genome_from_unit`, so a velocity component moves one gene of
    one EV by a bounded amount and the decoded schedule changes *smoothly*. That
    is the property the old priority-matrix PSO lacked entirely (Tier 0 item 9).

    ``horizon_slots`` is accepted for compatibility and unused; see
    :func:`optimize_ga`.
    """
    rng = np.random.default_rng(seed)
    n_ev = len(ev)
    shape = (n_ev, N_GENES)

    def decode(genome):
        return decode_genome(genome, ev, solar_kw, slot_offset=slot_offset)

    def assess(unit_position):
        genome = genome_from_unit(unit_position)
        return float(objective(decode(genome), ev, solar_kw, weights)), genome

    # Seed the swarm with the same warm starts the GA gets, then random keys.
    span = GENE_BOUNDS[:, 1] - GENE_BOUNDS[:, 0]
    seeds = [(heuristic_genome(ev, kind) - GENE_BOUNDS[:, 0]) / span
             for kind in WARM_START_KINDS]

    positions = np.empty((particles, *shape), dtype=float)
    seeded = len(seeds) if warm_start else 0
    for p in range(particles):
        positions[p] = seeds[p] if p < seeded else rng.random(shape)

    velocities = np.zeros_like(positions)
    pbest = positions.copy()
    pbest_scores = np.full(particles, np.inf)
    pbest_genomes = np.empty((particles, *shape), dtype=float)

    for p in range(particles):
        score, genome = assess(positions[p])
        pbest_scores[p] = score
        pbest_genomes[p] = genome

    g_index = int(np.argmin(pbest_scores))
    gbest_score = float(pbest_scores[g_index])
    gbest_genome = pbest_genomes[g_index].copy()
    gbest_position = pbest[g_index].copy()

    for _ in range(iterations):
        for p in range(particles):
            r1 = rng.random(shape)
            r2 = rng.random(shape)
            velocities[p] = (
                inertia * velocities[p]
                + cognitive * r1 * (pbest[p] - positions[p])
                + social * r2 * (gbest_position - positions[p])
            )
            velocities[p] = np.clip(velocities[p], -max_velocity, max_velocity)
            positions[p] = _clip_unit(positions[p] + velocities[p])

            score, genome = assess(positions[p])
            if score < pbest_scores[p]:
                pbest_scores[p] = score
                pbest[p] = positions[p].copy()
                pbest_genomes[p] = genome
                if score < gbest_score:
                    gbest_score = float(score)
                    gbest_genome = genome.copy()
                    gbest_position = positions[p].copy()

        if history is not None:
            history.append(gbest_score)

    if polish_evaluations and polish_evaluations > 0:
        def evaluate(genome):
            return float(objective(decode(genome), ev, solar_kw, weights))

        gbest_genome, gbest_score, _ = local_search(
            gbest_genome,
            evaluate=evaluate,
            rng=rng,
            evaluations=polish_evaluations,
            initial_score=gbest_score,
            n_ev=n_ev,
        )

    return decode(gbest_genome), float(gbest_score)
