"""
GA and PSO implementations that share one repair routine and one objective.

Both optimizers search a random-priority matrix of shape (n_ev, n_slots) in
[0, 1]. The matrix is only a *ranking hint*: ``repair_schedule`` converts it
into a schedule, and the fixed heuristic terms inside that repair (urgency,
tariff, solar) still supply roughly two thirds of the ranking score. See
docs/TIER0_CHANGES.md for why this makes the search weak and what Tier 2
changes about it.

Memory note (Tier 0): the previous implementation memoised every evaluation
with a 115 KB chromosome as the dict key and stored the 115 KB schedule as the
value, so a single GA run grew to hundreds of megabytes. The GA now carries
elite scores forward, which is both cheaper and equivalent: each chromosome is
still evaluated exactly once, and no cache is needed at all.
"""
from __future__ import annotations

import numpy as np

import config as cfg
from multiobjective_core import objective, repair_schedule


def optimize_ga(
    ev,
    solar_kw,
    population_size=cfg.GA_POPULATION_SIZE,
    generations=cfg.GA_GENERATIONS,
    seed=cfg.GA_SEED,
    weights=None,
    elite_count=2,
    mutation_rate=0.04,
    tournament_size=2,
    history=None,
    slot_offset=0,
    horizon_slots=None,
):
    """
    Run a real-coded GA over the EV x slot priority matrix.

    Returns ``(best_schedule, best_score)``. If ``history`` is a list, the best
    objective value of every generation is appended to it (for convergence
    plots and audits).

    ``slot_offset``/``horizon_slots`` are forwarded to the shared repair routine
    and are only used by the rolling-horizon controller; leaving them at their
    defaults keeps full-horizon runs numerically unchanged.
    """
    rng = np.random.default_rng(seed)
    n_ev, n_slots = len(ev), len(solar_kw)

    def assess(chromosome):
        """Score -> chromosome -> schedule, so nothing needs to be re-decoded."""
        schedule = repair_schedule(
            chromosome, ev, solar_kw,
            slot_offset=slot_offset, horizon_slots=horizon_slots,
        )
        return (objective(schedule, ev, solar_kw, weights), chromosome, schedule)

    population = [rng.random((n_ev, n_slots)) for _ in range(population_size)]
    scored = [assess(x) for x in population]

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
                picks = rng.choice(len(scored), tournament_size, replace=False)
                # Tie-breaking note: the last minimum wins, which reproduces the
                # pre-Tier-0 behaviour exactly. Using min() here would pick the
                # first minimum and change results on this flat landscape.
                best_index = picks[0]
                for candidate in picks[1:]:
                    if scored[candidate][0] <= scored[best_index][0]:
                        best_index = candidate
                parents.append(scored[best_index][1])

            p1, p2 = parents
            mask = rng.random((n_ev, n_slots)) < 0.5
            child = np.where(mask, p1, p2)

            mut = rng.random((n_ev, n_slots)) < mutation_rate
            child[mut] = rng.random(int(np.sum(mut)))
            children.append(child)

        scored = elite + [assess(c) for c in children]

    scored.sort(key=lambda entry: entry[0])
    best_score, _, best_schedule = scored[0]
    return best_schedule, float(best_score)


def optimize_pso(
    ev,
    solar_kw,
    particles=cfg.PSO_PARTICLES,
    iterations=cfg.PSO_ITERATIONS,
    seed=cfg.PSO_SEED,
    weights=None,
    inertia=0.7,
    cognitive=1.5,
    social=1.5,
    max_velocity=0.25,
    history=None,
    slot_offset=0,
    horizon_slots=None,
):
    """
    Run a continuous PSO over the same priority matrix and return the best
    schedule found.

    Positions live in [0, 1] and are clipped there; velocities are clipped to
    +/-``max_velocity``. NOTE: because the schedule is produced by
    ``repair_schedule`` rather than read directly from the position, a small
    velocity step does not correspond to a small change in the schedule, so the
    swarm dynamics are only loosely meaningful. Tier 2 replaces this with a
    random-key representation.
    """
    rng = np.random.default_rng(seed)
    shape = (len(ev), len(solar_kw))

    def assess(position):
        schedule = repair_schedule(
            position, ev, solar_kw,
            slot_offset=slot_offset, horizon_slots=horizon_slots,
        )
        return objective(schedule, ev, solar_kw, weights), schedule

    positions = rng.random((particles, *shape))
    velocities = np.zeros_like(positions)
    pbest = positions.copy()
    pbest_score = np.full(particles, np.inf)
    pbest_schedule = np.zeros((particles, *shape), dtype=float)

    for p in range(particles):
        score, schedule = assess(positions[p])
        pbest_score[p] = score
        pbest_schedule[p] = schedule

    g_index = int(np.argmin(pbest_score))
    gbest = pbest[g_index].copy()
    gbest_schedule = pbest_schedule[g_index].copy()
    gbest_score = float(pbest_score[g_index])

    for _ in range(iterations):
        for p in range(particles):
            r1 = rng.random(shape)
            r2 = rng.random(shape)
            velocities[p] = (
                inertia * velocities[p]
                + cognitive * r1 * (pbest[p] - positions[p])
                + social * r2 * (gbest - positions[p])
            )
            velocities[p] = np.clip(velocities[p], -max_velocity, max_velocity)
            positions[p] = np.clip(positions[p] + velocities[p], 0.0, 1.0)

            score, schedule = assess(positions[p])
            if score < pbest_score[p]:
                pbest_score[p] = score
                pbest[p] = positions[p].copy()
                pbest_schedule[p] = schedule
                if score < gbest_score:
                    gbest_score = float(score)
                    gbest = positions[p].copy()
                    gbest_schedule = schedule.copy()

        if history is not None:
            history.append(gbest_score)

    return gbest_schedule, float(gbest_score)
