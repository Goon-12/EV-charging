"""
The Tier 2 genome: a small, per-EV decision vector with a feasible decoder.

Why this module exists
----------------------
Tier 0/1 searched a ``(n_ev, n_slots)`` priority matrix: 14,400 genes, 560
evaluations, and a decoder that supplied roughly two thirds of its own ranking
score from fixed heuristics (urgency, tariff, solar). The genome therefore
controlled a minority of the decision, and what it did control was smeared over
a search space far too large to sample. On top of that, a small change to one
gene moved one cell of a 14,400-cell matrix, so a "small step" had almost no
effect on the schedule - which is why a velocity update in PSO meant nothing.

This module replaces that with four numbers per EV:

    rank     base priority when more EVs want a port than there are ports
    urgency  how strongly the EV's own deadline pressure raises that priority
    tariff   how strongly the time-of-use price lowers it
    power    charging rate as a fraction of the EV's rated power

That is 400 genes for the 100-EV scenario, they are all on a known interval,
and each one changes the realised schedule in a direct and local way.

The decoder is a slot-major greedy walk
---------------------------------------
For every slot, in chronological order, the connected EVs with energy left are
scored; the best-scoring ones take the available ports and charge at their
requested rate, until ports, grid capacity or their own remaining energy run
out. An EV whose score is *negative* is deferred rather than charged. Because
``rank`` and ``urgency`` can only ever add to the score and ``tariff`` can only
subtract from it, the sign has a clean meaning: an EV charges whenever it has any
reason to, and waits only when the current slot's price outweighs both its base
priority and its deadline pressure. Waiting is therefore expressed by the same
three genes that express everything else - there is no separate start-slot gene.

Feasibility is a property of the constructor, not of a repair step: the walk
never places power outside ``[arrival, departure)``, never exceeds an EV's rated
power or remaining energy, never uses more than ``n_chargers`` ports in a slot
and never exceeds the slot capacity ``max_grid_kw + solar``. Every schedule this
module returns is feasible, so the objective never needs a penalty term.

Known limitation (documented in docs/TIER2_CHANGES.md): the decoder is greedy
and chronological, so it cannot look ahead. It is a decoder, not a solver - the
comparison against the MILP reference in ``reference_optimum.py`` measures
exactly how much that costs.
"""
from __future__ import annotations

import numpy as np

import config as cfg
from multiobjective_core import make_windows, tariff_vector

# Canonical gene order. Index i of this tuple is column i of the genome.
GENOME_COLUMNS = ("rank", "urgency", "tariff", "power")
N_GENES = len(GENOME_COLUMNS)

# (lower, upper) per gene. ``rank`` and ``power`` are already fractions;
# ``urgency`` and ``tariff`` are weights on normalised quantities, so 3.0 is
# "strong enough to override a full unit of rank".
GENE_BOUNDS = np.array([
    [0.0, 1.0],   # rank
    [0.0, 3.0],   # urgency
    [0.0, 3.0],   # tariff
    [0.0, 1.0],   # power
])

# Deadline pressure is clipped so that one very tight EV cannot swamp the score.
MAX_URGENCY = 5.0

# Urgency is meaningless below this fraction of rated power; guards division by 0.
_POWER_FLOOR = 1e-9

# An EV counts as finished at this residual energy (kWh).
_ENERGY_FLOOR = 1e-9


def empty_genome(n_ev):
    """A genome of zeros. It decodes to an empty schedule - a valid, useless one."""
    return np.zeros((n_ev, N_GENES), dtype=float)


def random_genome(rng, n_ev):
    """Uniform random genome inside :data:`GENE_BOUNDS`."""
    lo, hi = GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]
    return lo + rng.random((n_ev, N_GENES)) * (hi - lo)


def genome_to_unit(genome):
    """Map a genome into the unit box (what PSO actually moves in)."""
    lo, hi = GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]
    return (np.asarray(genome, dtype=float) - lo) / (hi - lo)


def genome_from_unit(unit):
    """Map a point of the unit box back to a genome."""
    lo, hi = GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]
    unit = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
    return lo + unit * (hi - lo)


def clip_genome(genome):
    """Keep every gene inside its bounds."""
    lo, hi = GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]
    return np.clip(np.asarray(genome, dtype=float), lo, hi)


def heuristic_genome(ev, kind="power"):
    """
    A structured warm start, not a search result.

    The optimizers put a handful of these in their initial population. Without
    one, the first generation is 40 random 400-gene vectors and the search is
    spent rediscovering that ports are the scarce resource; with one, it starts
    from a policy that already works and improves on it. ``kind`` selects the
    ranking quantity:

    ``power``     highest rated power first - the throughput-aware rule
    ``deadline``  earliest departure first - the EDF rule
    ``need``      largest remaining demand first
    ``arrival``   earliest arrival first - the FCFS rule

    All of them charge at rated power (``power = 1``) and give the tariff a
    moderate weight, so the optimizers are compared against a fair starting
    point rather than against a strawman.
    """
    n_ev = len(ev)
    genome = empty_genome(n_ev)
    genome[:, 1] = 1.0   # urgency
    genome[:, 2] = 1.0   # tariff
    genome[:, 3] = 1.0   # power

    def normalise(values):
        values = np.asarray(values, dtype=float)
        spread = values.max() - values.min()
        if spread <= 1e-12:
            return np.full(n_ev, 0.5)
        return (values - values.min()) / spread

    if kind == "power":
        genome[:, 0] = normalise(ev["max_power_kw"].to_numpy(float))
    elif kind == "deadline":
        genome[:, 0] = normalise(-ev["departure_slot"].to_numpy(float))
    elif kind == "need":
        genome[:, 0] = normalise(ev["required_energy_kwh"].to_numpy(float))
    elif kind == "arrival":
        genome[:, 0] = normalise(-ev["arrival_slot"].to_numpy(float))
    else:
        raise ValueError(
            f"Unknown warm start: {kind!r} "
            "(expected 'power', 'deadline', 'need' or 'arrival')"
        )

    # Break exact ties by EV index so the warm start is deterministic and each
    # EV keeps its own identity instead of collapsing into a block. The ranking
    # is compressed to [0, 0.99] first so the tie-break cannot push it past 1.
    genome[:, 0] = 0.99 * genome[:, 0] + np.arange(n_ev) * 1e-6
    return genome


def decode_genome(
    genome,
    ev,
    solar_kw,
    max_grid_kw=cfg.MAX_GRID_KW,
    n_chargers=cfg.N_CHARGERS,
    slot_offset=0,
):
    """
    Turn a genome into a feasible charging schedule.

    ``slot_offset`` is the absolute slot index of column 0. The rolling-horizon
    controller works on a slice of the horizon and must pass it so that the
    tariff weight sees the real time of day instead of restarting at midnight
    (the Tier 0 tariff-clock bug).

    Returns ``schedule``, an ``(n_ev, n_slots)`` array of kW. Constraints are
    satisfied by construction; see the module docstring.
    """
    genome = np.asarray(genome, dtype=float)
    solar = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)
    n_ev, n_slots = len(ev), len(solar)
    if genome.shape != (n_ev, N_GENES):
        raise ValueError(
            f"Genome shape {genome.shape} does not match ({n_ev}, {N_GENES})."
        )

    rank = genome[:, 0]
    urgency_weight = genome[:, 1]
    tariff_weight = genome[:, 2]
    power_fraction = np.clip(genome[:, 3], 0.0, 1.0)

    arrival, departure = make_windows(ev, n_slots)
    required = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)

    # Normalised slot attributes: both live in [0, 1] so the gene weights have
    # a comparable meaning across slots and across scenarios.
    tariff = tariff_vector(n_slots, slot_offset)
    tariff_span = max(cfg.MAX_TARIFF - cfg.TARIFF_OFFPEAK, 1e-9)
    tariff_norm = (tariff - cfg.TARIFF_OFFPEAK) / tariff_span

    capacity = np.maximum(max_grid_kw + solar, 0.0)
    remaining = required.copy()
    schedule = np.zeros((n_ev, n_slots), dtype=float)

    # EVs that can never be served are dropped once instead of being retried in
    # every slot.
    usable = (required > _ENERGY_FLOOR) & (pmax > _POWER_FLOOR)
    usable &= departure > arrival

    for t in range(n_slots):
        eligible = np.flatnonzero(usable & (remaining > _ENERGY_FLOOR))
        eligible = eligible[arrival[eligible] <= t]
        eligible = eligible[t < departure[eligible]]
        if eligible.size == 0:
            continue

        slack = np.maximum(departure[eligible] - t, 1)
        urgency = np.minimum(
            remaining[eligible] / (slack * cfg.STEP_HOURS * pmax[eligible]),
            MAX_URGENCY,
        )
        score = (
            rank[eligible]
            + urgency_weight[eligible] * urgency
            - tariff_weight[eligible] * tariff_norm[t]
        )

        # Highest score first; ties keep the lower EV index (stable sort).
        order = np.argsort(-score, kind="stable")

        used = 0.0
        ports = 0
        for position in order:
            if score[position] < 0.0:
                break  # every remaining EV prefers to wait for a cheaper slot
            if ports >= n_chargers:
                break
            room = capacity[t] - used
            if room <= 1e-9:
                break

            i = eligible[position]
            power = min(
                pmax[i] * power_fraction[i],
                remaining[i] / cfg.STEP_HOURS,
                room,
            )
            if power <= 1e-9:
                continue

            schedule[i, t] = power
            remaining[i] -= power * cfg.STEP_HOURS
            used += power
            ports += 1

    return schedule


def local_search(
    genome,
    evaluate,
    rng,
    evaluations,
    initial_score=None,
    n_ev=None,
):
    """
    Memetic polish: greedy single-gene hill climbing from ``genome``.

    ``evaluate`` maps a genome to a scalar objective (lower is better) and is
    the *same* function the optimizers use, so a polish step is comparable with
    a search step. Half the moves resample a gene uniformly (exploration), half
    apply a small Gaussian step (refinement); a move is kept only if it strictly
    improves.

    Returns ``(genome, score, evaluations_used)``. The count is exact, so the
    reported budget of a memetic run is not an estimate.
    """
    if evaluations <= 0:
        score = evaluate(genome) if initial_score is None else initial_score
        return genome, score, 0

    best = np.array(genome, dtype=float, copy=True)
    if initial_score is None:
        # The evaluation above is a real objective call, so it counts.
        best_score = float(evaluate(best))
        used = 1
    else:
        best_score = float(initial_score)
        used = 0

    n_ev = len(best) if n_ev is None else n_ev
    gene_count = best.shape[1]
    lo, hi = GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1]

    while used < evaluations:
        candidate = best.copy()
        gene = int(rng.integers(0, gene_count))
        row = int(rng.integers(0, n_ev))
        mutate = rng.random()
        if mutate < 0.5:
            value = lo[gene] + rng.random() * (hi[gene] - lo[gene])
        else:
            step = 0.15 * (hi[gene] - lo[gene])
            value = best[row, gene] + rng.normal(0.0, step)
        candidate[row, gene] = np.clip(value, lo[gene], hi[gene])

        score = float(evaluate(candidate))
        used += 1
        if score < best_score - 1e-12:
            best, best_score = candidate, score

    return best, best_score, used
