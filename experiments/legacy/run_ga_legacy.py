# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md
"""
Frozen copy of the pre-Tier-0 ``src/run_ga.py`` GA engine.

This is an exact snapshot of the engine that produced the archived
``*_ga_*``/``*_cost_*``/``*_solar_*``/``*_battery_aware_*`` result files. It is
kept so those archived runs stay reproducible. It is NOT part of the pipeline:
it uses a fitness function with no cost term at all

    fitness = 1e6 * unmet_kwh + 1e3 * not_fully_charged + peak_kw

which is why the "Original GA" row in older tables reports the most energy and
the highest cost of any method.

Its repair routine also *adds* energy that the candidate never asked for (it
greedily fills every slot up to the grid cap), which is one of the reasons the
search has a flat fitness landscape. See docs/TIER0_CHANGES.md.

Archived runners write into ``results/legacy/`` so that running them can never
overwrite a pipeline artifact.
"""
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow imports from other files inside src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from load_scenario import load_scenario

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "legacy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Simulation settings
STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0

# GA settings (small initial run)
POPULATION_SIZE = 30
GENERATIONS = 50
ELITE_COUNT = 2
MUTATION_RATE = 0.08
SEED = int(os.environ.get("EV_OPT_SEED", "46"))
rng = np.random.default_rng(SEED)


def create_schedule(scenario):
    """Create a feasible starting schedule using earliest-arrival priority."""
    n_ev = len(scenario)
    schedule = np.zeros((n_ev, N_SLOTS))

    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)
    required = scenario["required_energy_kwh"].to_numpy(dtype=float)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)

    # Randomize ties while preserving approximate arrival priority.
    if rng.random() < 0.5:
        order = np.argsort(departure)
    else:
        order = rng.permutation(n_ev)

    for i in order:
        remaining = required[i]

        for t in range(arrival[i], min(departure[i], N_SLOTS)):
            active_count = np.count_nonzero(schedule[:, t] > 1e-9)
            remaining_grid = MAX_GRID_KW - schedule[:, t].sum()

            if active_count >= N_CHARGERS or remaining_grid <= 1e-9:
                continue

            power = min(max_power[i], remaining / STEP_HOURS, remaining_grid)
            if power > 0:
                schedule[i, t] = power
                remaining -= power * STEP_HOURS

            if remaining <= 1e-9:
                break

    return schedule


def evaluate_fitness(schedule, scenario):
    """Prioritize unmet energy, then fully charged EVs, then peak demand."""
    required = scenario["required_energy_kwh"].to_numpy(dtype=float)

    delivered = schedule.sum(axis=1) * STEP_HOURS
    unmet = np.maximum(required - delivered, 0.0)

    total_unmet = float(unmet.sum())
    not_fully_charged = int(np.count_nonzero(unmet > 0.05))

    peak = float(schedule.sum(axis=0).max())

    return 1_000_000.0 * total_unmet + 1_000.0 * not_fully_charged + peak


def mutate(schedule, scenario):
    """Create a changed candidate by rebuilding one EV's charging plan."""
    child = schedule.copy()

    n_ev = len(scenario)
    i = int(rng.integers(0, n_ev))

    # Remove this EV's old plan.
    child[i, :] = 0.0

    arrival = int(scenario.iloc[i]["arrival_slot"])
    departure = min(int(scenario.iloc[i]["departure_slot"]), N_SLOTS)
    required = float(scenario.iloc[i]["required_energy_kwh"])
    max_power = float(scenario.iloc[i]["max_power_kw"])

    # Try charging this EV in a randomly selected order of available slots.
    slots = np.arange(arrival, departure)
    rng.shuffle(slots)

    for t in slots:
        active_count = np.count_nonzero(child[:, t] > 1e-9)
        remaining_grid = MAX_GRID_KW - child[:, t].sum()

        if active_count >= N_CHARGERS or remaining_grid <= 1e-9:
            continue

        delivered = child[i, :].sum() * STEP_HOURS
        remaining = required - delivered

        if remaining <= 1e-9:
            break

        power = min(max_power, remaining / STEP_HOURS, remaining_grid)
        child[i, t] = power

    return child


def crossover(parent1, parent2, scenario):
    """Combine parent plans and repair the resulting schedule."""
    n_ev = parent1.shape[0]
    choose_parent1 = rng.random(n_ev) < 0.5

    child = np.where(choose_parent1[:, None], parent1, parent2)

    return repair_schedule(child, scenario)


def repair_schedule(schedule, scenario):
    """Preserve candidate preferences while repairing hard constraints."""
    n_ev = len(scenario)

    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)
    required = scenario["required_energy_kwh"].to_numpy(dtype=float)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)

    starts = np.clip(arrival, 0, N_SLOTS)
    ends = np.clip(departure, 0, N_SLOTS)

    # Keep the candidate's requested slot preferences.
    repaired = np.asarray(schedule, dtype=float).copy()
    repaired = np.maximum(repaired, 0.0)
    repaired = np.minimum(repaired, max_power[:, None])

    # No charging before arrival or at/after departure.
    for i in range(n_ev):
        repaired[i, :starts[i]] = 0.0
        repaired[i, ends[i]:] = 0.0

    # Remove energy beyond each EV's request.
    for i in range(n_ev):
        row_energy = repaired[i].sum() * STEP_HOURS
        if row_energy > required[i] and row_energy > 0:
            repaired[i] *= required[i] / row_energy

    # Enforce charger and grid limits in every time slot.
    for t in range(N_SLOTS):
        active = np.flatnonzero(repaired[:, t] > 1e-9)

        if len(active) > N_CHARGERS:
            # Retain the EVs with the strongest candidate preferences.
            keep = active[np.argsort(repaired[active, t])[-N_CHARGERS:]]
            keep_set = set(keep.tolist())

            for i in active:
                if i not in keep_set:
                    repaired[i, t] = 0.0

        total_power = repaired[:, t].sum()
        if total_power > MAX_GRID_KW:
            repaired[:, t] *= MAX_GRID_KW / total_power

    # Fill remaining capacity, prioritizing urgent EVs.
    delivered = repaired.sum(axis=1) * STEP_HOURS

    for t in range(N_SLOTS):
        eligible = [
            i for i in range(n_ev)
            if starts[i] <= t < ends[i]
            and required[i] - delivered[i] > 1e-8
            and repaired[i, t] <= 1e-9
        ]

        eligible.sort(key=lambda i: (ends[i], -(required[i] - delivered[i]), i))

        active_count = np.count_nonzero(repaired[:, t] > 1e-9)
        grid_remaining = MAX_GRID_KW - repaired[:, t].sum()

        for i in eligible:
            if active_count >= N_CHARGERS or grid_remaining <= 1e-9:
                break

            remaining_energy = required[i] - delivered[i]

            power = min(
                max_power[i],
                remaining_energy / STEP_HOURS,
                grid_remaining,
            )

            if power > 1e-9:
                repaired[i, t] = power
                delivered[i] += power * STEP_HOURS
                grid_remaining -= power
                active_count += 1

    return repaired


def main():
    scenario = load_scenario()

    population = [create_schedule(scenario) for _ in range(POPULATION_SIZE)]

    best_history = []

    for generation in range(GENERATIONS):
        fitness = np.array(
            [evaluate_fitness(individual, scenario) for individual in population]
        )

        ranked = np.argsort(fitness)
        population = [population[i] for i in ranked]
        fitness = fitness[ranked]

        best_history.append(fitness[0])

        if generation % 10 == 0:
            print(f"Generation {generation:3d} | Best fitness: {fitness[0]:.3f}")

        next_population = [individual.copy() for individual in population[:ELITE_COUNT]]

        while len(next_population) < POPULATION_SIZE:
            # Select parents from the better half of the population.
            parent1 = population[int(rng.integers(0, max(1, POPULATION_SIZE // 2)))]
            parent2 = population[int(rng.integers(0, max(1, POPULATION_SIZE // 2)))]

            child = crossover(parent1, parent2, scenario)

            if rng.random() < MUTATION_RATE:
                child = mutate(child, scenario)
                child = repair_schedule(child, scenario)

            next_population.append(child)

        population = next_population

    fitness = np.array(
        [evaluate_fitness(individual, scenario) for individual in population]
    )
    best_idx = int(np.argmin(fitness))
    best_schedule = population[best_idx]

    schedule_df = pd.DataFrame(
        best_schedule, columns=[f"slot_{t}" for t in range(N_SLOTS)]
    )
    schedule_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())

    schedule_path = RESULTS_DIR / "ga_best_schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    history_df = pd.DataFrame(
        {"generation": np.arange(len(best_history)), "best_fitness": best_history}
    )
    history_csv = RESULTS_DIR / "ga_convergence.csv"
    history_df.to_csv(history_csv, index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(history_df["generation"], history_df["best_fitness"], label="GA best fitness")
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (lower is better)")
    plt.title("Genetic Algorithm Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS_DIR / "ga_convergence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nGA run completed.")
    print("Best fitness:", round(float(fitness[best_idx]), 3))
    print("Schedule saved to:", schedule_path)
    print("Convergence plot saved to:", plot_path)


if __name__ == "__main__":
    main()
