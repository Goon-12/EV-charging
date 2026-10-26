
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from load_scenario import load_scenario
from run_ga import (
    STEP_HOURS,
    N_SLOTS,
    N_CHARGERS,
    MAX_GRID_KW,
    POPULATION_SIZE,
    GENERATIONS,
    ELITE_COUNT,
    MUTATION_RATE,
    create_schedule,
    mutate,
)
from repair_schedule_cost import repair_schedule_cost
from evaluate_objective import evaluate_objective

# Reproducible experimental seed.
SEED = 42
rng = np.random.default_rng(SEED)

# Same illustrative tariff used in the cost comparison.
def illustrative_tariff(slot_hours):
    hour = slot_hours % 24
    return np.select(
        [
            (hour >= 0) & (hour < 6),
            (hour >= 17) & (hour < 22),
        ],
        [4.0, 10.0],
        default=6.0,
    ).astype(float)


def main():
    scenario = load_scenario()
    required = scenario["required_energy_kwh"].to_numpy(dtype=float)

    prices = illustrative_tariff(
        np.arange(N_SLOTS) * STEP_HOURS
    )

    def crossover_cost(parent1, parent2):
        choose_parent1 = rng.random(len(scenario)) < 0.5

        child = np.where(
            choose_parent1[:, None],
            parent1,
            parent2,
        )

        return repair_schedule_cost(
            child, scenario, prices,
            step_hours=STEP_HOURS,
            n_slots=N_SLOTS,
            max_chargers=N_CHARGERS,
            max_grid_kw=MAX_GRID_KW,
        )

    def fitness(schedule):
        metrics = evaluate_objective(
            schedule=schedule,
            required_energy_kwh=required,
            price_per_kwh=prices,
            step_hours=STEP_HOURS,
            max_grid_kw=MAX_GRID_KW,
        )
        return metrics["objective"]

    # Use the same scenario and existing schedule-generation operators.
    population = [
        repair_schedule_cost(
            create_schedule(scenario),
            scenario,
            prices,
            step_hours=STEP_HOURS,
            n_slots=N_SLOTS,
            max_chargers=N_CHARGERS,
            max_grid_kw=MAX_GRID_KW,
        )
        for _ in range(POPULATION_SIZE)
    ]

    history = []

    for generation in range(GENERATIONS):
        scores = np.array([fitness(ind) for ind in population])
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = scores[order]

        history.append(float(scores[0]))

        if generation % 10 == 0:
            print(
                f"Generation {generation:3d} | "
                f"Best objective: {scores[0]:.6f}"
            )

        next_population = [
            individual.copy()
            for individual in population[:ELITE_COUNT]
        ]

        while len(next_population) < POPULATION_SIZE:
            parent_pool = max(1, POPULATION_SIZE // 2)
            p1 = population[int(rng.integers(0, parent_pool))]
            p2 = population[int(rng.integers(0, parent_pool))]

            child = crossover_cost(p1, p2)

            if rng.random() < MUTATION_RATE:
                child = mutate(child, scenario)
                child = repair_schedule_cost(
                    child,
                    scenario,
                    prices,
                    step_hours=STEP_HOURS,
                    n_slots=N_SLOTS,
                    max_chargers=N_CHARGERS,
                    max_grid_kw=MAX_GRID_KW,
                )

            next_population.append(child)

        population = next_population

    # Select the best individual from the final population.
    scores = np.array([fitness(ind) for ind in population])
    best_idx = int(np.argmin(scores))
    best_schedule = population[best_idx]

    schedule_df = pd.DataFrame(
        best_schedule,
        columns=[f"slot_{t}" for t in range(N_SLOTS)]
    )
    schedule_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())

    schedule_path = RESULTS_DIR / "ga_cost_best_schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    metrics = evaluate_objective(
        schedule=best_schedule,
        required_energy_kwh=required,
        price_per_kwh=prices,
        step_hours=STEP_HOURS,
        max_grid_kw=MAX_GRID_KW,
    )

    pd.DataFrame([metrics]).to_csv(
        RESULTS_DIR / "ga_cost_metrics.csv",
        index=False
    )

    plt.figure(figsize=(9, 4))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best objective (lower is better)")
    plt.title("Experimental Cost-Aware GA Convergence")
    plt.grid(True)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "ga_cost_convergence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nEXPERIMENTAL COST-AWARE GA")
    print("-" * 45)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print(f"\nSchedule saved to: {schedule_path}")
    print(f"Metrics saved to: {RESULTS_DIR / 'ga_cost_metrics.csv'}")
    print(f"Convergence plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
