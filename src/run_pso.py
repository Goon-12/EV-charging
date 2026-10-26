
from pathlib import Path
import sys
import os

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
    repair_schedule,
    evaluate_fitness,
)

# PSO settings
N_PARTICLES = 20
ITERATIONS = 50
W = 0.7       # Inertia: preserves part of previous velocity
C1 = 1.5      # Cognitive coefficient: learns from particle's best
C2 = 1.5      # Social coefficient: learns from swarm's best
SEED = int(os.environ.get("EV_OPT_SEED", "42"))
rng = np.random.default_rng(SEED)


def create_random_particle(scenario):
    """Create a random candidate and repair it into a feasible schedule."""
    n_ev = len(scenario)
    position = np.zeros((n_ev, N_SLOTS), dtype=float)

    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)

    for i in range(n_ev):
        start = max(0, arrival[i])
        end = min(N_SLOTS, departure[i])

        if end > start:
            position[i, start:end] = (
                rng.random(end - start) * max_power[i]
            )

    return repair_schedule(position, scenario)


def main():
    scenario = load_scenario()
    n_ev = len(scenario)

    print(f"Loaded {n_ev} EVs from saved scenario.")
    print(f"Scenario file: {PROJECT_DIR / 'data' / 'processed' / 'ev_scenario.csv'}")
    print(f"Particles: {N_PARTICLES}, iterations: {ITERATIONS}")

    # Initialize particles and their velocities.
    positions = [
        create_random_particle(scenario)
        for _ in range(N_PARTICLES)
    ]

    velocities = [
        rng.uniform(-0.5, 0.5, size=(n_ev, N_SLOTS))
        for _ in range(N_PARTICLES)
    ]

    personal_best_positions = [p.copy() for p in positions]
    personal_best_fitness = np.array([
        evaluate_fitness(p, scenario) for p in positions
    ])

    best_idx = int(np.argmin(personal_best_fitness))
    global_best_position = personal_best_positions[best_idx].copy()
    global_best_fitness = float(personal_best_fitness[best_idx])

    history = []

    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)

    # Mask ensures particles do not charge outside each EV's time window.
    valid_window = np.zeros((n_ev, N_SLOTS), dtype=bool)
    for i in range(n_ev):
        start = max(0, arrival[i])
        end = min(N_SLOTS, departure[i])
        valid_window[i, start:end] = True

    for iteration in range(ITERATIONS):
        for p in range(N_PARTICLES):
            r1 = rng.random((n_ev, N_SLOTS))
            r2 = rng.random((n_ev, N_SLOTS))

            velocities[p] = (
                W * velocities[p]
                + C1 * r1 * (personal_best_positions[p] - positions[p])
                + C2 * r2 * (global_best_position - positions[p])
            )

            # Update continuous position, then project to valid power bounds.
            candidate = positions[p] + velocities[p]
            candidate = np.maximum(candidate, 0.0)
            candidate = np.minimum(candidate, max_power[:, None])
            candidate[~valid_window] = 0.0

            # Repair shared grid, charger, and energy constraints.
            candidate = repair_schedule(candidate, scenario)
            positions[p] = candidate

            fitness = evaluate_fitness(candidate, scenario)

            if fitness < personal_best_fitness[p]:
                personal_best_fitness[p] = fitness
                personal_best_positions[p] = candidate.copy()

            if fitness < global_best_fitness:
                global_best_fitness = float(fitness)
                global_best_position = candidate.copy()

        history.append(global_best_fitness)

        if iteration % 10 == 0:
            print(
                f"Iteration {iteration:3d} | "
                f"Best fitness: {global_best_fitness:.3f}"
            )

    # Save best PSO schedule.
    schedule_df = pd.DataFrame(
        global_best_position,
        columns=[f"slot_{t}" for t in range(N_SLOTS)]
    )
    schedule_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())

    schedule_path = RESULTS_DIR / "pso_best_schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    # Save convergence history and plot.
    pd.DataFrame({
        "iteration": np.arange(ITERATIONS),
        "best_fitness": history,
    }).to_csv(RESULTS_DIR / "pso_convergence.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(range(ITERATIONS), history)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness (lower is better)")
    plt.title("Particle Swarm Optimization Convergence")
    plt.grid(True)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "pso_convergence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nPSO run completed.")
    print(f"Best fitness: {global_best_fitness:.3f}")
    print(f"Schedule saved to: {schedule_path}")
    print(f"Convergence plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
