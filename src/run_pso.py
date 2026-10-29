"""
Standalone particle-swarm run.

    .venv/Scripts/python src/run_pso.py

Writes results/pso_best_schedule.csv, results/pso_convergence.csv,
results/pso_convergence.png, results/pso_metrics.csv and
results/pso_load_profile.png.

This used to be a second, independent PSO implementation. Unlike the shared one
it did not clip the velocity, it searched in kW space and then let the repair
routine dominate the result, and it measured itself with
run_ga.evaluate_fitness (no cost term) rather than the shared objective. It now
runs the single shared objective and repair, so its numbers are comparable with
every other row in results/.

See docs/TIER0_CHANGES.md. Tier 2 replaces the encoding itself, because a
velocity step in this representation does not correspond to a small change in
the schedule.
"""
from __future__ import annotations

import argparse

import config as cfg
from run_final_experiments import run_single_algorithm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=cfg.PSO_SEED)
    parser.add_argument("--particles", type=int, default=cfg.PSO_PARTICLES)
    parser.add_argument("--iterations", type=int, default=cfg.PSO_ITERATIONS)
    parser.add_argument("--out-dir", default=str(cfg.RESULTS_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="Tiny settings for a smoke test.")
    args = parser.parse_args()

    run_single_algorithm(
        "PSO",
        quick=args.quick,
        seed=args.seed,
        out_dir=args.out_dir,
        overrides={
            "particles": args.particles,
            "iterations": args.iterations,
        },
    )


if __name__ == "__main__":
    main()
