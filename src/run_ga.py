"""
Standalone genetic-algorithm run.

    .venv/Scripts/python src/run_ga.py
    .venv/Scripts/python src/run_ga.py --generations 200 --population-size 60

Writes results/ga_best_schedule.csv, results/ga_convergence.csv,
results/ga_convergence.png, results/ga_metrics.csv and
results/ga_load_profile.png.

This used to be a second, independent GA implementation with its own repair
routine and its own fitness function

    fitness = 1e6 * unmet_kwh + 1e3 * not_fully_charged + peak_kw

which contains no cost term at all. Results from it were nevertheless reported
in the same tables as the cost-aware variants, so the "Original GA" row meant
something different from every other row. It now runs the single shared
objective and repair (multiobjective_core) and is therefore comparable with
everything else in results/.

The frozen old implementation is archived at
experiments/legacy/run_ga_legacy.py.
"""
from __future__ import annotations

import argparse

import config as cfg
from run_final_experiments import run_single_algorithm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=cfg.GA_SEED)
    parser.add_argument("--population-size", type=int, default=cfg.GA_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=cfg.GA_GENERATIONS)
    parser.add_argument("--out-dir", default=str(cfg.RESULTS_DIR))
    parser.add_argument("--quick", action="store_true",
                        help="Tiny settings for a smoke test.")
    args = parser.parse_args()

    run_single_algorithm(
        "GA",
        quick=args.quick,
        seed=args.seed,
        out_dir=args.out_dir,
        overrides={
            "population_size": args.population_size,
            "generations": args.generations,
        },
    )


if __name__ == "__main__":
    main()
