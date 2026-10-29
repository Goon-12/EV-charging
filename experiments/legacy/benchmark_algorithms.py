# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md

from pathlib import Path
import os
import re
import subprocess
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SEEDS = [42, 43, 44, 45, 46]


def run_script(script_name, seed):
    """Run a project script with a reproducible seed."""
    env = os.environ.copy()
    env["EV_OPT_SEED"] = str(seed)

    result = subprocess.run(
        [PYTHON, str(PROJECT_DIR / "src" / script_name)],
        cwd=PROJECT_DIR,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed for seed {seed}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout


def extract_metric(output, pattern, metric_name):
    match = re.search(pattern, output, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not find {metric_name} in output:\n{output}"
        )
    return float(match.group(1))


def extract_ga_metrics(output):
    return {
        "peak_grid_kw": extract_metric(
            output, r"peak_grid_kw:\s*([0-9.eE+-]+)", "peak_grid_kw"
        ),
        "total_energy_kwh": extract_metric(
            output, r"total_energy_kwh:\s*([0-9.eE+-]+)",
            "total_energy_kwh"
        ),
        "total_unmet_energy_kwh": extract_metric(
            output, r"total_unmet_energy_kwh:\s*([0-9.eE+-]+)",
            "total_unmet_energy_kwh"
        ),
        "evs_fully_charged": int(extract_metric(
            output, r"evs_fully_charged:\s*(\d+)",
            "evs_fully_charged"
        )),
    }


def extract_pso_metrics(output):
    return {
        "peak_grid_kw": extract_metric(
            output, r"Peak grid demand:\s*([0-9.eE+-]+)",
            "peak_grid_kw"
        ),
        "total_energy_kwh": extract_metric(
            output, r"Total delivered energy:\s*([0-9.eE+-]+)",
            "total_energy_kwh"
        ),
        "total_unmet_energy_kwh": extract_metric(
            output, r"Total unmet energy:\s*([0-9.eE+-]+)",
            "total_unmet_energy_kwh"
        ),
        "evs_fully_charged": int(extract_metric(
            output, r"Fully charged EVs.*?:\s*(\d+)",
            "evs_fully_charged"
        )),
    }


def main():
    scenario_file = (
        PROJECT_DIR / "data" / "processed" / "ev_scenario.csv"
    )
    if not scenario_file.exists():
        raise FileNotFoundError(
            f"Saved scenario not found: {scenario_file}"
        )

    records = []

    for seed in SEEDS:
        print(f"\n========== SEED {seed} ==========")

        # GA run and validation.
        run_script("run_ga.py", seed)
        ga_output = run_script("validate_ga.py", seed)
        ga_metrics = extract_ga_metrics(ga_output)
        records.append({
            "method": "GA",
            "seed": seed,
            **ga_metrics,
        })
        print("GA:", ga_metrics)

        # PSO run and validation.
        run_script("run_pso.py", seed)
        pso_output = run_script("validate_pso.py", seed)
        pso_metrics = extract_pso_metrics(pso_output)
        records.append({
            "method": "PSO",
            "seed": seed,
            **pso_metrics,
        })
        print("PSO:", pso_metrics)

    results = pd.DataFrame(records)
    detailed_file = RESULTS_DIR / "multi_seed_results.csv"
    results.to_csv(detailed_file, index=False)

    summary = results.groupby("method").agg(
        peak_grid_kw_mean=("peak_grid_kw", "mean"),
        peak_grid_kw_std=("peak_grid_kw", "std"),
        total_energy_kwh_mean=("total_energy_kwh", "mean"),
        total_energy_kwh_std=("total_energy_kwh", "std"),
        unmet_energy_kwh_mean=("total_unmet_energy_kwh", "mean"),
        unmet_energy_kwh_std=("total_unmet_energy_kwh", "std"),
        fully_charged_mean=("evs_fully_charged", "mean"),
        fully_charged_std=("evs_fully_charged", "std"),
    ).reset_index()

    summary_file = RESULTS_DIR / "multi_seed_summary.csv"
    summary.to_csv(summary_file, index=False)

    print("\nMULTI-SEED SUMMARY (mean and standard deviation)")
    print(summary.round(3).to_string(index=False))
    print(f"\nDetailed results saved to: {detailed_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
