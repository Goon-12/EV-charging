"""
The single command that regenerates every headline result.

    .venv/Scripts/python src/run_final_experiments.py

Produces, all through one evaluator and one repair routine:

* ``results/final_unified/comparison_metrics.csv`` - every method side by side
  (volume metrics, audit/provenance)
* ``results/final_unified/equal_service_comparison.csv`` - the same methods
  judged at equal service: cost and peak per delivered kWh, service attained,
  and Pareto dominance flags
* ``results/final_unified/equal_service_frontier.png`` - service vs intensity
* ``results/final_unified/schedules.npz``         - the schedules themselves
* ``results/final_unified/run_manifest.csv``      - mode, settings, seeds, input hashes
* ``results/ga_*`` / ``results/pso_*``            - per-algorithm schedule,
  convergence history/plot, metrics and load profile
* ``results/baseline_metrics.csv``                - all conventional policies
* with ``--multi-seed``: ``multi_seed_results.csv`` and ``multi_seed_summary.csv``

Before Tier 0 this file and ``final_comparision.py`` produced different tables
from different evaluators, and a third script (``benchmark_algorithms.py``)
produced the multi-seed table by launching the two legacy pipelines as
subprocesses and scraping their stdout with regular expressions. Both are now
archived in ``experiments/legacy/``.

``run_ga.py`` and ``run_pso.py`` delegate to ``run_single_algorithm`` below so
there is exactly one implementation of "run an optimizer and write its files".
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from artifacts import (
    RESULT_COLUMNS,
    file_digest,
    metrics_row,
    save_convergence_csv,
    save_convergence_comparison_plot,
    save_convergence_plot,
    save_cost_service_plot,
    save_equal_service_plot,
    save_load_profile_plot,
    save_method_profiles_plot,
    save_metrics_csv,
    save_schedule_csv,
    save_seed_spread_plot,
    save_service_comparison_plot,
)
from baselines import all_baselines
from equal_service import (
    equal_service_table,
    frontier_rows,
    service_accounting,
)
from multiobjective_core import load_inputs
from multiobjective_optimizers import optimize_ga, optimize_pso
from reference_optimum import (
    attach_reference,
    capacity_bounds,
    reference_table,
)
from run_rolling_horizon import optimize_rolling


def settings(quick):
    """Optimizer settings for a full or smoke-test run."""
    if quick:
        return {
            "ga": dict(population_size=8, generations=5, polish_evaluations=20),
            "pso": dict(particles=6, iterations=4, polish_evaluations=20),
            "rolling": dict(population_size=6, generations=3, particles=4, iterations=2),
        }
    return {
        "ga": dict(
            population_size=cfg.GA_POPULATION_SIZE,
            generations=cfg.GA_GENERATIONS,
            polish_evaluations=cfg.GA_POLISH_EVALUATIONS,
        ),
        "pso": dict(
            particles=cfg.PSO_PARTICLES,
            iterations=cfg.PSO_ITERATIONS,
            polish_evaluations=cfg.PSO_POLISH_EVALUATIONS,
        ),
        "rolling": dict(
            population_size=cfg.ROLLING_GA_POPULATION_SIZE,
            generations=cfg.ROLLING_GA_GENERATIONS,
            particles=cfg.ROLLING_PSO_PARTICLES,
            iterations=cfg.ROLLING_PSO_ITERATIONS,
        ),
    }


def count_ga_evaluations(
    population_size, generations, elite_count=cfg.GA_ELITE_COUNT, polish_evaluations=0
):
    """
    Exact number of objective evaluations a GA run performs.

    The initial population is evaluated once, each generation evaluates only its
    children (elites keep their score), and the memetic polish spends a fixed
    number of further evaluations on the final best genome.
    """
    return (
        population_size
        + generations * (population_size - min(elite_count, population_size))
        + polish_evaluations
    )


def count_pso_evaluations(particles, iterations, polish_evaluations=0):
    """Exact number of objective evaluations a PSO run performs."""
    return particles * (1 + iterations) + polish_evaluations


def algorithm_spec(name, conf, seed=None):
    """Return ``(optimizer, kwargs, evaluation_budget)`` for ``name``."""
    key = name.strip().lower()
    if key == "ga":
        kwargs = dict(conf["ga"])
        if seed is not None:
            kwargs["seed"] = seed
        return optimize_ga, kwargs, count_ga_evaluations(**conf["ga"])
    if key == "pso":
        kwargs = dict(conf["pso"])
        if seed is not None:
            kwargs["seed"] = seed
        return optimize_pso, kwargs, count_pso_evaluations(**conf["pso"])
    raise ValueError(f"Unknown algorithm: {name!r} (expected 'GA' or 'PSO')")


def write_algorithm_artifacts(name, schedule, history, row, ev, solar, out_dir):
    """Write the per-algorithm schedule, convergence, metrics and profile files."""
    stem = name.strip().lower().replace(" ", "_")
    out_dir = Path(out_dir)
    save_schedule_csv(out_dir / f"{stem}_best_schedule.csv", schedule, ev)
    if history:
        save_convergence_csv(out_dir / f"{stem}_convergence.csv", history)
        save_convergence_plot(
            out_dir / f"{stem}_convergence.png", history, f"{name} convergence"
        )
    save_metrics_csv(out_dir / f"{stem}_metrics.csv", row, name)
    save_load_profile_plot(
        out_dir / f"{stem}_load_profile.png", schedule, solar,
        f"{name} charging load profile",
    )


def run_single_algorithm(
    name, quick=False, seed=None, out_dir=None, verbose=True, overrides=None
):
    """
    Run one metaheuristic end to end and write its artifacts.

    Returns ``(metrics_row, schedule)``. This is what ``run_ga.py`` and
    ``run_pso.py`` call, so there is one implementation for both.
    ``overrides`` replaces individual optimizer settings (e.g. a larger
    population) and recomputes the reported evaluation budget accordingly.
    """
    out_dir = cfg.RESULTS_DIR if out_dir is None else Path(out_dir)
    ev, solar = load_inputs()
    conf = settings(quick)
    optimizer, kwargs, evaluations = algorithm_spec(name, conf, seed)
    if overrides:
        kwargs.update(overrides)
        if name.strip().lower() == "ga":
            evaluations = count_ga_evaluations(
                kwargs["population_size"],
                kwargs["generations"],
                polish_evaluations=kwargs.get("polish_evaluations", 0),
            )
        else:
            evaluations = count_pso_evaluations(
                kwargs["particles"],
                kwargs["iterations"],
                polish_evaluations=kwargs.get("polish_evaluations", 0),
            )

    history = []
    started = time.perf_counter()
    schedule, score = optimizer(ev, solar, history=history, **kwargs)
    elapsed = time.perf_counter() - started

    row = metrics_row(schedule, ev, solar, name, evaluations, elapsed)
    write_algorithm_artifacts(name, schedule, history, row, ev, solar, out_dir)

    if verbose:
        print(f"\n{name} finished in {elapsed:.1f} s "
              f"({evaluations} evaluations, objective {score:.6f})")
        print(f"  delivered      : {row['delivered_energy_kwh']:.3f} kWh "
              f"({row['demand_met_pct']:.2f}% of required)")
        print(f"  unmet energy   : {row['unmet_energy_kwh']:.3f} kWh")
        print(f"  fully charged  : {row['fully_charged_evs']} / {len(ev)} EVs")
        print(f"  peak EV load   : {row['peak_ev_load_kw']:.3f} kW "
              f"(grid limit {cfg.MAX_GRID_KW:.0f} kW)")
        print(f"  tariff cost    : {row['grid_cost_units']:.3f} units "
              f"({row['cost_per_delivered_kwh']:.3f} per kWh)")
        print(f"  violations     : {row['constraint_violations']}")
        print(f"  artifacts      : {out_dir}")
    return row, schedule


def run_rolling(name, method, ev, solar, rolling_kwargs):
    """Run the rolling-horizon controller for one optimizer."""
    started = time.perf_counter()
    schedule = optimize_rolling(ev, solar, method=method, **rolling_kwargs)
    elapsed = time.perf_counter() - started
    return metrics_row(schedule, ev, solar, name, 0, elapsed), schedule


def write_run_manifest(out_dir, ev, solar, conf, args):
    """
    Record exactly how these results were produced.

    The committed ``comparison_metrics.csv`` that Tier 0 investigated turned out
    to be a ``--quick`` smoke-test output: nothing in the file said so, and a
    38-evaluation GA (1089 kWh) looks just as plausible as a 560-evaluation one
    (1116 kWh) when read on its own. Every run now writes its mode, settings,
    seeds and input hashes next to its numbers.
    """
    scenario = cfg.DEFAULT_SCENARIO if args.scenario is None else args.scenario
    solar_path = cfg.DEFAULT_SOLAR if args.solar is None else args.solar
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "quick" if args.quick else "full",
        "rolling_horizon": not args.skip_rolling,
        "multi_seed": args.multi_seed,
        "scenario_path": str(scenario),
        "scenario_sha256_16": file_digest(scenario),
        "solar_path": str(solar_path),
        "solar_sha256_16": file_digest(solar_path),
        "n_ev": len(ev),
        "n_slots": len(solar),
        "n_chargers": cfg.N_CHARGERS,
        "max_grid_kw": cfg.MAX_GRID_KW,
        "step_hours": cfg.STEP_HOURS,
        "objective_weights": json.dumps(cfg.OBJECTIVE_WEIGHTS),
        "ga_settings": json.dumps(conf["ga"]),
        "ga_seed": cfg.GA_SEED,
        "pso_settings": json.dumps(conf["pso"]),
        "pso_seed": cfg.PSO_SEED,
        "rolling_settings": json.dumps(conf["rolling"]),
        "rolling_seed": cfg.ROLLING_SEED,
        "multi_seed_seeds": json.dumps(list(cfg.SEEDS)),
    }
    path = Path(out_dir) / "run_manifest.csv"
    pd.DataFrame([manifest]).to_csv(path, index=False)
    return path


def run_multi_seed(ev, solar, out_dir, quick=False):
    """
    Repeat GA and PSO over the configured seed list, in process.

    The pre-Tier-0 version of this ran ``run_ga.py``/``run_pso.py`` as
    subprocesses and recovered metrics by regex-matching their console output,
    so any change to a printed label silently broke the benchmark. Everything
    now runs in process against the shared evaluator.
    """
    conf = settings(quick)
    rows = []
    for seed in cfg.SEEDS:
        for name in ("GA", "PSO"):
            optimizer, kwargs, evaluations = algorithm_spec(name, conf, seed)
            schedule, _ = optimizer(ev, solar, **kwargs)
            row = metrics_row(schedule, ev, solar, name, evaluations)
            row["seed"] = seed
            rows.append(row)

    frame = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    frame.to_csv(out_dir / "multi_seed_results.csv", index=False)

    summary = frame.groupby("method").agg(
        runs=("seed", "count"),
        delivered_mean=("delivered_energy_kwh", "mean"),
        delivered_std=("delivered_energy_kwh", "std"),
        peak_mean=("peak_ev_load_kw", "mean"),
        peak_std=("peak_ev_load_kw", "std"),
        cost_mean=("grid_cost_units", "mean"),
        cost_std=("grid_cost_units", "std"),
        unmet_mean=("unmet_energy_kwh", "mean"),
        unmet_std=("unmet_energy_kwh", "std"),
    ).reset_index()
    summary.to_csv(out_dir / "multi_seed_summary.csv", index=False)

    print(f"\nMULTI-SEED SUMMARY (mean +/- std over seeds {list(cfg.SEEDS)})")
    print(summary.round(3).to_string(index=False))
    return frame, summary


def best_conventional_row(table):
    """The strongest conventional policy on delivered energy, for the figures."""
    conventional = table[
        table["method"].str.startswith(("FCFS", "EDF")) & table["feasible"]
    ]
    return conventional.sort_values("service_delivered_kwh").iloc[-1]


def resolve_reference(ev, solar, out_dir, args):
    """
    Load the cached MILP reference or solve it now.

    Solving is the expensive part of a full run (five to six MILPs at the
    configured time limit), so ``--reuse-reference`` exists for iterating on the
    analysis without re-solving. It refuses a cache that was solved on different
    inputs, because a stale optimum would silently price the wrong scenario.
    """
    path = Path(out_dir) / "reference_optimum.csv"
    scenario = cfg.DEFAULT_SCENARIO if args.scenario is None else Path(args.scenario)
    solar_path = cfg.DEFAULT_SOLAR if args.solar is None else Path(args.solar)

    if args.reuse_reference and path.exists():
        cached = pd.read_csv(path)
        if (
            (cached["scenario_sha256_16"] == file_digest(scenario)).all()
            and (cached["solar_sha256_16"] == file_digest(solar_path)).all()
        ):
            print(f"\nReusing reference optima: {path}")
            return cached
        print(
            "\nCached reference optima were solved on different inputs - "
            "re-solving instead of reusing them."
        )

    # A smoke test must not spend minutes proving an optimum it will not report.
    time_limit = min(args.reference_time_limit, 10.0) if args.quick else args.reference_time_limit
    frame, _ = reference_table(
        ev,
        solar,
        time_limit=time_limit,
        mip_gap=cfg.REFERENCE_MIP_GAP,
        scenario_path=scenario,
        solar_path=solar_path,
    )
    frame.to_csv(path, index=False)
    return frame


def print_verdict(service_optimal, reference, conventional):
    """Say plainly whether the optimizers beat the conventional policies."""
    best_service = float(reference["delivered_energy_kwh"].max())
    table = service_optimal.set_index("method")
    print("\n" + "=" * 118)
    print("TIER 2 VERDICT - optimizers vs conventional policies vs the MILP reference")
    print("=" * 118)
    print(
        f"Best conventional policy: {conventional['method']} at "
        f"{conventional['service_delivered_kwh']:.1f} kWh "
        f"({conventional['cost_per_delivered_kwh']:.4f} units/kWh). "
        f"Reference optimum: {best_service:.1f} kWh."
    )
    for name in ("GA", "PSO", "GA rolling horizon", "PSO rolling horizon"):
        if name not in table.index:
            continue
        row = table.loc[name]
        delta = row["service_delivered_kwh"] - conventional["service_delivered_kwh"]
        gap = row["cost_gap_vs_optimal_pct"]
        gap_text = f"{gap:+.1f}%" if pd.notna(gap) else "n/a"
        print(
            f"  {name:<22} {row['service_delivered_kwh']:7.1f} kWh "
            f"({row['pct_of_best_service']:5.1f}% of best, {delta:+7.1f} vs conventional), "            f" cost {row['cost_units']:7.1f} ({gap_text} vs cheapest at equal service)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--solar", default=None)
    parser.add_argument("--out", default=str(cfg.RESULTS_DIR / "final_unified"))
    parser.add_argument("--quick", action="store_true",
                        help="Tiny GA/PSO settings for a smoke test.")
    parser.add_argument("--skip-rolling", action="store_true",
                        help="Skip the rolling-horizon controller (slow).")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Also repeat GA/PSO over config.SEEDS.")
    parser.add_argument("--reuse-reference", action="store_true",
                        help="Reuse a reference_optimum.csv solved on the same inputs.")
    parser.add_argument("--reference-time-limit", type=float,
                        default=cfg.REFERENCE_TIME_LIMIT_S,
                        help="Time limit per reference MILP, in seconds.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev, solar = load_inputs(args.scenario, args.solar)
    conf = settings(args.quick)
    print(f"Mode: {'QUICK (smoke test)' if args.quick else 'FULL'}")
    print(f"Scenario: {len(ev)} EVs, {len(solar)} slots, "
          f"required energy {ev['required_energy_kwh'].sum():.1f} kWh")
    print(f"Site limits: {cfg.N_CHARGERS} chargers, {cfg.MAX_GRID_KW:.0f} kW grid")

    rows, schedules = [], {}

    for name in ("GA", "PSO"):
        row, schedule = run_single_algorithm(name, args.quick, out_dir=cfg.RESULTS_DIR)
        rows.append(row)
        schedules[name] = schedule

    if not args.skip_rolling:
        for method, name in (("ga", "GA rolling horizon"), ("pso", "PSO rolling horizon")):
            print(f"\nRunning {name}...")
            row, schedule = run_rolling(name, method, ev, solar, conf["rolling"])
            rows.append(row)
            schedules[name] = schedule
            print(f"  {name}: delivered {row['delivered_energy_kwh']:.1f} kWh, "
                  f"peak {row['peak_ev_load_kw']:.1f} kW, "
                  f"violations {row['constraint_violations']}")

    print("\nRunning conventional baselines...")
    baselines = all_baselines(ev, solar)
    baseline_rows = []
    for name, schedule in baselines.items():
        row = metrics_row(schedule, ev, solar, name)
        baseline_rows.append(row)
        rows.append(row)
        schedules[name] = schedule

    table = pd.DataFrame(rows)
    table = table[RESULT_COLUMNS].sort_values("method").reset_index(drop=True)
    table.to_csv(out_dir / "comparison_metrics.csv", index=False)
    np.savez_compressed(out_dir / "schedules.npz", **schedules)

    # The volume table above is the audit record; this is the comparison. Total
    # cost and total peak are volume metrics, so a method that delivers less
    # energy always looks cheaper. Judging at equal service removes that effect.
    service = equal_service_table(schedules, ev, solar)
    service.to_csv(out_dir / "equal_service_comparison.csv", index=False)
    save_equal_service_plot(out_dir / "equal_service_frontier.png", service)

    # Tier 2: the same comparison, but measured against what is achievable
    # rather than against the other methods. Delivered energy as a share of the
    # MILP optimum, and cost against the cheapest schedule known to deliver the
    # same amount of energy.
    reference = resolve_reference(ev, solar, out_dir, args)
    service_optimal = attach_reference(
        service,
        reference,
        service_column="service_delivered_kwh",
        cost_column="cost_units",
    )
    # Carry the evaluation budgets into this table too: a row that claims a gap
    # to the optimum should also say what the method spent to get there, and the
    # counts otherwise live only in comparison_metrics.csv.
    budgets = table.set_index("method")["evaluations"]
    service_optimal["evaluations"] = (
        service_optimal["method"].map(budgets).fillna(0).astype(int)
    )
    service_optimal.to_csv(out_dir / "service_vs_optimal.csv", index=False)
    conventional = best_conventional_row(service)

    save_service_comparison_plot(
        out_dir / "service_vs_methods.png",
        service,
        "service_delivered_kwh",
        reference=reference,
        best_conventional=float(conventional["service_delivered_kwh"]),
    )
    save_cost_service_plot(
        out_dir / "cost_vs_service.png",
        service_optimal,
        reference,
        service_column="service_delivered_kwh",
        cost_column="cost_units",
    )
    save_method_profiles_plot(
        out_dir / "comparison_load_profiles.png",
        schedules,
        solar,
        [
            "FCFS (chronological)",
            "EDF (chronological)",
            "GA",
            "PSO",
        ],
        title="Charging load: conventional policies against the optimizers",
    )
    histories = {}
    for stem, label in (("ga", "GA"), ("pso", "PSO")):
        path = cfg.RESULTS_DIR / f"{stem}_convergence.csv"
        if path.exists():
            histories[label] = pd.read_csv(path).iloc[:, 1].to_numpy()
    if len(histories) == 2:
        save_convergence_comparison_plot(
            out_dir / "convergence_comparison.png", histories
        )

    pd.DataFrame(baseline_rows)[RESULT_COLUMNS].to_csv(
        cfg.RESULTS_DIR / "baseline_metrics.csv", index=False
    )

    manifest_path = write_run_manifest(out_dir, ev, solar, conf, args)

    accounting = service_accounting(ev, len(solar))

    print("\n" + "=" * 110)
    print("VOLUME METRICS (audit table - total cost and total peak are NOT comparable across methods)")
    print("=" * 110)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 118)
    print("EQUAL-SERVICE COMPARISON (cost and peak per delivered kWh)")
    print("=" * 118)
    print(
        f"Attainable service: {accounting['attainable_energy_kwh']:.1f} of "
        f"{accounting['required_energy_kwh']:.1f} kWh requested "
        f"({accounting['infeasible_evs']} EVs cannot be fully charged in their own "
        f"window, {accounting['infeasible_energy_kwh']:.1f} kWh)"
    )
    shown = [
        "method",
        "service_delivered_kwh",
        "service_pct_of_attainable",
        "fully_charged_evs",
        "unmet_congestion_kwh",
        "cost_per_delivered_kwh",
        "peak_per_delivered_kwh",
        "peak_to_average_ratio",
        "pareto_status",
    ]
    print(service[shown].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nNon-dominated methods, cheapest service first:")
    for _, row in frontier_rows(service).iterrows():
        print(
            f"  {row['method']:<34} service {row['service_pct_of_attainable']:6.2f}% "
            f" cost/kWh {row['cost_per_delivered_kwh']:.4f} "
            f" peak/kWh {row['peak_per_delivered_kwh']:.5f}"
        )
    for _, row in service[service["pareto_status"] == "dominated"].iterrows():
        print(f"  {row['method']:<34} dominated by {row['dominated_by']}")

    print("\n" + "=" * 118)
    print("REFERENCE OPTIMA (MILP, see reference_optimum.csv for status and MIP gap)")
    print("=" * 118)
    bounds = capacity_bounds(ev, solar)
    print(
        f"Rigorous ceilings: per-EV {bounds['per_ev_bound_kwh']:.1f} kWh, "
        f"per-slot {bounds['per_slot_bound_kwh']:.1f} kWh; "
        f"{bounds['port_slots_available']:.0f} port-hours over "
        f"{bounds['connected_slots']} occupied slots"
    )
    print(reference.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print_verdict(service_optimal, reference, conventional)

    print(f"\nSaved: {out_dir / 'comparison_metrics.csv'}")
    print(f"Saved: {out_dir / 'equal_service_comparison.csv'}")
    print(f"Saved: {out_dir / 'equal_service_frontier.png'}")
    print(f"Saved: {out_dir / 'service_vs_optimal.csv'}")
    print(f"Saved: {out_dir / 'reference_optimum.csv'}")
    print(f"Saved: {out_dir / 'service_vs_methods.png'}")
    print(f"Saved: {out_dir / 'cost_vs_service.png'}")
    print(f"Saved: {out_dir / 'comparison_load_profiles.png'}")
    print(f"Saved: {out_dir / 'convergence_comparison.png'}")
    print(f"Saved: {out_dir / 'schedules.npz'}")
    print(f"Saved: {manifest_path}")
    print("Note: 'Uncontrolled' is deliberately infeasible and is a reference only.")
    print("Note: read the volume table for provenance, the equal-service table to "
          "compare methods (docs/TIER1_CHANGES.md) and service_vs_optimal.csv for "
          "the gap to the reference optimum (docs/TIER2_CHANGES.md).")

    if args.multi_seed:
        seeds, _ = run_multi_seed(ev, solar, out_dir, args.quick)
        save_seed_spread_plot(
            out_dir / "multi_seed_spread.png",
            seeds,
            conventional=pd.DataFrame([
                {
                    "method": conventional["method"],
                    "delivered_energy_kwh": conventional["service_delivered_kwh"],
                }
            ]),
            reference_service=float(reference["delivered_energy_kwh"].max()),
        )
        print(f"Saved: {out_dir / 'multi_seed_spread.png'}")


if __name__ == "__main__":
    main()
