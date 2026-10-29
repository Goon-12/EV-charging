# Archived (pre-Tier-0) experiments

Everything in this folder is **frozen history**. These scripts are kept so the
result files they produced can still be regenerated and audited, but they are
**not part of the pipeline** and their numbers must never be mixed into a
comparison table with the pipeline results.

## Why they were moved out of `src/`

Before Tier 0 the repository held nine different GA runners, two PSO runners and
six different fitness functions. Two consequences made the headline research
claim ("GA/PSO beat conventional policies") unverifiable:

1. **Different objectives were being compared as if they were the same method.**
   `run_ga_legacy.evaluate_fitness` has no cost term at all
   (`1e6*unmet + 1e3*not_fully_charged + peak`), while `run_ga_cost` optimised a
   tariff term and `run_ga_battery_aware_*` optimised a battery proxy. The
   "Original GA" row in the old comparison tables therefore reports the largest
   delivered energy *and* the highest cost of any method — not because the GA
   was better, but because this fitness function simply does not see cost.
2. **Duplicate evaluators drifted apart.** `evaluate_schedule.py`,
   `evaluate_objective.py` and `evaluate_cost.py` each computed peak, cost and
   unmet energy slightly differently, so the same schedule could score
   differently depending on which module measured it.

`src/` now contains exactly one repair routine, one evaluator and one objective
(`multiobjective_core.py`), and every algorithm goes through them.

## Contents

| Group | Files | Superseded by |
|---|---|---|
| Alternate-objective GA runners | `run_ga_cost.py`, `run_ga_solar.py`, `run_ga_peak_constrained.py`, `run_ga_battery_aware.py`, `run_ga_battery_aware_v2.py`, `run_ga_battery_aware_v3.py` | `src/run_ga.py` |
| Frozen engine they depend on | `run_ga_legacy.py` | `src/multiobjective_core.py` + `src/multiobjective_optimizers.py` |
| Duplicate evaluators | `evaluate_schedule.py`, `evaluate_objective.py`, `evaluate_cost.py`, `repair_schedule_cost.py` | `src/multiobjective_core.py` |
| Archived validators | `validate_ga.py`, `validate_ga_cost.py`, `validate_ga_solar.py`, `validate_ga_battery_aware*.py`, `validate_pso.py`, `validate_baselines.py` | `tests/` |
| Superseded comparison scripts | `final_comparision.py`, `benchmark_algorithms.py` | `src/run_final_experiments.py` |
| Other one-off heuristics | `run_solar_scheduler.py` (solar-aware rule), `run_feasibility_baseline.py` (tightest-window-first rule) | `src/baselines.py` |

## Running them

```bash
# from the repository root (EV-charging/)
.venv/Scripts/python experiments/legacy/run_ga_cost.py
```

* Archived **runners** write into `results/legacy/`, so running one can never
  overwrite a pipeline artifact.
* Archived **validators** still read the archived schedules that remain in
  `results/` (`ga_cost_best_schedule.csv`, `ga_battery_aware_*`, ...).
  `validate_ga.py` points at `results/ga_best_schedule.csv`, which after Tier 0
  is the *pipeline's* GA schedule — running it there is a useful independent
  cross-check of the new engine with the old evaluator, not an error.
* The scripts resolved their own imports from `src/`. That path is now appended
  instead of prepended so sibling frozen modules win; `load_scenario` still
  resolves from `src/`.
* Every file here had its project root corrected from `parents[1]` to
  `parents[2]` because the folder is one level deeper.

## Do not use these for new results

If you need a cost-aware or battery-aware variant, add it to
`src/multiobjective_optimizers.py` as a *weight set* over the single shared
objective instead of writing another runner. Optimising a different objective
and then reporting it next to the pipeline's numbers is exactly the mistake this
folder exists to prevent.
