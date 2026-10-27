# Tier 0 changes — correctness, provenance and reproducibility

**Scope of Tier 0:** make the project *trustworthy*, not *better*. No scientific
claim was to change, no optimizer was to be re-designed, and no objective was to
be re-tuned. Tier 1 (honest comparison methodology) and Tier 2 (making the
metaheuristics actually optimize) are explicitly out of scope and are listed at
the end as deferred work.

**Reproduce everything in one command (~2m50s):**

```bash
cd EV-charging
.venv/Scripts/python src/create_solar_profile.py      # inputs (idempotent)
.venv/Scripts/python src/simulate_baselines.py        # inputs (idempotent)
.venv/Scripts/python src/run_final_experiments.py --multi-seed
.venv/Scripts/python -m pytest tests -q               # 101 passed, 2 skipped
```

---

## 1. Corrections to the earlier review

Three claims in the review that motivated this work turned out to be **wrong or
overstated** once I checked them. They are corrected here first, because the rest
of this document depends on them.

### 1.1 `comparison_metrics.csv` was not "stale" — it was a smoke test

The review said the committed headline table (GA 1088.98 kWh) was produced by an
older version of the optimizer. It was not. Running the *current* code with
`--quick` reproduces that file **exactly**:

| | committed file | `--quick` today | full run today |
|---|---|---|---|
| GA delivered kWh | 1088.984 | 1088.984 | 1116.084 |
| GA peak kW | 102.700 | 102.700 | 98.296 |
| GA cost units | 6905.192 | 6905.192 | 7070.608 |
| GA evaluations | *(not recorded)* | 38 | 560 |

So the defect was **provenance, not staleness**: a 38-evaluation smoke test was
committed as the headline result and nothing in the file said so. Fixed by
writing a run manifest (§2.6).

### 1.2 The rolling-horizon "departure clipping" bug was overstated

The review claimed `clip(dep - t, 0, width)` made every EV look like it left at
the end of the look-ahead window and *inflated* urgency. On inspection the
clipping is **required** (`repair_schedule` receives a priority matrix only
`width` slots wide) and its effect was the opposite of inflated — it *deflated*
urgency to zero for every EV departing after the window, so the controller
deferred far-deadline EVs and then charged them late and urgently.

The real bug was the tariff clock (§2.5), and the urgency defect was fixed in a
way that does not perturb full-horizon runs at all (§2.5).

### 1.3 Scenario numbers corrected

| Quantity | Review said | Measured | Consequence |
|---|---|---|---|
| Port-hours available while EVs are connected | 268 | **310** (20 ports × 62 connected slots × 0.25 h) | Still below the 342.6 port-hours of work at rated power, so the congestion conclusion stands, but 268 (from `results/capacity_feasibility_summary.csv`) was not the right derivation. |
| EVs that cannot be charged even alone | not mentioned | **2 of 100** (need 57.81 kWh, can physically receive 55.52 kWh in their own window) | `demand_met_pct` can never reach 100%, `fully_charged_evs` is capped at 98, and ~2.3 kWh of "unmet" energy is unfixable by any scheduler. |
| Deliverable ceiling | ≈1.2 MWh | Not a single ceiling: per-slot port capacity sums to **1535.5 kWh**, but that bound assumes only the 20 highest-power EVs ever charge. | The binding constraint is concurrency (100 EVs vs 20 ports), not aggregate energy. |
| Grid limit reachable? | "decorative", 20 ports × ~4.3 kW | The theoretical max is **20 × 6.548 = 130.96 kW > 125 kW**, so it is structurally reachable; the *mean* 20-port ceiling is 84.7 kW and every observed peak is 95–99 kW. | Still never approaches binding, but for a subtler reason than stated. |

None of these change the conclusions; all of them are now asserted in
`tests/test_scenario.py` so they cannot drift silently.

---

## 2. Changes

### 2.1 Environment and repository hygiene

**Was:** `.venv/` was committed to git (22,727 of the repository's 22,898 tracked
files) — a Windows virtual environment built on another machine, whose
`pyvenv.cfg` pointed at `C:\Users\DELL PC\...` and `D:\EV optimizer`.
`__pycache__` was committed too, and there was no `.gitignore` and no
`requirements.txt`. The venv's interpreter did not exist on this machine, so the
project could not be run as checked out.

**Now:**

* `.gitignore` added (`.venv/`, `__pycache__/`, `*.py[cod]`, caches, `data/raw/`).
* `requirements.txt` pins the versions the results were produced with:
  `numpy==2.4.6`, `pandas==3.0.6`, `matplotlib==3.11.2`, `pytest==8.4.2`.
* `.venv/` and every `__pycache__/` untracked (`git rm --cached`) and the venv
  rebuilt on the local CPython 3.11.15.
* **Tracked files: 22,898 → 172.**

### 2.2 One engine instead of six

**Was:** six different fitness functions and three different evaluators.

| Module | Fitness / objective | Cost term? |
|---|---|---|
| `run_ga.py` | `1e6·unmet + 1e3·not_charged + peak` | **none** |
| `run_ga_cost.py` | cost + unmet | yes |
| `run_ga_solar.py` | solar-aware | partly |
| `run_ga_peak_constrained.py` | hard peak cap | no |
| `run_ga_battery_aware{,_v2,_v3}.py` | battery proxy | no |
| `evaluate_schedule.py` / `evaluate_objective.py` / `evaluate_cost.py` | three slightly different cost/peak definitions | — |

The "Original GA" row in old tables delivered the most energy *and* cost the
most, because its fitness cannot see cost. Meanwhile the same schedule could be
scored three different ways depending on which evaluator measured it.

**Now:** `src/multiobjective_core.py` is the only repair routine, only evaluator
and only objective. The six alternate-objective runners, the four duplicate
evaluators, the seven archived validators and the two superseded comparison
scripts are moved to `experiments/legacy/` (22 files, with a
`run_ga_legacy.py` frozen snapshot of the old engine so those archived
experiments remain reproducible). `experiments/legacy/README.md` explains what
each one was and why it must not be mixed with pipeline results.

`src/run_ga.py` and `src/run_pso.py` are now thin entry points over the shared
engine — one implementation, shared with `run_final_experiments.py`.

### 2.3 One source of constants

**Was:** 15-minute step, 144 slots, 20 chargers, 125 kW cap, tariff boundaries,
seeds and objective weights were copy-pasted into ~20 scripts. Two scripts could
disagree about the scenario they were comparing.

**Now:** `src/config.py` holds all of it, with absolute paths so scripts run from
any directory. Every module imports from it; `tests/test_config.py` asserts the
derived values (`N_SLOTS == 144`) and that the objective weights form a valid
convex scalarisation summing to 1.

**Numerically neutral:** the refactor was verified to be *bit-identical* — see §3.1.

### 2.4 One scenario loader

**Was:** `load_scenario.py` validated and loaded the CSV; `multiobjective_core.load_inputs()`
silently re-read the same CSV with a *different* validation set. `load_scenario()`
also printed three lines on every call, which made it unusable in tests.

**Now:** `load_scenario()` is the only loader; `load_inputs()` delegates to it and
adds the solar profile. Prints removed (informational output belongs in `__main__`).
Extra checks added: windows must lie inside `[0, N_SLOTS]`.

### 2.5 Rolling-horizon controller bugs

**Bug A — tariff clock.** `optimize_rolling` sliced the solar array for its
look-ahead window but rebuilt the tariff with `tariff_vector(width)`, i.e. every
window believed it started at midnight and applied the 4.0 off-peak rate — even
across the 17:00–22:00 peak. The controller was structurally blind to the tariff
it was supposed to be optimising.

Fixed by threading an absolute `slot_offset` through
`repair_schedule(..., slot_offset=t)`. Verified as a property: for every tested
`(t, k)`, `tariff_vector(k, slot_offset=t) == tariff_vector(N_SLOTS)[t:t+k]`.
For `t = 68` (17:00) the old call returned `4.0` and the fixed call returns `10.0`.

**Bug B — deadline urgency scaled by the window.** Urgency inside the repair
routine is `1 - slack/n_slots`, where `n_slots` was the width of the priority
matrix. In a 24-slot window that collapsed urgency towards zero for every EV
departing after the window, so the controller deferred far-deadline EVs.

Fixed with an explicit `horizon_slots` argument (slots left in the *whole*
simulation). **It defaults to `n_slots`, so full-horizon runs are numerically
unchanged** — this is why the one-shot GA/PSO results in §3.1 are bit-identical.

**Also added:** the assembled trajectory is validated against every hard
constraint, and an unknown `method` now raises instead of silently running the GA.

**Rolling-horizon results changed by design** (they must — the clock was wrong):

| Method | delivered kWh | peak kW | cost units | violations |
|---|---|---|---|---|
| GA rolling horizon | 1125.553 | 98.845 | 7185.903 | 0 |
| PSO rolling horizon | 1121.014 | 95.535 | 7143.325 | 0 |

Whether that is *better* needs the Tier 1 equal-service analysis; Tier 0 only
makes it correct and feasible.

### 2.6 Provenance: a run manifest

Every run now writes `results/final_unified/run_manifest.csv` with the mode
(`full`/`quick`), whether rolling horizon ran, per-optimizer settings and seeds,
the objective weights, the scenario shape, and a SHA-256 (truncated) of the
scenario and solar files. `tests/test_pipeline_outputs.py` fails if the committed
headline results are a `quick` run or if the input hashes no longer match.

### 2.7 Baseline consolidation, and a genuine FCFS

**Was:** the table's "FCFS" row was not FCFS. It was a constant per-EV priority
vector (`rank` by arrival) fed into the *same* tariff- and solar-aware repair
heuristic the optimizers use — i.e. not a conventional policy at all.
`simulate_baselines.py` contained a *different* FCFS that used
`max_grid_kw` instead of `max_grid_kw + solar` for its per-slot capacity, so the
baseline and the optimizers were not even solving the same problem.

**Now:** `src/baselines.py` holds five policies in two clearly separated families:

* **Policy baselines** (commit each slot as time advances): `uncontrolled`,
  `chronological_fcfs` (new — genuine first-come-first-served), `chronological_edf` (new).
* **Priority baselines** (a fixed ranking handed to the repair heuristic, renamed
  to say so): `priority_fcfs`, `priority_edf`.

All use the same per-slot capacity rule as the repair routine, so every row of a
comparison table faces an identical constraint set. `tests/test_baselines.py`
proves the two families are different policies on a two-EV case where FCFS keeps
the only port for the earliest arrival while EDF gives it to the tighter deadline.

**This changes the headline story.** The genuine chronological EDF is now the
best feasible method on the shared objective:

| Method | objective | delivered kWh | peak kW | cost | cost/kWh |
|---|---|---|---|---|---|
| **EDF (chronological)** | **0.302** | **1133.740** | **96.902** | 7226.694 | 6.374 |
| GA | 0.311 | 1116.084 | 98.296 | 7070.608 | 6.335 |
| PSO | 0.318 | 1102.284 | 97.029 | 7036.148 | 6.383 |
| FCFS (chronological) | 0.345 | 1051.981 | 98.893 | 6872.956 | 6.533 |
| FCFS-priority (repair) | 0.385 | 962.530 | 96.131 | 5605.794 | 5.824 |

Both metaheuristics now lose to a one-pass rule-based policy with no search at
all — which is a much sharper version of the question that motivated this work.

### 2.8 Hot-loop hygiene

* **Prints removed from `objective()`.** It printed three diagnostic lines per
  call: ~1,700 lines of console noise per GA run and unusable inside tests. Now
  covered by a regression test.
* **Evaluation cache removed.** The GA memoised with a 115 KB chromosome as the
  dict key and the 115 KB schedule as the value (~230 KB per evaluation, ~130 MB
  for a single 560-evaluation run, and worse for larger budgets). Elites now
  carry their score forward instead, so each chromosome is evaluated **exactly
  once** and no cache exists at all. Memory is bounded by the population
  (~2.3 MB) and the result is identical.
* **PSO's cache removed** — positions change every iteration, so it never hit.
* **Reported budget:** the table now has an `evaluations` column (GA 560,
  PSO 192 for a full run) so a budget cannot be inferred from a generation count.

### 2.9 Validation that can actually fail

**Was:** six `test_*.py` and five `validate_*.py` scripts in `src/` were
print-only. `test_repair_quality.py` "tested" the repair by printing metrics and
always exited 0; `validate_ga.py` printed `WARNING: ...` and exited 0.

**Now:** a real pytest suite in `tests/` — **101 tests, 2 skipped** (the raw
dataset is not committed):

| File | Covers | Replaces |
|---|---|---|
| `test_config.py` | constants, weight validity, budget regression | — |
| `test_scenario.py` | scenario integrity + the three structural characterisations | `validate_scenarios.py`, `test_grid_capacity.py` |
| `test_constraints.py` | repair can never violate a hard constraint, random + adversarial inputs | `test_repair_quality.py` |
| `test_objective.py` | cost/peak/energy on known schedules, solar offset, tariff by time of day, infeasibility dominance, no-print regression | `test_objective.py`, `test_cost_evaluation.py` |
| `test_battery_proxy.py` | proxy arithmetic (half power ⇒ 0.25, full ⇒ 1.0) | `test_battery_proxy*.py` |
| `test_baselines.py` | feasibility, determinism, chronological ordering | `validate_baselines.py` |
| `test_rolling_horizon.py` | tariff clock property, urgency scale, trajectory feasibility | — |
| `test_optimizers.py` | determinism under seed, feasibility, history, budget arithmetic | — |
| `test_pipeline_outputs.py` | manifest mode + input hashes, table↔schedules agreement, evaluation budget, no `*_latest*` files | `validate_ga.py`, `validate_pso.py` |
| `test_session_data.py` | raw ACN session parsing; derived data present | `validate_session_data.py` |

`test_pipeline_outputs.py` and `test_session_data.py` skip with an explicit
reason when their inputs are absent, rather than failing misleadingly.

### 2.10 Artifacts deleted

Nine files whose producers were archived or deleted and which could not be
regenerated, including two mislabelled ones:

| File | Why |
|---|---|
| `results/ga_cost_best_schedule_latest.csv`, `results/ga_cost_metrics_latest.csv` | Duplicates of the non-`_latest` files. Verified identical (1115.843 kWh, `max abs difference: 0.0`); the 5 analysis scripts that read `_latest` were repointed to the canonical file, a true no-op. |
| `results/multi_seed_results.csv`, `results/multi_seed_summary.csv` | Produced by the archived `benchmark_algorithms.py` from the **legacy** engine, and directly contradicted the new authoritative table (stale: GA 1172.8 kWh mean, peak 100.8 kW; shared engine: GA 1114.5 kWh, peak 97.6 kW). Multi-seed results now come from `run_final_experiments.py --multi-seed` into `results/final_unified/`. |
| `results/repair_test_schedule.csv` | Diagnostic output of the deleted print-only `test_repair_quality.py`. |
| `results/baseline_validation.csv` | Produced by the archived `validate_baselines.py`; superseded by `baseline_metrics.csv`. |
| `results/final_algorithm_comparison.csv` + 4 `final_*.png` | Produced by the archived `final_comparision.py`, and it labelled a repair-based priority vector as "FCFS" and used a different evaluator. Honest comparison plots belong in Tier 1. |

---

## 3. Verification

### 3.1 The refactor is numerically neutral

Captured from the pre-change code, then re-run after every change, with
`--skip-rolling` (full settings):

| Method | delivered kWh before | after | peak kW before | after | cost before | after |
|---|---|---|---|---|---|---|
| GA | 1116.084 | **1116.084** | 98.296 | **98.296** | 7070.608 | **7070.608** |
| PSO | 1102.284 | **1102.284** | 97.029 | **97.029** | 7036.148 | **7036.148** |
| FCFS-priority | 962.530 | **962.530** | 96.131 | **96.131** | 5605.794 | **5605.794** |
| EDF-priority | 1064.368 | **1064.368** | 97.106 | **97.106** | 6556.383 | **6556.383** |

Bit-identical, including a subtle detail: the original tournament selection broke
ties by choosing the *last* minimum (`a if s[a] < s[b] else b`). The obvious
`min()` refactor would have chosen the *first* and silently changed results on
this flat landscape, so the behaviour is reproduced explicitly and commented.

### 3.2 Inputs are unchanged

* `create_solar_profile.py` → `solar_profile_36h.csv` **byte-identical**
  (381.835 + 184.668 = 566.503 kWh).
* `simulate_baselines.py` → `ev_scenario.csv` **byte-identical**. The draw order
  (uniform arrivals, then uniform stay durations) is preserved exactly and
  commented as such, because regenerating the scenario would invalidate every
  stored result.

### 3.3 Independent cross-check of the new engine

The archived legacy evaluator, run on the *new* pipeline's GA schedule, agrees
exactly: 1116.084 kWh delivered, 98.296 kW peak, 0 grid/charger/power/window
violations. Two independently written evaluators, one answer.

### 3.4 Test suite and full run

```
101 passed, 2 skipped in 24.68s

MULTI-SEED SUMMARY (mean +/- std over seeds [42, 43, 44, 45, 46])
method  runs  delivered_mean  delivered_std  peak_mean  peak_std  cost_mean  cost_std
    GA     5        1114.509          2.091     97.559     0.790   7105.478    50.792
   PSO     5        1097.928          5.641     97.403     2.488   6976.164    60.180
```

Multi-seed is now computed in-process. It previously ran `run_ga.py`/`run_pso.py`
as subprocesses and recovered metrics by regex-matching their console output, so
renaming a printed label would silently corrupt the benchmark — and it was
measuring the *legacy* engine (GA 1,231.7 kWh), not the pipeline's.

### 3.5 Nothing is left broken

* Static import scan over `src/`: no unresolved imports.
* All 22 archived scripts byte-compile; the frozen legacy engine imports and
  exposes every symbol the archived variants expect.
* No references to any deleted artifact remain in `src/`.

---

## 4. Deferred to Tier 1 and Tier 2 (not done here)

Classed explicitly as **not** completed, so they are not mistaken for finished work.

**Tier 1 — make the comparison honest**

1. Compare at equal service: cost and peak *per delivered kWh*, or an
   ε-constraint frontier. Total cost and peak remain volume metrics.
2. Put `P_base` back into the grid model (`P_grid = P_base + P_ev - P_solar`).
   Without it the "125 kW grid limit" constrains only EV load.
3. Re-normalise the objective: cost is currently compressed into 0.47–1.0 by
   `cost/(required × 10)`, so its 6% weight is nearly inert.
4. Fix or drop the solar term: only 43.8 of 566.5 kWh of PV coincides with any
   connected EV, so its maximum attainable contribution is ~7.7% at 5% weight.
5. Rebuild the scenario with slack (more ports or fewer EVs) so scheduling has
   room to matter, and report the 2 individually infeasible EVs explicitly.
6. Re-tune the objective weights and run a sensitivity sweep.

**Tier 2 — make the metaheuristics actually optimize**

7. Replace the encoding. 14,400 genes with 560 evaluations is hopeless, and the
   genome controls only ~32% of the ranking score because the repair routine's
   own weights total 0.95. Per-EV `(start slot, power level)` genes (~200) or a
   permutation decoder would fix the root cause.
8. Remove the "fill remaining capacity" stage that the legacy repair used; the
   shared repair does not add unrequested energy, and a memetic local search
   over slot swaps should now be viable.
9. Give PSO a representation where a velocity step is meaningful (random-key or
   discrete/permutation PSO), or drop it.
10. Add an LP/MILP reference optimum (PuLP, OR-Tools, `scipy.optimize.linprog`).
    Without a benchmark, "GA beat FCFS" is not a claim about optimality.
11. Actually report budgets: the `evaluations` column exists; the real fix is
    10⁴–10⁵ evaluations rather than 560.

**Also still open:** ~20 one-off `analyse_*`/`audit_*`/`compare_*` scripts remain
in `src/`; they are analysis, not pipeline, and belong in `experiments/`.
Module 12 (dashboard), 13 and 14 (literature matrix, paper) have no artifacts.
The raw ACN dataset is not committed, so `create_clean_datasets.py` cannot be
re-run from this checkout.
