# Tier 1 changes — making the comparison honest

**Scope of Tier 1:** stop the *comparison* from flattering or penalising a method
because of how much energy it happened to move. Tier 1 does not change the
scenario, the objective weights or the optimizers; those are Tier 2. Six items
were deferred from Tier 0 and are listed at the end with their status.

**Item 1 — judge methods at equal service, with cost and peak per delivered kWh —
is the only item completed here.** Everything below describes that work.

**Reproduce (~3-6 min, machine-dependent):**

```bash
cd EV-charging
.venv/Scripts/python src/run_final_experiments.py --multi-seed
.venv/Scripts/python -m pytest tests -q
```

---

## 1. The problem: total cost and total peak are volume metrics

`results/final_unified/comparison_metrics.csv` reports **total** tariff cost and
**total** peak demand. Under this scenario those totals are dominated by how much
energy each method delivered, not by how well it scheduled:

| Observed spread across the 8 feasible methods | value |
|---|---|
| Delivered energy | 962.5 → 1133.7 kWh (**17.8%**) |
| Peak EV load | 95.5 → 98.9 kW (**3.5%**) |
| Total tariff cost | 5605.8 → 7226.7 units (28.9%) |

No method exceeds **75.02%** of the attainable energy, and 2 of the 100 EVs
cannot be charged at all (Tier 0, §1.3), so full service is impossible for every
method and "deliver less" is always available as a way to look cheap.

The concrete distortion: ranking by total cost puts **FCFS-priority first**
(5605.8 units) — but it is also the **worst** method on service, delivering
63.69% against EDF's 75.02%. It looks cheapest because it moved 171 kWh less.

The peak column has the opposite problem: it is nearly constant. Every method
peaks at 95.5-98.9 kW against a 125 kW cap, because peak demand is set by
20 ports times the fleet's average power rather than by the grid. It carries
almost no information about scheduling quality, yet it sits in the table as if it
were a result.

## 2. What changed

| File | Change |
|---|---|
| `src/equal_service.py` | **New.** The equal-service comparison: service-ceiling accounting, the two intensity metrics, and Pareto dominance. Exports `EQUAL_SERVICE_COLUMNS`, `per_ev_attainable_energy`, `service_accounting`, `equal_service_row`, `flag_pareto`, `equal_service_table`, `frontier_rows`. |
| `src/artifacts.py` | Added `save_equal_service_plot`. Tier 0 deleted the misleading comparison plots on the grounds that "honest comparison plots belong in Tier 1"; this is that plot. |
| `src/run_final_experiments.py` | Builds and writes the equal-service table and plot; the console output now prints the volume table as an audit record and the equal-service table as *the* comparison, followed by the frontier in cost order and the dominated methods with their dominators. |
| `tests/test_equal_service.py` | **New.** Locks the arithmetic invariants, the dominance semantics, and the committed artifact's consistency. |
| `results/final_unified/equal_service_comparison.csv` | **New artifact** — the comparison. |
| `results/final_unified/equal_service_frontier.png` | **New artifact** — service attained against both intensities. |
| `results/final_unified/comparison_metrics.csv` | **Unchanged in substance.** It stays as the audit/provenance table: raw volumes, evaluation budgets, runtimes, violations. |

Nothing in the optimization path changed. The optimizer rows are bit-identical to
the committed ones (GA 1116.084 kWh / objective 0.311073, PSO 1102.284 kWh), and
§8 verifies it exhaustively.

## 3. The two metrics

```
cost_per_delivered_kwh = grid_cost_units / delivered_energy_kwh
peak_per_delivered_kwh = peak_ev_load_kw  / delivered_energy_kwh
```

Both are also accompanied by `peak_to_average_ratio`, which is **the same number
scaled by the horizon** — `peak_per_delivered_kwh × 36 h`. Both are reported
because one is what the brief asked for and the other is dimensionless, but they
are not two pieces of evidence:

```
assert peak_to_average_ratio == peak_per_delivered_kwh * len(solar) * STEP_HOURS
```

(`tests/test_equal_service.py::test_intensity_metrics_are_ratios_of_the_evaluated_totals`)

## 4. Why this is an "equal service" comparison — and where it stops being one

**The property.** Scaling every power in a schedule by a factor β scales delivered
energy by β and peak demand by β, so both ratios are unchanged: *a method gets no
credit for delivering less energy and no credit for delivering more*. That is the
formal sense in which cost and peak per delivered kWh are judged at equal service,
and it is asserted directly rather than asserted in prose
(`test_intensity_is_invariant_to_uniform_scaling_without_solar`).

**The caveat on cost.** The cancellation is exact for the peak. For cost it is
exact only while grid import equals EV load: with solar the import is
`max(load - solar, 0)`, which is convex rather than linear, so a small residual
volume effect survives. The invariance test therefore runs on a solar-free case.

**The caveat that matters more.** Cost per kWh is an *intensity*, not the cost of
serving a mandated quantity of energy. The marginal kWh is harder to deliver than
the first one, because it belongs to EVs with tighter deadlines, so at equal
efficiency a method that delivers more energy still shows a slightly higher cost
per kWh. The evidence is right there on the frontier (§6): cost per delivered kWh
rises **monotonically** with service, 5.824 → 6.374, across the five non-dominated
methods. A true equal-service frontier needs the ε-constraint sweep in §9, which
is not done here.

## 5. Service accounting: the denominator was wrong before

`service_pct_of_required` can never reach 100%, which makes it a misleading
headline. The scenario's real ceiling is computed per EV as
`min(required, window_hours × max_power)`:

| Quantity | kWh |
|---|---|
| Required energy | 1513.561 |
| **Attainable** energy (per-EV ceiling) | **1511.277** |
| Physically impossible energy (2 EVs) | 2.284 |

So every service figure in the new table is available in two forms:
`service_pct_of_required` and `service_pct_of_attainable` (EDF: 74.91% vs 75.02%).
`unmet_congestion_kwh` separates the unfixable shortfall from the schedulable one
— for EDF, 377.5 kWh of the 379.8 kWh shortfall is congestion, which is what a
scheduler can actually influence.

The ceiling is cross-checked against an independent construction: the
`Uncontrolled` policy charges each EV at its own maximum power for as long as it
is connected, which *is* the per-EV ceiling, and it reaches exactly 100.00% of
`attainable_energy_kwh`. If the two definitions ever drift apart, one of them is
wrong (`test_the_infeasible_reference_reaches_the_service_ceiling`).

## 6. Dominance, and the resulting comparison

Even at equal service, cost per kWh and peak per kWh can trade off, so the table
does not pretend to produce a single winner. What it does produce is the set that
*needs* a trade-off argument: a method is flagged `dominated` when another
feasible method is at least as good on service, cost per kWh **and** peak per kWh,
and strictly better on one. Such a method cannot be preferred under any weighting
of those three and can be dropped without discussion.

`results/final_unified/equal_service_comparison.csv` (feasible methods):

| Method | Service attained | cost/kWh | peak/kWh | Verdict |
|---|---|---|---|---|
| EDF (chronological) | **75.02%** | 6.3742 | 0.08547 | frontier |
| GA rolling horizon | 74.48% | 6.3843 | 0.08782 | **dominated** by EDF (chronological) |
| PSO rolling horizon | 74.18% | **6.3722** | **0.08522** | frontier |
| GA | 73.85% | 6.3352 | 0.08807 | frontier |
| PSO | 72.94% | 6.3832 | 0.08802 | **dominated** by EDF (chronological); PSO rolling horizon |
| EDF-priority (repair-based) | 70.43% | 6.1599 | 0.09123 | frontier |
| FCFS (chronological) | 69.61% | 6.5333 | 0.09400 | **dominated** by 6 methods |
| FCFS-priority (repair-based) | 63.69% | **5.8240** | 0.09987 | frontier |

Non-dominated methods, cheapest service first — the trade-off is one-dimensional:

| | service | cost/kWh |
|---|---|---|
| FCFS-priority (repair-based) | 63.69% | 5.8240 |
| EDF-priority (repair-based) | 70.43% | 6.1599 |
| GA | 73.85% | 6.3352 |
| PSO rolling horizon | 74.18% | 6.3722 |
| EDF (chronological) | 75.02% | 6.3742 |

**What this changes about the Tier 0 headline.** Tier 0 reported that the
chronological EDF beats both metaheuristics. At equal service that is a *service*
win, not a uniformly better schedule: EDF delivers 1.6% more energy than the GA
but pays **0.6% more per delivered kWh** (6.3742 vs 6.3352), and PSO's rolling
controller achieves a lower peak per kWh (0.08522) than EDF. The claim that
survives is narrower and safe: EDF attains the most service, and the GA is the
most cost-efficient of the high-service methods.

**What this changes about the objective's ranking.** The scalarized objective
(`comparison_metrics.csv`, `objective_score`) ranked GA rolling horizon 2nd of 9
(0.3066). At equal service that row is **strictly dominated** by EDF: EDF delivers
more energy, cheaper per kWh, at a lower peak per kWh. The objective did not
express a preference for a method that is worse on all three criteria — it could
not, because 96% of its weight sits on unmet energy and only 6% on cost. Reading
the new table removes three of eight methods from the argument outright.

## 7. The peak metrics are nearly inert — do not over-read them

`peak_per_delivered_kwh` was explicitly requested, so it is reported, but in this
scenario it must not be read as evidence about peak shaving:

* Peak EV load spans 95.535-98.893 kW (**3.5%**) while delivered energy spans
  17.8%, so a ratio of the two is close to `constant / delivered_energy`.
  By the squared log-spread ratio, of the order of **4%** of the variation in
  `peak_per_delivered_kwh` can come from its numerator. *(Approximation, flagged
  as such: it treats the log-spreads as the relevant scales.)*
* Consistently, `peak_per_delivered_kwh` is almost a monotone function of service
  (Spearman **-0.905**), i.e. methods that serve more have a *lower* peak per kWh.
* `cost_per_delivered_kwh` does **not** share that defect: its rank correlation
  with service is only **+0.310**, so it carries genuine independent information.

The conclusion to carry forward is that peak demand is not a competing objective
in this scenario at all — it is Tier 0's "the 125 kW cap never binds" seen from a
different angle. A volume-free peak metric would have to be normalised against a
scenario-specific flat ideal, which is not attempted here.

## 8. Verification

* **The optimizers are untouched, and this is checked exhaustively.** The
  regenerated `comparison_metrics.csv` was compared against the committed one
  column by column, row by row, excluding `runtime_seconds`: **zero differences**
  across all 9 methods and 19 columns, with exact equality (`atol=0`), and
  `results/final_unified/schedules.npz` is **byte-identical**. The equal-service
  module is only ever called after optimization has finished. The multi-seed
  means reproduce Tier 0's exactly as well (GA 1114.509 ± 2.091 kWh, peak
  97.559 ± 0.790 kW).
* **The new module cannot silently disagree with the evaluator.** The intensity
  metrics are recomputed from `evaluate_schedule` in-test, and the committed
  artifact is checked against the committed volume table (they must carry the
  same delivered energy and the same cost per kWh to 1e-12).
* **Every dominance verdict is re-derived semantically.** The test does not call
  `flag_pareto`; it re-reads the shipped schedules, rebuilds the keys, and checks
  that each `dominated_by` name really is at least as good on all three keys and
  strictly better on one, and that each `frontier` row has no dominator at all
  (`test_every_dominance_verdict_holds_under_recomputation`).
* **Infeasible rows can never dominate.** `Uncontrolled` delivers the most energy
  of any row precisely because it ignores the port and grid limits; it is labelled
  `infeasible reference`, excluded from the ranking, and separately asserted to
  appear in no `dominated_by` field.
* **Exact ties are not dominance** (`test_exact_ties_are_not_dominance`) — the
  comparison operator needs a strict improvement, not a `>=`.

Test suite: **120 passed, 2 skipped, 0 failed** (`pytest tests -q`, exit 0). The
two skips are the raw-ACN-data tests, which need a dataset that is not committed;
they skip with an explicit reason rather than failing misleadingly.

## 9. Deferred (Tier 1 items 2-6, and the ε-constraint)

Listed explicitly as **not** done, so they are not mistaken for finished work.

1. **The ε-constraint / service-matched frontier.** The rigorous complement to §4:
   sweep a mandated service requirement, re-run every method at each level, and
   compare cost and peak at *identical* service. This needs re-runs - the rolling
   controller alone costs 80-155 s per pass and the GA 27-63 s on the current
   machine (§10) - so it is a deliberate second step rather than something folded
   into the default run.
2. **Put `P_base` back into the grid model** (`P_grid = P_base + P_ev - P_solar`).
   Without it the 125 kW limit constrains only EV load, which is why §7 finds the
   peak objective inert.
3. **Re-normalise the objective.** Cost is compressed into 0.47-1.0 by
   `cost/(required × 10)`, so its 6% weight is nearly inert — visible directly in
   §6, where the objective cannot express the cost-per-kWh preference at all.
4. **Fix or drop the solar term.** Only 43.8 of 566.5 kWh of PV coincides with any
   connected EV, so its maximum attainable contribution is ~7.7% at 5% weight.
5. **Rebuild the scenario with slack** (more ports or fewer EVs) so scheduling has
   room to matter, and report the 2 individually infeasible EVs explicitly.
6. **Re-tune the objective weights and run a sensitivity sweep.**

Also still open from Tier 0: ~20 one-off `analyse_*`/`audit_*`/`compare_*` scripts
remain in `src/`, and the raw ACN dataset is not committed.

## 10. Test and run record

From the run that produced the artifacts in this revision. See
`results/final_unified/run_manifest.csv` for the exact provenance: mode `full`,
rolling horizon enabled, multi-seed enabled, GA settings
`population_size=20, generations=30`, PSO settings `particles=12, iterations=15`,
and input hashes `658fcc66a11fcef0` (scenario) / `63dba68e5466b222` (solar).

```
120 passed, 2 skipped in 15.26s

MULTI-SEED SUMMARY (mean +/- std over seeds [42, 43, 44, 45, 46])
method  runs  delivered_mean  delivered_std  peak_mean  peak_std  cost_mean  cost_std  unmet_mean  unmet_std
    GA     5        1114.509          2.091     97.559     0.790   7105.478    50.792     399.053      2.091
   PSO     5        1097.928          5.641     97.403     2.488   6976.164    60.180     415.633      5.641
```

**A note on the timing columns.** Runtimes differ only because of machine load,
not because of this change: three runs of *identical* work on the same machine
produced GA 27.4 s / 48.8 s / 63.2 s and PSO 4.5 s / 5.7 s / 18.3 s, with every
result bit-identical. `runtime_seconds` and the wall-clock cost of a full run are
therefore not comparable between runs. The first `--multi-seed` attempt was cut
short by a tool timeout for this reason and was re-run to completion; the
committed artifacts come from the completed run.
