# Project overview — EV charging optimization

What this project is, what was built, what we learned. For the detailed history see
`docs/TIER0_CHANGES.md`, `TIER1_CHANGES.md`, `TIER2_CHANGES.md`; for the academic
deliverables see `academic/`.

---

## 1. The problem

A public charging site oversubscribed against both its hardware and its grid contract:

- **Demand:** 100 EV sessions needing **1,513.6 kWh** over a **36-hour horizon**
  (144 slots of 15 min), calibrated to ACN-Data (Caltech ACN) session statistics.
- **Site:** **20 ports** (per-EV rated power 3.3–7.2 kW), grid contract **125 kW**,
  566.5 kWh of horizon PV (only 43.8 kWh of it coincides with a connected EV).
- **Tariff:** time-of-use — off-peak 4.0, mid [6.0], peak 10.0 units/kWh.

Decide per-slot charging power `s[i,t]` under five constraints (connection window,
per-EV rate ≤ rated, ≤ 20 ports per slot, site cap `≤ 125 kW + PV`, no over-charge),
minimising a weighted sum: unmet 0.55, cost-per-kWh 0.30, peak 0.08, solar 0.05,
battery 0.02.

**Why scheduling matters here (the certified fact):** the site offers **268 port-hours**
but the sessions need **342** — ports, not the grid, are the binding resource. Which EVs
get a port in which slot is the whole game.

---

## 2. What was built

| Layer | Files | Role |
|---|---|---|
| Data | `load_scenario.py`, `create_solar_profile.py`, `validate_*.py` | scenario + PV prep, sanity checks, SHA-256 provenance |
| Constraints | `multiobjective_core.py` | **one** evaluator (`evaluate_schedule`) + `constraint_report` — every method scored through it |
| Search | `genome.py`, `multiobjective_optimizers.py` | random-key genome, feasible decoder, memetic GA + PSO |
| Reference | `reference_optimum.py` | MILP referee (HiGHS via `scipy.optimize.milp`), ε-constraint cost frontier, hand bounds |
| Baselines | `baselines.py`, `equal_service.py` | FCFS/EDF chronological + repair-priority variants, rolling-horizon controllers, equal-service pricing |
| Reporting | `run_final_experiments.py`, `artifacts.py` | pipeline orchestration, 5 figures, provenance CSVs, run manifest |

Plus one Python orchestrator (`run_final_experiments.py --multi-seed` reproduces
everything in ~25–30 min) and a **155-test quality gate** (2 skipped: need an
uncommitted raw dataset; 0 failed).

### The genome (Tier 2's core idea)

4 interpretable genes per EV (400 total, replacing an unsearchable 14,400-cell
priority matrix): `rank`, `urgency`, `tariff`, `power`. A slot-major decoder awards
ports by `score = rank + urgency·deadline − tariff·price`; **a negative score defers
the EV** — feasibility is structural, no repair stage, no penalty. Both GA
(40×240, elitism 4, mutation 0.02) and PSO (30×300) search this same genome with
4 shared warm starts, plus a memetic polish counted exactly: **GA 12,680 / PSO 13,030
evaluations**.

### The MILP referee

One binary per feasible (EV, slot) pair (3,703 of 14,400), solved to proven optimality
(gap ≤ 0.8%): max-service optimum plus a 5-point ε-constraint min-cost frontier
(960→1,260 kWh). Hand bounds: per-EV 1,511.3 kWh, per-slot 1,535.5 kWh.

---

## 3. How the project progressed (three documented tiers)

| Tier | Focus | Outcome |
|---|---|---|
| **0** | Audit everything honestly | Found the real defect list: three inconsistent cost evaluators (same schedule, three prices), a printing evaluator (~1,700 log lines per 560-eval run), an infeasible uncontrolled row, solar term scoring wrongly. Consolidated one evaluator, one repair, one metric set. |
| **1** | Fair benchmark + prerequisites | 9-policy benchmark on one evaluator; rolled-horizon controller (144 replans); cost term turned into an intensity (per kWh). Finding: **EDF beat both metaheuristics** — GA 1,116.1 vs EDF 1,133.7 kWh — because 560 evaluations on a 14,400-gene genome cannot search anything. Tier 1 wrote that down as the hypothesis Tier 2 tested. |
| **2** | Rebuild the search, certify the ceiling | New 400-gene genome + memetic GA/PSO + MILP referee. Re-tuned mutation from measured ablations (0.08→0.02 alone: +65.5 kWh). **Reversal:** both optimizers now beat every conventional policy. |

---

## 4. Headline results (seed 46, full budget, one evaluator)

| Method | Delivered kWh | % of demand | % of best known | Cost/kWh | Fully charged |
|---|---|---|---|---|---|
| **PSO** | **1,254.7** | **82.9** | **97.0** | 6.332 | 25 |
| **GA** | 1,239.5 | 81.9 | 95.8 | 6.348 | 16 |
| EDF (best conventional) | 1,133.7 | 74.9 | 87.6 | 6.374 | **61** |
| FCFS | 1,052.0 | 69.5 | 81.3 | 6.533 | 62 |
| *MILP max-service (referee)* | *1,293.7* | *85.5* | *100* | *6.377* | 32 |

- **+10.7% / +9.3% energy over EDF** at equal feasibility and identical constraints.
- **Cost at equal service:** PSO is 7.2% above the cheapest attainable schedule,
  EDF 16.4% — the optimizers halve the conventional premium.
- **Stable:** 5 seeds each — PSO 1,253.1 ± 7.3 kWh, GA 1,249.0 ± 7.3; all 10 runs beat
  EDF (worst by +105.8 kWh).
- **Mechanism** (`comparison_load_profiles.png`): FCFS/EDF idle 20–30 kW of the site all
  evening; the optimizers hold ~100 kW near the 125 kW cap.

### What the optimizers honestly do NOT win

- **Completed EVs:** EDF finishes 61 of 100, PSO only 25, even the max-service MILP just
  32. Total-energy optimisation structurally favours high-power EVs — a fairness-weighted
  objective is the top deferred item.
- **Peak:** 100–102 kW vs EDF's 96.9 — serving 10% more energy through fixed ports means
  running the site harder. No peak-shaving claim is made.

---

## 5. Known limitations (tracked, not hidden)

1. Fairness not in the objective; completed-EV count suffers.
2. No base load in the grid cap — EV-only.
3. Scenario has zero slack (268 vs 342 port-hours saturates everything).
4. Solar term nearly inert at this arrival mix (43.8/566.5 kWh coincidence).
5. ε-frontier cost comparisons are chords between solved levels, not per-level solves.
6. Rolling controller deliberately under-budgeted (reported as a controller, not a tuned optimizer).
7. ~20 one-off `analyse_*`/`audit_*` scripts remain in `src/`; raw ACN dataset not committed.

---

## 6. Verification (how we know it's right)

- Every MILP solution re-validated through the project's own `constraint_report`;
  this check caught a real bug during development (a silently mis-folded model produced
  a "feasible" 1,511 kWh schedule using 36 ports in a 20-port slot).
- Budgets, genome feasibility (incl. degenerate inputs), objective invariants,
  provenance staleness, and the headline claim are all **assertions in the test suite**,
  not documentation.
- All result rows carry scenario/solar SHA-256 digests; a stale reference optimum
  cannot silently price a regenerated scenario.

## 7. How to run

```bash
cd EV-charging
.venv/Scripts/python src/run_final_experiments.py --multi-seed   # full artifact set
.venv/Scripts/python -m pytest tests -q                          # quality gate
.venv/Scripts/python webapp/app.py                               # browser demo (127.0.0.1:8000)
```

`--reuse-reference` skips the six MILPs when inputs are unchanged.

## 8. Where the documents live

| Artifact | Path |
|---|---|
| Tier change histories | `docs/TIER{0,1,2}_CHANGES.md` |
| Result CSVs + figures | `results/final_unified/` |
| Review II deck + viva | `academic/REVIEW2_PRESENTATION.md` |
| Research article draft | `academic/RESEARCH_ARTICLE_DRAFT.md` |
| Final project report | `academic/PROJECT_REPORT.md` |
| Interactive demo app | `webapp/` (see `webapp/README.md`) |
