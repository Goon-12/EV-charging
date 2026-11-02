# Review II — Design & Implementation (Target: 30/30)

**Project title:** Soft-Computing-Based Scheduling of Public Electric Vehicle Charging Under Port,
Grid and Deadline Constraints — *A Random-Key Genome GA/PSO Framework with a MILP-Certified
Reference Optimum*
**Author:** [Author name], [Roll number], [Department, Institute]
**Supervisor:** [Supervisor name]
**Stage presented at this review:** ≈ 65% complete (design frozen, core implementation validated,
preliminary benchmark vs. conventional policies and a MILP reference; full multi-seed study and
article submission scheduled before the Final Review).

Every number in this deck is a measured result from the committed code base and result CSVs
(`results/final_unified/*.csv`), not an estimate.

---

## Slide 1 — Title (120 s)

- Problem in one sentence: a 20-port, 125 kW public station is oversubscribed (100 EVs,
  1,513.6 kWh demanded); *who charges when, and at what rate, decides how many kWh the site
  actually delivers and what they cost.*
- What I built: an evaluation + benchmarking pipeline, two soft-computing schedulers (GA, PSO)
  over a compact random-key genome, and a MILP reference optimum that certifies how much is
  achievable.
- Why it matters: the default policies (FCFS, EDF) under-use the site all evening. Measured
  over a 36-h horizon they deliver **74.9% / 69.5%** of the energy demanded; the scheduler
  I propose delivers **82.9%** of demand, i.e. **97% of the best schedule that exists**.

**Talking points**
- State the deliverable status plainly: rubric-wise I am exactly where the 50–70% guideline
  expects — methodology and implementation are validated, experiments are preliminary, article
  drafting has started.
- Name the three artefacts produced so far: (1) a 155-test quality gate on the whole pipeline,
  (2) a benchmark suite of 9 scheduling policies on one shared evaluator, (3) a MILP
  ε-constraint frontier that prices every service level.

---

## Slide 2 — Problem statement, objectives & formulation

- **Inputs:** per-EV `arrival, departure, rated_power, required_energy` for 100 sessions;
  15-min slot grid (144 slots = 36 h); tariff ToU curve (off-peak 4.0, mid [6.0], peak 10.0
  units/kWh); horizon PV profile 566.5 kWh.
- **Constraints:** per-EV charging only inside its connection window; per-EV rate ≤ rated power;
  site-side **20 simultaneous ports**; site-level grid cap `P_ev ≤ 125 kW + P_solar(t)`;
  per-EV energy in ≤ required (no over-charge).
- **Objectives (weighted sum, exact weights in `config.py`):**
  `unmet 0.55 · peak 0.08 · cost 0.30 (cost per delivered kWh) · battery 0.02 · solar 0.05`.
  Service still leads, but cost is measured as an *intensity* so a cheap-feeling schedule
  cannot hide a bad one.
- **Formally:** mixed-integer non-convex scheduling; decomposed into (a) an exactly evaluated
  discrete objective for the metaheuristics and (b) a MILP relaxation used only as a
  *reference optimum*.

**Method sentence for viva:** "It is a bi-level design — the metaheuristic searches schedules,
the MILP prices optimality; both consume the same constraint model."

---

## Slide 3 — Literature survey & refinement of the research gap (Rubric: 3)

Clusters surveyed (12+ primary references — full list in the report, §References):

| Cluster | Representative works | What they give us |
|---|---|---|
| Coordinated/smart charging | Sortomme et al. (2011); Mohsenian-Rad & Leon-Garcia (2010) | ToU-aware charging formulations; Why naive FCFS is expensive at scale |
| EV charging datasets | Lee et al., ACN-Data (arXiv:1905.04817) | Realistic session statistics my synthetic scenario is calibrated to; port-utilisation regime |
| GA for scheduling | Holland (1975); Goldberg (1989); Deb et al. NSGA-II (2002) | Representation theory — the key argument for a random-key genome over a priority matrix |
| PSO & multi-objective PSO | Kennedy & Eberhart (1995); Coello et al. (2004) | Velocity semantics on discrete-vs-continuous codes; why my PSO operates on the same genome |
| MBO vs MILP comparison | Wolpert & Macready (1997) no-free-lunch; classical MILP literature | The legitimacy of *certified bounds* when a metaheuristic claims near-optimality |

**Refined research gap (the refinement claim this review is about):**

1. Public *multi-port* charging under a simultaneity cap (20 ports) is under-modelled:
   most smart-charging work constrains the total power only, which silently admits schedules
   no physical charger can host.
2. Metaheuristic scheduling literature rarely validates against a **certified optimum**, so
   "we improved 12%" is reported with no idea whether 30% was available. My design makes the
   achieved/(achievable) ratio the headline metric.
3. Genomes sized by *(EV × slot)* (the common encoding, `n_ev × n_slots = 14,400` variables)
   scale quadratically and starve the budget: with 560 objective calls the search cannot make
   one coherent move per EV. I encode the scheduler as 4 interpretable genes **per EV** (400).

**Gap statement one-liner for the panel:** "Existing soft-computing EV schedulers are either
under-constrained or unbenchmarked; I deliver both a structurally-feasible genome and a
MILP-certified yardstick."

---

## Slide 4 — Detailed methodology / system design (Rubric: 4)

Pipeline view (all stages committed and unit-tested):

1. **Data layer** (`load_scenario.py`, `create_solar_profile.py`): EV table + PV horizon →
   aligned arrays; SHA-256 provenance digests on the inputs.
2. **Constraint layer** (`multiobjective_core.evaluate_schedule` + `constraint_report`):
   single source of truth for feasibility; every policy — baselines, GA, PSO, rolling, MILP —
   is scored through the *same* evaluator, so the benchmark is apples-to-apples.
3. **Search layer** (`genome.py`, `multiobjective_optimizers.py`):
   - Genome = 4 genes/EV: `rank ∈ [0,1]`, `urgency ∈ [0,3]`, `tariff ∈ [0,3]`, `power ∈ [0,1]`.
   - Decoder: each slot scores connected EVs by
     `score = rank + urgency · deadline_pressure(t) − tariff · price(t)`,
     awards the 20 ports by score, caps at grid and remaining energy; **negative score ⇒ defer**,
     which is how "wait for the cheap hours" is encoded without a start-time gene.
   - GA: population 40, 240 generations, elitism, mutation 0.02 (tuned; 0.08 destroyed the
     elite — measured 1,174 kWh vs 1,239.5 kWh at equal budget).
   - PSO: 30 particles, 300 iterations on the same random-key genome (velocity on gene space,
     not schedule space).
   - **Memetic polish:** 4,000 hill-climb evaluations on the best genome; budgets count exactly
     — GA 12,680 and PSO 13,030 objective calls (asserted in tests).
4. **Reference layer** (`reference_optimum.py`): MILP with one binary per feasible (EV, slot)
   pair — 3,703 of 14,400 — solved by HiGHS via `scipy.optimize.milp`; ε-constraint frontier at
   service levels {960, 1080, 1,140, 1,200, 1,260 kWh} + max-service solve.
5. **Reporting layer** (`artifacts.py`): 5 figures + provenance CSVs + run manifest.

**Design justification for viva (soft-computing justification, Rubric: 2):**
- Why nature-inspired at all? The port-constrained problem is combinatorial (which EVs get the
  20 ports in each of 144 slots); exact per-slot DP explodes, and the MILP grows with the
  feasible-pair count. Metaheuristics give a near-optimal schedule in ~1 min on a laptop.
- Why *memetic*? Pure GA/PSO plateaus ~10 kWh short; the canopy-and-refine structure recovers
  a measured extra 9.3 kWh from 3,000 extra polish evaluations.
- Why *not* MILP end-to-end? Fair point for this scenario — my MILP solves in 6 s — but the
  MILP needs the port binaries per (EV, slot); at 10× sessions or with linear battery
  degradation dynamics the metaheuristic keeps its budget while the MILP becomes intractable.
  MILP is used as referee, not engine — and Referee + Engine is exactly the novelty.

---

## Slides 5–7 — Implementation progress (Rubric: 8)

Present as a three-tier build log; each tier retired a documented defect list
(TIER0/TIER1/TIER2_CHANGES.md).

**Tier 0–1 (completed, tested): evaluated everything honestly first**
- Consolidated three inconsistent cost evaluators into `evaluate_schedule` — the benchmark
  earlier priced the same schedule three different ways; now one cost definition exists.
- Repair-based feasibility (`repair_schedule`) + `constraint_report` metric adopted everywhere.
- Rolling-horizon controller (144 replans over 36 h) — both genomes and baselines.
- defect ledger closed: printing evaluator (three lines per call × 1700 lines/run), grid
  capacity semantics, solar-time-invariant scoring, cost term normalisation.

**Tier 2 (this review's work): made the metaheuristics actually optimise**
- Replaced the 14,400-gene priority-matrix genome with the 400-gene per-EV random-key genome;
  feasibility is *structural* (decoder cannot emit an infeasible schedule) instead of
  *penalised*.
- Rewrote GA and PSO on that genome; both warm-started with the same four structured genomes
  (`power / deadline / need / arrival` rankings) so no method starts ahead — measured warm-start
  service 1,165.7 / 1,130.2 / 1,102.3 / 1,161.6 kWh; the GA then found a further 73.8 kWh.
- Added the MILP reference (max-service solve attains **1,293.7 kWh**, gap ≤ 1%, 6 s) and the
  ε-frontier (five more optimal solves, ≤ 41 s each).
- 155 tests, 2 skipped (need an uncommitted raw dataset), 0 failed; the headline claim —
  *"both optimizers beat every conventional policy"* — is itself an assertion
  (`test_the_optimizers_beat_every_conventional_policy`).

**Remaining for Final Review (≈35%)**
- Five-seed replication (writing up; runs finished and archived in `multi_seed_*`).
- Article submission package; patent claims refine; present artefacts regenerated at the final
  settings for the demo.
- Known deferred items, deliberately tracked (state them, examiners reward honesty): fairness
  term absent from objective; base load not in the grid cap; solar term momentarily inert;
  ε-frontier interpolation is a chord at intermediate service levels.

---

## Slide 8 — Dataset / data preparation (Rubric: 3)

- **Source:** ACN-Data (Caltech ACN public charger network) session statistics — the raw CSV is
  not committed (licensing); the derived `data/processed/ev_scenario.csv` is, and its digest
  (`658fcc66a11fcef0…`) is recorded in every result row.
- **Prepared scenario:** 100 sessions straddling a 36-h horizon in 144 slots of 15 min;
  rated powers per session; total demanded 1,513.6 kWh; site: 20 ports, 125 kW grid cap.
- **Solar:** `solar_profile_36h.csv` (566.5 kWh over the horizon; 43.8 kWh coincides with a
  connected EV — measured, and the reason the solar objective weight is a small constant here).
- **Precursor QC, committed as validation scripts** (this is the "data preparation" evidence):
  `validate_session_data.py` (window/energy sanity, pre-arrival refusal), `validate_scenarios.py`
  (constraint feasibility of the scenario itself), `validate_baselines.py` (baselines under the
  same evaluator).
- **Reproducibility:** seeds 42–46 fixed for the multi-seed study; input digests included in all
  committed result CSVs — a stale reference optimum cannot silently price a regenerated scenario.

**One-line viva answer:** "The dataset is an ACN-Data-calibrated scenario with 100 real-session
statistics; every artifact row carries the SHA-256 of the exact inputs that produced it."

---

## Slide 9 — Preliminary results & analysis (Rubric: 4)

Benchmark: 9 policies, one evaluator, one scenario, seed 42 budgets 12,680/13,030 calls.
All numbers from `results/final_unified/comparison_metrics.csv` / `service_vs_optimal.csv`.

| Policy | Delivered (kWh) | % of demand | % of best known | Cost/kWh | Peak (kW) | Feasible |
|---|---|---|---|---|---|---|
| **PSO (proposed)** | **1,254.7** | **82.89** | **97.0** | 6.332 | 101.5 | ✔ |
| **GA (proposed)** | 1,239.5 | 81.89 | 95.8 | 6.348 | 100.2 | ✔ |
| EDF chronological | 1,133.7 | 74.91 | 87.6 | 6.374 | 96.9 | ✔ |
| FCFS chronological | 1,052.0 | 69.50 | 81.3 | 6.533 | 98.9 | ✔ |
| Uncontrolled (reference) | 1,511.3 | 99.8 | — | 8.368 | 305.2 | ✘ (52 violations) |
| MILP max-service (referee) | 1,293.7 | 85.47 | 100 | 6.377 | 109.2 | ✔ |

Reading order for the panel:
1. **The claim:** +10.7% energy over the best conventional policy at equal feasibility, and
   equal-service cost still **+7.2% more efficient than EDF's +16.4% gap** to the frontier —
   i.e. PSO halves EDF's distance to the cheapest possible schedule for the same service.
2. **Why:** conventional policies idle 20–30 kW of the site all evening; the optimizers run
   ~100 kW against a 125 kW cap (load-profile figure `comparison_load_profiles.png`).
3. **Where it is limited (say this before they ask):** completed-EV count — EDF completes 61
   EVs, PSO only 25 — total-energy maximisation systematically dis-favours low-power EVs;
   fairness-weighted objective is a documented future item.
4. **Preliminary sensitivity:** mutation-rate ablation (0.08 → 0.02 at fixed budget) alone worth
   +65.5 kWh; polish budget 1,000 → 4,000 adds +9.3 kWh. Both recorded in the ablation table.

Presented figures (already in `results/final_unified/`): `service_vs_methods.png`,
`cost_vs_service.png`, `comparison_load_profiles.png`, `convergence_comparison.png`,
`multi_seed_spread.png`.

---

## Slide 10 — Article / patent / startup development (Rubric: 4)

**Article** — draft exists (`academic/RESEARCH_ARTICLE_DRAFT.md`):
- Working title: *"A Random-Key Genome Soft-Computing Framework for Port-Constrained
  EV Charging Scheduling with MILP-Certified Optimality Bounds"*
- Target journals (shortlisted, fit-for-timeline): **IEEE Access** (fast review, strong applied
  prosumer/entity fit) → fallback **IET Smart Grid** / **Energies (MDPI)**; conference track
  **ACM e-Energy** as a backup.
- Package planned: methodology + benchmark + certified-gap metric + reproduction seeds.

**Patent/IPR fragment** (canvas filed in the report, not yet drafted fully):
- Novel core: the **decoder architecture** — port allocation driven by an interpretable
  per-EV gene vector with a sign-rule deferral (negative score ⇒ defer), making feasibility
  structural rather than penalty-based; plus the **certified-gap benchmark harness**.
- Prior-art scan (informal, to be formalised with a search report before filing): gap found in
  combining (a) random-key, per-vehicle genome + (b) feasibility-by-construction decoder +
  (c) MILP ε-constraint referee in one pipeline for multi-port sites.

**Startup/industry path (proposal-level):**
- Problem: departure-uncertain, port-limited public charging delivers avoidable under-service;
  site operators over-provision hardware before trying optimisation.
- Product: a scheduling module shipped as a service (station HID/API) that references a
  station-owner-tunable objective (energy vs cost vs fairness).
- USP: **optimality transparency** — every produced schedule is reported with its distance to
  the certified optimum, not just "better than FCFS".
- Target users: fleet depots, workplace campuses, highway plazas; revenue via SaaS subscription;
  go-to-market: co-pilot plugin to existing CSMS vendors.

---

## Slide 11 — Why Soft Computing, justified (Rubric: 2)

- The combinatorial core — assign 20 ports per slot under deadlines — is NP-hard in spirit;
  an exact model would grow binarily with each feasible (EV, slot) pair, exactly what my MILP
  needs 3,703 binaries to express; scaling sessions 5–10× makes the MILP intractable while the
  metaheuristic's cost is linear in the evaluation budget.
- **Structure exploitation:** the genome is *interpretable* (4 genes/EV with semantic meaning:
  priority, urgency, price-sensitivity, aggressiveness) — this is what makes memetic operators
  and ablation studies meaningful rather than black-box tuning.
- **Feasibility-by-construction decoder** is the crack design decision that makes GA/PSO
  practical here: no penalty landscape, no repair distortion, every evaluation counts.
- Memetic hybridisation belongs to the mature soft-computing tradition (local-improvement
  + global search) and is measured, not assumed: +9.3 kWh attributable to the polish alone.
- Soft-computing legitimacy check: I'm not claiming GA > MILP, I'm claiming *metaheuristic with
  certified near-optimality* (97.0% PSO), which is the correct scientific position and defensible
  in the viva.

---

## Slide 12 — Timeline to the Final Review

| Week | Milestone |
|---|---|
| 1 | Five-seed full-budget replication written up (`multi_seed_*`); article results section frozen |
| 2 | Article submission package (IEEE Access template); patent claims draft circulated to supervisor |
| 3 | Deferred items triaged: pick 1 (fairness-weighted objective) as stretch ablation only if time allows; else document as future scope |
| 4 | Demo-ready artefacts regenerated; final report + review ppt locked; viva rehearsals |

---

## Viva talking points (save for examiners' questions)

- **"Why not just MILP?"** — This exact scenario *is* MILP-tractable (6 s), and I say so; the
  argument is scaling (feasible-pair count grows with connected EVs per slot) and extensibility
  (degradation dynamics, forecast error, asymmetric tariffs). MILP as *referee* gives my
  metaheuristic a credibility most soft-computing papers lack.
- **"How do you know the result isn't a lucky seed?"** — five seeds, mean 1,253.1 ± 7.3 kWh
  (PSO); worst run still beats EDF by +105.8 kWh; spread ±0.6%, gap to EDF +10.7%.
- **"Is the comparison fair — do the baselines get the same constraints?"** — yes, single
  evaluator `evaluate_schedule` + identical `constraint_report`; every schedule is validated for
  the same 52-constraint envelope. Uncontrolled is present as a marked *infeasible* reference
  row, not hidden.
- **"Why do optimizers finish fewer EVs?"** — I published this trade-off myself (report §):
  maximising total kWh structurally prefers high-power EVs; the fix (fairness floor) is
  a documented design direction, not an oversight.
- **"What's the intellectual core?"** — the sign-rule decoder + certified reference gap metric;
  both are the article's claimed novelty and the patent's inventive step.

## Anticipated Q&A bank (short answers)

1. *Why random-key instead of binary/integer genes?* — Keeps PSO continuous; one decoder serves
   both GA and PSO; genes map to interpretable EV-level levers.
2. *What exactly is "MILP-certified optimality"?* — an independently solved MILP on the same
   constraint model with proven optimality (gap ≤ 0.8%) is used as the denominator of my
   achieved/(achievable) ratio.
3. *How is the objective scalarised and why defensible?* — convex weighted sum; weights are
   stated, bounded and ablated (0.55/0.30/0.08/0.02/0.05); no scalar hides a dead term because
   each is measured as a standalone column in the results CSVs.
4. *Convergence evidence?* — `convergence_comparison.png` — best-so-far objective per
   generation/iteration, log scale; runs plateau, informing the polish budget.
5. *Is the solar term doing anything?* — honest answer: at this scenario only 43.8 of 566.5 kWh
   of PV coincides with charging demand, so it scores as nearly constant; I documented this; on a
   solar-coincident profile it becomes load-bearing.
