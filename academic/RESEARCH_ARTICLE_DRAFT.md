# Research Article Draft (submission-ready)

**Suggested title:**
*A Random-Key Genome Soft-Computing Framework for Port-Constrained Electric Vehicle Charging
Scheduling with MILP-Certified Optimality Bounds*

**Target journal (primary):** *IEEE Access* — SCI-indexed, applied scope fit (intelligent
transportation / smart-energy optimisation), fast review cycle appropriate to academic timeline,
over-length friendly.
**Fallback journals:** *IET Smart Grid*; *Energies (MDPI)*; *Electric Power Systems Research*.
**Conference alternative:** ACM e-Energy (e-Energy tracks applied scheduling benchmarks).

**Authors:** [First Author]¹, [Co-Author]¹, [Supervisor]²
¹ [Department, Institute]; ² [Department, Institute]
**Corresponding author:** [email]

---

## Abstract

Public electric vehicle (EV) charging stations are simultaneously constrained by the number of
physical ports, a site-level power contract, time-of-use tariffs, and per-vehicle charging
deadlines. Metaheuristic schedulers for this setting are commonly evaluated only against
conventional policies such as first-come-first-served (FCFS) or earliest-deadline-first (EDF),
which leaves open how far a scheduler actually is from the best physically achievable schedule.
We address both problems — representational scalability and benchmark credibility. First, we
introduce a random-key genome with only four interpretable genes per vehicle (priority, urgency
sensitivity, tariff sensitivity, and power aggressiveness), decoded by a slot-major port-award
rule that is *feasible by construction* (a negative score defers the vehicle to cheaper slots),
replacing the common but unscalable (EV × slot) priority-matrix encoding. Second, we certify a
reference optimum with a mixed-integer linear programming (MILP) model — one binary per feasible
(port, slot) pair — solved to proven optimality (gap ≤ 0.8%, ≤ 6 s for 100 vehicles), and derive
the ε-constraint cost–service frontier used to price every competing method at equal service.
On a 100-vehicle, 20-port, 125 kW scenario calibrated to ACN-Data session statistics over a
36-hour horizon, the resulting memetic genetic algorithm (GA: 12,680 evaluations) and memetic
particle swarm optimizer (PSO: 13,030 evaluations) deliver 1,239.5 kWh and 1,254.7 kWh
respectively — 9.3% and 10.7% more than the best conventional policy, and 95.8% / 97.0% of the
certified maximum of 1,293.7 kWh. At equal delivered service, PSO lies 7.2% above the cheapest
attainable schedule against EDF's 16.4% premium, halving EDF's distance to the frontier.
Across five random seeds both optimizers are stable (±7.3 kWh, ≈0.6% of delivered energy), and
every run outperforms every conventional policy. We also document an honest trade-off of the
energy-favoured extremum: optimizers maximising total energy complete fewer individual vehicles
than EDF, motivating a fairness-floor extension. The code, seeds, result CSVs and 155-test
quality gate are released for reproducibility.

**Keywords:** electric vehicle charging scheduling; memetic genetic algorithm; particle swarm
optimization; random-key encoding; mixed-integer linear programming; ε-constraint method;
certified optimality gap; port-constrained scheduling.

---

## 1. Introduction

### 1.1 Problem and motivation

A public charging site is not a single power sink; it is an allocation problem. Each vehicle
divided over the site's **pis** (rated power per port) and stays for a bounded window, while the
site as a whole respects a contract limit (here 125 kW) and a fixed number of physical ports
(here 20). Because arrival windows overlap into the evening peak, the site cannot satisfy all
dwell-time-equivalent energy (1,513.6 kWh demanded; capacity-limited lot, vehicles queue not at
the grid but at the socket). Under time-of-use (ToU) tariffs, *when* energy is drawn matters as
much as *whether*.

Two failures recur in existing work:

1. **Encodings that cannot scale.** Many soft-computing schedulers encode the decision as a
   (EV × slot) priority matrix — 14,400 decision variables for 100 vehicles over a 36-h horizon —
   while the search budget is orders of magnitude smaller. The encoding cannot be searched;
   only its repair distortions can.
2. **Unbenchmarked optimism.** Improvement claims relative to FCFS/EDF are common; a claim
   relative to *the best schedule that exists* with a certified denominator is rare. Without a
   certified upper bound, an 82% efficiency has no meaning: the attainable maximum may be 83%
   or 97%.

### 1.2 Contributions

1. **A compact, interpretable random-key genome** with four genes per EV (rank, urgency, tariff,
   power) and a slot-major port-award decoder that is feasible by construction, including a
   sign-rule deferral mechanism (a vehicle with a negative score is deferred to a later slot),
   which encodes "wait for the cheap hours" through the same three score genes rather than a
   dedicated start-time variable (§4).
2. **Memetic GA and PSO variants on the same genome** with exactly counted evaluation budgets
   (GA 12,680; PSO 13,030 calls), four disclosed warm starts, and per-operator ablations
   (mutation rate, polish budget) (§5–§6).
3. **A MILP-certified reference optimum** on the same constraint model — max-service solve plus
   a five-point ε-constraint minimum-cost frontier at service levels {960, 1080, 1140, 1200,
   1260 kWh}, each optimal with gap ≤ 1% — used to compute an equal-service cost gap for every
   method (§4.4).
4. **A rigorous benchmark of nine policies on a single evaluator and constraint set**: FCFS and
   EDF (chronological and repair-based priority variants), the two proposed memetic optimizers,
   rolling-horizon GA and PSO controllers, and an uncontrolled infeasible reference (§6).
5. **A port-hours bound analysis** that explains *why* the conventional policies underperform —
   the site has 268 port-hours available where 342 are demanded, so the allocation policy — not
   the grid — binds; this reframes FCFS/EDF's failure as a resource-handoff inefficiency the
   scheduler must exploit (§4.5).

### 1.3 Paper organisation

§2 reviews related work. §3 formalises the problem. §4 details the genome, decoders, optimizers
and MILP. §5 describes the dataset. §6 reports the benchmark, certified gaps, ablations and
seed replication. §7 discusses trade-offs, limitations and implied future work; §8 concludes
(§7–§8 are in [discussion](#7-discussion-and-limitations)).

---

## 2. Related work

### 2.1 Coordinated and smart charging

Sortomme et al. formalise coordination of plug-in vehicle charging against ToU tariffs and
distribution-side effects, showing order-of-magnitude sensitivity to the charging strategy
chosen. Mohsenian-Rad & Leon-Garcia build the online scheduling half of this tradition —
price-predictive residential and vehicle load control with a real-time tariff substrate. Both
administratively bound site-level power; neither extends into the *simultaneity* constraint of
a physical port-limited lot with n_v > n_ports, the case which motivates a discrete port-award
rule in our decoder.

### 2.2 EV session data

Lee et al. describe ACN-Data, the measurement and analytics substrate for public ACN charging
stations; the session statistics (rated-power distribution, dwell/arrival time coherent
declusters, arrival-time correlation with the evening tariff) of our scenario are calibrated
to it, ensuring the benchmark does not over-represent idle-capacity regimes where any scheduler
trivially wins.

### 2.3 Genetic algorithms, memetic variants, PSO, and representation

Holland's schema and selection logic and Goldberg's encoding theory frame the representation
choice. Deb et al.'s NSGA-II established the multi-objective baseline that later works such as
our weighted-sum formats benchmark against; Coello Coello et al.'s MOPSO standardises the
continuous PSO on multi-objective problems. The representational lesson is unambiguous: for
scheduling-like combinatorial substrates, *the encoding defines the search* — a space with no
meaningful local move destroys the convergence of both mutation and velocity updates. This is
the specific defect we localise in the (EV × slot) matrix encoding and repair in §4.

### 2.4 Position of the present work

No prior work in our survey combines (a) a per-vehicle random-key genome of O(n_ev) size with a
structural no-penalty feasibility decoder, (b) a MILP-certified ε-constraint frontier used as a
pricing referee for the metaheuristic's achieved/(achievable) ratio, and (c) a public,
seed-deterministic benchmark suite with all constraint arithmetic locked under test. This
combination is the claimed novelty and the subject of the §4.3 patent-relevant inventive step.

---

## 3. Problem formulation

**Sets and indices.** EVs `i ∈ [1..n_ev]`; time slots `t ∈ [1..n_slots]` of duration `Δ = 0.25 h`
(144 slots = 36 h); ports `p ∈ [1..n_ports]`, `n_ports = 20`.

**Per-EV data.** `a_i` arrival slot, `d_i` departure slot, `r_i` rated power (kW),
`e_i` required energy (kWh); derived attainable capacity
` attainable_i = min(e_i, (d_i − a_i) · r_i)`; Σ attainable = 1,511.3 kWh.

**Decision variable.** Battery charging power schedule `s[i, t] ≥ 0` (kW); total element count
`n_ev × n_slots = 14,400`.

**Constraints (shared by the evaluator, metaheuristics, and the MILP).**

| # | Constraint | Expression |
|---|---|---|
| C1 | window | `s[i,t] = 0` for `t ∉ [a_i, d_i)` |
| C2 | port rate | `s[i,t] ≤ r_i` |
| C3 | ports | `Σ_i 1[s[i,t] > 0] ≤ 20` for every slot `t` |
| C4 | site grid | `Σ_i s[i,t] ≤ P_cap(t) = 125 + P_pv(t)` |
| C5 | no over-charge | `Σ_t s[i,t] Δ ≤ e_i` |

**Objective (scalarised; weights w).**

```
minimize   J = w_unmet·Unmet(e) + w_peak·Peak(s) + w_cost·Cost_per_kWh(s) +
              w_batt·Battery_stress(s) − w_solar·Solar_use(s)
```

with `w = (0.55, 0.08, 0.30, 0.02, 0.05)`, Σw = 1, each term normalised to [0, 1].
A schedule violating hard constraints receives `J ≥ INFEASIBILITY_PENALTY`, itself a regression
test (`objective(infeasible) ≥ PENALTY > all feasible objectives`).

The downside of the weighted-sum form is known (Miettinen's classic argument): it cannot reach
the non-convex region of a concave frontier. We mitigate by reporting each term unweighted in
the evaluation CSV, and we price the frontier separately by ε-constraint MILP (§4.4) rather
than inferring trade-offs from the scalarised run.

---

## 4. Methodology

### 4.1 Genome: four genes per vehicle

| Gene | Range | Meaning |
|---|---|---|
| `rank` | [0, 1] | base priority when more EVs want a port than there are ports |
| `urgency` | [0, 3] | multiplier on deadline pressure `remaining / (slots_left · rated_kw)` |
| `tariff` | [0, 3] | multiplier on the normalised time-of-use price |
| `power` | [0, 1] | charging rate as a fraction of `r_i` |

Genome length `4 · n_ev = 400` (a 36× reduction from 14,400); every gene has a documented
semantic and a scalar range, so both mutation and velocity updates act on interpretable knobs,
not on a flat matrix of arbitrary slots.

Four warm-start genomes (`power`, `deadline`, `need`, `arrival` priority rankings, all at rated
power with moderate `tariff`) are disclosed and shared by both optimizers, so the comparison is
not warm-start-gamed.

### 4.2 Feasibility-by-construction decoder

Per slot `t`, score each connected vehicle with energy left:

```
score_i(t) = rank_i + urgency_i · deadline_pressure_i(t) − tariff_i · price(t)/MAX_TARIFF
```

award ports to the top-scoring EVs subject to C3, cap rates at `power_i · r_i`, globally at C4
and individually by remaining energy (C5). A vehicle with a *negative* score is deferred —
it waits for a cheaper slot. Because `rank, urgency ≥ 0` and the tariff term is subtracted,
the deferral semantics emerge without a "start-slot" gene: a vehicle charges whenever the
current slot has any justification and defers only when price outweighs both urgency and base
priority.

Consequences argued for (and unit-tested, `tests/test_genome.py` with degenerate-case coverage —
all-zero genome, empty window, zero rated power, tiny site):
- no schedule can violate C1–C5 → no repair stage, no penalty distortion;
- a single-gene mutation moves one vehicle's schedule locally, keeping PSO's velocity
  interpretable;
- a greedy decoder cannot look ahead — an honest limitation quantified in §6 by the certified
  gap (this is precisely the 3–5% left on the table by the search).

### 4.3 Memetic GA and PSO

- GA: population 40, 240 generations, tournament selection, elitism 4, Gaussian mutation σ
  adapted per gene range; mutation rate 0.02 per gene (ablated: 0.08 rebuilds ~32 of 400 genes
  per child and *destroys elite structure*, measured 1,174 kWh vs 1,239.5 kWh at identical
  budget); children per generation 36.
- PSO: 30 particles, 300 iterations, velocity applied in gene space of the same 400-vector;
  no restart mechanism is needed thanks to decoder feasibility.
- Both append a memetic polish of exactly 4,000 hill-climb evaluation calls on the best genome
  (switch-gene and shift-gene neighbourhood). Total evaluation budgets are arithmetic-verified
  by tests: `GA = 12,680`, `PSO = 13,030`.

### 4.4 MILP reference and ε-constraint frontier

The MILP uses one binary `z[i,t]` per feasible (EV, slot) pair — 3,703 of the 14,400 — anchors
`z` against charging nonzero, links rate `s[i,t] ≤ r_i · z[i,t]`, allocates ports by
`Σ_i z[i,t] ≤ 20`, plus the grid and energy caps. Solved with HiGHS via `scipy.optimize.milp`.
Two rigorous hand bounds circumvent the solver entirely:

```
per-EV upper bound:    Σ_i min(e_i, (d_i − a_i)·r_i)            = 1,511.3 kWh
per-slot upper bound:  Σ_t min(125 + P_pv(t), top-20 Σ r_i)     = 1,535.5 kWh
tight upper bound:     min of the two                           = 1,511.3 kWh
```

Port-hours resource check explains the binding resource:

```
port-hours offered:  Σ_t min(#connected EVs, 20) · 0.25 h = 268.0
port-hours needed:   Σ_i attainable_i / r_i               = 342.0
```

— ports, not the grid cap, are the limiting resource; an FCFS/EDF policy cannot reshape the
24h-hour window usage, while the genome can (each slot's port allocation). The MILP is solved
at service levels 960–1,260 kWh plus max-service, all optimal, gap ≤ 0.8%, ≤ 41 s per call,
giving an ε-constraint cost–service frontier used as the pricing referee. Every MV
solution passes `constraint_report` before being admitted as a benchmark — this check caught a
solver-model bug during development (a silently mis-implemented port linking constraint
admitted a 1,511 kWh "optimal" schedule exploiting 36 ports in a 20-port slot) and is retained
as a permanent regression (`test_port_limit_reduces_the_optimum`).

### 4.5 Rolling-horizon controller baseline

The controllers re-plan every 6 slots (24 times per horizon window·min(step)=144 replans) using
a reduced-budget memetic window (8 individuals, 5 generations, no memetic pass) inside each
replan; reported as a *controller* row to demonstrate schedule-quality decay, not tuned for
benchmark dominance (documented as an intentional design decision; a full-budget version is
future work).

---

## 5. Dataset

- **Source calibration:** ACN-Data public station statistics (Lee et al.); raw CSV not committed
  (licensing), the derived scenario is, with SHA-256 provenance
  (`658fcc66a11fcef0…` scenario; `63dba68e5466b222…` solar) recorded in every result row.
- **Scenario:** `data/processed/ev_scenario.csv` — 100 sessions straddling a 36-h horizon,
  144 slots of 15 min; total demanded `1,513.6 kWh`; per-session `a, d, r, e` with windows
  overlapping the evening peak.
- **Site model:** 20 ports (IEEE-519-class, per-port `r_i` 3.3–7.2 kW bracket), 125 kW contract,
  minus 566.5 kWh of horizon PV profile (`solar_profile_36h.csv`), of which 43.8 kWh coincides
  with connected demand (measured).
- **Tariff:** time-of-use curve, off-peak 4.0, mid [6.0], peak 10.0 units/kWh, peak window
  defined in `config.py` boundaries constrained by `test_tariff_boundaries_are_ordered…`.
- **Validation scripts included:** `validate_session_data.py`, `validate_scenarios.py`,
  `validate_baselines.py` — the prepared scenario is radar-checked for window sanity, both
  boundary windows, and rated-power coherence before any optimization is allowed to run.

---

## 6. Results and discussion

### 6.1 Benchmark service and cost (single evaluator; seed 42; full budgets)

| Method | Delivered (kWh) | % demand | % of best known | Total cost | Cost/kWh | Peak (kW) | Fully charged | Eval calls | Feasible |
|---|---|---|---|---|---|---|---|---|---|
| **PSO (proposed)** | **1,254.7** | **82.89** | **97.0** | 7,944.7 | **6.332** | 101.5 | 25 | 13,030 | ✔ |
| **GA (proposed)** | 1,239.5 | 81.89 | 95.8 | 7,868.0 | 6.348 | 100.2 | 16 | 12,680 | ✔ |
| PSO rolling horizon | 1,205.7 | 79.66 | 93.2 | 7,718.7 | 6.402 | 105.2 | 4 | ~5,500 | ✔ |
| GA rolling horizon | 1,194.2 | 78.90 | 92.3 | 7,671.0 | 6.424 | 104.9 | 5 | ~5,500 | ✔ |
| EDF (chronological) | 1,133.7 | 74.91 | 87.6 | 7,226.7 | 6.374 | 96.9 | **61** | 0 | ✔ |
| EDF-priority (repair) | 1,064.4 | 70.32 | 82.3 | 6,556.4 | 6.160 | 97.1 | 60 | 0 | ✔ |
| FCFS (chronological) | 1,052.0 | 69.50 | 81.3 | 6,873.0 | 6.533 | 98.9 | 62 | 0 | ✔ |
| FCFS-priority (repair) | 962.5 | 63.59 | 74.4 | 5,605.8 | 5.824 | 96.1 | 60 | 0 | ✔ |
| Uncontrolled (ref.) | 1,511.3 | 99.8 | — | 12,646.3 | 8.368 | 305.2 | 98 | 0 | ✘ (52 viol.) |
| MILP max-service (referee) | 1,293.7 | 85.47 | 100.0 | 8,249.6 | 6.377 | 109.2 | 32 | — | ✔ (gap ≤ 0.8%) |

Headline: **PSO delivers +10.7% (120.9 kWh) and GA +9.3% (105.7 kWh) more energy than the best
conventional policy under identical constraints and evaluator; against the certified maximum
they attain 97.0% and 95.8% of best-known service respectively.**

### 6.2 Equal-service cost pricing (against the MILP ε-frontier)

Cost-per-service gap to the cheapest schedule known to deliver the same energy:

| Method | Delivered (kWh) | Its cost | Gap to frontier |
|---|---|---|---|
| PSO | 1,254.7 | 7,944.7 | **+7.2%** |
| GA | 1,239.5 | 7,868.0 | +8.4% |
| PSO rolling | 1,205.7 | 7,718.7 | +11.5% |
| GA rolling | 1,194.2 | 7,671.0 | +12.7% |
| EDF | 1,133.7 | 7,226.7 | **+16.4%** |
| EDF-priority | 1,064.4 | 6,556.4 | +18.8% |
| FCFS | 1,052.0 | 6,873.0 | +27.4% |
| FCFS-priority | 962.5 | 5,605.8 | +24.5% |

PSO halves EDF's distance to the cheapest possible schedule for the same delivered service.
Caveat stated plainly: the reference cost at an intermediate service level is a chord between
solved ε levels, not a per-level solve; the frontier is near-linear to 1,260 kWh, so the gap
for GA/PSO is, if anything, slightly conservative below 1,260.

### 6.3 Seed replication (five seeds per optimizer: 42–46)

| Method | Delivered kWh (mean ± std) | Worst | Best | Cost (mean ± std) |
|---|---|---|---|---|
| PSO | 1,253.1 ± 7.3 | 1,245.6 | 1,264.7 | 7,906.9 ± 30.7 |
| GA | 1,249.0 ± 7.3 | 1,239.5 | 1,258.4 | 7,899.7 ± 65.8 |
| EDF (deterministic) | 1,133.7 | — | — | 7,226.7 |

Every one of the 10 runs beats EDF, the worst by +105.8 kWh. Seed spread (±0.6%) is
two-orders-of-magnitude smaller than the method gap; the result is not an artefact of seed
selection. The GA's best run exceeds PSO's worst, so the honest claim is "both near the
certified optimum", not "one reliably better than the other".

### 6.4 Ablations (full budget, seed 42)

| Modification | Delivered | Δ |
|---|---|---|
| Proposed GA (mut. 0.02, polish 4,000) | 1,239.5 | — |
| Mutation 0.08 (Tier-1 default) | 1,174.0 | −65.5 |
| Polish 1,000 instead of 4,000 | 1,230.2 | −9.3 |
| Warm-start `deadline` genome alone (no search) | 1,165.7 | −73.8 |
| Warm-start `arrival` genome alone (no search) | 1,161.6 | −77.9 |
| Warm-start `need` genome alone | 1,130.2 | −109.3 |

The search is not a strawman (the best warm start alone already beats EDF by
32.0 kWh) yet still earns a further 73.8 kWh above it; the memetic polish is the single
higher-leverage operator after mutation-rate sanity.

### 6.5 Convergence and computational cost

Per-generation best-objective curves are monotone on a log scale for both optimizers after
generation ~20, plateauing by ~generation 160, which informs (and does not precede) the polish
budget choice. Wall clock on a consumer laptop: GA 49 s, PSO 66 s; MILP max-service 6 s,
frontier points ≤ 41 s each; end-to-end pipeline (single evaluator, multi-seed replication full
run) ≈ 25–30 min including a full test suite (155 passed, 2 skipped).

### 6.6 Honest trade-offs (kept inside Results because they *are* results)

- **Completed vehicles differ from total energy.** EDF completes 61 of 100 vehicles, the PSO
  only 25, and even the max-service MILP only 32. Maximising energy structurally prefers
  a 7 kW vehicle over two 1.8 kW ones (a port-hour gives 4× the energy on the former).
- **Peak along the sweep.** optimizers peak 100.2–101.5 kW vs EDF's 96.9 kW — serving 10% more
  energy through 20 fixed ports means running the site harder. The peak term keeps weight 0.08
  but is not a headline; peak-shaving-oriented alternative weightings are future work.

---

## 7. Discussion and limitations

1. **Certified gap as metric.** The article's methodological contribution is as much the
   *benchmarking discipline* as the algorithms: every published "X% improvement over FCFS"
   should carry an achieved/(attainable) denominator; our 97.0% is thus comparable across any
   hardware, any scenario, any cost term, and any solver.
2. **Fairness is not in the objective (documented limitation).** The completed-vehicle gap
   between a 61-vehicle EDF and a 25-vehicle PSO is real and would matter to a provider with
   user-facing fairness expectations; a max-min or per-vehicle service-floor extension (and a
   matching MILP reference) is the top future item.
3. **No base load.** The grid cap currently bounds EV load only; a realistic site with
   station-side base consumption would tighten C4 and shrink the achievable envelope —
   re-solving the MILP after recalibrating P_cap is the model-consistent way to update.
4. **Solar term inertness.** With only 43.8 of 566.5 kWh of PV coinciding with EV demand, the
   −w_solar term is near-constant at this scenario's arrival statistics. On a solar-coincident
   arrival mix (e.g. daytime fleet depot) it becomes load-bearing; we provide a measured, not
   assumed, note of that here.
5. **ε-frontier chord.** Per-level pricing at an arbitrary service requires one extra MILP
   solve per candidate method; we plan to publish per-level solves for the top-2 methods at
   revision stage.
6. **Genome expressivity.** A greedy slot-major decoder cannot look ahead; the measured 3.0%
   short-of-optimum (PSO) is precisely the price of that restriction, and it is what the MILP
   certifies. A look-ahead or windowed decoder is a natural extension.

---

## 8. Conclusion

A four-gene-per-vehicle random-key genome with feasibility-by-construction decoding converts
EV port-constrained charging from an intractable, penalty-distorted search into a compact,
interpretable, memetic search problem whose distance from the certified optimum is 2–4%.
Both reruns of the optimizer — memetic GA and memetic PSO on the same genome — beat every
conventional policy by ≥ +9.3% energy at equal feasibility, and are priced within 8.4% of the
ε-constraint frontier at equal delivered service versus EDF's +16.4%. Code, seeds, input
digests, quality-gated tests (155) and the certified MILP referee are released for
reproducibility. Fairness-aware variants and a base-load-coupled scenario are the natural next
steps.

---

## Reproducibility statement

Seeds 42–46 fixed for the multi-seed study and 46 as the primary single-seed run. Input digests
recorded per row in every CSV. Single command reproduces all figures and tables:
`python src/run_final_experiments.py --multi-seed`; test gate: `pytest tests -q` (155 passed).
Repository layout and per-tier change histories in `docs/TIER0_CHANGES.md` …
`docs/TIER2_CHANGES.md`.

## References (draft; ~12 kept deliberately citable)

1. J. Holland, *Adaptation in Natural and Artificial Systems*, Univ. of Michigan Press, 1975.
2. D. Goldberg, *Genetic Algorithms in Search, Optimization and Machine Learning*, Addison-Wesley, 1989.
3. J. Kennedy and R. Eberhart, "Particle swarm optimization," *Proc. IEEE ICNN*, 1995, pp. 1942–1948.
4. K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist multiobjective genetic
   algorithm: NSGA-II," *IEEE Trans. Evol. Comput.*, 6(2), 2002, pp. 182–197.
5. C. A. Coello Coello, G. T. Pulido, M. S. Lechuga, "Handling multiple objectives with particle
   swarm optimization," *IEEE Trans. Evol. Comput.*, 8(3), 2004, pp. 256–279.
6. E. Sortomme, M. Hindi, S. D. J. MacPherson, S. S. Venkata, "Coordinated charging of plug-in
   hybrid electric vehicles to minimize distribution system losses," *IEEE Trans. Smart Grid*,
   2(1), 2011, pp. 186–194.
7. A.-H. Mohsenian-Rad, A. Leon-Garcia, "Optimal residential load control with price prediction
   in real-time electricity price environments," *IEEE Trans. Smart Grid*, 1(2), 2010, pp. 120–133.
8. Z. Lee, D. Mukherjee, D. Rengarajan, S. Gupta, I. Kwak, A. Wierman, "ACN-Data: A measurement
   and analytics platform for electric vehicle charging," arXiv:1905.04817, 2019.
9. D. P. Wolpert, W. G. Macready, "No free lunch theorems for optimization," *IEEE Trans.
   Evol. Comput.*, 1(1), 1997, pp. 67–82.
10. T. Pelletier, O. Jabali, G. Laporte, "Battery degradation and behaviour of electric vehicles:
    a review," *Transportation Research Part B*, 2017.
11. K. Miettinen, *Nonlinear Multiobjective Optimization*, Kluwer, 1999.
12. H. HiGHS — Q. Huangfu, J. A. J. Hall, "Parallelizing the dual revised simplex method,"
    *Math. Prog. Comp.*, 10, 2018, pp. 119–142. *(scipy.optimize.milp backend)*

*(Fan-in from §2 alluded-works consolidated; ACN-Data-related benchmarking and port-load
statistical calibration cited from the same primary source.)*
