"""
Single source of truth for the EV charging optimization prototype.

Why this module exists
----------------------
Before Tier 0 the same constants (15-minute step, 144 slots, 20 chargers,
125 kW grid limit, tariff boundaries, seeds, objective weights) were
copy-pasted into roughly twenty scripts. Two scripts could therefore
disagree about the scenario they were comparing, which silently invalidated
cross-file comparisons. Every module must now import its constants from here.

Conventions
-----------
* Slot ``t`` covers the half-open interval ``[t*STEP_HOURS, (t+1)*STEP_HOURS)``.
* An EV is available when ``arrival_slot <= t < departure_slot`` (exclusive end).
* Tariff values are synthetic illustrative units, NOT real electricity prices.
* All paths are absolute so scripts can run from any working directory.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"

DEFAULT_SCENARIO = DATA_DIR / "ev_scenario.csv"
DEFAULT_SOLAR = DATA_DIR / "solar_profile_36h.csv"
CLEAN_SESSIONS = DATA_DIR / "clean_office_sessions.csv"

# --------------------------------------------------------------------------
# Time discretization
# --------------------------------------------------------------------------
STEP_HOURS = 0.25
HORIZON_HOURS = 36
N_SLOTS = int(round(HORIZON_HOURS / STEP_HOURS))  # 144

# --------------------------------------------------------------------------
# Site limits
# --------------------------------------------------------------------------
N_CHARGERS = 20
MAX_GRID_KW = 125.0

# --------------------------------------------------------------------------
# Illustrative tariff (synthetic units per kWh)
# --------------------------------------------------------------------------
TARIFF_OFFPEAK = 4.0   # 00:00 - 06:00
TARIFF_MID = 6.0       # 06:00 - 17:00 and 22:00 - 24:00
TARIFF_PEAK = 10.0     # 17:00 - 22:00
OFFPEAK_END_HOUR = 6.0
PEAK_START_HOUR = 17.0
PEAK_END_HOUR = 22.0
MAX_TARIFF = TARIFF_PEAK

# --------------------------------------------------------------------------
# Scenario generation (data/processed/ev_scenario.csv)
# --------------------------------------------------------------------------
SCENARIO_SEED = 42
N_EV = 100
ENERGY_MIN_KWH = 2.0
ENERGY_MAX_KWH = 30.0
POWER_MIN_KW = 1.0
POWER_MAX_KW = 7.0
ARRIVAL_HOUR_MIN = 17.0
ARRIVAL_HOUR_MAX = 21.0
STAY_HOURS_MIN = 6.0
STAY_HOURS_MAX = 12.0
SOLAR_PEAK_KW = 50.0

# --------------------------------------------------------------------------
# Stochastic optimizers
# --------------------------------------------------------------------------
# Tier 2 budgets. The genome is 4 genes per EV (400 for this scenario), so the
# search needs 10^4 evaluations rather than the 560 of Tier 0/1:
#
#   GA  : 40 + 240 * (40 - 4)  = 8,680 + 4,000 polish = 12,680
#   PSO : 30 * (1 + 300)       = 9,030 + 4,000 polish = 13,030
#
# Both counts are asserted against the optimizers' real behaviour in
# tests/test_optimizers.py, so the reported budget cannot silently drift.
GA_SEED = 46
PSO_SEED = 42
ROLLING_SEED = 46

GA_POPULATION_SIZE = 40
GA_GENERATIONS = 240
# Mutation and elitism were tuned against the full budget, not guessed: a 400-gene
# genome needs a *low* per-gene mutation rate. At 0.08 (the Tier 0/1 value) some
# 32 genes of every child are re-drawn, which destroys the structure that makes a
# genome good - that run delivered 1,165 kWh against 1,240 kWh at 0.02 with the
# same budget.
GA_ELITE_COUNT = 4
GA_MUTATION_RATE = 0.02
GA_POLISH_EVALUATIONS = 4_000

PSO_PARTICLES = 30
PSO_ITERATIONS = 300
PSO_POLISH_EVALUATIONS = 4_000

ROLLING_WINDOW_SLOTS = 24
ROLLING_GA_POPULATION_SIZE = 8
ROLLING_GA_GENERATIONS = 5
ROLLING_PSO_PARTICLES = 6
ROLLING_PSO_ITERATIONS = 4

SEEDS = (42, 43, 44, 45, 46)

# --------------------------------------------------------------------------
# Objective weights
# --------------------------------------------------------------------------
# J = w_unmet * unmet_n + w_peak * peak_n + w_cost * cost_n
#     + w_battery * battery_n - w_solar * solar_n
#
# Tier 2 (was Tier 0's 0.80 / 0.07 / 0.06 / 0.02 / 0.05): the cost term is now
# cost per delivered kWh scaled by the peak tariff, so it is no longer pinned
# into 0.47-1.0 by the size of the schedule, and it carries a weight that the
# optimizer can actually act on. Service still leads, because a schedule that
# serves less is not a schedule.
#
# The battery and solar terms stay small and are effectively inert in this
# scenario (battery stress is ~0.98 for every method, and only 43.8 of 566.5 kWh
# of PV coincides with a connected EV). They are kept for continuity with the
# model description, not because they discriminate between methods.
OBJECTIVE_WEIGHTS = {
    "unmet": 0.55,
    "peak": 0.08,
    "cost": 0.30,
    "battery": 0.02,
    "solar": 0.05,
}

# Feasibility must dominate the score: an infeasible schedule must never
# look better than a feasible one just because it moved more energy.
INFEASIBILITY_PENALTY = 10.0

# Tolerance (kWh) below which an EV counts as fully charged.
FULLY_CHARGED_TOLERANCE_KWH = 1e-3

# --------------------------------------------------------------------------
# Reference optima (src/reference_optimum.py)
# --------------------------------------------------------------------------
# The MILP that bounds what any scheduler can achieve. It is solved once per
# artifact set, not per experiment, so the time limit is generous but finite:
# a reported optimum is always the best feasible solution *found within the
# limit*, together with the solver's own MIP gap.
REFERENCE_TIME_LIMIT_S = 90.0
REFERENCE_MIP_GAP = 0.01
# Service levels (kWh delivered) at which the minimum-cost reference is solved.
# Levels above the max-service optimum are skipped at run time, so this is an
# upper envelope rather than a set of hard requests.
REFERENCE_SERVICE_LEVELS_KWH = (960.0, 1080.0, 1140.0, 1200.0, 1260.0)
