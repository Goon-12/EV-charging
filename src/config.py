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
GA_SEED = 46
PSO_SEED = 42
ROLLING_SEED = 46

GA_POPULATION_SIZE = 20
GA_GENERATIONS = 30
PSO_PARTICLES = 12
PSO_ITERATIONS = 15

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
# NOTE (Tier 0): this remains a throughput-dominated weighting. Tier 1
# re-normalises the cost and peak terms and re-tunes these weights so that
# cost and peak are not drowned out by the unmet-energy term.
OBJECTIVE_WEIGHTS = {
    "unmet": 0.80,
    "peak": 0.07,
    "cost": 0.06,
    "battery": 0.02,
    "solar": 0.05,
}

# Feasibility must dominate the score: an infeasible schedule must never
# look better than a feasible one just because it moved more energy.
INFEASIBILITY_PENALTY = 10.0

# Tolerance (kWh) below which an EV counts as fully charged.
FULLY_CHARGED_TOLERANCE_KWH = 1e-3
