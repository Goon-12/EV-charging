from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))

from load_scenario import load_scenario

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
GRID_IMPORT_LIMIT_KW = 125.0
TOL = 1e-9
FULL_CHARGE_TOL_KWH = 0.05


def load_solar():
    path = DATA / "solar_profile_36h.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Solar profile not found: {path}. Run src/create_solar_profile.py first."
        )
    df = pd.read_csv(path)
    if "solar_power_kw" not in df.columns or len(df) != N_SLOTS:
        raise ValueError(
            "Expected solar_profile_36h.csv to contain 144 rows and a "
            "'solar_power_kw' column. Check the column names and time resolution."
        )
    solar = df["solar_power_kw"].to_numpy(dtype=float)
    if not np.isfinite(solar).all() or np.any(solar < 0):
        raise ValueError("Solar profile contains invalid values.")
    return solar


def main():
    scenario = load_scenario()
    n_ev = len(scenario)
    required = scenario["required_energy_kwh"].to_numpy(dtype=float)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)
    arrival = np.clip(scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS)
    departure = np.clip(scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS)
    solar = load_solar()

    if np.any(required < 0) or np.any(max_power < 0):
        raise ValueError("Energy requests and power limits must be nonnegative.")

    # Per-EV upper bound: energy request, individual power x available slots.
    available_slots = np.maximum(departure - arrival, 0)
    individual_power_bound = np.minimum(
        required,
        max_power * available_slots * STEP_HOURS
    )
    individually_unreachable = individual_power_bound < required - FULL_CHARGE_TOL_KWH

    # Capacity upper bound per slot:
    # EV load <= grid-import limit + available solar power.
    # Also limited by the charger count and the sum of the largest eligible EV rates.
    eligible_count = np.zeros(N_SLOTS, dtype=int)
    slot_power_upper_bound = np.zeros(N_SLOTS, dtype=float)
    for t in range(N_SLOTS):
        eligible = np.flatnonzero((arrival <= t) & (t < departure))
        eligible_count[t] = len(eligible)
        if len(eligible):
            powers = np.sort(max_power[eligible])[::-1]
            charger_limited_power = powers[:N_CHARGERS].sum()
        else:
            charger_limited_power = 0.0
        slot_power_upper_bound[t] = min(
            charger_limited_power,
            GRID_IMPORT_LIMIT_KW + solar[t]
        )

    # Aggregate energy upper bound from slot capacities, while also respecting
    # each EV's individual upper bound. This is an optimistic bound, not a
    # proof that a joint schedule attaining it exists.
    aggregate_slot_energy_bound = float(slot_power_upper_bound.sum() * STEP_HOURS)
    sum_individual_energy_bound = float(individual_power_bound.sum())
    requested_total = float(required.sum())
    optimistic_total_bound = min(
        requested_total,
        aggregate_slot_energy_bound,
        sum_individual_energy_bound
    )

    per_ev = scenario[[
        "ev_id", "required_energy_kwh", "max_power_kw",
        "arrival_slot", "departure_slot"
    ]].copy()
    per_ev["available_window_hours"] = available_slots * STEP_HOURS
    per_ev["individual_energy_upper_bound_kwh"] = individual_power_bound
    per_ev["individually_unreachable"] = individually_unreachable
    per_ev["individual_shortfall_lower_bound_kwh"] = np.maximum(
        required - individual_power_bound, 0.0
    )
    per_ev["individual_demand_met_upper_bound_pct"] = np.where(
        required > 0, 100 * individual_power_bound / required, 100.0
    )
    per_ev_path = RESULTS / "capacity_feasibility_per_ev.csv"
    per_ev.to_csv(per_ev_path, index=False)

    summary = pd.DataFrame([{
        "ev_count": n_ev,
        "time_slots": N_SLOTS,
        "slot_hours": STEP_HOURS,
        "charger_limit": N_CHARGERS,
        "grid_import_limit_after_solar_kw": GRID_IMPORT_LIMIT_KW,
        "requested_energy_kwh": requested_total,
        "sum_individual_energy_upper_bounds_kwh": sum_individual_energy_bound,
        "sum_individual_shortfall_lower_bounds_kwh": float(
            np.maximum(required - individual_power_bound, 0.0).sum()
        ),
        "evs_individually_unable_to_meet_request": int(individually_unreachable.sum()),
        "solar_available_kwh": float(solar.sum() * STEP_HOURS),
        "aggregate_slot_capacity_upper_bound_kwh": aggregate_slot_energy_bound,
        "optimistic_total_deliverable_upper_bound_kwh": optimistic_total_bound,
        "minimum_unmet_energy_lower_bound_kwh": max(
            requested_total - optimistic_total_bound, 0.0
        ),
        "note": (
            "Upper bounds are optimistic. The aggregate slot bound does not "
            "solve the coupled scheduling problem and may overestimate feasible delivery."
        ),
    }])
    summary_path = RESULTS / "capacity_feasibility_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nCAPACITY AND FEASIBILITY ANALYSIS")
    print("=" * 72)
    for key, value in summary.iloc[0].items():
        print(f"{key}: {value}")
    print("\nEVs with largest individual energy shortfalls:")
    display_cols = [
        "ev_id", "required_energy_kwh", "max_power_kw",
        "available_window_hours", "individual_energy_upper_bound_kwh",
        "individual_shortfall_lower_bound_kwh",
        "individual_demand_met_upper_bound_pct"
    ]
    print(
        per_ev.sort_values("individual_shortfall_lower_bound_kwh", ascending=False)
        .head(12)[display_cols].round(3).to_string(index=False)
    )
    print(f"\nPer-EV diagnostics saved to: {per_ev_path}")
    print(f"Summary saved to: {summary_path}")
    print(
        "\nInterpretation: an EV marked individually unreachable cannot meet its "
        "request even if it receives its maximum permitted power for every slot "
        "in its available window. The aggregate capacity figure is only an upper bound."
    )


if __name__ == "__main__":
    main()
