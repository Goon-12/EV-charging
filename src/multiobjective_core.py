"""
Shared multi-objective core for EV charging optimization.
All algorithms must use this evaluator/repair so results are comparable.
Assumptions: 15-minute slots; solar/tariff profiles are illustrative scenario inputs.
Battery stress is a relative charging-rate proxy, NOT a battery-life prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

STEP_HOURS = 0.25
N_CHARGERS = 20
MAX_GRID_KW = 125.0
DEFAULT_SCENARIO = Path("data/processed/ev_scenario.csv")
DEFAULT_SOLAR = Path("data/processed/solar_profile_36h.csv")

def load_inputs(scenario_path=DEFAULT_SCENARIO, solar_path=DEFAULT_SOLAR):
    scenario_path, solar_path = Path(scenario_path), Path(solar_path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}. Put ev_scenario.csv in data/processed/.")
    if not solar_path.exists():
        raise FileNotFoundError(f"Solar profile not found: {solar_path}. Put solar_profile_36h.csv in data/processed/.")
    ev = pd.read_csv(scenario_path).copy()
    solar_df = pd.read_csv(solar_path)
    required = {"arrival_slot","departure_slot","required_energy_kwh","max_power_kw"}
    if not required.issubset(ev.columns):
        raise ValueError(f"Scenario missing columns: {sorted(required-set(ev.columns))}")
    solar_col = "solar_power_kw" if "solar_power_kw" in solar_df.columns else None
    if solar_col is None:
        raise ValueError("Solar CSV must contain solar_power_kw.")
    solar = solar_df[solar_col].to_numpy(float)
    n_slots = len(solar)
    ev["arrival_slot"] = ev["arrival_slot"].astype(int).clip(0,n_slots)
    ev["departure_slot"] = ev["departure_slot"].astype(int).clip(0,n_slots)
    ev["required_energy_kwh"] = ev["required_energy_kwh"].astype(float).clip(lower=0)
    ev["max_power_kw"] = ev["max_power_kw"].astype(float).clip(lower=0)
    return ev, solar

def tariff_vector(n_slots):
    """Synthetic illustrative tariff: 4 overnight, 10 evening peak, 6 otherwise."""
    hours = (np.arange(n_slots) * STEP_HOURS) % 24
    return np.where((hours >= 17) & (hours < 22), 10.0,
                    np.where((hours >= 0) & (hours < 6), 4.0, 6.0))

def make_windows(ev, n_slots):
    a = np.clip(ev["arrival_slot"].to_numpy(int), 0, n_slots)
    d = np.clip(ev["departure_slot"].to_numpy(int), 0, n_slots)
    # A departure must be after arrival. Invalid/zero-length windows remain unchargeable.
    d = np.maximum(d, a)
    return a, d

def repair_schedule(
    priority,
    ev,
    solar_kw,
    max_grid_kw=MAX_GRID_KW,
    n_chargers=N_CHARGERS,
):
    """
    Convert an EV x slot priority matrix into a feasible schedule.

    Ranks EV-slot opportunities across the horizon instead of filling
    slots chronologically. This allows solar availability and tariff
    to influence charging time.

    Hard constraints:
      - EV arrival/departure window
      - Individual maximum charging power
      - Required energy not exceeded
      - Charger count per slot
      - Grid import <= max_grid_kw after solar
    """
    priority = np.asarray(priority, dtype=float)
    solar_kw = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)

    n_ev, n_slots = priority.shape

    if n_ev != len(ev) or n_slots != len(solar_kw):
        raise ValueError(
            "Priority dimensions must match EV rows and solar slots."
        )

    a, d = make_windows(ev, n_slots)
    need = ev["required_energy_kwh"].to_numpy(float).copy()
    pmax = ev["max_power_kw"].to_numpy(float)

    sched = np.zeros((n_ev, n_slots), dtype=float)
    tariff = tariff_vector(n_slots)

    # Build every feasible EV-slot charging opportunity.
    ev_indices = []
    slot_indices = []
    scores = []

    max_tariff = 10.0

    for i in range(n_ev):
        if need[i] <= 1e-8 or pmax[i] <= 0:
            continue

        for t in range(a[i], d[i]):
            # Higher urgency as departure approaches.
            slack = max(d[i] - t, 1)
            urgency = 1.0 - (slack / max(n_slots, 1))

            # Global slot properties now affect ranking across time.
            solar_bonus = min(solar_kw[t] / max(max_grid_kw, 1e-9), 1.0)
            tariff_penalty = (
                (tariff[t] - 4.0) / max(max_tariff - 4.0, 1e-9)
            )

            score = (
                0.45 * priority[i, t]
                + 0.20 * urgency
                + 0.45 * solar_bonus
                - 0.30 * tariff_penalty
            )

            ev_indices.append(i)
            slot_indices.append(t)
            scores.append(score)

    # Process the most attractive EV-slot opportunities first.
    order = np.argsort(np.asarray(scores))[::-1]

    slot_grid_capacity = np.maximum(
        max_grid_kw + solar_kw, 0.0
    )
    slot_used_power = np.zeros(n_slots, dtype=float)
    slot_chargers_used = np.zeros(n_slots, dtype=int)

    for idx in order:
        i = ev_indices[idx]
        t = slot_indices[idx]

        if need[i] <= 1e-8:
            continue

        if slot_chargers_used[t] >= n_chargers:
            continue

        remaining_capacity = (
            slot_grid_capacity[t] - slot_used_power[t]
        )

        if remaining_capacity <= 1e-8:
            continue

        power = min(
            pmax[i],
            need[i] / STEP_HOURS,
            remaining_capacity,
        )

        if power <= 1e-8:
            continue

        sched[i, t] = power
        need[i] -= power * STEP_HOURS
        slot_used_power[t] += power
        slot_chargers_used[t] += 1

    return sched

def evaluate_schedule(schedule, ev, solar_kw, max_grid_kw=MAX_GRID_KW, n_chargers=N_CHARGERS):
    s = np.asarray(schedule, dtype=float)
    n_ev, n_slots = s.shape
    req = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)
    a,d = make_windows(ev,n_slots)
    delivered_by_ev = s.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(req-delivered_by_ev,0)
    total_req = float(req.sum())
    total_delivered = float(delivered_by_ev.sum())
    total_unmet = float(unmet_by_ev.sum())
    ev_load = s.sum(axis=0)
    grid_import = np.maximum(ev_load - np.maximum(np.asarray(solar_kw),0),0)
    tariff = tariff_vector(n_slots)
    cost = float(np.sum(grid_import * STEP_HOURS * tariff))
    peak_ev = float(ev_load.max(initial=0))
    peak_grid = float(grid_import.max(initial=0))
    solar_used = float(np.sum(np.minimum(ev_load, np.maximum(solar_kw,0))) * STEP_HOURS)
    
    solar_available = float(np.sum(np.maximum(solar_kw,0))*STEP_HOURS)
    # Relative charging-rate stress proxy: energy-weighted squared fraction of rated power.

        # Diagnostic: solar availability during EV connection windows
    connected_count = np.zeros(n_slots, dtype=int)

    for i in range(n_ev):
        connected_count[a[i]:d[i]] += 1

    solar_during_connection = float(
        np.sum(
            np.where(connected_count > 0, np.maximum(solar_kw, 0), 0)
        ) * STEP_HOURS
    )

    solar_with_charger_capacity = float(
        np.sum(
            np.minimum(
                np.maximum(solar_kw, 0),
                N_CHARGERS * np.max(pmax, initial=0)
            )
        ) * STEP_HOURS
    )

   
    rate_fraction = np.divide(s, pmax[:,None], out=np.zeros_like(s), where=pmax[:,None] > 1e-9)
    energy = s * STEP_HOURS
    stress = float(np.sum(energy * np.clip(rate_fraction,0,1)**2) / max(total_delivered,1e-9))
    full = int(np.sum(unmet_by_ev <= 1e-3))
    window_violation = 0
    for i in range(n_ev):
        if a[i] > 0 and np.any(s[i,:a[i]] > 1e-6): window_violation += int(np.sum(s[i,:a[i]] > 1e-6))
        if d[i] < n_slots and np.any(s[i,d[i]:] > 1e-6): window_violation += int(np.sum(s[i,d[i]:] > 1e-6))
    power_violation = int(np.sum(s > pmax[:,None] + 1e-6))
    energy_over = int(np.sum(delivered_by_ev > req + 1e-3))
    charger_violation = int(np.sum(np.sum(s > 1e-6, axis=0) > n_chargers))
    grid_violation = int(np.sum(grid_import > max_grid_kw + 1e-6))
    violations = window_violation + power_violation + energy_over + charger_violation + grid_violation
    return {
        "required_energy_kwh": total_req, "delivered_energy_kwh": total_delivered,
        "unmet_energy_kwh": total_unmet, "demand_met_pct": 100*total_delivered/max(total_req,1e-9),
        "fully_charged_evs": full, "peak_ev_load_kw": peak_ev, "peak_grid_import_kw": peak_grid,
        "grid_cost_units": cost, "solar_used_kwh": solar_used,
        "solar_available_kwh": solar_available,
        "solar_utilization_pct": 100*solar_used/max(solar_available,1e-9),
        "battery_stress_proxy": stress, "constraint_violations": int(violations),
        "solar_during_connection_kwh": solar_during_connection,
        "solar_with_charger_capacity_kwh": solar_with_charger_capacity,
    }

def objective(schedule, ev, solar_kw, weights=None):
    """
    Normalized weighted objective; hard constraints are handled by repair.
    Energy shortfall is emphasized. Weights are documented and shared by GA/PSO/RH.
    """
    w = weights or {"unmet":0.80, "peak":0.07, "cost":0.06, "battery":0.02, "solar":0.05}
    m = evaluate_schedule(schedule, ev, solar_kw)
    print("Solar available:", round(m["solar_available_kwh"], 2))
    print("Solar during connection windows:",round(m["solar_during_connection_kwh"], 2))
    print("Solar used by EVs:", round(m["solar_used_kwh"], 2))
    req = max(m["required_energy_kwh"],1e-9)
    cost_ref = max(req * 10.0, 1.0)
    unmet_n = m["unmet_energy_kwh"]/req
    peak_n = m["peak_grid_import_kw"]/MAX_GRID_KW
    cost_n = min(m["grid_cost_units"]/cost_ref, 1.0)
    battery_n = m["battery_stress_proxy"]
    solar_n = m["solar_used_kwh"]/max(m["solar_available_kwh"],1e-9)
    # Solar use is a reward (subtract from cost); clip final score only for stability.
    soft_score = (w["unmet"]*unmet_n + w["peak"]*peak_n + w["cost"]*cost_n
                  + w["battery"]*battery_n - w["solar"]*solar_n)
    # Feasibility must dominate the score: an infeasible uncontrolled baseline
    # must never appear better simply because it delivers more energy.
    return float(soft_score + 10.0 * m["constraint_violations"])

def validate_schedule(schedule, ev, solar_kw):
    m = evaluate_schedule(schedule, ev, solar_kw)
    if m["constraint_violations"] != 0:
        raise ValueError(f"Schedule has {m['constraint_violations']} constraint violations: {m}")
    return m
