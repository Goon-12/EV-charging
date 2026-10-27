"""
Shared multi-objective core for EV charging optimization.

This module is the ONLY repair routine, evaluator and objective in the
project. Every algorithm (GA, PSO, rolling horizon, baselines) must go
through it so that results are comparable. See docs/TIER0_CHANGES.md.

Assumptions
-----------
* 15-minute slots; slot ``t`` covers ``[t*STEP_HOURS, (t+1)*STEP_HOURS)``.
* An EV is available when ``arrival_slot <= t < departure_slot``.
* ``max_power_kw`` is the EV's own limit; the charger adds no further cap.
* Solar/tariff profiles are illustrative scenario inputs, not measurements.
* Battery stress is a relative charging-rate proxy, NOT a battery-life
  prediction, and is dominated by a single fixed reference value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from load_scenario import load_scenario


# --------------------------------------------------------------------------
# Input loading
# --------------------------------------------------------------------------
def load_inputs(scenario_path=None, solar_path=None):
    """Load the EV scenario and the solar profile into (DataFrame, ndarray)."""
    scenario_path = cfg.DEFAULT_SCENARIO if scenario_path is None else scenario_path
    solar_path = cfg.DEFAULT_SOLAR if solar_path is None else solar_path
    scenario_path, solar_path = Path(scenario_path), Path(solar_path)

    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Scenario not found: {scenario_path}. Put ev_scenario.csv in data/processed/."
        )
    if not solar_path.exists():
        raise FileNotFoundError(
            f"Solar profile not found: {solar_path}. Put solar_profile_36h.csv in data/processed/."
        )

    # Delegate to the single validated loader rather than reading the CSV here.
    ev = load_scenario(scenario_path)
    solar_df = pd.read_csv(solar_path)

    if "solar_power_kw" not in solar_df.columns:
        raise ValueError("Solar CSV must contain solar_power_kw.")
    solar = solar_df["solar_power_kw"].to_numpy(float)

    n_slots = len(solar)
    if n_slots != cfg.N_SLOTS:
        raise ValueError(
            f"Solar profile has {n_slots} slots but config expects {cfg.N_SLOTS}."
        )

    ev["arrival_slot"] = ev["arrival_slot"].astype(int).clip(0, n_slots)
    ev["departure_slot"] = ev["departure_slot"].astype(int).clip(0, n_slots)
    ev["required_energy_kwh"] = ev["required_energy_kwh"].astype(float).clip(lower=0)
    ev["max_power_kw"] = ev["max_power_kw"].astype(float).clip(lower=0)
    return ev, solar


def tariff_vector(n_slots, slot_offset=0):
    """
    Synthetic illustrative tariff: 4 overnight, 10 evening peak, 6 otherwise.

    ``slot_offset`` is the absolute slot index of ``n_slots[0]``. It exists so
    that a sub-window of the horizon gets the tariff of the real time of day
    instead of restarting at midnight. Rolling horizon MUST pass its offset.
    """
    hours = ((np.arange(n_slots) + slot_offset) * cfg.STEP_HOURS) % 24.0
    return np.where(
        (hours >= cfg.PEAK_START_HOUR) & (hours < cfg.PEAK_END_HOUR),
        cfg.TARIFF_PEAK,
        np.where(hours < cfg.OFFPEAK_END_HOUR, cfg.TARIFF_OFFPEAK, cfg.TARIFF_MID),
    )


def make_windows(ev, n_slots):
    """Return clipped arrival/departure slot arrays; empty windows stay empty."""
    a = np.clip(ev["arrival_slot"].to_numpy(int), 0, n_slots)
    d = np.clip(ev["departure_slot"].to_numpy(int), 0, n_slots)
    # A departure must not precede an arrival.
    d = np.maximum(d, a)
    return a, d


def window_mask(ev, n_slots):
    """Boolean (n_ev, n_slots) matrix of allowed charging slots."""
    a, d = make_windows(ev, n_slots)
    mask = np.zeros((len(ev), n_slots), dtype=bool)
    for i in range(len(ev)):
        mask[i, a[i]:d[i]] = True
    return mask


# --------------------------------------------------------------------------
# Repair (decoder)
# --------------------------------------------------------------------------
def repair_schedule(
    priority,
    ev,
    solar_kw,
    max_grid_kw=cfg.MAX_GRID_KW,
    n_chargers=cfg.N_CHARGERS,
    slot_offset=0,
    horizon_slots=None,
):
    """
    Convert an EV x slot priority matrix into a feasible schedule.

    Ranks EV-slot opportunities across the horizon instead of filling slots
    chronologically, so solar availability and tariff can influence the
    charging time.

    Hard constraints enforced (never violated by the output):
      - EV arrival/departure window
      - Individual maximum charging power
      - Required energy not exceeded
      - Charger count per slot
      - Grid import <= max_grid_kw after solar

    Parameters used only by the rolling-horizon controller:
      ``slot_offset``   absolute slot index of column 0, so the tariff matches
                        the real time of day instead of restarting at midnight.
      ``horizon_slots`` number of slots remaining in the *whole* simulation,
                        used to scale the deadline-urgency term. When omitted it
                        defaults to the width of this priority matrix, which
                        keeps full-horizon (one-shot) runs numerically unchanged.
    """
    priority = np.asarray(priority, dtype=float)
    solar_kw = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)

    n_ev, n_slots = priority.shape
    if n_ev != len(ev) or n_slots != len(solar_kw):
        raise ValueError("Priority dimensions must match EV rows and solar slots.")

    a, d = make_windows(ev, n_slots)
    need = ev["required_energy_kwh"].to_numpy(float).copy()
    pmax = ev["max_power_kw"].to_numpy(float)

    sched = np.zeros((n_ev, n_slots), dtype=float)
    tariff = tariff_vector(n_slots, slot_offset)

    # Build every feasible EV-slot charging opportunity.
    ev_indices, slot_indices, scores = [], [], []

    # Deadline urgency is measured against the slots actually remaining in the
    # simulation, not just the width of this priority matrix.
    urgency_scale = n_slots if horizon_slots is None else max(int(horizon_slots), 1)

    for i in range(n_ev):
        if need[i] <= 1e-8 or pmax[i] <= 0:
            continue

        for t in range(a[i], d[i]):
            # Higher urgency as departure approaches.
            slack = max(d[i] - t, 1)
            urgency = 1.0 - (slack / max(urgency_scale, 1))

            # Global slot properties affect ranking across time.
            solar_bonus = min(solar_kw[t] / max(max_grid_kw, 1e-9), 1.0)
            tariff_penalty = (tariff[t] - cfg.TARIFF_OFFPEAK) / max(
                cfg.MAX_TARIFF - cfg.TARIFF_OFFPEAK, 1e-9
            )

            scores.append(
                0.45 * priority[i, t]
                + 0.20 * urgency
                + 0.45 * solar_bonus
                - 0.30 * tariff_penalty
            )
            ev_indices.append(i)
            slot_indices.append(t)

    if not scores:
        return sched

    # Process the most attractive EV-slot opportunities first.
    order = np.argsort(np.asarray(scores))[::-1]

    slot_grid_capacity = np.maximum(max_grid_kw + solar_kw, 0.0)
    slot_used_power = np.zeros(n_slots, dtype=float)
    slot_chargers_used = np.zeros(n_slots, dtype=int)

    for idx in order:
        i = ev_indices[idx]
        t = slot_indices[idx]

        if need[i] <= 1e-8:
            continue
        if slot_chargers_used[t] >= n_chargers:
            continue

        remaining_capacity = slot_grid_capacity[t] - slot_used_power[t]
        if remaining_capacity <= 1e-8:
            continue

        power = min(pmax[i], need[i] / cfg.STEP_HOURS, remaining_capacity)
        if power <= 1e-8:
            continue

        sched[i, t] = power
        need[i] -= power * cfg.STEP_HOURS
        slot_used_power[t] += power
        slot_chargers_used[t] += 1

    return sched


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_schedule(
    schedule,
    ev,
    solar_kw,
    max_grid_kw=cfg.MAX_GRID_KW,
    n_chargers=cfg.N_CHARGERS,
):
    """
    Compute every reported metric for one schedule.

    ``schedule[i, t]`` is charging power for EV i in slot t, in kW.
    """
    s = np.asarray(schedule, dtype=float)
    n_ev, n_slots = s.shape
    req = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)
    a, d = make_windows(ev, n_slots)
    solar = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)

    delivered_by_ev = s.sum(axis=1) * cfg.STEP_HOURS
    unmet_by_ev = np.maximum(req - delivered_by_ev, 0.0)
    total_req = float(req.sum())
    total_delivered = float(delivered_by_ev.sum())
    total_unmet = float(unmet_by_ev.sum())

    ev_load = s.sum(axis=0)
    grid_import = np.maximum(ev_load - solar, 0.0)
    tariff = tariff_vector(n_slots)
    cost = float(np.sum(grid_import * cfg.STEP_HOURS * tariff))
    peak_ev = float(ev_load.max(initial=0))
    peak_grid = float(grid_import.max(initial=0))
    solar_used = float(np.sum(np.minimum(ev_load, solar)) * cfg.STEP_HOURS)
    solar_available = float(np.sum(solar) * cfg.STEP_HOURS)

    # Diagnostics: how much solar actually coincides with connected EVs.
    connected_count = np.zeros(n_slots, dtype=int)
    for i in range(n_ev):
        connected_count[a[i]:d[i]] += 1

    solar_during_connection = float(
        np.sum(np.where(connected_count > 0, solar, 0.0)) * cfg.STEP_HOURS
    )
    solar_with_charger_capacity = float(
        np.sum(np.minimum(solar, n_chargers * np.max(pmax, initial=0))) * cfg.STEP_HOURS
    )

    # Relative charging-rate stress proxy: energy-weighted squared
    # fraction of rated power. 1.0 = everything at rated power.
    rate_fraction = np.divide(
        s, pmax[:, None], out=np.zeros_like(s), where=pmax[:, None] > 1e-9
    )
    energy = s * cfg.STEP_HOURS
    stress = float(
        np.sum(energy * np.clip(rate_fraction, 0, 1) ** 2) / max(total_delivered, 1e-9)
    )

    full = int(np.sum(unmet_by_ev <= cfg.FULLY_CHARGED_TOLERANCE_KWH))

    window_violation = 0
    for i in range(n_ev):
        if a[i] > 0:
            window_violation += int(np.sum(s[i, :a[i]] > 1e-6))
        if d[i] < n_slots:
            window_violation += int(np.sum(s[i, d[i]:] > 1e-6))
    power_violation = int(np.sum(s > pmax[:, None] + 1e-6))
    energy_over = int(np.sum(delivered_by_ev > req + 1e-3))
    charger_violation = int(np.sum(np.sum(s > 1e-6, axis=0) > n_chargers))
    grid_violation = int(np.sum(grid_import > max_grid_kw + 1e-6))
    violations = (
        window_violation
        + power_violation
        + energy_over
        + charger_violation
        + grid_violation
    )

    return {
        "required_energy_kwh": total_req,
        "delivered_energy_kwh": total_delivered,
        "unmet_energy_kwh": total_unmet,
        "demand_met_pct": 100 * total_delivered / max(total_req, 1e-9),
        "fully_charged_evs": full,
        "peak_ev_load_kw": peak_ev,
        "peak_grid_import_kw": peak_grid,
        "grid_cost_units": cost,
        "solar_used_kwh": solar_used,
        "solar_available_kwh": solar_available,
        "solar_utilization_pct": 100 * solar_used / max(solar_available, 1e-9),
        "battery_stress_proxy": stress,
        "constraint_violations": int(violations),
        "solar_during_connection_kwh": solar_during_connection,
        "solar_with_charger_capacity_kwh": solar_with_charger_capacity,
    }


def objective(schedule, ev, solar_kw, weights=None):
    """
    Normalized weighted objective; hard constraints are handled by repair.

    Lower is better. Energy shortfall is emphasized. This function is called
    ~10^4 times per experiment, so it must stay silent: it printed three
    diagnostic lines per call before Tier 0. Use ``evaluate_schedule`` if you
    need those numbers.
    """
    w = cfg.OBJECTIVE_WEIGHTS if weights is None else weights
    m = evaluate_schedule(schedule, ev, solar_kw)
    req = max(m["required_energy_kwh"], 1e-9)
    cost_ref = max(req * cfg.TARIFF_PEAK, 1.0)

    unmet_n = m["unmet_energy_kwh"] / req
    peak_n = m["peak_grid_import_kw"] / cfg.MAX_GRID_KW
    cost_n = min(m["grid_cost_units"] / cost_ref, 1.0)
    battery_n = m["battery_stress_proxy"]
    solar_n = m["solar_used_kwh"] / max(m["solar_available_kwh"], 1e-9)

    # Solar use is a reward (subtracted).
    soft_score = (
        w["unmet"] * unmet_n
        + w["peak"] * peak_n
        + w["cost"] * cost_n
        + w["battery"] * battery_n
        - w["solar"] * solar_n
    )
    # Feasibility must dominate: an infeasible schedule must never look better
    # than a feasible one merely because it delivered more energy.
    return float(soft_score + cfg.INFEASIBILITY_PENALTY * m["constraint_violations"])


def constraint_report(schedule, ev, solar_kw, max_grid_kw=cfg.MAX_GRID_KW,
                      n_chargers=cfg.N_CHARGERS):
    """
    Per-constraint violation counts, for tests and audits.

    ``evaluate_schedule`` returns only the total, which is enough to drive the
    objective but not enough to diagnose which constraint broke.
    """
    s = np.asarray(schedule, dtype=float)
    n_ev, n_slots = s.shape
    a, d = make_windows(ev, n_slots)
    pmax = ev["max_power_kw"].to_numpy(float)
    req = ev["required_energy_kwh"].to_numpy(float)
    solar = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)

    ev_load = s.sum(axis=0)
    grid_import = np.maximum(ev_load - solar, 0.0)

    window = 0
    for i in range(n_ev):
        if a[i] > 0:
            window += int(np.sum(s[i, :a[i]] > 1e-6))
        if d[i] < n_slots:
            window += int(np.sum(s[i, d[i]:] > 1e-6))

    report = {
        "window_violations": window,
        "power_violations": int(np.sum(s > pmax[:, None] + 1e-6)),
        "energy_over_violations": int(np.sum(s.sum(axis=1) * cfg.STEP_HOURS > req + 1e-3)),
        "charger_violations": int(np.sum(np.sum(s > 1e-6, axis=0) > n_chargers)),
        "grid_violations": int(np.sum(grid_import > max_grid_kw + 1e-6)),
        "negative_power": int(np.sum(s < -1e-9)),
        "nonfinite_power": int(np.sum(~np.isfinite(s))),
    }
    report["total"] = sum(v for k, v in report.items() if k != "total")
    return report


def validate_schedule(schedule, ev, solar_kw):
    """Raise if the schedule breaks any hard constraint, else return metrics."""
    report = constraint_report(schedule, ev, solar_kw)
    if report["total"] != 0:
        raise ValueError(f"Schedule violates hard constraints: {report}")
    return evaluate_schedule(schedule, ev, solar_kw)
