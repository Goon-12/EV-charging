"""
Reference optima: what a perfect scheduler could actually achieve.

Why this module exists
----------------------
Tier 0/1 could only say "method A beat method B". That is not a claim about
quality, and Tier 0 item 10 said so explicitly: without a benchmark, a ranking is
just an ordering. This module supplies the benchmark by solving the scenario as a
mixed-integer linear program with HiGHS (``scipy.optimize.milp``):

* **Max service** - how much of the requested energy can be delivered at all,
  subject to the same hard constraints every method faces (connection windows,
  per-EV rated power, 20 ports per slot, the grid cap after solar).
* **Min cost at a service floor** - the cheapest possible way to deliver at least
  a given amount of energy. Swept over a grid of floors this produces the
  epsilon-constraint frontier Tier 1 deferred: cost compared at *identical*
  service, instead of cost compared across different service levels.

The result is a denominator for every other method. "EDF delivered 1,133.7 kWh"
becomes "EDF delivered 87.6% of the best known service", and "GA cost 7,070 units"
becomes a gap to the cheapest schedule known to deliver the same energy.

Model
-----
The port limit is the only combinatorial part: at most ``n_chargers`` EVs may
charge in a slot, which needs one binary per feasible (EV, slot) pair.

    s[i,t]  kW delivered to EV i in slot t          (free only inside its window)
    y[i,t]  1 if EV i uses a port in slot t         (binary)
    g[t]    grid import in slot t                   (cost model only)

    power       s[i,t] <= pmax_i * y[i,t]
    energy      sum_t s[i,t] * dt <= required_i
    ports       sum_i y[i,t] <= n_chargers
    grid        sum_i s[i,t] <= max_grid_kw + solar[t]
    cost model  g[t] >= sum_i s[i,t] - solar[t],  g[t] >= 0
    floor       sum_{i,t} s[i,t] * dt >= service_floor

Only 3,703 of the 14,400 (EV, slot) pairs are feasible, so 7,406 variables with
3,703 binaries is enough - a solve takes seconds to a couple of minutes.

Honesty about "optimal"
-----------------------
The solver runs under a time limit and a MIP gap tolerance set in ``config.py``,
and every row it produces carries ``status``, ``mip_gap`` and ``solve_seconds``. A
row labelled ``optimal`` proved optimality; a row labelled ``feasible`` is the
best schedule found within the limit and the reported gap is its certificate. No
number here is presented as exact unless the solver says so.

Every solved schedule is re-checked with
:func:`multiobjective_core.constraint_report` before it is returned. That check is
what caught a modelling bug during development - a linking constraint merged into
the grid row, which produced a "feasible" 1,511 kWh optimum that used 36 ports.
It stays in the module rather than the test suite because a silently infeasible
optimum would invalidate every gap computed from it.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

import config as cfg
from artifacts import file_digest
from multiobjective_core import (
    constraint_report,
    evaluate_schedule,
    make_windows,
    tariff_vector,
)

# Canonical column order of results/final_unified/reference_optimum.csv.
REFERENCE_COLUMNS = [
    "label",
    "objective",
    "service_floor_kwh",
    "delivered_energy_kwh",
    "demand_met_pct",
    "fully_charged_evs",
    "peak_ev_load_kw",
    "grid_cost_units",
    "cost_per_delivered_kwh",
    "constraint_violations",
    "status",
    "mip_gap",
    "solve_seconds",
    # Provenance: the reference is only valid for the inputs it was solved on, so
    # every row names them. Without this a stale optimum would silently price a
    # regenerated scenario, which is the failure Tier 0 fixed with a run manifest.
    "scenario_sha256_16",
    "solar_sha256_16",
]

MAX_SERVICE_LABEL = "Max service (MILP)"
COST_PREFIX = "Min cost at"


@dataclass
class ReferenceSolution:
    """One solved (or partially solved) reference point."""

    label: str
    schedule: np.ndarray
    objective: float
    status: str
    mip_gap: float
    seconds: float
    service_floor_kwh: float = 0.0


def _status_name(status):
    return {
        0: "optimal",
        1: "feasible",       # time limit / gap tolerance reached, incumbent kept
        2: "infeasible",
        3: "unbounded",
        4: "failed",
    }.get(int(status), f"status {status}")


def _build_model(
    ev, solar, with_grid_cost=False, max_grid_kw=None, n_chargers=None
):
    """
    Assemble the constraint matrix once, in CSR form.

    Row blocks: ``[EV energy | grid | ports | linking | floor | grid cost]``.
    Keeping the linking rows in their own block is deliberate - folding them into
    the grid block silently removes the port constraint, because ``s - pmax*y``
    is then bounded by the grid instead of by zero.
    """
    max_grid_kw = cfg.MAX_GRID_KW if max_grid_kw is None else float(max_grid_kw)
    n_chargers = cfg.N_CHARGERS if n_chargers is None else int(n_chargers)

    solar = np.maximum(np.asarray(solar, dtype=float), 0.0)
    n_ev, n_slots = len(ev), len(solar)
    arrival, departure = make_windows(ev, n_slots)
    required = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)

    pair_ev, pair_slot, pair_pmax = [], [], []
    for i in range(n_ev):
        if required[i] <= 0 or pmax[i] <= 0 or departure[i] <= arrival[i]:
            continue
        for t in range(arrival[i], departure[i]):
            pair_ev.append(i)
            pair_slot.append(t)
            pair_pmax.append(pmax[i])

    pair_ev = np.asarray(pair_ev, dtype=int)
    pair_slot = np.asarray(pair_slot, dtype=int)
    pair_pmax = np.asarray(pair_pmax, dtype=float)
    n_pairs = len(pair_ev)

    var_s = np.arange(n_pairs)
    var_y = np.arange(n_pairs, 2 * n_pairs)
    nvars = 2 * n_pairs
    var_g = np.array([], dtype=int)
    if with_grid_cost:
        var_g = np.arange(nvars, nvars + n_slots)
        nvars += n_slots

    row_energy = 0
    row_grid = n_ev
    row_ports = row_grid + n_slots
    row_link = row_ports + n_slots
    row_floor = row_link + n_pairs
    row_cost = row_floor + 1
    nrows = row_cost + (n_slots if with_grid_cost else 0)

    rows, cols, vals = [], [], []

    def add(row, col, value):
        rows.append(int(row))
        cols.append(int(col))
        vals.append(float(value))

    dt = cfg.STEP_HOURS
    for j in range(n_pairs):
        add(row_energy + pair_ev[j], var_s[j], dt)
        add(row_grid + pair_slot[j], var_s[j], 1.0)
        add(row_ports + pair_slot[j], var_y[j], 1.0)
        add(row_link + j, var_s[j], 1.0)
        add(row_link + j, var_y[j], -pair_pmax[j])
        add(row_floor, var_s[j], -dt)

    if with_grid_cost:
        for j in range(n_pairs):
            add(row_cost + pair_slot[j], var_s[j], 1.0)
        for t in range(n_slots):
            add(row_cost + t, var_g[t], -1.0)

    matrix = csr_matrix((vals, (rows, cols)), shape=(nrows, nvars))

    upper = np.zeros(nrows)
    lower = np.full(nrows, -np.inf)
    upper[row_energy:row_energy + n_ev] = required
    upper[row_grid:row_grid + n_slots] = np.maximum(max_grid_kw + solar, 0.0)
    upper[row_ports:row_ports + n_slots] = n_chargers
    upper[row_link:row_link + n_pairs] = 0.0
    upper[row_floor] = np.inf          # inactive until a floor is requested
    if with_grid_cost:
        upper[row_cost:row_cost + n_slots] = solar

    integrality = np.zeros(nvars)
    integrality[var_y] = 1
    lb = np.zeros(nvars)
    ub = np.full(nvars, np.inf)
    ub[var_s] = pair_pmax
    ub[var_y] = 1.0
    if with_grid_cost:
        ub[var_g] = np.maximum(max_grid_kw, 0.0) + solar

    return {
        "ev": ev,
        "solar": solar,
        "matrix": matrix,
        "upper": upper,
        "lower": lower,
        "integrality": integrality,
        "bounds": Bounds(lb, ub),
        "var_s": var_s,
        "var_g": var_g,
        "pair_ev": pair_ev,
        "pair_slot": pair_slot,
        "row_floor": row_floor,
        "max_grid_kw": max_grid_kw,
        "n_chargers": n_chargers,
        "n_ev": n_ev,
        "n_slots": n_slots,
        "n_pairs": n_pairs,
        "tariff": tariff_vector(n_slots),
    }


def _solve(model, cost, floor=None, time_limit=None, mip_gap=None, allow_missing=False):
    """
    Run HiGHS, verify the schedule against the project's constraints.

    With ``allow_missing`` a solve that finds no integer-feasible schedule inside
    the time limit returns ``None`` instead of raising. That matters for the
    cost frontier: a hard floor near the maximum service is a tight problem, and
    a missing point must show up as a hole in the frontier, not as a crash and
    certainly not as an invented number.
    """
    time_limit = cfg.REFERENCE_TIME_LIMIT_S if time_limit is None else time_limit
    mip_gap = cfg.REFERENCE_MIP_GAP if mip_gap is None else mip_gap

    upper, lower = model["upper"], model["lower"]
    if floor is not None:
        upper = upper.copy()
        upper[model["row_floor"]] = -float(floor)

    started = time.perf_counter()
    result = milp(
        c=cost,
        constraints=LinearConstraint(model["matrix"], lower, upper),
        integrality=model["integrality"],
        bounds=model["bounds"],
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_gap),
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - started

    if result.x is None:
        if allow_missing:
            return None
        raise RuntimeError(
            f"Reference MILP returned no solution (status {result.status}: "
            f"{result.message})."
        )

    schedule = np.zeros((model["n_ev"], model["n_slots"]), dtype=float)
    schedule[model["pair_ev"], model["pair_slot"]] = result.x[model["var_s"]]

    report = constraint_report(
        schedule,
        model["ev"],
        model["solar"],
        max_grid_kw=model["max_grid_kw"],
        n_chargers=model["n_chargers"],
    )
    if report["total"] != 0:
        raise RuntimeError(
            "Reference MILP produced an infeasible schedule, so the model does not "
            f"match the project's constraints: {report}"
        )

    return ReferenceSolution(
        label="",
        schedule=schedule,
        objective=float(result.fun),
        status=_status_name(result.status),
        mip_gap=float(getattr(result, "mip_gap", float("nan"))),
        seconds=elapsed,
        service_floor_kwh=float(floor) if floor is not None else 0.0,
    )


def solve_max_service(
    ev, solar, time_limit=None, mip_gap=None, max_grid_kw=None, n_chargers=None
):
    """Deliver as much energy as the site allows, ignoring price."""
    model = _build_model(
        ev, solar, max_grid_kw=max_grid_kw, n_chargers=n_chargers
    )
    cost = np.zeros(model["matrix"].shape[1])
    cost[model["var_s"]] = -cfg.STEP_HOURS
    solution = _solve(model, cost, None, time_limit, mip_gap)
    solution.label = MAX_SERVICE_LABEL
    return solution


def solve_min_cost(
    ev,
    solar,
    service_floor_kwh,
    time_limit=None,
    mip_gap=None,
    max_grid_kw=None,
    n_chargers=None,
):
    """Cheapest schedule that delivers at least ``service_floor_kwh``."""
    model = _build_model(
        ev,
        solar,
        with_grid_cost=True,
        max_grid_kw=max_grid_kw,
        n_chargers=n_chargers,
    )
    cost = np.zeros(model["matrix"].shape[1])
    cost[model["var_g"]] = model["tariff"] * cfg.STEP_HOURS
    solution = _solve(
        model, cost, service_floor_kwh, time_limit, mip_gap, allow_missing=True
    )
    if solution is None:
        return None
    solution.label = f"{COST_PREFIX} {service_floor_kwh:.0f} kWh"
    return solution


def capacity_bounds(ev, solar, max_grid_kw=None, n_chargers=None):
    """
    Two rigorous ceilings that do not need a solver.

    * ``per_ev_bound``    - each EV alone: ``min(required, window * pmax)`` summed.
    * ``per_slot_bound``  - each slot alone: ``min(grid + solar, sum of the
      ``n_chargers`` largest rated powers connected in that slot)``, summed.

    Both relax at least one real coupling, so no feasible schedule can exceed
    either. ``upper_bound`` is their minimum. The MILP optimum sits below it, and
    the distance between the two is how much the *interaction* of the port limit
    with the connection windows costs - which is exactly what a better scheduler
    can try to exploit.
    """
    max_grid_kw = cfg.MAX_GRID_KW if max_grid_kw is None else float(max_grid_kw)
    n_chargers = cfg.N_CHARGERS if n_chargers is None else int(n_chargers)

    solar = np.maximum(np.asarray(solar, dtype=float), 0.0)
    n_ev, n_slots = len(ev), len(solar)
    arrival, departure = make_windows(ev, n_slots)
    required = ev["required_energy_kwh"].to_numpy(float)
    pmax = ev["max_power_kw"].to_numpy(float)
    dt = cfg.STEP_HOURS

    per_ev = np.minimum(required, np.maximum(departure - arrival, 0) * dt * pmax)

    per_slot = np.zeros(n_slots)
    connected = np.zeros(n_slots, dtype=int)
    for t in range(n_slots):
        eligible = np.flatnonzero((arrival <= t) & (t < departure) & (pmax > 0))
        connected[t] = eligible.size
        if eligible.size == 0:
            continue
        top = np.sort(pmax[eligible])[::-1][:n_chargers].sum()
        per_slot[t] = min(max_grid_kw + solar[t], top)

    return {
        "per_ev_bound_kwh": float(per_ev.sum()),
        "per_slot_bound_kwh": float(per_slot.sum() * dt),
        "upper_bound_kwh": float(min(per_ev.sum(), per_slot.sum() * dt)),
        "connected_slots": int((connected > 0).sum()),
        "port_slots_available": float(np.minimum(connected, n_chargers).sum() * dt),
    }


def reference_row(solution, ev, solar, scenario_path=None, solar_path=None):
    """One row of ``reference_optimum.csv`` from a solved reference point."""
    scenario_path = cfg.DEFAULT_SCENARIO if scenario_path is None else scenario_path
    solar_path = cfg.DEFAULT_SOLAR if solar_path is None else solar_path
    metrics = evaluate_schedule(solution.schedule, ev, solar)
    delivered = metrics["delivered_energy_kwh"]
    return {
        "label": solution.label,
        "objective": solution.objective,
        "service_floor_kwh": solution.service_floor_kwh,
        "delivered_energy_kwh": delivered,
        "demand_met_pct": metrics["demand_met_pct"],
        "fully_charged_evs": metrics["fully_charged_evs"],
        "peak_ev_load_kw": metrics["peak_ev_load_kw"],
        "grid_cost_units": metrics["grid_cost_units"],
        "cost_per_delivered_kwh": metrics["grid_cost_units"] / max(delivered, 1e-9),
        "constraint_violations": metrics["constraint_violations"],
        "status": solution.status,
        "mip_gap": solution.mip_gap,
        "solve_seconds": solution.seconds,
        "scenario_sha256_16": file_digest(scenario_path),
        "solar_sha256_16": file_digest(solar_path),
    }


def reference_table(
    ev,
    solar,
    levels=None,
    time_limit=None,
    mip_gap=None,
    verbose=True,
    scenario_path=None,
    solar_path=None,
):
    """
    Solve the max-service optimum and the min-cost frontier.

    Returns ``(frame, schedules)``: the table in :data:`REFERENCE_COLUMNS` order
    and ``{label: schedule}`` so the runner can plot the reference profiles.
    The levels actually solved are the configured grid filtered to what is
    physically reachable, plus one point on the optimum itself.
    """
    levels = cfg.REFERENCE_SERVICE_LEVELS_KWH if levels is None else levels
    upper = capacity_bounds(ev, solar)["upper_bound_kwh"]

    rows, schedules = [], {}

    limit = cfg.REFERENCE_TIME_LIMIT_S if time_limit is None else time_limit
    gap = cfg.REFERENCE_MIP_GAP if mip_gap is None else mip_gap
    if verbose:
        print(f"\nSolving reference optima (time limit {limit:.0f}s per solve, "
              f"MIP gap {gap:.1%})...")

    best = solve_max_service(ev, solar, time_limit, mip_gap)
    rows.append(reference_row(best, ev, solar, scenario_path, solar_path))
    schedules[best.label] = best.schedule
    if verbose:
        print(f"  {best.label:<28} {rows[-1]['delivered_energy_kwh']:8.1f} kWh "
              f"({best.status}, gap {best.mip_gap:.3%}, {best.seconds:.1f}s)")

    for level in levels:
        if level >= rows[0]["delivered_energy_kwh"]:
            continue  # not reachable, the max-service solve already covers it
        if level > upper:
            continue
        solution = solve_min_cost(ev, solar, level, time_limit, mip_gap)
        if solution is None:
            if verbose:
                print(f"  {COST_PREFIX} {level:.0f} kWh{'':<10} no feasible schedule "
                      f"found inside the time limit - level skipped")
            continue
        row = reference_row(solution, ev, solar, scenario_path, solar_path)
        rows.append(row)
        schedules[solution.label] = solution.schedule
        if verbose:
            print(f"  {solution.label:<28} {row['delivered_energy_kwh']:8.1f} kWh "
                  f"cost {row['grid_cost_units']:8.1f} "
                  f"({solution.status}, gap {solution.mip_gap:.3%}, "
                  f"{solution.seconds:.1f}s)")

    frame = pd.DataFrame(rows)[REFERENCE_COLUMNS]
    frame = frame.sort_values("delivered_energy_kwh").reset_index(drop=True)
    return frame, schedules


def frontier_points(reference):
    """
    ``(service, cost)`` pairs of the epsilon-constraint frontier, cheapest first.

    The min-cost rows are monotone in the floor by construction (a cheaper
    schedule that delivers more would have been feasible for the tighter floor),
    so the frontier is the rows sorted by service. The max-service optimum is the
    most expensive point on it.
    """
    frame = reference.dropna(subset=["delivered_energy_kwh", "grid_cost_units"])
    frame = frame[frame["constraint_violations"] == 0]
    frame = frame.sort_values("delivered_energy_kwh")
    return frame[["delivered_energy_kwh", "grid_cost_units"]].to_numpy(float)


def interpolated_optimal_cost(reference, service_kwh):
    """
    Cheapest cost known to deliver ``service_kwh``, interpolated from the frontier.

    Returns ``nan`` outside the solved range instead of extrapolating: below the
    cheapest solved floor no reference exists, and above the max-service optimum
    nothing can deliver that much at all. An interpolated value is a bound only up
    to the convexity of the frontier, which is exactly what the docstring of the
    figure says - it is not a proof of optimality at that service level.
    """
    points = frontier_points(reference)
    if len(points) < 2:
        return float("nan")
    service, cost = points[:, 0], points[:, 1]
    if service_kwh < service[0] - 1e-9 or service_kwh > service[-1] + 1e-9:
        return float("nan")
    return float(np.interp(service_kwh, service, cost))


def attach_reference(
    table,
    reference,
    service_column="delivered_energy_kwh",
    cost_column="grid_cost_units",
):
    """
    Add the optimality columns every method table should carry.

    * ``pct_of_best_service`` - delivered energy as a share of the best known
      service (what the conventional "% of required" column cannot express).
    * ``service_gap_kwh``     - what that method left on the table.
    * ``optimal_cost_at_service`` / ``cost_gap_vs_optimal_pct`` - the method's
      total cost against the cheapest known schedule delivering as much energy.
      ``nan`` where the frontier does not cover the method's service level.
    """
    table = table.copy()
    best_service = float(reference["delivered_energy_kwh"].max())
    table["pct_of_best_service"] = 100.0 * table[service_column] / best_service
    table["service_gap_kwh"] = best_service - table[service_column]

    optimal_cost = [
        interpolated_optimal_cost(reference, value) for value in table[service_column]
    ]
    table["optimal_cost_at_service"] = optimal_cost
    table["cost_gap_vs_optimal_pct"] = [
        100.0 * (cost - opt) / opt if np.isfinite(opt) else np.nan
        for cost, opt in zip(table[cost_column], optimal_cost)
    ]
    return table


def save_reference(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_reference(path):
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=cfg.REFERENCE_TIME_LIMIT_S)
    parser.add_argument("--mip-gap", type=float, default=cfg.REFERENCE_MIP_GAP)
    parser.add_argument(
        "--out",
        default=str(cfg.RESULTS_DIR / "final_unified" / "reference_optimum.csv"),
    )
    args = parser.parse_args()

    from multiobjective_core import load_inputs

    ev, solar = load_inputs()
    bounds = capacity_bounds(ev, solar)
    print("Rigorous service ceilings (no solver needed):")
    print(f"  per-EV bound   : {bounds['per_ev_bound_kwh']:.1f} kWh")
    print(f"  per-slot bound : {bounds['per_slot_bound_kwh']:.1f} kWh")
    print(f"  upper bound    : {bounds['upper_bound_kwh']:.1f} kWh")
    print(f"  port-hours     : {bounds['port_slots_available']:.1f} h")

    frame, _ = reference_table(
        ev, solar, time_limit=args.time_limit, mip_gap=args.mip_gap
    )
    from pathlib import Path

    save_reference(frame, Path(args.out))
    print(f"\nSaved: {args.out}")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
