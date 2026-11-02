"""
Data and experiment engine behind the demo web app.

The app has two jobs: show the committed Tier 2 evidence honestly, and let a
visitor run the optimizers live without leaving the browser. Both go through
:mod:`multiobjective_core` and :mod:`multiobjective_optimizers`, i.e. the same
evaluator and the same decoder the experiment scripts use. Nothing here
re-implements the model:

* committed numbers are read from ``results/final_unified/*.csv`` and the
  schedules from ``schedules.npz``;
* live numbers are computed by running the real baselines and the real
  optimizers, then evaluating with :func:`evaluate_schedule`.

Live runs are deliberately budget-limited (hundreds to ~1,200 evaluations)
because they happen inside an HTTP request. The committed full-budget runs used
12,680 (GA) and 13,030 (PSO) evaluations; the app says which is which so a live
result is never mistaken for the reported one.

The module is import-safe: no server, no plotting backend, no files written.
"""
from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config as cfg  # noqa: E402
from baselines import POLICIES  # noqa: E402
from multiobjective_core import (  # noqa: E402
    evaluate_schedule,
    load_inputs,
    make_windows,
    objective,
    tariff_vector,
)
from multiobjective_optimizers import optimize_ga, optimize_pso  # noqa: E402

RESULTS_DIR = PROJECT_DIR / "results" / "final_unified"
WEBAPP_DIR = Path(__file__).resolve().parent

# Power is sent to the browser as one byte per (EV, slot) cell. 8 kW covers the
# scenario's 7 kW maximum rated power with a little headroom.
CELL_SCALE_KW = 8.0

# Live budgets. These are small on purpose: an HTTP request should answer in
# seconds. The full committed run is ~12,700 evaluations per optimizer.
BUDGETS = {
    "fast": {
        "label": "Quick",
        "ga": dict(population_size=10, generations=18, polish_evaluations=60),
        "pso": dict(particles=10, iterations=18, polish_evaluations=60),
    },
    "standard": {
        "label": "Standard",
        "ga": dict(population_size=14, generations=30, polish_evaluations=150),
        "pso": dict(particles=14, iterations=30, polish_evaluations=150),
    },
    "deep": {
        "label": "Deep",
        "ga": dict(population_size=18, generations=50, polish_evaluations=300),
        "pso": dict(particles=16, iterations=40, polish_evaluations=300),
    },
}

# The committed full-budget evaluation counts, asserted against the formulas in
# tests/test_webapp.py so the app cannot quietly quote a different budget.
FULL_BUDGET = {"GA": 12680, "PSO": 13030}


def count_ga_evaluations(
    population_size, generations, polish_evaluations=0,
    elite_count=cfg.GA_ELITE_COUNT,
):
    """Exact GA evaluation count (mirrors run_final_experiments)."""
    return (
        population_size
        + generations * (population_size - min(elite_count, population_size))
        + polish_evaluations
    )


def count_pso_evaluations(particles, iterations, polish_evaluations=0):
    """Exact PSO evaluation count (mirrors run_final_experiments)."""
    return particles * (1 + iterations) + polish_evaluations


def _f(value, digits=3):
    """JSON-safe float (None for NaN/inf)."""
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _i(value):
    return int(value)


def _record(row):
    """Convert one pandas row to a JSON-safe dict."""
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.bool_, bool)):
            out[key] = bool(value)
        elif isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            out[key] = _f(value, 6)
        elif value is None or (isinstance(value, float) and math.isnan(value)):
            out[key] = None
        else:
            try:
                if pd.isna(value):
                    out[key] = None
                    continue
            except (TypeError, ValueError):
                pass
            out[key] = value
    return out


def _method_family(name):
    """Same family classification as artifacts.method_family (kept local so the
    web app does not import matplotlib)."""
    if name.startswith("GA rolling") or name.startswith("PSO rolling"):
        return "rolling"
    if name.startswith("GA") or name.startswith("PSO"):
        return "metaheuristic"
    if "priority" in name:
        return "priority"
    return "conventional"


def _encode_cells(schedule):
    """Quantise an EV x slot power matrix to bytes for the browser canvas."""
    cells = np.clip(schedule / CELL_SCALE_KW, 0.0, 1.0)
    quantised = np.round(cells * 255).astype(np.uint8)
    return base64.b64encode(quantised.tobytes()).decode("ascii")


class DemoEngine:
    """Loads the committed artifacts once and runs live experiments on demand."""

    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.ev, self.solar = load_inputs()
        self.hours = np.arange(len(self.solar)) * cfg.STEP_HOURS
        self.tariff = tariff_vector(len(self.solar))
        self._lock = threading.Lock()
        self._test_count = None
        self._test_count_checked = False
        self._schedules = {}
        npz_path = self.results_dir / "schedules.npz"
        if npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as data:
                self._schedules = {key: data[key].copy() for key in data.files}
        self._tables = self._read_tables()
        self.reference_service_kwh = float(
            self._tables["reference"]["delivered_energy_kwh"].max()
        )

    # ------------------------------------------------------------------
    # Committed artifacts
    # ------------------------------------------------------------------
    def _read_tables(self):
        read = lambda name: pd.read_csv(self.results_dir / name)
        tables = {
            "service": read("service_vs_optimal.csv"),
            "reference": read("reference_optimum.csv"),
            "multi_seed": read("multi_seed_results.csv"),
            "multi_seed_summary": read("multi_seed_summary.csv"),
            "manifest": read("run_manifest.csv"),
        }
        return tables

    def _conventional_rows(self):
        table = self._tables["service"]
        return table[
            table["method"].str.startswith(("FCFS", "EDF")) & table["feasible"]
        ]

    def _best_conventional(self):
        return self._conventional_rows().sort_values(
            "service_delivered_kwh"
        ).iloc[-1]

    def port_pressure(self):
        """Port-hours available vs port-hours the sessions need at rated power."""
        arrival, departure = make_windows(self.ev, len(self.solar))
        required = self.ev["required_energy_kwh"].to_numpy(float)
        pmax = self.ev["max_power_kw"].to_numpy(float)

        connected = np.zeros(len(self.solar), dtype=int)
        for slot in range(len(self.solar)):
            connected[slot] = int(
                np.sum((arrival <= slot) & (slot < departure) & (pmax > 0))
            )
        port_hours_available = float(
            np.minimum(connected, cfg.N_CHARGERS).sum() * cfg.STEP_HOURS
        )
        port_hours_needed = float(np.sum(required / np.maximum(pmax, 1e-9)))
        return {
            "port_hours_available": _f(port_hours_available, 1),
            "port_hours_needed": _f(port_hours_needed, 1),
            "connected_slots": int((connected > 0).sum()),
            "mean_window_hours": _f(
                np.mean((departure - arrival) * cfg.STEP_HOURS), 2
            ),
            "required_kwh": _f(required.sum(), 2),
            "solar_available_kwh": _f(
                np.sum(np.maximum(self.solar, 0.0)) * cfg.STEP_HOURS, 1
            ),
        }

    def test_count(self):
        """Number of tests pytest collects, or None if it cannot be determined."""
        if not self._test_count_checked:
            self._test_count_checked = True
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                    cwd=PROJECT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                for line in reversed(completed.stdout.splitlines()):
                    if " tests collected" in line:
                        self._test_count = int(line.split()[0])
                        break
            except Exception:  # pragma: no cover - best effort only
                self._test_count = None
        return self._test_count

    def overview(self):
        """Everything the dashboard needs that does not depend on a live run."""
        service = self._tables["service"]
        best_conventional = self._best_conventional()
        pso = service[service["method"] == "PSO"].iloc[0]
        ga = service[service["method"] == "GA"].iloc[0]
        optimum = float(self._tables["reference"]["delivered_energy_kwh"].max())

        seeded = self._tables["multi_seed"]
        beats = seeded[
            seeded["delivered_energy_kwh"] > best_conventional["service_delivered_kwh"]
        ]
        min_margin = (
            100.0
            * (
                beats["delivered_energy_kwh"].min()
                / best_conventional["service_delivered_kwh"]
                - 1.0
            )
            if len(beats)
            else None
        )

        artifacts = []
        for path in sorted(self.results_dir.iterdir()):
            if path.is_file():
                artifacts.append(
                    {
                        "name": path.name,
                        "suffix": path.suffix.lstrip("."),
                        "size_kb": _f(path.stat().st_size / 1024, 1),
                    }
                )

        manifest = _record(self._tables["manifest"].iloc[0])
        for key in ("objective_weights", "ga_settings", "pso_settings",
                    "rolling_settings", "multi_seed_seeds"):
            if isinstance(manifest.get(key), str):
                try:
                    manifest[key] = json.loads(manifest[key])
                except ValueError:
                    pass

        methods = []
        for _, row in service.iterrows():
            record = _record(row)
            record["family"] = _method_family(record["method"])
            methods.append(record)

        reference_rows = self._tables["reference"]
        return {
            "site": {
                "n_ev": int(len(self.ev)),
                "n_slots": int(len(self.solar)),
                "n_chargers": int(cfg.N_CHARGERS),
                "max_grid_kw": float(cfg.MAX_GRID_KW),
                "step_hours": float(cfg.STEP_HOURS),
                "horizon_hours": float(cfg.HORIZON_HOURS),
                "required_kwh": _f(self.ev["required_energy_kwh"].sum(), 2),
                "arrival_range_hours": [
                    _f(self.ev["arrival_slot"].min() * cfg.STEP_HOURS, 2),
                    _f(self.ev["arrival_slot"].max() * cfg.STEP_HOURS, 2),
                ],
                "solar_available_kwh": _f(
                    float(np.sum(self.solar)) * cfg.STEP_HOURS, 1
                ),
            },
            "objective_weights": {k: float(v) for k, v in cfg.OBJECTIVE_WEIGHTS.items()},
            "port_pressure": self.port_pressure(),
            "headline": {
                "reference_kwh": _f(optimum, 2),
                "reference_gap_pct": _f(
                    reference_rows.loc[
                        reference_rows["label"].str.startswith("Max service"),
                        "mip_gap",
                    ].iloc[0]
                    * 100,
                    2,
                ),
                "best_conventional": {
                    "method": str(best_conventional["method"]),
                    "delivered_kwh": _f(
                        best_conventional["service_delivered_kwh"], 2
                    ),
                    "cost_per_kwh": _f(
                        best_conventional["cost_per_delivered_kwh"], 4
                    ),
                },
                "pso": {
                    "delivered_kwh": _f(pso["service_delivered_kwh"], 2),
                    "pct_of_optimum": _f(pso["pct_of_best_service"], 2),
                    "delta_vs_conventional_pct": _f(
                        100
                        * (
                            pso["service_delivered_kwh"]
                            / best_conventional["service_delivered_kwh"]
                            - 1
                        ),
                        2,
                    ),
                    "evaluations": int(pso["evaluations"]),
                },
                "ga": {
                    "delivered_kwh": _f(ga["service_delivered_kwh"], 2),
                    "pct_of_optimum": _f(ga["pct_of_best_service"], 2),
                    "delta_vs_conventional_pct": _f(
                        100
                        * (
                            ga["service_delivered_kwh"]
                            / best_conventional["service_delivered_kwh"]
                            - 1
                        ),
                        2,
                    ),
                    "evaluations": int(ga["evaluations"]),
                },
                "seeded_runs": {
                    "beating_conventional": int(len(beats)),
                    "total": int(len(seeded)),
                    "min_margin_pct": _f(min_margin, 2) if min_margin is not None else None,
                },
                # The uncontrolled reference is infeasible by construction and is
                # excluded; this is the violation count of the methods that claim
                # to be feasible.
                "violations": int(
                    service.loc[service["feasible"], "constraint_violations"].sum()
                ),
            },
            "methods": methods,
            "reference": [_record(row) for _, row in reference_rows.iterrows()],
            "multi_seed": {
                "runs": [_record(row) for _, row in seeded.iterrows()],
                "summary": [
                    _record(row)
                    for _, row in self._tables["multi_seed_summary"].iterrows()
                ],
            },
            "manifest": manifest,
            "artifacts": artifacts,
            "tests": {"collected": self.test_count()},
            "series": {
                "hours": [_f(h, 2) for h in self.hours],
                "solar_kw": [_f(v, 2) for v in self.solar],
                "tariff": [_f(v, 2) for v in self.tariff],
            },
            "budgets": [
                {
                    "key": key,
                    "label": spec["label"],
                    "ga_evaluations": count_ga_evaluations(
                        spec["ga"]["population_size"],
                        spec["ga"]["generations"],
                        spec["ga"]["polish_evaluations"],
                    ),
                    "pso_evaluations": count_pso_evaluations(
                        spec["pso"]["particles"],
                        spec["pso"]["iterations"],
                        spec["pso"]["polish_evaluations"],
                    ),
                }
                for key, spec in BUDGETS.items()
            ],
            "full_budget": FULL_BUDGET,
        }

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------
    def schedule(self, method):
        """Payload for one committed schedule (load profile, EVs, heatmap cells)."""
        if method not in self._schedules:
            raise KeyError(f"Unknown schedule: {method}")
        table = self._tables["service"].set_index("method")
        metrics = None
        if method in table.index:
            metrics = _record(table.loc[method])
        return self._schedule_payload(method, self._schedules[method], metrics)

    def _schedule_payload(self, method, schedule, metrics=None):
        schedule = np.asarray(schedule, dtype=float)
        n_ev, n_slots = schedule.shape
        arrival, departure = make_windows(self.ev, n_slots)
        delivered = schedule.sum(axis=1) * cfg.STEP_HOURS
        required = self.ev["required_energy_kwh"].to_numpy(float)
        ev_load = schedule.sum(axis=0)
        occupancy = (schedule > 1e-9).sum(axis=0)

        return {
            "method": method,
            "family": _method_family(method),
            "metrics": metrics,
            "shape": [int(n_ev), int(n_slots)],
            "hours": [_f(h, 2) for h in self.hours],
            "load_kw": [_f(v, 2) for v in ev_load],
            "occupancy": [int(v) for v in occupancy],
            "evs": {
                "id": [int(v) for v in self.ev["ev_id"].to_numpy()],
                "arrival": [int(v) for v in arrival],
                "departure": [int(v) for v in departure],
                "required": [_f(v, 2) for v in required],
                "delivered": [_f(v, 2) for v in delivered],
                "peak_kw": [_f(v, 2) for v in schedule.max(axis=1)],
            },
            "cells": _encode_cells(schedule),
            "cell_scale_kw": CELL_SCALE_KW,
        }

    # ------------------------------------------------------------------
    # Live experiments
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_weights(payload):
        weights = dict(cfg.OBJECTIVE_WEIGHTS)
        supplied = payload.get("weights") or {}
        for key in ("unmet", "peak", "cost", "battery", "solar"):
            if key in supplied:
                try:
                    value = float(supplied[key])
                except (TypeError, ValueError):
                    raise ValueError(f"Weight {key!r} must be a number.")
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Weight {key!r} must be non-negative.")
                weights[key] = value
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Objective weights must not all be zero.")
        return {key: round(value / total, 6) for key, value in weights.items()}, total

    def _live_row(self, method, schedule, weights, evaluations=0, seconds=0.0):
        metrics = evaluate_schedule(schedule, self.ev, self.solar)
        delivered = metrics["delivered_energy_kwh"]
        row = {
            "method": method,
            "family": _method_family(method),
            "objective_score": _f(objective(schedule, self.ev, self.solar, weights), 6),
            "evaluations": int(evaluations),
            "seconds": _f(seconds, 2),
            "delivered_kwh": _f(delivered, 2),
            "pct_of_optimum": _f(
                100 * delivered / max(self.reference_service_kwh, 1e-9), 2
            ),
            "fully_charged": int(metrics["fully_charged_evs"]),
            "unmet_kwh": _f(metrics["unmet_energy_kwh"], 2),
            "peak_kw": _f(metrics["peak_ev_load_kw"], 2),
            "cost_units": _f(metrics["grid_cost_units"], 2),
            "cost_per_kwh": _f(
                metrics["grid_cost_units"] / max(delivered, 1e-9), 4
            ),
            "solar_used_kwh": _f(metrics["solar_used_kwh"], 2),
            "violations": int(metrics["constraint_violations"]),
            "feasible": bool(metrics["constraint_violations"] == 0),
        }
        return row

    def run(self, payload):
        """Run the conventional policies and the selected optimizers live."""
        if payload is None or not isinstance(payload, dict):
            raise ValueError("Expected a JSON object body.")

        weights, weight_sum = self._clean_weights(payload)
        budget_key = str(payload.get("budget", "fast"))
        if budget_key not in BUDGETS:
            raise ValueError(
                f"Unknown budget {budget_key!r}; choose from {sorted(BUDGETS)}."
            )
        algorithms = payload.get("algorithms")
        if algorithms is None:
            algorithms = ["GA", "PSO"]
        if not isinstance(algorithms, list):
            raise ValueError("Algorithms must be a list of 'GA' and/or 'PSO'.")
        algorithms = [str(name).upper() for name in algorithms]
        if not algorithms:
            raise ValueError("Select at least one optimizer.")
        if any(name not in ("GA", "PSO") for name in algorithms):
            raise ValueError("Algorithms must be GA, PSO, or both.")
        try:
            seed = int(payload.get("seed", cfg.GA_SEED))
        except (TypeError, ValueError):
            raise ValueError("Seed must be an integer.")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be between 0 and 2^32 - 1.")

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A live run is already in progress.")
        try:
            started = time.perf_counter()
            rows, series, schedules = [], {}, {}

            for name, policy in POLICIES.items():
                began = time.perf_counter()
                schedule = policy(self.ev, self.solar)
                rows.append(
                    self._live_row(
                        name, schedule, weights,
                        seconds=time.perf_counter() - began,
                    )
                )
                series[name] = [_f(v, 2) for v in schedule.sum(axis=0)]
                # Only the optimizers get a full schedule in the response; the
                # conventional policies already have committed schedules.

            spec = BUDGETS[budget_key]
            for name in algorithms:
                history = []
                began = time.perf_counter()
                if name == "GA":
                    schedule, _ = optimize_ga(
                        self.ev, self.solar, seed=seed, weights=weights,
                        history=history, **spec["ga"],
                    )
                    evaluations = count_ga_evaluations(
                        spec["ga"]["population_size"],
                        spec["ga"]["generations"],
                        spec["ga"]["polish_evaluations"],
                    )
                else:
                    schedule, _ = optimize_pso(
                        self.ev, self.solar, seed=seed, weights=weights,
                        history=history, **spec["pso"],
                    )
                    evaluations = count_pso_evaluations(
                        spec["pso"]["particles"],
                        spec["pso"]["iterations"],
                        spec["pso"]["polish_evaluations"],
                    )
                elapsed = time.perf_counter() - began
                row = self._live_row(
                    name, schedule, weights, evaluations, elapsed
                )
                rows.append(row)
                series[name] = [_f(v, 2) for v in schedule.sum(axis=0)]
                schedules[name] = self._schedule_payload(name, schedule, row)
                schedules[name]["history"] = [_f(v, 6) for v in history]

            # Comparison anchors are computed from this run, never hard-coded.
            conventional = [
                row for row in rows
                if row["method"].startswith(("FCFS", "EDF"))
                and row["method"] != "Uncontrolled (infeasible reference)"
            ]
            anchor = max(conventional, key=lambda row: row["delivered_kwh"])
            for row in rows:
                row["delta_vs_conventional_pct"] = _f(
                    100 * (row["delivered_kwh"] / anchor["delivered_kwh"] - 1), 2
                )

            optimizers = [
                row for row in rows if row["method"] in ("GA", "PSO")
            ]
            best = max(optimizers, key=lambda row: row["delivered_kwh"])

            return {
                "weights": weights,
                "weight_sum_before_normalisation": _f(weight_sum, 4),
                "budget": budget_key,
                "budget_label": spec["label"],
                "seed": seed,
                "elapsed_seconds": _f(time.perf_counter() - started, 2),
                "rows": rows,
                "series": series,
                "schedules": schedules,
                "anchor": {
                    "method": anchor["method"],
                    "delivered_kwh": anchor["delivered_kwh"],
                },
                "best_optimizer": {
                    "method": best["method"],
                    "delivered_kwh": best["delivered_kwh"],
                    "pct_of_optimum": best["pct_of_optimum"],
                    "delta_vs_conventional_pct": best["delta_vs_conventional_pct"],
                },
                "reference_kwh": _f(self.reference_service_kwh, 2),
            }
        finally:
            self._lock.release()
