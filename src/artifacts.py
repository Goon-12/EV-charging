"""
Shared helpers for reading and writing experiment artifacts.

Before Tier 0 each runner re-implemented its own schedule CSV writer and its own
convergence plot. Schedule CSVs are the interface between the optimizers and the
audit scripts, so a single pair of ``save_schedule_csv`` / ``load_schedule_csv``
functions keeps that format honest: loading re-aligns rows to the scenario by
``ev_id`` and refuses to return an array that does not match the scenario
exactly, instead of silently trusting row order the way the old validators did.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no GUI window, savefig only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from multiobjective_core import evaluate_schedule, objective

SLOT_COLUMNS = [f"slot_{t}" for t in range(cfg.N_SLOTS)]

# Canonical column order of every comparison table in results/.
RESULT_COLUMNS = [
    "method",
    "objective_score",
    "evaluations",
    "required_energy_kwh",
    "delivered_energy_kwh",
    "unmet_energy_kwh",
    "demand_met_pct",
    "fully_charged_evs",
    "peak_ev_load_kw",
    "peak_grid_import_kw",
    "peak_to_average_ratio",
    "grid_cost_units",
    "cost_per_delivered_kwh",
    "solar_used_kwh",
    "solar_utilization_pct",
    "battery_stress_proxy",
    "constraint_violations",
    "feasible",
    "runtime_seconds",
]


def metrics_row(schedule, ev, solar, method, evaluations=0, seconds=0.0):
    """
    Build one comparison-table row from the shared evaluator.

    ``peak_to_average_ratio`` and ``cost_per_delivered_kwh`` are included on
    purpose: total cost and peak demand are *volume* metrics, so a scheduler
    that simply charges less energy always looks cheaper and flatter. Every
    table must be read together with delivered energy.
    """
    metrics = evaluate_schedule(schedule, ev, solar)
    delivered = metrics["delivered_energy_kwh"]
    peak = metrics["peak_ev_load_kw"]

    return {
        "method": method,
        "objective_score": objective(schedule, ev, solar),
        "evaluations": int(evaluations),
        **metrics,
        "peak_to_average_ratio": (
            peak / (delivered / (len(solar) * cfg.STEP_HOURS)) if delivered > 0 else np.nan
        ),
        "cost_per_delivered_kwh": (
            metrics["grid_cost_units"] / delivered if delivered > 0 else np.nan
        ),
        "feasible": bool(metrics["constraint_violations"] == 0),
        "runtime_seconds": seconds,
    }


def schedule_frame(schedule, ev):
    """Build the canonical schedule DataFrame (ev_id + one column per slot)."""
    schedule = np.asarray(schedule, dtype=float)
    if schedule.shape != (len(ev), cfg.N_SLOTS):
        raise ValueError(
            f"Schedule shape {schedule.shape} does not match "
            f"({len(ev)}, {cfg.N_SLOTS})."
        )
    frame = pd.DataFrame(schedule, columns=SLOT_COLUMNS)
    frame.insert(0, "ev_id", ev["ev_id"].to_numpy())
    return frame


def save_schedule_csv(path, schedule, ev):
    """Write a schedule to CSV in the canonical format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schedule_frame(schedule, ev).to_csv(path, index=False)
    return path


def load_schedule_csv(path, ev):
    """
    Read a schedule CSV and align it to ``ev`` by ``ev_id``.

    Raises ValueError if the file does not describe this exact scenario.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schedule not found: {path}")

    frame = pd.read_csv(path)
    missing = [c for c in SLOT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing slot columns: {missing[:5]}")
    if "ev_id" not in frame.columns:
        raise ValueError(f"{path.name} has no ev_id column.")
    if frame["ev_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate EV IDs.")

    expected = ev["ev_id"].astype(str)
    got = frame["ev_id"].astype(str)
    if set(got) != set(expected):
        raise ValueError(f"{path.name} EV IDs do not match the scenario.")
    if len(frame) != len(ev):
        raise ValueError(f"{path.name} row count does not match the scenario.")

    frame = frame.set_index(got).loc[expected.to_numpy()].reset_index(drop=True)

    schedule = frame[SLOT_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(schedule).all():
        raise ValueError(f"{path.name} contains NaN or infinite values.")
    if (schedule < -1e-9).any():
        raise ValueError(f"{path.name} contains negative power.")
    return schedule


def save_convergence_csv(path, values, column="best_objective"):
    """Write a convergence history (one row per generation/iteration)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"step": np.arange(len(values)), column: values}).to_csv(
        path, index=False
    )
    return path


def save_convergence_plot(path, values, title, ylabel="Best objective (lower is better)"):
    """Save a convergence plot. Uses the Agg backend, never blocks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(len(values)), values, marker="o", markersize=3)
    ax.set_xlabel("Generation / iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_load_profile_plot(path, schedule, solar_kw, title, grid_limit=cfg.MAX_GRID_KW):
    """Save an aggregate load profile plot against the site limits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hours = np.arange(schedule.shape[1]) * cfg.STEP_HOURS
    ev_load = schedule.sum(axis=0)
    solar = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hours, ev_load, label="EV charging load", linewidth=1.6)
    ax.plot(hours, solar, label="Solar available", linewidth=1.2, alpha=0.8)
    ax.axhline(
        grid_limit, linestyle="--", color="tab:red",
        label=f"Grid limit ({grid_limit:.0f} kW)",
    )
    ax.set_xlabel("Hours from simulation start")
    ax.set_ylabel("Power (kW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_metrics_csv(path, metrics, method):
    """Write a single row of metrics for one method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"method": method, **{k: v for k, v in metrics.items() if k != "method"}}
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def load_solar(path=None):
    """Load just the solar profile as an ndarray."""
    path = cfg.DEFAULT_SOLAR if path is None else Path(path)
    frame = pd.read_csv(path)
    if "solar_power_kw" not in frame.columns:
        raise ValueError(f"{path} has no solar_power_kw column.")
    return frame["solar_power_kw"].to_numpy(float)
