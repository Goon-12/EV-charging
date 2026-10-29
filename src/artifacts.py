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

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no GUI window, savefig only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from multiobjective_core import evaluate_schedule, objective

SLOT_COLUMNS = [f"slot_{t}" for t in range(cfg.N_SLOTS)]


def file_digest(path, length=16):
    """Short content hash so an artifact can be tied to the exact input used."""
    path = Path(path)
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:length]

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


def save_equal_service_plot(path, frame, title="Equal-service comparison"):
    """
    Plot cost and peak per delivered kWh against service attained.

    This is the picture the equal-service table describes: x is how much of the
    scenario's attainable energy a method delivered, y is what one delivered kWh
    cost it (left) and the peak demand it needed to achieve that (right).
    Frontier methods are filled, dominated ones hollow, and the deliberately
    infeasible reference is not drawn at all - it would sit far off-scale.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    feasible = frame[frame["feasible"]].reset_index(drop=True)
    dominated = feasible[feasible["pareto_status"] == "dominated"]
    frontier = feasible[feasible["pareto_status"] == "frontier"]

    panels = (
        ("cost_per_delivered_kwh", "Tariff cost per delivered kWh (units/kWh)"),
        ("peak_per_delivered_kwh", "Peak EV load per delivered kWh (kW/kWh)"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (column, ylabel) in zip(axes, panels):
        ax.scatter(
            dominated["service_pct_of_attainable"], dominated[column],
            s=55, facecolors="none", edgecolors="tab:gray",
            label="dominated",
        )
        ax.scatter(
            frontier["service_pct_of_attainable"], frontier[column],
            s=70, color="tab:blue", label="Pareto frontier",
        )
        for _, row in feasible.iterrows():
            ax.annotate(
                row["method"],
                (row["service_pct_of_attainable"], row[column]),
                fontsize=7, xytext=(4, 3), textcoords="offset points",
            )
        ax.set_xlabel("Service attained (% of attainable energy)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Tier 2 comparison figures
# --------------------------------------------------------------------------
# One place that decides what each family of method looks like in every figure,
# so the same colour means the same thing across the whole artifact set.
METHOD_COLORS = {
    "metaheuristic": "tab:blue",
    "rolling": "tab:cyan",
    "conventional": "tab:orange",
    "priority": "tab:gray",
}


def method_family(name):
    """Classify a method name for plotting."""
    if name.startswith("GA rolling") or name.startswith("PSO rolling"):
        return "rolling"
    if name.startswith("GA") or name.startswith("PSO"):
        return "metaheuristic"
    if "priority" in name:
        return "priority"
    return "conventional"


def save_service_comparison_plot(
    path,
    frame,
    service_column,
    reference=None,
    best_conventional=None,
    title="Service delivered: optimizers against conventional policies",
):
    """
    Horizontal bars of delivered energy per method, with the achievable ceiling.

    This is the figure that answers "are the optimizers better than the
    conventional policies?" without any volume confound: every bar is the same
    quantity, and the vertical lines say what a perfect scheduler could reach
    and what the best conventional policy reached.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = frame[frame["feasible"]] if "feasible" in frame else frame
    data = data.sort_values(service_column)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    colors = [METHOD_COLORS[method_family(name)] for name in data["method"]]
    ax.barh(data["method"], data[service_column], color=colors, alpha=0.9)

    if best_conventional is not None:
        ax.axvline(
            best_conventional, color="tab:orange", linestyle="--", linewidth=1.6,
            label=f"Best conventional policy ({best_conventional:.1f} kWh)",
        )
    if reference is not None and len(reference):
        best = float(reference["delivered_energy_kwh"].max())
        ax.axvline(
            best, color="black", linestyle=":", linewidth=1.8,
            label=f"MILP reference optimum ({best:.1f} kWh)",
        )

    for y, value in enumerate(data[service_column]):
        label = f"{value:.1f}"
        if reference is not None and len(reference):
            label += f"  ({100 * value / float(reference['delivered_energy_kwh'].max()):.1f}% of best)"
        ax.text(value, y, "  " + label, va="center", fontsize=8)

    ax.set_xlabel("Delivered energy (kWh)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    # Below the axes: the bars are labelled end-to-end, so any in-plot legend
    # position covers a label. Extra bottom margin keeps it on the canvas.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, fontsize=8)
    ax.set_xlim(0, data[service_column].max() * 1.28)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_cost_service_plot(
    path,
    frame,
    reference,
    service_column="delivered_energy_kwh",
    cost_column="grid_cost_units",
    title="Cost against service, with the epsilon-constraint frontier",
):
    """
    Scatter every method as (service, cost) against the min-cost frontier.

    The frontier is the set of cheapest schedules known to deliver at least each
    service level. A method *above* the line pays more for the energy it does
    deliver; a method *below* it would be a better schedule than the reference and
    is impossible unless the solver missed something - which is why the reference
    rows carry a MIP gap.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = frame[frame["feasible"]] if "feasible" in frame else frame

    fig, ax = plt.subplots(figsize=(11, 5.6))

    if reference is not None and len(reference):
        # The reference frame always carries the canonical column names, even
        # when the method table is an equal-service table with its own names.
        ref = reference.sort_values("delivered_energy_kwh")
        ax.plot(
            ref["delivered_energy_kwh"], ref["grid_cost_units"],
            color="black", linestyle=":", marker="x", linewidth=1.5,
            label="MILP min-cost frontier / optimum",
        )

    # Label offsets alternate up/down around each point: methods cluster tightly
    # (GA and PSO differ by a few kWh), and a single offset makes the labels
    # unreadable exactly where the comparison matters.
    for index, (_, row) in enumerate(data.sort_values(service_column).iterrows()):
        family = method_family(row["method"])
        ax.scatter(
            row[service_column], row[cost_column],
            s=70, color=METHOD_COLORS[family], edgecolors="black", zorder=3,
        )
        ax.annotate(
            row["method"], (row[service_column], row[cost_column]),
            fontsize=7.5, xytext=(6, 6 if index % 2 == 0 else -12),
            textcoords="offset points",
        )

    for family, color in METHOD_COLORS.items():
        ax.scatter([], [], color=color, label=family, s=60)

    ax.set_xlabel("Delivered energy (kWh)")
    ax.set_ylabel("Total tariff cost (units)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_method_profiles_plot(
    path,
    schedules,
    solar_kw,
    methods,
    title="Charging load profiles",
    grid_limit=cfg.MAX_GRID_KW,
):
    """Overlay the aggregate load profile of a few methods on one axis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    first = np.asarray(schedules[methods[0]], dtype=float)
    hours = np.arange(first.shape[1]) * cfg.STEP_HOURS
    solar = np.maximum(np.asarray(solar_kw, dtype=float), 0.0)

    # Four profiles on one axis need four distinguishable styles: FCFS and EDF
    # overlap almost exactly (they fill every port), as do GA and PSO, so colour
    # alone would make the figure look like two lines instead of four.
    palette = ("#e6862b", "#b8431f", "#1f77b4", "#6a3d9a", "#2ca02c", "#7f7f7f")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for index, name in enumerate(methods):
        schedule = np.asarray(schedules[name], dtype=float)
        delivered = schedule.sum() * cfg.STEP_HOURS
        ax.plot(
            hours, schedule.sum(axis=0),
            linewidth=2.0 if index < 2 else 1.5,
            color=palette[index % len(palette)],
            linestyle="-" if index % 2 == 0 else "--",
            label=f"{name} ({delivered:.0f} kWh)",
            alpha=0.9,
        )
    ax.plot(hours, solar, color="tab:green", linewidth=1.0, alpha=0.5, label="Solar available")
    ax.axhline(
        grid_limit, linestyle=":", color="tab:red", linewidth=1.8,
        label=f"Grid limit ({grid_limit:.0f} kW)",
    )
    ax.set_xlabel("Hours from simulation start")
    ax.set_ylabel("Power (kW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_seed_spread_plot(
    path,
    multi_seed,
    conventional=None,
    reference_service=None,
    title="Seed-to-seed spread against the conventional policies",
):
    """
    Box plot of delivered energy over seeds, with conventional policies as lines.

    A single run beating EDF proves little; this shows whether the win survives
    a change of random seed, which is the difference between a result and a lucky
    draw.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = multi_seed.copy()
    names = sorted(frame["method"].unique())
    values = [frame.loc[frame["method"] == name, "delivered_energy_kwh"].to_numpy() for name in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(values, patch_artist=True, widths=0.45)
    # Set ticks separately: ``labels=`` was renamed to ``tick_labels=`` in
    # matplotlib 3.9, and this way the figure works on both.
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    for patch, name in zip(box["boxes"], names):
        patch.set_facecolor(METHOD_COLORS[method_family(name)])
        patch.set_alpha(0.55)
    for median in box["medians"]:
        median.set_color("black")

    if conventional is not None and len(conventional):
        for _, row in conventional.sort_values("delivered_energy_kwh").iterrows():
            ax.axhline(
                row["delivered_energy_kwh"], color=METHOD_COLORS["conventional"],
                linestyle="--", linewidth=1.2,
            )
            # Annotated at the left edge: the boxes occupy the middle, so the
            # right side is where the legend would collide with the label.
            ax.annotate(
                f"{row['method']}: {row['delivered_energy_kwh']:.1f} kWh",
                (0.45, row["delivered_energy_kwh"]),
                fontsize=8, va="bottom", color="tab:orange",
            )
    if reference_service is not None:
        ax.axhline(
            reference_service, color="black", linestyle=":", linewidth=1.6,
            label=f"MILP optimum ({reference_service:.1f} kWh)",
        )
    ax.legend(fontsize=8, loc="upper left")

    ax.set_ylabel("Delivered energy (kWh)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(0.4, len(names) + 1.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_convergence_comparison_plot(path, histories, title="Convergence (best objective)"):
    """Overlay several convergence histories on a log axis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for (name, history), color in zip(histories.items(), ("tab:blue", "tab:red", "tab:green")):
        history = np.asarray(history, dtype=float)
        if history.size == 0:
            continue
        steps = np.arange(len(history))
        ax.plot(steps, history, linewidth=1.6, color=color, label=name)
    ax.set_xlabel("Generation / iteration")
    ax.set_ylabel("Best objective (lower is better)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
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
