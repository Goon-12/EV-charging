"""
Validated loader for the frozen EV charging scenario.

There is exactly one scenario loader in the project. ``multiobjective_core``
delegates to this module rather than reading the CSV itself, so every entry
point sees the same validation and the same column contract. Before Tier 0 the
loader also printed three lines on every call, which made it unusable inside a
test suite.

The scenario file is a build product: it is generated once by
``simulate_baselines.py`` from the cleaned ACN sessions and then frozen, because
every algorithm must be compared on byte-identical inputs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as cfg

REQUIRED_COLUMNS = [
    "ev_id",
    "arrival_hour",
    "departure_hour",
    "arrival_slot",
    "departure_slot",
    "required_energy_kwh",
    "max_power_kw",
]


def load_scenario(scenario_path=None):
    """Load and validate the EV scenario; returns a DataFrame of EV requests."""
    path = cfg.DEFAULT_SCENARIO if scenario_path is None else Path(scenario_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Scenario missing columns: {missing}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Scenario contains missing values.")

    if df["ev_id"].duplicated().any():
        raise ValueError("Duplicate EV IDs found.")

    if (df["departure_slot"] <= df["arrival_slot"]).any():
        raise ValueError("Invalid arrival/departure slots.")

    if (df["required_energy_kwh"] <= 0).any():
        raise ValueError("Energy requirements must be positive.")

    if (df["max_power_kw"] <= 0).any():
        raise ValueError("Maximum power must be positive.")

    if (df["arrival_slot"] < 0).any() or (df["departure_slot"] > cfg.N_SLOTS).any():
        raise ValueError(
            f"Scenario windows must lie inside [0, {cfg.N_SLOTS}] slots."
        )

    return df


if __name__ == "__main__":
    scenario = load_scenario()

    print("First five EVs:")
    print(scenario.head().to_string(index=False))
    print("\nTotal required energy (kWh):",
          round(scenario["required_energy_kwh"].sum(), 3))
