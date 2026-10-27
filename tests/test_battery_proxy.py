"""
Battery-stress proxy properties.

The proxy is ``sum(E * (P/P_max)^2) / sum(E)``: an energy-weighted squared
charging rate. It is NOT a battery-aging model and must never be described as
one. These tests pin the arithmetic that ``test_battery_proxy.py`` and
``test_battery_proxy_behaviour.py`` checked, now against the shared evaluator
instead of two private copies of the formula.
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import make_inputs
from multiobjective_core import evaluate_schedule


def stress_for(powers, pmax, required):
    """Run one EV's power profile through the shared evaluator."""
    schedule = np.asarray(powers, dtype=float).reshape(1, -1)
    ev, solar = make_inputs(
        required=[required], pmax=[pmax],
        arrival=[0], departure=[len(powers)], n_slots=len(powers),
    )
    return evaluate_schedule(schedule, ev, solar)["battery_stress_proxy"]


def test_half_power_charging_scores_a_quarter():
    assert stress_for([2.0, 2.0, 0.0, 0.0], pmax=4.0, required=1.0) == pytest.approx(0.25)


def test_full_power_charging_scores_one():
    assert stress_for([4.0, 4.0, 0.0, 0.0], pmax=4.0, required=2.0) == pytest.approx(1.0)


def test_lower_power_is_preferred_at_equal_energy():
    """3.5 kWh at half rate must score below 3.5 kWh at full rate."""
    high = stress_for([7.0, 7.0, 0.0, 0.0], pmax=7.0, required=3.5)
    low = stress_for([3.5, 3.5, 3.5, 3.5], pmax=7.0, required=3.5)
    assert low < high
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(1.0)


def test_no_energy_gives_zero_stress():
    assert stress_for([0.0, 0.0, 0.0, 0.0], pmax=4.0, required=1.0) == 0.0


def test_proxy_cannot_exceed_one(ev, solar):
    rng = np.random.default_rng(2)
    from multiobjective_core import repair_schedule

    schedule = repair_schedule(rng.random((len(ev), len(solar))), ev, solar)
    assert evaluate_schedule(schedule, ev, solar)["battery_stress_proxy"] <= 1.0 + 1e-9
