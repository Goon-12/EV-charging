"""
Rolling-horizon controller.

The controller plans over a look-ahead window, executes only its first slot and
replans. Two things went wrong before Tier 0 and both are pinned here:

1. every window rebuilt its tariff from slot 0, so the controller believed it was
   always just after midnight and applied the off-peak rate to the evening peak;
2. the deadline-urgency term was scaled by the window width instead of the slots
   actually remaining, which compressed urgency towards zero and made the
   controller defer far-deadline EVs.
"""
from __future__ import annotations

import numpy as np
import pytest

import config as cfg
from multiobjective_core import constraint_report, make_windows, tariff_vector
from run_rolling_horizon import optimize_rolling

TINY = dict(window_slots=12, population_size=4, generations=1, particles=3, iterations=1)


def test_window_tariff_equals_the_full_horizon_slice():
    """A window's tariff must match what the one-shot solver sees for those slots."""
    full = tariff_vector(cfg.N_SLOTS)
    for start in (0, 24, 60, 68, 88, 120, 140):
        width = min(24, cfg.N_SLOTS - start)
        window = tariff_vector(width, slot_offset=start)
        assert np.array_equal(window, full[start:start + width])


def test_every_slot_in_a_window_gets_its_real_tariff():
    """Regression: without an offset the whole evening window came back off-peak."""
    evening = 68  # slot 68 is 17:00, the first peak slot

    assert tariff_vector(cfg.N_SLOTS)[evening] == cfg.TARIFF_PEAK
    assert tariff_vector(4, slot_offset=evening)[0] == cfg.TARIFF_PEAK
    # The pre-Tier-0 call, with no offset, returned the off-peak rate instead.
    assert tariff_vector(4)[0] == cfg.TARIFF_OFFPEAK


@pytest.mark.parametrize("method", ["ga", "pso"])
def test_rolling_horizon_trajectory_is_feasible(method, ev, solar):
    schedule = optimize_rolling(ev, solar, method=method, **TINY)

    report = constraint_report(schedule, ev, solar)
    assert report["total"] == 0, report
    assert schedule.sum() > 0


@pytest.mark.parametrize("method", ["ga", "pso"])
def test_rolling_horizon_only_charges_inside_connection_windows(method, ev, solar):
    schedule = optimize_rolling(ev, solar, method=method, **TINY)
    arrival, departure = make_windows(ev, len(solar))

    for i in range(len(ev)):
        assert np.all(schedule[i, : arrival[i]] == 0)
        assert np.all(schedule[i, departure[i]:] == 0)


@pytest.mark.parametrize("method", ["ga", "pso"])
def test_rolling_horizon_respects_site_limits(method, ev, solar):
    schedule = optimize_rolling(ev, solar, method=method, **TINY)

    assert (schedule > 1e-9).sum(axis=0).max() <= cfg.N_CHARGERS
    grid_import = np.maximum(schedule.sum(axis=0) - np.maximum(solar, 0), 0)
    assert (grid_import <= cfg.MAX_GRID_KW + 1e-6).all()


def test_rolling_horizon_rejects_an_unknown_method(ev, solar):
    with pytest.raises(ValueError):
        optimize_rolling(ev, solar, method="annealing", **TINY)
