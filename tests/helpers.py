"""Small builders for tiny, hand-checkable scenarios."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def make_inputs(required, pmax, arrival, departure, n_slots=4, solar_kw=None):
    """
    Build a minimal ``(ev, solar)`` pair.

    Every argument is per-EV except ``n_slots`` and ``solar_kw``. Slots are
    absolute, so ``arrival``/``departure`` must lie inside ``[0, n_slots]``.
    """
    required = np.atleast_1d(np.asarray(required, dtype=float))
    pmax = np.atleast_1d(np.asarray(pmax, dtype=float))
    arrival = np.atleast_1d(np.asarray(arrival, dtype=int))
    departure = np.atleast_1d(np.asarray(departure, dtype=int))

    ev = pd.DataFrame({
        "ev_id": np.arange(len(required)),
        "arrival_hour": arrival * cfg.STEP_HOURS,
        "departure_hour": departure * cfg.STEP_HOURS,
        "arrival_slot": arrival,
        "departure_slot": departure,
        "required_energy_kwh": required,
        "max_power_kw": pmax,
    })
    solar = (
        np.zeros(n_slots, dtype=float)
        if solar_kw is None
        else np.asarray(solar_kw, dtype=float)
    )
    return ev, solar


def random_priority(rng, n_ev, n_slots):
    """A random priority matrix, the search representation both optimizers use."""
    return rng.random((n_ev, n_slots))
