"""
Generate the synthetic solar generation profile.

    .venv/Scripts/python src/create_solar_profile.py

Writes ``data/processed/solar_profile_36h.csv`` and
``results/solar_profile_36h.png``.

Illustrative only: this is a clean sine curve peaking at midday, NOT measured
weather data. It is generated before ``simulate_baselines.py`` because the
baselines and every optimized schedule are evaluated against it.

Tier 0 change: constants moved into ``config.py``. Generated values are
unchanged, so the committed CSV is untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from artifacts import save_load_profile_plot

DAY_START_HOUR = 6.0
DAY_END_HOUR = 18.0
DAY_LENGTH_HOURS = DAY_END_HOUR - DAY_START_HOUR


def build_solar_profile():
    """Return ``(solar_kw, time_hours, hour_of_day)`` for the full horizon."""
    time_hours = np.arange(cfg.N_SLOTS) * cfg.STEP_HOURS
    hour_of_day = time_hours % 24.0

    # Smooth daylight curve: zero overnight, peaking at noon, zero after 18:00.
    solar_kw = np.zeros(cfg.N_SLOTS, dtype=float)
    daylight = (hour_of_day >= DAY_START_HOUR) & (hour_of_day <= DAY_END_HOUR)
    solar_kw[daylight] = cfg.SOLAR_PEAK_KW * np.sin(
        np.pi * (hour_of_day[daylight] - DAY_START_HOUR) / DAY_LENGTH_HOURS
    )
    return np.maximum(solar_kw, 0.0), time_hours, hour_of_day


def main():
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    solar_kw, time_hours, hour_of_day = build_solar_profile()

    frame = pd.DataFrame({
        "time_hours": time_hours,
        "hour_of_day": hour_of_day,
        "solar_power_kw": solar_kw,
        "solar_energy_kwh": solar_kw * cfg.STEP_HOURS,
    })
    frame.to_csv(cfg.DEFAULT_SOLAR, index=False)

    # Integrity checks.
    assert len(frame) == cfg.N_SLOTS, "Unexpected number of time slots."
    assert np.isfinite(solar_kw).all(), "Solar profile contains invalid values."
    assert (solar_kw >= 0).all(), "Solar generation cannot be negative."
    assert solar_kw.max() <= cfg.SOLAR_PEAK_KW + 1e-9, "Solar peak exceeds assumption."
    outside = (hour_of_day < DAY_START_HOUR) | (hour_of_day > DAY_END_HOUR)
    assert np.all(solar_kw[outside] == 0), "Solar must be zero outside daylight."

    daily = [
        float((solar_kw * cfg.STEP_HOURS)[
            (time_hours >= start) & (time_hours < start + 24.0)
        ].sum())
        for start in (0.0, 24.0)
    ]

    save_load_profile_plot(
        cfg.RESULTS_DIR / "solar_profile_36h.png",
        np.zeros((1, cfg.N_SLOTS)),  # no EV load: plot the solar curve alone
        solar_kw,
        f"Synthetic solar generation profile ({cfg.SOLAR_PEAK_KW:.0f} kW peak)",
    )

    print("SOLAR PROFILE CREATED")
    print(f"  slots        : {cfg.N_SLOTS} x {cfg.STEP_HOURS * 60:.0f} min")
    print(f"  assumed peak : {cfg.SOLAR_PEAK_KW:.1f} kW")
    print(f"  day 1 energy : {daily[0]:.3f} kWh")
    print(f"  day 2 energy : {daily[1]:.3f} kWh")
    print(f"  saved        : {cfg.DEFAULT_SOLAR}")
    print("Note: synthetic illustrative profile, not measured solar data.")


if __name__ == "__main__":
    main()
