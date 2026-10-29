"""
Raw ACN session quality audit (from ``validate_session_data.py``).

The raw dataset is roughly a gigabyte of gzipped per-session traces and is not
committed (see ``.gitignore``), so this module skips unless the data is present
locally. It is kept because ``data/processed/clean_office_sessions.csv`` - and
therefore the whole scenario - is derived from it, and the derived product is
only as good as the checks in ``create_clean_datasets.py``.

NOTE: the assertions here could not be executed during the Tier 0 work because
the raw dataset is not in the repository. They are deliberately conservative and
describe shape rather than exact values.
"""
from __future__ import annotations

import pandas as pd
import pytest

import config as cfg

TIMESERIES_DIR = (
    cfg.RAW_DATA_DIR / "ACN-Data-Static-main" / "time series data"
)
SESSION_FILE = TIMESERIES_DIR / "office_01" / "Parking_Lot_01"

requires_raw_data = pytest.mark.skipif(
    not TIMESERIES_DIR.exists(),
    reason="raw ACN dataset not present (not committed - see .gitignore)",
)


def find_session_files():
    return sorted(TIMESERIES_DIR.rglob("*.csv.gz"))


@requires_raw_data
def test_session_files_exist():
    assert find_session_files(), f"no session files under {TIMESERIES_DIR}"


@requires_raw_data
def test_one_session_parses_into_a_usable_power_series():
    """Interval-based power must be derivable from an arbitrary real session."""
    target = find_session_files()[0]

    frame = pd.read_csv(target, compression="gzip", index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]

    assert len(frame) > 1
    assert "Energy Delivered (kWh)" in frame.columns
    assert "Charging Current (A)" in frame.columns

    energy = pd.to_numeric(frame["Energy Delivered (kWh)"], errors="coerce")
    interval_seconds = frame.index.to_series().diff().dt.total_seconds()
    energy_change = energy.diff()

    valid = interval_seconds.gt(0) & energy_change.ge(0)
    assert valid.sum() > 0

    power = energy_change[valid] / (interval_seconds[valid] / 3600)
    assert power.notna().any()


def test_derived_cleaned_sessions_are_available():
    """The committed build product the scenario is sampled from."""
    assert cfg.CLEAN_SESSIONS.exists()
    cleaned = pd.read_csv(cfg.CLEAN_SESSIONS)
    assert len(cleaned) >= cfg.N_EV
    assert "energy_delivered_kwh_provisional" in cleaned.columns
    assert "mean_positive_power_kw" in cleaned.columns
