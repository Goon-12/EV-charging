
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start with 500 sessions to test the processing pipeline.
# We can expand after checking the output.
MAX_SESSIONS = 500

files = sorted(TIMESERIES_DIR.rglob("*.csv.gz"))[:MAX_SESSIONS]

if not files:
    raise FileNotFoundError("No .csv.gz files found.")

records = []

for i, file in enumerate(files, start=1):
    try:
        df = pd.read_csv(
            file,
            compression="gzip",
            index_col=0,
            parse_dates=True
        )

        if df.empty or "Energy Delivered (kWh)" not in df.columns:
            continue

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        energy = pd.to_numeric(
            df["Energy Delivered (kWh)"],
            errors="coerce"
        )

        current = pd.to_numeric(
            df["Charging Current (A)"],
            errors="coerce"
        )

        state = df["Charging State"].astype(str).str.upper()

        # Estimate total delivered energy from valid cumulative readings.
        # Counter resets are excluded from interval differences.
        energy_change = energy.diff()
        unplugged = state.str.contains("UNPLUGGED", na=False)

        valid_delta = (
            energy_change.ge(0)
            & energy.notna()
            & energy.shift().notna()
            & ~unplugged
        )

        total_energy = energy_change[valid_delta].sum()

        # The cumulative counter's maximum provides a useful cross-check
        # for sessions where it starts at zero and resets only at the end.
        if (
            energy.notna().any()
            and energy.iloc[0] == 0
            and energy.max() >= total_energy
        ):
            total_energy = energy.max()

        duration_minutes = (
            (df.index.max() - df.index.min()).total_seconds() / 60
        )

        # Calculate interval-average power without crossing resets.
        seconds = df.index.to_series().diff().dt.total_seconds()
        power = energy_change / (seconds / 3600)

        valid_power = (
            energy_change.ge(0)
            & seconds.gt(0)
            & energy_change.notna()
            & ~unplugged
        )

        estimated_power = power[valid_power].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        records.append({
            "file": file.name,
            "site": str(file.parent.relative_to(TIMESERIES_DIR)),
            "start_time_utc": df.index.min(),
            "end_time_utc": df.index.max(),
            "duration_minutes": duration_minutes,
            "energy_delivered_kwh": total_energy,
            "mean_estimated_power_kw": (
                estimated_power.mean()
                if not estimated_power.empty else np.nan
            ),
            "p95_estimated_power_kw": (
                estimated_power.quantile(0.95)
                if not estimated_power.empty else np.nan
            ),
            "max_current_a": current.max(),
            "measurement_rows": len(df),
            "negative_energy_changes": int(
                energy_change.lt(0).sum()
            )
        })

    except Exception as exc:
        print(f"Skipped {file.name}: {exc}")

    if i % 100 == 0:
        print(f"Processed {i}/{len(files)} files")

summary = pd.DataFrame(records)

output_file = OUTPUT_DIR / "session_summary_500.csv"
summary.to_csv(output_file, index=False)

print("\nSummary created:", output_file)
print("Sessions summarized:", len(summary))

if not summary.empty:
    print("\nFirst five sessions:")
    print(summary.head().to_string(index=False))

    print("\nSummary statistics:")
    print(
        summary[
            [
                "duration_minutes",
                "energy_delivered_kwh",
                "mean_estimated_power_kw",
                "max_current_a"
            ]
        ].describe().round(3)
    )

    print("\nSessions by site:")
    print(summary["site"].value_counts())
