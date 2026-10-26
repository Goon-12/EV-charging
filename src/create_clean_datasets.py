
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
SITE_DIR = DATA_DIR / "time series data" / "office_01" / "Parking_Lot_01"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 300
RANDOM_SEED = 42

files = sorted(SITE_DIR.glob("*.csv.gz"))

if not files:
    raise FileNotFoundError(f"No sessions found in {SITE_DIR}")

rng = np.random.default_rng(RANDOM_SEED)
selected = rng.choice(
    files,
    size=min(SAMPLE_SIZE, len(files)),
    replace=False
)

records = []

for i, file in enumerate(selected, start=1):
    try:
        df = pd.read_csv(
            file,
            compression="gzip",
            index_col=0,
            parse_dates=True
        )

        if df.empty:
            continue

        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        energy = pd.to_numeric(
            df["Energy Delivered (kWh)"], errors="coerce"
        )
        power = pd.to_numeric(
            df["Power (kW)"], errors="coerce"
        )
        current = pd.to_numeric(
            df["Charging Current (A)"], errors="coerce"
        )
        state = df["Charging State"].astype(str).str.upper()

        valid_energy = energy.dropna()
        valid_power = power.dropna()

        if valid_energy.empty:
            continue

        # Provisional total for this individual session.
        # The counter may reset at unplugging, so use the maximum
        # reading rather than the final reading or sum of differences.
        energy_total = float(valid_energy.max())

        energy_delta = energy.diff()
        negative = energy_delta < -1e-9
        negative_unplugged = (
            negative & state.str.contains("UNPLUGGED", na=False)
        )
        negative_other = negative & ~negative_unplugged

        positive_power = valid_power[valid_power > 0]

        # Active charging is defined here by positive measured power.
        # This is an operational estimate, not a vehicle's connection time.
        active = power > 0
        active_times = df.index[active.fillna(False)]

        active_duration_minutes = (
            (active_times.max() - active_times.min()).total_seconds() / 60
            if len(active_times) >= 2 else 0.0
        )

        connection_duration_minutes = (
            df.index.max() - df.index.min()
        ).total_seconds() / 60

        power_completeness = float(power.notna().mean())
        energy_completeness = float(energy.notna().mean())

        # Flag unusual energy behavior; do not silently delete it.
        quality_flag = (
            "review"
            if negative_other.any()
            or power_completeness < 0.90
            or energy_completeness < 0.90
            else "provisional_ok"
        )

        records.append({
            "session_file": file.name,
            "site": "office_01/Parking_Lot_01",
            "start_time_utc": df.index.min(),
            "end_time_utc": df.index.max(),
            "connection_duration_minutes": connection_duration_minutes,
            "active_charging_span_minutes": active_duration_minutes,
            "energy_delivered_kwh_provisional": energy_total,
            "mean_positive_power_kw": (
                positive_power.mean()
                if not positive_power.empty else np.nan
            ),
            "p95_positive_power_kw": (
                positive_power.quantile(0.95)
                if not positive_power.empty else np.nan
            ),
            "peak_measured_power_kw": (
                positive_power.max()
                if not positive_power.empty else np.nan
            ),
            "max_current_a": current.max(),
            "power_completeness_pct": 100 * power_completeness,
            "energy_completeness_pct": 100 * energy_completeness,
            "negative_changes_at_unplugged": int(
                negative_unplugged.sum()
            ),
            "negative_changes_other_states": int(
                negative_other.sum()
            ),
            "quality_flag": quality_flag,
            "measurement_rows": len(df)
        })

    except Exception as exc:
        print(f"Skipped {file.name}: {exc}")

    if i % 50 == 0:
        print(f"Processed {i}/{len(selected)} sessions")

clean = pd.DataFrame(records)

output_file = OUTPUT_DIR / "clean_office_sessions.csv"
clean.to_csv(output_file, index=False)

print("\nCleaned dataset saved to:", output_file)
print("Sessions processed:", len(clean))

if not clean.empty:
    print("\nQuality flags:")
    print(clean["quality_flag"].value_counts().to_string())

    print("\nSummary statistics:")
    print(
        clean[
            [
                "connection_duration_minutes",
                "active_charging_span_minutes",
                "energy_delivered_kwh_provisional",
                "mean_positive_power_kw",
                "peak_measured_power_kw"
            ]
        ].describe().round(3).to_string()
    )
