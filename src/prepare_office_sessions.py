
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
    raise FileNotFoundError(f"No session files found in {SITE_DIR}")

rng = np.random.default_rng(RANDOM_SEED)
selected_files = rng.choice(
    files,
    size=min(SAMPLE_SIZE, len(files)),
    replace=False
)

records = []

for number, file in enumerate(selected_files, start=1):
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
        current = pd.to_numeric(
            df["Charging Current (A)"], errors="coerce"
        )
        power = pd.to_numeric(
            df["Power (kW)"], errors="coerce"
        )
        state = df["Charging State"].astype(str).str.upper()

        energy_valid = energy.dropna()
        energy_delta = energy.diff()

        # Detect negative increments; do not count them as delivered energy.
        negative_delta = energy_delta < -1e-9
        negative_at_unplugged = (
            negative_delta & state.str.contains("UNPLUGGED", na=False)
        )
        negative_other = negative_delta & ~negative_at_unplugged

        starts_at_zero = (
            not energy_valid.empty
            and abs(energy_valid.iloc[0]) < 1e-9
        )

        # Only assign a provisional total if the counter starts at zero.
        # We will validate reset behavior before using these totals.
        energy_total = (
            energy_valid.max()
            if starts_at_zero else np.nan
        )

        positive_power = power[power > 0].dropna()

        records.append({
            "file": file.name,
            "site": "office_01/Parking_Lot_01",
            "start_time_utc": df.index.min(),
            "end_time_utc": df.index.max(),
            "duration_minutes": (
                df.index.max() - df.index.min()
            ).total_seconds() / 60,
            "energy_delivered_kwh_provisional": energy_total,
            "energy_counter_starts_at_zero": starts_at_zero,
            "energy_counter_max_kwh": (
                energy_valid.max() if not energy_valid.empty else np.nan
            ),
            "energy_counter_min_kwh": (
                energy_valid.min() if not energy_valid.empty else np.nan
            ),
            "energy_negative_changes": int(negative_delta.sum()),
            "negative_changes_at_unplugged": int(
                negative_at_unplugged.sum()
            ),
            "negative_changes_other_states": int(negative_other.sum()),
            "power_valid_pct": 100 * power.notna().mean(),
            "current_valid_pct": 100 * current.notna().mean(),
            "median_positive_power_kw": (
                positive_power.median()
                if not positive_power.empty else np.nan
            ),
            "p95_positive_power_kw": (
                positive_power.quantile(0.95)
                if not positive_power.empty else np.nan
            ),
            "max_current_a": current.max(),
            "measurement_rows": len(df)
        })

    except Exception as exc:
        print(f"Skipped {file.name}: {exc}")

    if number % 50 == 0:
        print(f"Checked {number}/{len(selected_files)} selected sessions")

summary = pd.DataFrame(records)

output_file = OUTPUT_DIR / "office_sessions_sample.csv"
summary.to_csv(output_file, index=False)

print("\nSample saved to:", output_file)
print("Sessions processed:", len(summary))

if not summary.empty:
    print("\nData quality:")
    print(
        summary[
            [
                "duration_minutes",
                "energy_delivered_kwh_provisional",
                "power_valid_pct",
                "median_positive_power_kw",
                "energy_negative_changes",
                "negative_changes_other_states"
            ]
        ].describe().round(3).to_string()
    )

    print("\nCounter starts at zero:")
    print(summary["energy_counter_starts_at_zero"].value_counts())

    print("\nSessions with negative changes outside UNPLUGGED:")
    print((summary["negative_changes_other_states"] > 0).sum())
