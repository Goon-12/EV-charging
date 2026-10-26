
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS_PER_SITE = 100
records = []

# Each immediate subfolder of a provider is a site.
site_dirs = sorted(
    p for p in TIMESERIES_DIR.glob("*/*")
    if p.is_dir()
)

print("Sites found:", len(site_dirs))

for site_dir in site_dirs:
    files = sorted(site_dir.glob("*.csv.gz"))[:SESSIONS_PER_SITE]
    print(f"\n{site_dir.name}: checking {len(files)} files")

    for file in files:
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
                df["Energy Delivered (kWh)"], errors="coerce"
            )
            current = pd.to_numeric(
                df["Charging Current (A)"], errors="coerce"
            )
            state = df["Charging State"].astype(str).str.upper()

            energy_change = energy.diff()
            seconds = df.index.to_series().diff().dt.total_seconds()

            # Ignore negative deltas and the unplugged/reset row.
            valid_delta = (
                energy_change.ge(0)
                & seconds.gt(0)
                & energy_change.notna()
                & ~state.str.contains("UNPLUGGED", na=False)
            )

            power = energy_change / (seconds / 3600)
            valid_power = power[valid_delta].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            # For ordinary sessions starting at zero, the maximum
            # cumulative reading is the total delivered energy.
            # Flag sessions that may need special review.
            starts_at_zero = (
                energy.notna().any()
                and abs(energy.dropna().iloc[0]) < 1e-9
            )
            has_negative_delta = bool(energy_change.lt(0).any())

            total_energy = (
                float(energy.max())
                if starts_at_zero and energy.notna().any()
                else np.nan
            )

            records.append({
                "file": file.name,
                "site": str(site_dir.relative_to(TIMESERIES_DIR)),
                "start_time_utc": df.index.min(),
                "end_time_utc": df.index.max(),
                "duration_minutes": (
                    df.index.max() - df.index.min()
                ).total_seconds() / 60,
                "energy_delivered_kwh": total_energy,
                "mean_estimated_power_kw": (
                    valid_power.mean() if len(valid_power) else np.nan
                ),
                "p95_estimated_power_kw": (
                    valid_power.quantile(0.95)
                    if len(valid_power) else np.nan
                ),
                "max_current_a": current.max(),
                "measurement_rows": len(df),
                "negative_energy_change": has_negative_delta,
                "energy_starts_at_zero": starts_at_zero
            })

        except Exception as exc:
            print(f"Skipped {file.name}: {exc}")

summary = pd.DataFrame(records)
output_csv = OUTPUT_DIR / "site_comparison_sample.csv"
summary.to_csv(output_csv, index=False)

print("\nSample saved to:", output_csv)
print("Total sessions summarized:", len(summary))

if not summary.empty:
    print("\nSession counts by site:")
    print(summary["site"].value_counts())

    print("\nMedian metrics by site:")
    print(
        summary.groupby("site")[
            [
                "duration_minutes",
                "energy_delivered_kwh",
                "mean_estimated_power_kw",
                "max_current_a"
            ]
        ].median().round(2).to_string()
    )

    print("\nSessions with negative energy changes:")
    print(summary["negative_energy_change"].sum())

    # Compare energy distributions across sites.
    energy_data = summary.dropna(
        subset=["energy_delivered_kwh"]
    )

    if not energy_data.empty:
        plt.figure(figsize=(12, 6))
        energy_data.boxplot(
            column="energy_delivered_kwh",
            by="site",
            rot=35
        )
        plt.title("Energy Delivered by ACN Site")
        plt.suptitle("")
        plt.xlabel("Site")
        plt.ylabel("Energy delivered (kWh)")
        plt.tight_layout()

        graph_path = RESULTS_DIR / "energy_by_site.png"
        plt.savefig(graph_path, dpi=200)
        plt.close()

        print("\nGraph saved to:", graph_path)
