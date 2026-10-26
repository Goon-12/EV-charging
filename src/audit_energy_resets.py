
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"

records = []

site_dirs = sorted(
    p for p in TIMESERIES_DIR.glob("*/*")
    if p.is_dir()
)

for site_dir in site_dirs:
    files = sorted(site_dir.glob("*.csv.gz"))[:100]

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

            energy = pd.to_numeric(
                df["Energy Delivered (kWh)"], errors="coerce"
            )
            state = df["Charging State"].astype(str).str.upper()

            delta = energy.diff()
            negative = delta < -1e-9

            unplugged_negative = (
                negative & state.str.contains("UNPLUGGED", na=False)
            )
            other_negative = negative & ~unplugged_negative

            valid_energy = energy.dropna()
            starts_at_zero = (
                not valid_energy.empty
                and abs(valid_energy.iloc[0]) < 1e-9
            )

            records.append({
                "site": str(site_dir.relative_to(TIMESERIES_DIR)),
                "file": file.name,
                "starts_at_zero": starts_at_zero,
                "negative_changes": int(negative.sum()),
                "negative_at_unplugged": int(unplugged_negative.sum()),
                "negative_other_states": int(other_negative.sum()),
                "max_energy_kwh": (
                    valid_energy.max() if not valid_energy.empty else None
                ),
                "measurement_rows": len(df)
            })

        except Exception as exc:
            print("Skipped", file.name, ":", exc)

audit = pd.DataFrame(records)

if audit.empty:
    raise ValueError("No sessions could be audited.")

output_dir = PROJECT_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "energy_reset_audit.csv"
audit.to_csv(output_file, index=False)

print("Sessions audited:", len(audit))
print("\nOverall reset audit:")
print(audit[
    [
        "starts_at_zero",
        "negative_changes",
        "negative_at_unplugged",
        "negative_other_states"
    ]
].sum(numeric_only=True))

print("\nSummary by site:")
print(
    audit.groupby("site").agg(
        sessions=("file", "count"),
        start_at_zero=("starts_at_zero", "sum"),
        sessions_with_negative_changes=(
            "negative_changes", lambda x: (x > 0).sum()
        ),
        unplugged_resets=("negative_at_unplugged", "sum"),
        other_negative_changes=("negative_other_states", "sum"),
        median_max_energy_kwh=("max_energy_kwh", "median")
    ).round(3).to_string()
)

print("\nSaved audit:", output_file)
