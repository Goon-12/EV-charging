
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
                index_col=0
            )

            if df.empty:
                continue

            row = {
                "site": str(site_dir.relative_to(TIMESERIES_DIR)),
                "file": file.name,
                "rows": len(df)
            }

            for col in [
                "Energy Delivered (kWh)",
                "Charging Current (A)",
                "Voltage (V)",
                "Power (kW)",
                "Charging State"
            ]:
                if col in df.columns:
                    values = df[col]
                    row[f"{col}_valid_pct"] = (
                        values.notna().mean() * 100
                    )
                else:
                    row[f"{col}_valid_pct"] = 0.0

            if "Energy Delivered (kWh)" in df.columns:
                energy = pd.to_numeric(
                    df["Energy Delivered (kWh)"],
                    errors="coerce"
                ).dropna()

                row["energy_first_kwh"] = (
                    energy.iloc[0] if not energy.empty else None
                )
                row["energy_max_kwh"] = (
                    energy.max() if not energy.empty else None
                )
                row["energy_min_kwh"] = (
                    energy.min() if not energy.empty else None
                )
            else:
                row["energy_first_kwh"] = None
                row["energy_max_kwh"] = None
                row["energy_min_kwh"] = None

            records.append(row)

        except Exception as exc:
            print(f"Skipped {file.name}: {exc}")

quality = pd.DataFrame(records)

output_dir = PROJECT_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "site_data_quality.csv"
quality.to_csv(output_file, index=False)

percentage_cols = [
    "Energy Delivered (kWh)_valid_pct",
    "Charging Current (A)_valid_pct",
    "Voltage (V)_valid_pct",
    "Power (kW)_valid_pct",
    "Charging State_valid_pct"
]

print("Files checked:", len(quality))
print("\nMedian data completeness by site (%):")
print(
    quality.groupby("site")[percentage_cols]
    .median()
    .round(1)
    .to_string()
)

print("\nMedian maximum energy reading by site (kWh):")
print(
    quality.groupby("site")["energy_max_kwh"]
    .median()
    .round(3)
    .to_string()
)

print("\nSaved detailed report:", output_file)
