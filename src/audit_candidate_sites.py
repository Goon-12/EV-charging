
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_SITES = [
    ("caltech", "LIGO_01"),
    ("caltech", "N_Wilson_Garage_01"),
    ("caltech", "S_Wilson_Garage_01"),
    ("office_01", "Parking_Lot_01"),
]

records = []

for provider, site in CANDIDATE_SITES:
    site_dir = TIMESERIES_DIR / provider / site

    if not site_dir.exists():
        print("Site folder not found:", site_dir)
        continue

    files = sorted(site_dir.glob("*.csv.gz"))[:100]
    print(f"\n{site}: checking {len(files)} sessions")

    for file in files:
        try:
            df = pd.read_csv(
                file,
                compression="gzip",
                index_col=0,
                parse_dates=True
            )

            if df.empty:
                continue

            power = pd.to_numeric(
                df.get("Power (kW)", pd.Series(dtype=float)),
                errors="coerce"
            )
            energy = pd.to_numeric(
                df.get("Energy Delivered (kWh)", pd.Series(dtype=float)),
                errors="coerce"
            )
            current = pd.to_numeric(
                df.get("Charging Current (A)", pd.Series(dtype=float)),
                errors="coerce"
            )

            valid_power = power.dropna()
            # Positive measured power during charging.
            positive_power = valid_power[valid_power > 0]

            energy_valid = energy.dropna()
            energy_delta = energy.diff()

            records.append({
                "site": site,
                "file": file.name,
                "rows": len(df),
                "power_valid_pct": 100 * power.notna().mean(),
                "energy_valid_pct": 100 * energy.notna().mean(),
                "current_valid_pct": 100 * current.notna().mean(),
                "median_measured_power_kw": (
                    positive_power.median()
                    if not positive_power.empty else np.nan
                ),
                "p95_measured_power_kw": (
                    positive_power.quantile(0.95)
                    if not positive_power.empty else np.nan
                ),
                "max_measured_power_kw": (
                    positive_power.max()
                    if not positive_power.empty else np.nan
                ),
                "max_current_a": current.max(),
                "energy_starts_at_zero": (
                    not energy_valid.empty
                    and abs(energy_valid.iloc[0]) < 1e-9
                ),
                "max_energy_reading_kwh": (
                    energy_valid.max()
                    if not energy_valid.empty else np.nan
                ),
                "negative_energy_changes": int(
                    energy_delta.lt(-1e-9).sum()
                ),
            })

        except Exception as exc:
            print(f"Skipped {file.name}: {exc}")

audit = pd.DataFrame(records)

output_file = OUTPUT_DIR / "candidate_site_power_audit.csv"
audit.to_csv(output_file, index=False)

print("\nTotal sessions inspected:", len(audit))

if not audit.empty:
    print("\nSummary by site:")
    print(
        audit.groupby("site").agg(
            sessions=("file", "count"),
            median_power_completeness_pct=("power_valid_pct", "median"),
            median_energy_completeness_pct=("energy_valid_pct", "median"),
            median_positive_power_kw=("median_measured_power_kw", "median"),
            median_p95_power_kw=("p95_measured_power_kw", "median"),
            median_peak_power_kw=("max_measured_power_kw", "median"),
            sessions_starting_at_zero=("energy_starts_at_zero", "sum"),
            sessions_with_negative_energy_changes=(
                "negative_energy_changes",
                lambda x: (x > 0).sum()
            )
        ).round(3).to_string()
    )

print("\nSaved report:", output_file)
