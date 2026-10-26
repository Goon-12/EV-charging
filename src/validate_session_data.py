
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"

files = sorted(TIMESERIES_DIR.rglob("*.csv.gz"))

if not files:
    raise FileNotFoundError("No ACN session files found.")

# Use the same session as our previous analysis.
target = next(
    (f for f in files if f.name == "2-39-123-23-2018-05-01T17-11-32-081386.csv.gz"),
    None
)

if target is None:
    raise FileNotFoundError("The previously analyzed session was not found.")

df = pd.read_csv(
    target,
    compression="gzip",
    index_col=0,
    parse_dates=True
)

df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]

energy = pd.to_numeric(df["Energy Delivered (kWh)"], errors="coerce")
current = pd.to_numeric(df["Charging Current (A)"], errors="coerce")

interval_seconds = df.index.to_series().diff().dt.total_seconds()
energy_change = energy.diff()

valid = (
    interval_seconds.gt(0)
    & energy_change.ge(0)
    & interval_seconds.notna()
    & energy_change.notna()
)

estimated_power = (
    energy_change[valid] / (interval_seconds[valid] / 3600)
)

print("SESSION QUALITY REPORT")
print("-" * 45)
print("Rows:", len(df))
print("Time span:", df.index.min(), "to", df.index.max())
print("\nMeasurement interval (seconds):")
print(interval_seconds.describe(percentiles=[0.5, 0.9, 0.99]))

print("\nCharging current (A):")
print(current.describe(percentiles=[0.5, 0.9, 0.99]))
print("Missing current readings:", current.isna().sum())

print("\nEnergy change (kWh) per interval:")
print(energy_change.describe(percentiles=[0.5, 0.9, 0.99]))
print("Negative energy changes:", energy_change.lt(0).sum())

print("\nEstimated power (kW):")
print(estimated_power.describe(percentiles=[0.5, 0.9, 0.99]))

# Examine the 10 largest estimated-power intervals.
report = pd.DataFrame({
    "timestamp": df.index,
    "current_A": current.to_numpy(),
    "energy_kWh": energy.to_numpy(),
    "energy_change_kWh": energy_change.to_numpy(),
    "interval_seconds": interval_seconds.to_numpy(),
    "estimated_power_kW": (
        energy_change / (interval_seconds / 3600)
    ).to_numpy()
})

print("\n10 largest estimated-power intervals:")
print(
    report.nlargest(10, "estimated_power_kW")[
        [
            "timestamp",
            "current_A",
            "energy_change_kWh",
            "interval_seconds",
            "estimated_power_kW"
        ]
    ].to_string(index=False)
)
