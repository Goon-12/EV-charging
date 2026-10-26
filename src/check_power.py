
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(TIMESERIES_DIR.rglob("*.csv.gz"))

if not files:
    raise FileNotFoundError("No charging-session files found.")

# Check several files to find a session with valid measurements.
selected_file = None
df = None

for file in files[:500]:
    candidate = pd.read_csv(
        file,
        compression="gzip",
        index_col=0,
        parse_dates=True
    )

    if "Energy Delivered (kWh)" not in candidate.columns:
        continue

    energy = pd.to_numeric(
        candidate["Energy Delivered (kWh)"],
        errors="coerce"
    )

    if energy.notna().sum() >= 10 and energy.max() > energy.min():
        selected_file = file
        df = candidate
        break

if df is None:
    raise ValueError(
        "No suitable session found in the first 500 files."
    )

df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]

energy = pd.to_numeric(
    df["Energy Delivered (kWh)"],
    errors="coerce"
)

# Calculate changes in cumulative energy and elapsed time.
energy_change = energy.diff()
time_change_hours = (
    df.index.to_series().diff().dt.total_seconds() / 3600
)

# Reject invalid intervals and negative energy increments.
valid = (
    energy_change.ge(0)
    & time_change_hours.gt(0)
    & energy_change.notna()
    & time_change_hours.notna()
)

estimated_power = (
    energy_change[valid] / time_change_hours[valid]
)

print("Selected session:", selected_file.name)
print("Number of measurement rows:", len(df))
print("Valid power estimates:", len(estimated_power))
print("Energy delivered (kWh):", round(energy.max() - energy.min(), 3))

print("\nEstimated average charging power (kW):")
print(estimated_power.describe())

# Plot estimated power over time.
plt.figure(figsize=(11, 5))
plt.plot(
    df.index[valid],
    estimated_power,
    linewidth=1
)
plt.xlabel("Timestamp (UTC)")
plt.ylabel("Estimated average power (kW)")
plt.title("Estimated Charging Power from Energy Measurements")
plt.tight_layout()

output_path = RESULTS_DIR / "estimated_charging_power.png"
plt.savefig(output_path, dpi=200)
plt.close()

print("\nGraph saved to:", output_path)
