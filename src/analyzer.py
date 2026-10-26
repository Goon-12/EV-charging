
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Find the project and dataset folders.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"

# Create output folders.
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Locate charging-session files.
files = sorted(TIMESERIES_DIR.rglob("*.csv.gz"))

print("Total session files found:", len(files))

if not files:
    raise FileNotFoundError(
        f"No .csv.gz files found in {TIMESERIES_DIR}"
    )

# Start with one session to inspect its measurements.
sample_file = files[0]

df = pd.read_csv(
    sample_file,
    compression="gzip",
    index_col=0,
    parse_dates=True
)

df.index.name = "timestamp"
df = df.sort_index()

print("\nSample file:", sample_file.name)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isna().sum())

# Check whether the power measurement is available.
power_col = "Power (kW)"

if power_col in df.columns:
    power = pd.to_numeric(df[power_col], errors="coerce")

    print("\nPower statistics (kW):")
    print(power.describe())

    # Plot the measured power over time.
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, power, marker=".", linewidth=1)
    plt.xlabel("Timestamp")
    plt.ylabel("Charging power (kW)")
    plt.title("Power Profile of One ACN Charging Session")
    plt.xticks(rotation=30)
    plt.tight_layout()

    graph_path = RESULTS_DIR / "sample_session_power.png"
    plt.savefig(graph_path, dpi=200)
    plt.close()

    print("\nPower graph saved to:", graph_path)
else:
    print(f"\nColumn '{power_col}' was not found.")

# Save the sample in an ordinary CSV format.
sample_output = PROCESSED_DIR / "sample_session.csv"
df.to_csv(sample_output)

print("\nSample CSV saved to:", sample_output)
print("\nInitial data inspection completed.")
