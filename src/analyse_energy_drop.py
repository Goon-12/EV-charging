
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
TIMESERIES_DIR = DATA_DIR / "time series data"

filename = "2-39-123-23-2018-05-01T17-11-32-081386.csv.gz"

matches = list(TIMESERIES_DIR.rglob(filename))

if not matches:
    raise FileNotFoundError(f"Session file not found: {filename}")

df = pd.read_csv(
    matches[0],
    compression="gzip",
    index_col=0,
    parse_dates=True
)

df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()

energy = pd.to_numeric(
    df["Energy Delivered (kWh)"],
    errors="coerce"
)

df["energy_change_kWh"] = energy.diff()

# Locate all negative energy changes.
drop_rows = df.index[df["energy_change_kWh"] < 0]

print("Number of negative energy changes:", len(drop_rows))

for timestamp in drop_rows:
    position = df.index.get_loc(timestamp)
    start = max(0, position - 3)
    end = min(len(df), position + 4)

    print("\nRows around energy decrease:")
    print(
        df.iloc[start:end][
            [
                "Charging Current (A)",
                "Charging State",
                "Energy Delivered (kWh)",
                "energy_change_kWh"
            ]
        ].to_string()
    )

print("\nFirst and last energy readings:")
print("First:", energy.iloc[0])
print("Last:", energy.iloc[-1])
print("Minimum:", energy.min())
print("Maximum:", energy.max())
