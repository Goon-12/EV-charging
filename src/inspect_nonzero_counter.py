
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "ACN-Data-Static-main"
SITE_DIR = DATA_DIR / "time series data" / "office_01" / "Parking_Lot_01"

files = sorted(SITE_DIR.glob("*.csv.gz"))

found = False

for file in files:
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
    ).dropna()

    if energy.empty:
        continue

    if abs(energy.iloc[0]) > 1e-9:
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        print("Selected file:", file.name)
        print("Rows:", len(df))
        print("First energy reading:", energy.iloc[0])
        print("Minimum energy reading:", energy.min())
        print("Maximum energy reading:", energy.max())
        print("Last energy reading:", energy.iloc[-1])

        columns = [
            c for c in [
                "Charging Current (A)",
                "Voltage (V)",
                "Charging State",
                "Energy Delivered (kWh)",
                "Power (kW)"
            ] if c in df.columns
        ]

        print("\nFirst 10 rows:")
        print(df[columns].head(10).to_string())

        print("\nLast 10 rows:")
        print(df[columns].tail(10).to_string())

        print("\nCharging states:")
        print(df["Charging State"].value_counts(dropna=False).to_string())

        found = True
        break

if not found:
    print("No session with a nonzero initial energy reading was found.")
