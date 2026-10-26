
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCENARIO_FILE = PROJECT_DIR / "data" / "processed" / "ev_scenario.csv"

STEP_HOURS = 0.25
HORIZON_HOURS = 36
N_CHARGERS = 20
MAX_GRID_KW = 125.0


def load_scenario():
    if not SCENARIO_FILE.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIO_FILE}"
        )

    df = pd.read_csv(SCENARIO_FILE)

    required_columns = [
        "ev_id",
        "arrival_hour",
        "departure_hour",
        "arrival_slot",
        "departure_slot",
        "required_energy_kwh",
        "max_power_kw",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required_columns].isna().any().any():
        raise ValueError("Scenario contains missing values.")

    if df["ev_id"].duplicated().any():
        raise ValueError("Duplicate EV IDs found.")

    if (df["departure_slot"] <= df["arrival_slot"]).any():
        raise ValueError("Invalid arrival/departure slots.")

    if (df["required_energy_kwh"] <= 0).any():
        raise ValueError("Energy requirements must be positive.")

    if (df["max_power_kw"] <= 0).any():
        raise ValueError("Maximum power must be positive.")

    print(f"Loaded {len(df)} EVs from saved scenario.")
    print(f"Scenario file: {SCENARIO_FILE}")

    return df


if __name__ == "__main__":
    scenario = load_scenario()

    print("\nFirst five EVs:")
    print(scenario.head().to_string(index=False))

    print("\nTotal required energy (kWh):",
          round(scenario["required_energy_kwh"].sum(), 2))
