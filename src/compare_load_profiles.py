
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

N_SLOTS = 144
GRID_LIMIT_KW = 125.0
STEP_HOURS = 0.25

def load_profile(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    slot_columns = sorted(
        [c for c in df.columns if c.startswith("slot_")],
        key=lambda c: int(c.split("_")[1])
    )

    if len(slot_columns) != N_SLOTS:
        raise ValueError(
            f"{filename}: expected {N_SLOTS} slots, "
            f"but found {len(slot_columns)}."
        )

    return df[slot_columns].to_numpy(dtype=float).sum(axis=0)

ga_profile = load_profile("ga_best_schedule.csv")
pso_profile = load_profile("pso_best_schedule.csv")

hours = np.arange(N_SLOTS) * STEP_HOURS

plt.figure(figsize=(12, 6))
plt.plot(hours, ga_profile, label="GA", linewidth=2)
plt.plot(hours, pso_profile, label="PSO", linewidth=2)
plt.axhline(
    GRID_LIMIT_KW,
    linestyle="--",
    label="100 kW grid limit"
)

plt.xlabel("Hour of simulation horizon")
plt.ylabel("Total charging load (kW)")
plt.title("EV Charging Load Profiles: GA vs PSO")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = RESULTS_DIR / "ga_pso_load_profiles.png"
plt.savefig(output_path, dpi=150)
plt.close()

print("GA peak demand:", round(float(ga_profile.max()), 3), "kW")
print("PSO peak demand:", round(float(pso_profile.max()), 3), "kW")
print("Graph saved to:", output_path)
