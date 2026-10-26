from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep these settings consistent with the EV simulation.
STEP_HOURS = 0.25
HORIZON_HOURS = 36
N_SLOTS = int(HORIZON_HOURS / STEP_HOURS)

# Initial synthetic assumption for the apartment's solar installation.
SOLAR_PEAK_KW = 50.0

# Slot-start times: 0.00, 0.25, ..., 35.75 hours.
time_hours = np.arange(N_SLOTS) * STEP_HOURS
hour_of_day = time_hours % 24.0

# Smooth daylight profile: zero overnight, rising after 06:00,
# peaking at noon, and falling to zero by 18:00.
solar_kw = np.zeros(N_SLOTS, dtype=float)
daylight = (hour_of_day >= 6.0) & (hour_of_day <= 18.0)

solar_kw[daylight] = SOLAR_PEAK_KW * np.sin(
    np.pi * (hour_of_day[daylight] - 6.0) / 12.0
)
solar_kw = np.maximum(solar_kw, 0.0)

# Energy generated during each 15-minute slot.
solar_energy_kwh = solar_kw * STEP_HOURS

solar_df = pd.DataFrame({
    "time_hours": time_hours,
    "hour_of_day": hour_of_day,
    "solar_power_kw": solar_kw,
    "solar_energy_kwh": solar_energy_kwh,
})

csv_path = DATA_DIR / "solar_profile_36h.csv"
solar_df.to_csv(csv_path, index=False)

# Plot the full 36-hour profile.
plt.figure(figsize=(11, 5))
plt.plot(
    solar_df["time_hours"],
    solar_df["solar_power_kw"],
    label=f"Synthetic solar generation ({SOLAR_PEAK_KW:.0f} kW peak)",
)
plt.xlabel("Hours from simulation start")
plt.ylabel("Solar power available (kW)")
plt.title("Synthetic Solar Generation Profile — 36-Hour Simulation")
plt.xticks(np.arange(0, HORIZON_HOURS + 1, 3))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = RESULTS_DIR / "solar_profile_36h.png"
plt.savefig(plot_path, dpi=200)
plt.close()

# Basic integrity checks.
assert len(solar_df) == N_SLOTS, "Unexpected number of time slots."
assert np.isfinite(solar_kw).all(), "Solar profile contains invalid values."
assert (solar_kw >= 0).all(), "Solar generation cannot be negative."
assert solar_kw.max() <= SOLAR_PEAK_KW + 1e-9, "Solar peak exceeds assumption."
assert np.all(solar_kw[(hour_of_day < 6) | (hour_of_day > 18)] == 0), (
    "Solar should be zero outside the daylight window."
)

daily_energy = []
for day_start in (0.0, 24.0):
    mask = (time_hours >= day_start) & (time_hours < day_start + 24.0)
    daily_energy.append(float(solar_energy_kwh[mask].sum()))

print("\nSOLAR PROFILE CREATED")
print("=" * 55)
print(f"Simulation horizon: {HORIZON_HOURS} hours")
print(f"Time step: {STEP_HOURS} hours ({STEP_HOURS * 60:.0f} minutes)")
print(f"Number of slots: {N_SLOTS}")
print(f"Assumed solar peak: {SOLAR_PEAK_KW:.1f} kW")
print(f"Maximum generated power: {solar_kw.max():.3f} kW")
print(f"Day 1 generated energy: {daily_energy[0]:.3f} kWh")
print(f"Day 2 generated energy (partial): {daily_energy[1]:.3f} kWh")
print(f"Solar CSV saved to: {csv_path}")
print(f"Solar graph saved to: {plot_path}")
print("\nNote: this is a synthetic illustrative profile, not measured solar data.")
