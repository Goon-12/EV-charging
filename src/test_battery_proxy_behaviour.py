
import numpy as np

STEP_HOURS = 0.25


def battery_stress_proxy(schedule, max_power_kw):
    schedule = np.asarray(schedule, dtype=float)
    max_power_kw = np.asarray(max_power_kw, dtype=float)

    energy = schedule * STEP_HOURS
    total_energy = energy.sum()

    if total_energy <= 0:
        return 0.0

    relative_rate = np.clip(
        schedule / max_power_kw[:, None], 0.0, 1.0
    )

    return float(
        (energy * relative_rate**2).sum() / total_energy
    )


max_power = np.array([7.0])

# Both schedules deliver 3.5 kWh over the same 2-hour window.
# A: 7 kW for two 15-minute slots = 3.5 kWh.
schedule_a = np.array([[
    7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]])

# B: 3.5 kW for four 15-minute slots = 3.5 kWh.
schedule_b = np.array([[
    3.5, 3.5, 3.5, 3.5, 0.0, 0.0, 0.0, 0.0
]])

energy_a = schedule_a.sum() * STEP_HOURS
energy_b = schedule_b.sum() * STEP_HOURS

score_a = battery_stress_proxy(schedule_a, max_power)
score_b = battery_stress_proxy(schedule_b, max_power)

print("BATTERY PROXY BEHAVIOR TEST")
print("=" * 45)
print(f"Schedule A energy: {energy_a:.3f} kWh")
print(f"Schedule B energy: {energy_b:.3f} kWh")
print(f"Schedule A proxy: {score_a:.4f}")
print(f"Schedule B proxy: {score_b:.4f}")

assert np.isclose(energy_a, energy_b), "Energy must match."
assert score_b < score_a, "Lower-power schedule should score lower."

print("PASS: proxy prefers lower-power charging for equal energy.")
