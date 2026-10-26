
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_schedule import evaluate_schedule

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "clean_office_sessions.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducible simulation settings
SEED = 42
N_EV = 100
N_CHARGERS = 20
STEP_HOURS = 0.25
HORIZON_HOURS = 36
MAX_GRID_KW = 125.0

rng = np.random.default_rng(SEED)

# Load the cleaned ACN sample
data = pd.read_csv(DATA_FILE)

# Use only sessions with usable provisional energy and power estimates.
usable = data[
    (data["energy_delivered_kwh_provisional"] > 0)
    & data["mean_positive_power_kw"].notna()
    & (data["mean_positive_power_kw"] > 0)
].copy()

if len(usable) < N_EV:
    raise ValueError(
        f"Only {len(usable)} usable sessions; need at least {N_EV}."
    )

# Sample session characteristics to construct a synthetic apartment scenario.
sample = usable.sample(
    n=N_EV,
    replace=True,
    random_state=SEED
).reset_index(drop=True)

energy = sample["energy_delivered_kwh_provisional"].to_numpy()
power = sample["mean_positive_power_kw"].to_numpy()

# Avoid unrealistic outliers dominating this first prototype.
energy = np.clip(energy, 2.0, 30.0)
power = np.clip(power, 1.0, 7.0)


# Each EV arrives between 17:00 and 21:00 on the first day.
arrival_hour = rng.uniform(17.0, 21.0, N_EV)

# Departure times are simulated assumptions, not ACN observations.
stay_hours = rng.uniform(6.0, 12.0, N_EV)
departure_hour = arrival_hour + stay_hours

# Convert times to 15-minute slots.
n_slots = int(HORIZON_HOURS / STEP_HOURS)
slot_hours = np.arange(n_slots) * STEP_HOURS

arrival_slot = np.floor(arrival_hour / STEP_HOURS).astype(int)
departure_slot = np.ceil(departure_hour / STEP_HOURS).astype(int)
departure_slot = np.clip(departure_slot, 1, n_slots)

# Per-EV power limits and energy requirements.
# Energy and power are inspired by ACN observations, while windows are synthetic.
max_power = power
required_energy = energy


# Save the EV scenario so every algorithm uses identical inputs.
scenario = pd.DataFrame({
    "ev_id": np.arange(N_EV),
    "arrival_hour": arrival_hour,
    "departure_hour": departure_hour,
    "arrival_slot": arrival_slot,
    "departure_slot": departure_slot,
    "required_energy_kwh": required_energy,
    "max_power_kw": max_power
})

scenario_file = PROJECT_DIR / "data" / "processed" / "ev_scenario.csv"
scenario.to_csv(scenario_file, index=False)

print("\nEV scenario saved to:", scenario_file)
print("\nFirst five EVs:")
print(scenario.head().to_string(index=False))


def run_uncontrolled():
    """Each EV starts at arrival and charges at maximum power."""
    schedule = np.zeros((N_EV, n_slots))

    for i in range(N_EV):
        remaining = required_energy[i]

        for t in range(arrival_slot[i], departure_slot[i]):
            if remaining <= 0:
                break

            p = min(max_power[i], remaining / STEP_HOURS)
            schedule[i, t] = p
            remaining -= p * STEP_HOURS

    return schedule



def run_fcfs():
    """
    FCFS priority baseline:
    - At most 20 EVs are selected in each time slot.
    - Earlier arrivals get priority.
    - The 100 kW grid limit is enforced.
    - Finished or departed EVs are no longer selected.
    """
    schedule = np.zeros((N_EV, n_slots))
    order = np.argsort(arrival_slot)

    for t in range(n_slots):
        eligible = []

        # Find EVs that have arrived, have not departed,
        # and still need energy.
        for i in order:
            if arrival_slot[i] <= t < departure_slot[i]:
                delivered = schedule[i, :t].sum() * STEP_HOURS
                remaining = required_energy[i] - delivered

                if remaining > 1e-9:
                    eligible.append(i)

        # Select the earliest-arriving EVs first.
        selected = eligible[:N_CHARGERS]

        remaining_grid_kw = MAX_GRID_KW

        for i in selected:
            delivered = schedule[i, :t].sum() * STEP_HOURS
            remaining_energy = required_energy[i] - delivered

            power_now = min(
                max_power[i],
                remaining_energy / STEP_HOURS,
                remaining_grid_kw
            )

            if power_now > 0:
                schedule[i, t] = power_now
                remaining_grid_kw -= power_now

            if remaining_grid_kw <= 1e-9:
                break

    return schedule




def evaluate(name, schedule):
    metrics = evaluate_schedule(
        schedule=schedule,
        required_energy_kwh=required_energy,
        step_hours=STEP_HOURS,
        max_grid_kw=MAX_GRID_KW,
        max_chargers=N_CHARGERS,
        max_power_kw=max_power,
        arrival_slot=arrival_slot,
        departure_slot=departure_slot,
    )

    metrics["method"] = name
    metrics["max_grid_limit_kw"] = MAX_GRID_KW

    grid_load = schedule.sum(axis=0)
    return metrics, grid_load



uncontrolled = run_uncontrolled()
fcfs = run_fcfs()

metrics_u, load_u = evaluate("Uncontrolled", uncontrolled)
metrics_f, load_f = evaluate("FCFS", fcfs)

# Save both schedules so their constraints can be audited directly.
slot_columns = [f"slot_{t}" for t in range(n_slots)]

uncontrolled_df = pd.DataFrame(
    uncontrolled,
    columns=slot_columns
)
uncontrolled_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
uncontrolled_path = RESULTS_DIR / "uncontrolled_schedule.csv"
uncontrolled_df.to_csv(uncontrolled_path, index=False)

fcfs_df = pd.DataFrame(
    fcfs,
    columns=slot_columns
)
fcfs_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
fcfs_path = RESULTS_DIR / "fcfs_schedule.csv"
fcfs_df.to_csv(fcfs_path, index=False)

print("\nBaseline schedules saved:")
print(uncontrolled_path)
print(fcfs_path)


print("\nFCFS diagnostic:")
print("Total required energy (kWh):", round(required_energy.sum(), 2))
print("Total FCFS energy (kWh):", round(fcfs.sum() * STEP_HOURS, 2))
print("Maximum FCFS load (kW):", round(fcfs.sum(axis=0).max(), 2))
print("Time slots with zero FCFS load:", int((fcfs.sum(axis=0) == 0).sum()))
print("Time slots with positive FCFS load:", int((fcfs.sum(axis=0) > 0).sum()))
print("Maximum individual EV power (kW):", round(fcfs.max(), 2))

results = pd.DataFrame([metrics_u, metrics_f])
results.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

print("\nBaseline comparison:")
print(results.to_string(index=False))

# Plot aggregate load profiles
plt.figure(figsize=(11, 5))
plt.plot(slot_hours, load_u, label="Uncontrolled")
plt.plot(slot_hours, load_f, label="FCFS")
plt.axhline(MAX_GRID_KW, linestyle="--", label="Grid capacity")
plt.xlabel("Hour of day")
plt.ylabel("Aggregate charging power (kW)")
plt.title("Apartment EV Charging: Baseline Load Profiles")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_file = RESULTS_DIR / "baseline_load_profiles.png"
plt.savefig(plot_file, dpi=150)
plt.close()

print("\nLoad profile saved to:", plot_file)
print("Metrics saved to:", RESULTS_DIR / "baseline_metrics.csv")
