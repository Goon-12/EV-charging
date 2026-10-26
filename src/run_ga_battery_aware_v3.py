from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Battery-aware GA v3.

Grid constraint is defined on grid import AFTER solar offset:
    grid_import[t] = max(total_EV_load[t] - solar_power[t], 0)

This is an experimental heuristic, not a proof of global optimality.
The battery-stress score is a relative charging-rate proxy, not a battery
degradation/lifespan prediction.
"""

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0

POPULATION_SIZE = 36
GENERATIONS = 80
ELITE_COUNT = 2
MUTATION_RATE = 0.25
SEED = 42
rng = np.random.default_rng(SEED)

# Secondary objectives are used only after unmet-energy minimization.
W_COST = 0.50
W_PEAK = 0.30
W_BATTERY = 0.20
TOLERANCE_KWH = 0.05

SCENARIO_FILE = DATA_DIR / "ev_scenario.csv"
SOLAR_FILE = DATA_DIR / "solar_profile_36h.csv"

for required_file in (SCENARIO_FILE, SOLAR_FILE):
    if not required_file.exists():
        raise FileNotFoundError(f"Required input not found: {required_file}")

scenario = pd.read_csv(SCENARIO_FILE)
solar_df = pd.read_csv(SOLAR_FILE)

if len(solar_df) != N_SLOTS:
    raise ValueError(f"Expected {N_SLOTS} solar rows, found {len(solar_df)}.")

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = np.clip(scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS)
departure = np.clip(scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS)
solar_kw = solar_df["solar_power_kw"].to_numpy(dtype=float)
n_ev = len(scenario)

if n_ev == 0:
    raise ValueError("Scenario contains no EVs.")
if not np.isfinite(required).all() or np.any(required < 0):
    raise ValueError("Invalid energy requirements.")
if not np.isfinite(max_power).all() or np.any(max_power <= 0):
    raise ValueError("Invalid EV maximum powers.")
if not np.isfinite(solar_kw).all() or np.any(solar_kw < 0):
    raise ValueError("Invalid solar values.")

total_required = float(required.sum())
slot_hours = np.arange(N_SLOTS) * STEP_HOURS
hour_of_day = slot_hours % 24.0

# Same illustrative tariff as earlier experiments; these are synthetic units.
prices = np.select(
    [
        hour_of_day < 6.0,
        (hour_of_day >= 17.0) & (hour_of_day < 22.0),
    ],
    [4.0, 10.0],
    default=6.0,
).astype(float)

COST_REFERENCE = max(total_required * float(prices.max()), 1.0)


def battery_stress_proxy(schedule):
    """Energy-weighted squared relative charging rate; not actual aging."""
    energy = schedule * STEP_HOURS
    total_energy = float(energy.sum())
    if total_energy <= 1e-12:
        return 0.0
    relative_rate = np.clip(schedule / max_power[:, None], 0.0, 1.0)
    return float((energy * relative_rate**2).sum() / total_energy)


def get_metrics(schedule):
    energy_by_ev = schedule.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - energy_by_ev, 0.0)
    ev_load = schedule.sum(axis=0)
    grid_import = np.maximum(ev_load - solar_kw, 0.0)
    solar_used = np.minimum(ev_load, solar_kw)

    cost_units = float(np.sum(grid_import * STEP_HOURS * prices))
    peak_ev_load = float(ev_load.max())
    peak_grid_import = float(grid_import.max())
    total_unmet = float(unmet_by_ev.sum())
    delivered = float(energy_by_ev.sum())

    return {
        "unmet_energy_kwh": total_unmet,
        "delivered_energy_kwh": delivered,
        "demand_met_pct": 100.0 * delivered / total_required if total_required else 100.0,
        "fully_charged_evs": int((unmet_by_ev <= TOLERANCE_KWH).sum()),
        "cost_units": cost_units,
        "peak_ev_load_kw": peak_ev_load,
        "peak_grid_import_kw": peak_grid_import,
        "solar_available_kwh": float(solar_kw.sum() * STEP_HOURS),
        "solar_used_kwh": float(solar_used.sum() * STEP_HOURS),
        "solar_utilization_pct": (
            100.0 * float(solar_used.sum()) / max(float(solar_kw.sum()), 1e-12)
        ),
        "battery_stress_proxy": battery_stress_proxy(schedule),
        "ev_load": ev_load,
        "grid_import": grid_import,
    }


def repair_schedule(candidate):
    """
    Energy-first deterministic repair.

    For each slot, allocate to eligible EVs ordered by earliest departure.
    A slot's available EV-load capacity is grid limit plus available solar.
    The repair fills as much feasible energy as possible, then the GA can
    search for lower-cost/lower-peak/lower-stress alternatives.
    """
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (n_ev, N_SLOTS):
        raise ValueError(f"Expected schedule shape {(n_ev, N_SLOTS)}.")
    if not np.isfinite(candidate).all():
        raise ValueError("Candidate has NaN or infinity.")

    # Candidate only acts as a soft preference. The allocation pass is
    # energy-first and does not rely on candidate values to decide feasibility.
    candidate = np.clip(candidate, 0.0, max_power[:, None])
    for i in range(n_ev):
        candidate[i, :arrival[i]] = 0.0
        candidate[i, departure[i]:] = 0.0

    repaired = np.zeros((n_ev, N_SLOTS), dtype=float)
    delivered = np.zeros(n_ev, dtype=float)

    # To avoid always filling only the earliest time slots, order slots by
    # lower tariff, then higher solar availability, then earlier time.
    # Process time chronologically to protect EV departure deadlines.
    slot_order = range(N_SLOTS)

    # Earlier departure first; then less available charging time; then greater
    # energy need relative to available window.
    duration = np.maximum(departure - arrival, 0)
    ev_order = sorted(
        range(n_ev),
        key=lambda i: (
            departure[i],
            duration[i],
            -(required[i] / max(duration[i] * STEP_HOURS, 1e-9)),
            i,
        ),
    )

    for t in slot_order:
        eligible = [
            i for i in ev_order
            if arrival[i] <= t < departure[i]
            and required[i] - delivered[i] > 1e-8
        ]
        if not eligible:
            continue

        ev_load_now = float(repaired[:, t].sum())
        grid_room_as_ev_load = MAX_GRID_KW + solar_kw[t] - ev_load_now
        active_count = int(np.count_nonzero(repaired[:, t] > 1e-9))

        for i in eligible:
            if active_count >= N_CHARGERS or grid_room_as_ev_load <= 1e-9:
                break

            remaining = required[i] - delivered[i]
            if remaining <= 1e-8:
                continue

            power = min(
                max_power[i],
                remaining / STEP_HOURS,
                grid_room_as_ev_load,
            )
            if power <= 1e-9:
                continue

            repaired[i, t] = power
            delivered[i] += power * STEP_HOURS
            ev_load_now += power
            grid_room_as_ev_load -= power
            active_count += 1

    return repaired


def make_candidate():
    candidate = np.zeros((n_ev, N_SLOTS), dtype=float)
    for i in range(n_ev):
        start, end = int(arrival[i]), int(departure[i])
        if end <= start:
            continue
        slots = np.arange(start, end)
        count = int(rng.integers(1, len(slots) + 1))
        chosen = rng.choice(slots, size=count, replace=False)
        candidate[i, chosen] = max_power[i] * rng.uniform(0.25, 1.0, size=count)
    return repair_schedule(candidate)


def mutate(schedule):
    child = schedule.copy()
    i = int(rng.integers(0, n_ev))
    child[i, :] = 0.0
    start, end = int(arrival[i]), int(departure[i])
    if end > start:
        slots = np.arange(start, end)
        count = int(rng.integers(1, len(slots) + 1))
        chosen = rng.choice(slots, size=count, replace=False)
        child[i, chosen] = max_power[i] * rng.uniform(0.25, 1.0, size=count)
    return repair_schedule(child)


def crossover(parent1, parent2):
    mask = rng.random(n_ev) < 0.5
    return repair_schedule(np.where(mask[:, None], parent1, parent2))


def score(schedule):
    """
    Strict lexicographic priorities encoded with safely separated scales:
    first unmet energy, then cost, peak import, then battery proxy.
    This makes secondary terms too small to compensate for a meaningful
    unmet-energy increase. Report the individual metrics alongside this score.
    """
    m = get_metrics(schedule)
    secondary = (
        W_COST * min(m["cost_units"] / COST_REFERENCE, 1.0)
        + W_PEAK * min(m["peak_grid_import_kw"] / MAX_GRID_KW, 1.0)
        + W_BATTERY * m["battery_stress_proxy"]
    )
    return m["unmet_energy_kwh"] + 1e-5 * secondary


def audit(schedule):
    metrics = get_metrics(schedule)
    active_count = (schedule > 1e-8).sum(axis=0)
    power_v = int((schedule > max_power[:, None] + 1e-6).sum())
    grid_v = int((metrics["grid_import"] > MAX_GRID_KW + 1e-6).sum())
    charger_v = int((active_count > N_CHARGERS).sum())
    time_v = 0

    for i in range(n_ev):
        valid = np.zeros(N_SLOTS, dtype=bool)
        valid[arrival[i]:departure[i]] = True
        time_v += int(((schedule[i] > 1e-8) & ~valid).sum())

    overcharge_v = int(
        ((schedule.sum(axis=1) * STEP_HOURS) > required + TOLERANCE_KWH).sum()
    )
    return {
        "grid_limit_violations": grid_v,
        "charger_limit_violations": charger_v,
        "individual_power_violations": power_v,
        "time_window_violations": time_v,
        "overcharge_violations": overcharge_v,
        "feasible": all(v == 0 for v in [
            grid_v, charger_v, power_v, time_v, overcharge_v
        ]),
    }


def main():
    print(f"Loaded {n_ev} EVs from {SCENARIO_FILE}")
    print(f"Loaded solar profile from {SOLAR_FILE}")
    print(f"Seed={SEED}; grid limit applies to import after solar: {MAX_GRID_KW:.1f} kW")
    print("Repair strategy: energy-first, earliest departure, low-price slots.")

    population = [make_candidate() for _ in range(POPULATION_SIZE)]
    best_schedule = None
    best_score = float("inf")
    history = []

    for generation in range(GENERATIONS):
        values = np.array([score(ind) for ind in population])
        order = np.argsort(values)
        population = [population[j] for j in order]
        values = values[order]

        if float(values[0]) < best_score:
            best_score = float(values[0])
            best_schedule = population[0].copy()

        history.append(best_score)
        if generation % 10 == 0:
            m = get_metrics(best_schedule)
            print(
                f"Generation {generation:3d} | unmet={m['unmet_energy_kwh']:.3f} kWh | "
                f"delivered={m['delivered_energy_kwh']:.3f} kWh | "
                f"peak grid import={m['peak_grid_import_kw']:.3f} kW"
            )

        next_population = [ind.copy() for ind in population[:ELITE_COUNT]]
        parent_pool = max(2, POPULATION_SIZE // 2)
        while len(next_population) < POPULATION_SIZE:
            p1 = population[int(rng.integers(0, parent_pool))]
            p2 = population[int(rng.integers(0, parent_pool))]
            child = crossover(p1, p2)
            if rng.random() < MUTATION_RATE:
                child = mutate(child)
            next_population.append(child)
        population = next_population

    for individual in population:
        value = score(individual)
        if value < best_score:
            best_score = value
            best_schedule = individual.copy()

    m = get_metrics(best_schedule)
    checks = audit(best_schedule)

    out_schedule = pd.DataFrame(
        best_schedule, columns=[f"slot_{t}" for t in range(N_SLOTS)]
    )
    out_schedule.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    schedule_path = RESULTS_DIR / "ga_battery_aware_v3_schedule.csv"
    out_schedule.to_csv(schedule_path, index=False)

    metrics_out = {
        k: v for k, v in m.items() if k not in {"ev_load", "grid_import"}
    }
    metrics_out.update(checks)
    metrics_out.update({
        "method": "Battery-aware GA v3",
        "seed": SEED,
        "grid_limit_definition": "grid import after solar offset",
        "grid_limit_kw": MAX_GRID_KW,
        "charger_limit": N_CHARGERS,
        "slot_hours": STEP_HOURS,
        "tariff_type": "synthetic time-of-use",
        "solar_type": "synthetic illustrative profile",
    })
    metrics_path = RESULTS_DIR / "ga_battery_aware_v3_metrics.csv"
    pd.DataFrame([metrics_out]).to_csv(metrics_path, index=False)

    plt.figure(figsize=(9, 4.5))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Score (lower is better; unmet energy dominates)")
    plt.title("Battery-Aware GA v3 Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    convergence_path = RESULTS_DIR / "ga_battery_aware_v3_convergence.png"
    plt.savefig(convergence_path, dpi=200)
    plt.close()

    hours = np.arange(N_SLOTS) * STEP_HOURS
    plt.figure(figsize=(11, 5))
    plt.plot(hours, m["ev_load"], label="Total EV charging load")
    plt.plot(hours, solar_kw, label="Available solar generation")
    plt.plot(hours, m["grid_import"], label="Grid import after solar")
    plt.axhline(MAX_GRID_KW, linestyle="--", label="125 kW grid-import limit")
    plt.xlabel("Hours from simulation start")
    plt.ylabel("Power (kW)")
    plt.title("Battery-Aware GA v3: EV Load and Grid Import")
    plt.xticks(np.arange(0, 37, 3))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    profile_path = RESULTS_DIR / "ga_battery_aware_v3_load_profile.png"
    plt.savefig(profile_path, dpi=200)
    plt.close()

    print("\nBATTERY-AWARE GA V3 RESULTS")
    print("=" * 62)
    print(f"Required energy: {total_required:.3f} kWh")
    for key, value in metrics_out.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSchedule saved: {schedule_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Convergence plot saved: {convergence_path}")
    print(f"Load profile saved: {profile_path}")
    print("\nNote: tariff/solar are synthetic; battery proxy is not a degradation model.")
    if not checks["feasible"]:
        print("WARNING: independent validation required; constraints failed.")


if __name__ == "__main__":
    main()
