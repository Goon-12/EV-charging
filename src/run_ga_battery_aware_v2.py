from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Battery-aware GA v2: energy-first lexicographic optimization.

This version:
1. Uses the same EV scenario, synthetic solar profile and synthetic tariff.
2. Repairs schedules with an explicit energy-first allocation pass.
3. Uses a lexicographic score: unmet energy first, then cost, peak, battery proxy.
4. Saves separate v2 outputs; it does not overwrite previous experiments.

Battery proxy is only an illustrative high-relative-power indicator, not a
validated battery-aging or lifespan model.
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

# Cost and secondary-objective weights; unmet energy is handled lexicographically.
W_COST = 0.50
W_PEAK = 0.30
W_BATTERY = 0.20

scenario_path = DATA_DIR / "ev_scenario.csv"
solar_path = DATA_DIR / "solar_profile_36h.csv"
scenario = pd.read_csv(scenario_path)
solar_df = pd.read_csv(solar_path)

if len(solar_df) != N_SLOTS:
    raise ValueError(f"Expected {N_SLOTS} solar rows, got {len(solar_df)}.")

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = np.clip(scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS)
departure = np.clip(scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS)
solar_kw = solar_df["solar_power_kw"].to_numpy(dtype=float)
n_ev = len(scenario)

if n_ev == 0:
    raise ValueError("Scenario contains no EVs.")
if not np.isfinite(required).all() or np.any(required < 0):
    raise ValueError("Invalid required-energy values.")
if not np.isfinite(max_power).all() or np.any(max_power <= 0):
    raise ValueError("Invalid EV power limits.")
if not np.isfinite(solar_kw).all() or np.any(solar_kw < 0):
    raise ValueError("Invalid solar profile.")

total_required = float(required.sum())
slot_hours = np.arange(N_SLOTS) * STEP_HOURS
hour_of_day = slot_hours % 24.0
# Same synthetic tariff used in previous experiments.
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
    energy = schedule * STEP_HOURS
    total_energy = float(energy.sum())
    if total_energy <= 1e-12:
        return 0.0
    relative_rate = np.clip(schedule / max_power[:, None], 0.0, 1.0)
    return float((energy * relative_rate**2).sum() / total_energy)


def evaluate(schedule):
    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)
    unmet_total = float(unmet_by_ev.sum())

    ev_load = schedule.sum(axis=0)
    grid_import = np.maximum(ev_load - solar_kw, 0.0)
    grid_energy = grid_import * STEP_HOURS
    cost = float(np.sum(grid_energy * prices))
    peak = float(grid_import.max())
    stress = battery_stress_proxy(schedule)
    delivered = float(delivered_by_ev.sum())

    # Lexicographic ordering: unmet dominates all secondary metrics.
    # Since unmet is in kWh, even a small change takes priority over secondary
    # normalized scores. The tiny terms only break ties at near-identical unmet.
    cost_norm = min(cost / COST_REFERENCE, 1.0)
    peak_norm = min(peak / MAX_GRID_KW, 1.0)
    secondary = W_COST * cost_norm + W_PEAK * peak_norm + W_BATTERY * stress
    score = unmet_total + 1e-4 * secondary

    return {
        "score": float(score),
        "unmet_energy_kwh": unmet_total,
        "delivered_energy_kwh": delivered,
        "demand_met_pct": 100.0 * delivered / total_required if total_required else 100.0,
        "cost_units": cost,
        "peak_grid_kw": peak,
        "battery_stress_proxy": stress,
        "solar_available_kwh": float(solar_kw.sum() * STEP_HOURS),
        "solar_used_kwh": float(np.minimum(ev_load, solar_kw).sum() * STEP_HOURS),
        "fully_charged_evs": int((unmet_by_ev <= 0.05).sum()),
        "ev_load": ev_load,
        "grid_import": grid_import,
    }


def repair_schedule(candidate):
    """
    Reconstruct a schedule in earliest-departure order.
    For each EV, use low-price/high-solar slots first and allocate enough power
    to meet its request when possible. Charger and grid constraints are checked
    at every slot. The algorithm is heuristic and may not find the global
    maximum delivered energy.
    """
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (n_ev, N_SLOTS):
        raise ValueError("Candidate schedule has an unexpected shape.")
    if not np.isfinite(candidate).all():
        raise ValueError("Candidate schedule contains NaN or infinity.")

    repaired = np.zeros((n_ev, N_SLOTS), dtype=float)
    delivered = np.zeros(n_ev, dtype=float)
    slot_order = sorted(
        range(N_SLOTS),
        key=lambda t: (prices[t] - 0.04 * solar_kw[t], t),
    )

    # Rank EVs by deadline, then by least flexibility and higher energy need.
    duration = np.maximum(departure - arrival, 0)
    ev_order = sorted(
        range(n_ev),
        key=lambda i: (
            departure[i],
            duration[i],
            -required[i],
            i,
        ),
    )

    # Candidate preferences influence tie-breaking, but energy completion is
    # the primary goal in the repair.
    for i in ev_order:
        start = int(arrival[i])
        end = int(departure[i])
        if end <= start:
            continue

        eligible_slots = [t for t in slot_order if start <= t < end]
        # Prefer candidate's active slots first; among them lower tariff and
        # higher solar output. Then use remaining available slots.
        eligible_slots.sort(
            key=lambda t: (
                -int(candidate[i, t] > 1e-9),
                prices[t] - 0.04 * solar_kw[t],
                t,
            )
        )

        remaining = required[i]
        for t in eligible_slots:
            if remaining <= 1e-7:
                break

            active = np.flatnonzero(repaired[:, t] > 1e-9)
            if i not in active and len(active) >= N_CHARGERS:
                continue

            ev_load_now = float(repaired[:, t].sum())
            # Solar offsets EV load first in this simplified model.
            grid_room_as_ev_load = MAX_GRID_KW + solar_kw[t] - ev_load_now
            if grid_room_as_ev_load <= 1e-9:
                continue

            power = min(
                max_power[i],
                remaining / STEP_HOURS,
                grid_room_as_ev_load,
            )
            if power <= 1e-9:
                continue

            # If candidate explicitly requested lower power, retain it when
            # enough remaining slots exist; otherwise charge faster to meet
            # the request by departure.
            candidate_power = candidate[i, t]
            slots_left_in_window = max(end - t, 1)
            avg_needed = remaining / (slots_left_in_window * STEP_HOURS)
            if candidate_power > 1e-9 and avg_needed <= candidate_power:
                power = min(power, max(candidate_power, 0.25 * max_power[i]))
            elif avg_needed < max_power[i] and candidate_power <= 1e-9:
                power = min(power, max(avg_needed, 0.35 * max_power[i]))

            power = min(power, remaining / STEP_HOURS, grid_room_as_ev_load)
            repaired[i, t] = power
            delivered[i] += power * STEP_HOURS
            remaining -= power * STEP_HOURS

    return repaired


def create_candidate():
    candidate = np.zeros((n_ev, N_SLOTS), dtype=float)
    for i in range(n_ev):
        start, end = int(arrival[i]), int(departure[i])
        if end <= start:
            continue
        slots = np.arange(start, end)
        # Random subset and charging rates; repair prioritizes energy completion.
        count = int(rng.integers(1, len(slots) + 1))
        chosen = rng.choice(slots, size=count, replace=False)
        candidate[i, chosen] = max_power[i] * rng.uniform(0.35, 1.0, size=count)
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
        child[i, chosen] = max_power[i] * rng.uniform(0.3, 1.0, size=count)
    return repair_schedule(child)


def crossover(parent1, parent2):
    mask = rng.random(n_ev) < 0.5
    return repair_schedule(np.where(mask[:, None], parent1, parent2))


def audit(schedule):
    active_count = (schedule > 1e-8).sum(axis=0)
    ev_load = schedule.sum(axis=0)
    grid_import = np.maximum(ev_load - solar_kw, 0.0)
    grid_v = int((grid_import > MAX_GRID_KW + 1e-6).sum())
    charger_v = int((active_count > N_CHARGERS).sum())
    power_v = int((schedule > max_power[:, None] + 1e-6).sum())

    time_v = 0
    for i in range(n_ev):
        valid = np.zeros(N_SLOTS, dtype=bool)
        valid[arrival[i]:departure[i]] = True
        time_v += int(((schedule[i] > 1e-8) & ~valid).sum())

    return {
        "grid_limit_violations": grid_v,
        "charger_limit_violations": charger_v,
        "individual_power_violations": power_v,
        "time_window_violations": time_v,
        "feasible": grid_v == charger_v == power_v == time_v == 0,
    }


def main():
    print(f"Loaded {n_ev} EVs; seed={SEED}")
    print("Energy-first lexicographic objective: unmet energy is primary.")
    population = [create_candidate() for _ in range(POPULATION_SIZE)]
    best_schedule = None
    best_score = float("inf")
    history = []

    for generation in range(GENERATIONS):
        scores = np.array([evaluate(ind)["score"] for ind in population])
        order = np.argsort(scores)
        population = [population[j] for j in order]
        scores = scores[order]

        if scores[0] < best_score:
            best_score = float(scores[0])
            best_schedule = population[0].copy()

        history.append(best_score)
        if generation % 10 == 0:
            m = evaluate(best_schedule)
            print(
                f"Generation {generation:3d} | unmet={m['unmet_energy_kwh']:.3f} kWh | "
                f"delivered={m['delivered_energy_kwh']:.3f} kWh | "
                f"stress={m['battery_stress_proxy']:.4f}"
            )

        next_population = [x.copy() for x in population[:ELITE_COUNT]]
        pool = max(2, POPULATION_SIZE // 2)
        while len(next_population) < POPULATION_SIZE:
            p1 = population[int(rng.integers(0, pool))]
            p2 = population[int(rng.integers(0, pool))]
            child = crossover(p1, p2)
            if rng.random() < MUTATION_RATE:
                child = mutate(child)
            next_population.append(child)
        population = next_population

    for individual in population:
        m = evaluate(individual)
        if m["score"] < best_score:
            best_score = m["score"]
            best_schedule = individual.copy()

    m = evaluate(best_schedule)
    checks = audit(best_schedule)
    schedule_df = pd.DataFrame(
        best_schedule, columns=[f"slot_{t}" for t in range(N_SLOTS)]
    )
    schedule_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    schedule_path = RESULTS_DIR / "ga_battery_aware_v2_schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    metrics = {
        key: value for key, value in m.items()
        if key not in {"ev_load", "grid_import"}
    }
    metrics.update(checks)
    metrics.update({
        "method": "Battery-aware GA v2",
        "seed": SEED,
        "objective_priority": "unmet energy, cost, peak, battery proxy",
        "weight_cost": W_COST,
        "weight_peak": W_PEAK,
        "weight_battery": W_BATTERY,
        "tariff_type": "synthetic time-of-use",
        "solar_type": "synthetic illustrative profile",
    })
    metrics_path = RESULTS_DIR / "ga_battery_aware_v2_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    plt.figure(figsize=(9, 4.5))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best lexicographic score (lower is better)")
    plt.title("Battery-Aware GA v2 Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    convergence_path = RESULTS_DIR / "ga_battery_aware_v2_convergence.png"
    plt.savefig(convergence_path, dpi=200)
    plt.close()

    time_hours = np.arange(N_SLOTS) * STEP_HOURS
    plt.figure(figsize=(11, 5))
    plt.plot(time_hours, m["ev_load"], label="Total EV charging load")
    plt.plot(time_hours, solar_kw, label="Solar available")
    plt.plot(time_hours, m["grid_import"], label="Estimated grid import")
    plt.axhline(MAX_GRID_KW, linestyle="--", label="Grid import limit")
    plt.xlabel("Hours from simulation start")
    plt.ylabel("Power (kW)")
    plt.title("Battery-Aware GA v2: Charging and Grid Import")
    plt.xticks(np.arange(0, 37, 3))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    profile_path = RESULTS_DIR / "ga_battery_aware_v2_load_profile.png"
    plt.savefig(profile_path, dpi=200)
    plt.close()

    print("\nBATTERY-AWARE GA V2 RESULTS")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, (float, np.floating)) else f"{key}: {value}")
    print(f"\nSchedule saved: {schedule_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Convergence plot saved: {convergence_path}")
    print(f"Load profile saved: {profile_path}")
    print("\nReminder: battery proxy is not a validated degradation model.")
    if not checks["feasible"]:
        print("WARNING: hard-constraint audit failed; do not report as feasible.")


if __name__ == "__main__":
    main()
