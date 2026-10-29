# ARCHIVED (pre-Tier-0) experiment: superseded objective/evaluator. See experiments/legacy/README.md
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Solar-aware, battery-stress-aware experimental GA.
# This is a separate experiment; it does not overwrite existing schedules.

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
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

# Objective weights: unmet energy is deliberately dominant.
W_UNMET = 0.70
W_COST = 0.12
W_PEAK = 0.08
W_BATTERY = 0.10

scenario_path = DATA_DIR / "ev_scenario.csv"
solar_path = DATA_DIR / "solar_profile_36h.csv"

if not scenario_path.exists():
    raise FileNotFoundError(f"Missing scenario: {scenario_path}")
if not solar_path.exists():
    raise FileNotFoundError(f"Missing solar profile: {solar_path}")

scenario = pd.read_csv(scenario_path)
solar_df = pd.read_csv(solar_path)

if len(solar_df) != N_SLOTS:
    raise ValueError(
        f"Expected {N_SLOTS} solar rows, found {len(solar_df)}."
    )

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = np.clip(
    scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS
)
departure = np.clip(
    scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS
)
solar_kw = solar_df["solar_power_kw"].to_numpy(dtype=float)

if (
    not np.isfinite(required).all()
    or not np.isfinite(max_power).all()
    or not np.isfinite(solar_kw).all()
    or np.any(required < 0)
    or np.any(max_power <= 0)
    or np.any(solar_kw < 0)
):
    raise ValueError("Scenario or solar profile has invalid values.")

N_EV = len(scenario)
if N_EV == 0:
    raise ValueError("Scenario contains no EVs.")

slot_hours = np.arange(N_SLOTS) * STEP_HOURS
hour_of_day = slot_hours % 24.0
# Same illustrative synthetic tariff used in earlier cost-aware experiments.
prices = np.select(
    [
        hour_of_day < 6.0,
        (hour_of_day >= 17.0) & (hour_of_day < 22.0),
    ],
    [4.0, 10.0],
    default=6.0,
).astype(float)

TOTAL_REQUIRED = float(required.sum())
# Explicit normalization references for the objective. Cost values remain
# synthetic units, not currency or actual electricity bills.
COST_REFERENCE = max(TOTAL_REQUIRED * float(prices.max()), 1.0)
PEAK_REFERENCE = MAX_GRID_KW


def battery_stress_proxy(schedule):
    """Energy-weighted squared relative charging rate; not real degradation."""
    energy = schedule * STEP_HOURS
    total_energy = float(energy.sum())
    if total_energy <= 1e-12:
        return 0.0
    relative_rate = np.clip(schedule / max_power[:, None], 0.0, 1.0)
    return float((energy * relative_rate**2).sum() / total_energy)


def evaluate(schedule):
    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)
    total_unmet = float(unmet_by_ev.sum())

    ev_load = schedule.sum(axis=0)
    solar_used = np.minimum(ev_load, solar_kw)
    grid_import = np.maximum(ev_load - solar_kw, 0.0)
    grid_energy = grid_import * STEP_HOURS

    cost_units = float(np.sum(grid_energy * prices))
    peak_grid = float(grid_import.max())
    battery_proxy = battery_stress_proxy(schedule)
    delivered = float(delivered_by_ev.sum())

    unmet_norm = total_unmet / max(TOTAL_REQUIRED, 1e-12)
    cost_norm = min(cost_units / COST_REFERENCE, 1.0)
    peak_norm = min(peak_grid / PEAK_REFERENCE, 1.0)

    objective = (
        W_UNMET * unmet_norm
        + W_COST * cost_norm
        + W_PEAK * peak_norm
        + W_BATTERY * battery_proxy
    )

    return {
        "objective": float(objective),
        "unmet_energy_kwh": total_unmet,
        "delivered_energy_kwh": delivered,
        "demand_met_pct": (
            100.0 * delivered / TOTAL_REQUIRED
            if TOTAL_REQUIRED > 0 else 100.0
        ),
        "cost_units": cost_units,
        "peak_grid_kw": peak_grid,
        "battery_stress_proxy": battery_proxy,
        "solar_available_kwh": float(solar_kw.sum() * STEP_HOURS),
        "solar_used_kwh": float(solar_used.sum() * STEP_HOURS),
        "solar_utilization_pct": (
            100.0 * float(solar_used.sum())
            / max(float(solar_kw.sum()), 1e-12)
        ),
        "fully_charged_evs": int((unmet_by_ev <= 0.05).sum()),
        "ev_load": ev_load,
        "grid_import": grid_import,
    }


def repair_schedule(candidate):
    """
    Repair a candidate while retaining its charging-rate preferences.
    Enforces arrival/departure windows, individual power, charger count,
    and grid-import capacity. The grid limit is applied to EV load minus solar.
    """
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (N_EV, N_SLOTS):
        raise ValueError("Candidate schedule has an unexpected shape.")
    if not np.isfinite(candidate).all():
        raise ValueError("Candidate schedule contains NaN or infinity.")

    candidate = np.maximum(candidate, 0.0)
    candidate = np.minimum(candidate, max_power[:, None])

    for i in range(N_EV):
        candidate[i, :arrival[i]] = 0.0
        candidate[i, departure[i]:] = 0.0

    # Remove excess energy per EV, retaining the candidate's slot distribution.
    for i in range(N_EV):
        energy = float(candidate[i].sum() * STEP_HOURS)
        if energy > required[i] and energy > 1e-12:
            candidate[i] *= required[i] / energy

    repaired = np.zeros_like(candidate)

    # First pass: retain requested slot/rate choices while enforcing limits.
    # For each slot, keep the candidate-preferred EVs and scale power if needed.
    for t in range(N_SLOTS):
        active = np.flatnonzero(candidate[:, t] > 1e-9)
        if len(active) == 0:
            continue

        # Candidate preference: higher requested power, then tighter departure.
        active = sorted(
            active.tolist(),
            key=lambda i: (
                -candidate[i, t],
                departure[i],
                i,
            ),
        )[:N_CHARGERS]

        grid_room_as_ev_load = MAX_GRID_KW + solar_kw[t]
        requested_total = float(sum(candidate[i, t] for i in active))
        scale = min(
            1.0,
            grid_room_as_ev_load / max(requested_total, 1e-12),
        )
        for i in active:
            repaired[i, t] = candidate[i, t] * scale

    # Second pass: allocate additional energy into available slots. Prefer
    # low-price/high-solar slots and urgent EVs. Use a partial power level
    # (rather than always max power) so the battery proxy can affect schedules.
    delivered = repaired.sum(axis=1) * STEP_HOURS
    slot_order = sorted(
        range(N_SLOTS),
        key=lambda t: (prices[t] - 0.04 * solar_kw[t], t),
    )

    for t in slot_order:
        active_now = np.flatnonzero(repaired[:, t] > 1e-9)
        active_count = len(active_now)
        ev_load_now = float(repaired[:, t].sum())
        if active_count >= N_CHARGERS:
            continue

        eligible = [
            i for i in range(N_EV)
            if arrival[i] <= t < departure[i]
            and required[i] - delivered[i] > 1e-7
            and repaired[i, t] <= 1e-9
        ]
        eligible.sort(
            key=lambda i: (
                departure[i],
                -(required[i] - delivered[i]),
                i,
            )
        )

        for i in eligible:
            if active_count >= N_CHARGERS:
                break
            remaining = required[i] - delivered[i]
            grid_room = MAX_GRID_KW + solar_kw[t] - ev_load_now
            if remaining <= 1e-7 or grid_room <= 1e-9:
                continue

            # Use a lower preferred rate when feasible. If time is running out,
            # raise rate to meet the EV's remaining energy request.
            slots_left = max(
                1,
                int(np.count_nonzero(
                    (np.arange(N_SLOTS) > t)
                    & (np.arange(N_SLOTS) < departure[i])
                )) + 1,
            )
            required_average_power = remaining / (slots_left * STEP_HOURS)
            target_rate = max(
                0.35 * max_power[i],
                min(max_power[i], required_average_power),
            )
            power = min(
                max_power[i],
                remaining / STEP_HOURS,
                grid_room,
                target_rate,
            )
            if power <= 1e-9:
                continue

            repaired[i, t] = power
            delivered[i] += power * STEP_HOURS
            ev_load_now += power
            active_count += 1

    # Final safety cleanup for numerical tolerances and any accidental excess.
    repaired = np.maximum(repaired, 0.0)
    repaired = np.minimum(repaired, max_power[:, None])
    for i in range(N_EV):
        repaired[i, :arrival[i]] = 0.0
        repaired[i, departure[i]:] = 0.0
        energy = repaired[i].sum() * STEP_HOURS
        if energy > required[i] + 1e-9:
            repaired[i] *= required[i] / energy

    # The per-slot constraints are enforced in the allocation passes. Check
    # explicitly rather than silently claiming feasibility.
    return repaired


def make_candidate():
    """Create a varied initial schedule with random rate and slot preferences."""
    candidate = np.zeros((N_EV, N_SLOTS), dtype=float)
    for i in rng.permutation(N_EV):
        slots = np.arange(arrival[i], departure[i])
        if len(slots) == 0:
            continue
        # Mix low-rate and high-rate candidate slots to create diversity.
        active_count = min(
            len(slots),
            max(1, int(np.ceil(required[i] / max(max_power[i] * STEP_HOURS, 1e-9)))),
        )
        chosen = rng.choice(slots, size=active_count, replace=False)
        rates = rng.uniform(0.35, 1.0, size=active_count)
        for t, rate in zip(chosen, rates):
            candidate[i, t] = min(
                max_power[i] * rate,
                required[i] / STEP_HOURS,
            )
    return repair_schedule(candidate)


def mutate(schedule):
    child = schedule.copy()
    i = int(rng.integers(0, N_EV))
    start, end = int(arrival[i]), int(departure[i])
    child[i, :] = 0.0
    if end > start:
        slots = np.arange(start, end)
        # Randomly change the chosen charging slots and rate level.
        count = int(rng.integers(1, len(slots) + 1))
        chosen = rng.choice(slots, size=count, replace=False)
        for t in chosen:
            child[i, t] = max_power[i] * rng.uniform(0.25, 1.0)
    return repair_schedule(child)


def crossover(parent1, parent2):
    mask = rng.random(N_EV) < 0.5
    child = np.where(mask[:, None], parent1, parent2)
    return repair_schedule(child)


def constraint_audit(schedule):
    active = (schedule > 1e-8).sum(axis=0)
    ev_load = schedule.sum(axis=0)
    grid_import = np.maximum(ev_load - solar_kw, 0.0)

    grid_violations = int((grid_import > MAX_GRID_KW + 1e-6).sum())
    charger_violations = int((active > N_CHARGERS).sum())
    power_violations = int(
        (schedule > max_power[:, None] + 1e-6).sum()
    )
    time_violations = 0
    for i in range(N_EV):
        valid = np.zeros(N_SLOTS, dtype=bool)
        valid[arrival[i]:departure[i]] = True
        time_violations += int(
            ((schedule[i] > 1e-8) & ~valid).sum()
        )

    return {
        "grid_limit_violations": grid_violations,
        "charger_limit_violations": charger_violations,
        "individual_power_violations": power_violations,
        "time_window_violations": time_violations,
        "feasible": (
            grid_violations == 0
            and charger_violations == 0
            and power_violations == 0
            and time_violations == 0
        ),
    }


def main():
    print(f"Loaded {N_EV} EVs from: {scenario_path}")
    print(f"Loaded solar profile from: {solar_path}")
    print(f"Seed: {SEED}; weights U/C/P/B = "
          f"{W_UNMET:.2f}/{W_COST:.2f}/{W_PEAK:.2f}/{W_BATTERY:.2f}")

    population = [make_candidate() for _ in range(POPULATION_SIZE)]
    best_schedule = None
    best_score = float("inf")
    history = []

    for generation in range(GENERATIONS):
        scores = np.array([evaluate(ind)["objective"] for ind in population])
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = scores[order]

        if float(scores[0]) < best_score:
            best_score = float(scores[0])
            best_schedule = population[0].copy()

        history.append(best_score)
        if generation % 10 == 0:
            m = evaluate(best_schedule)
            print(
                f"Generation {generation:3d} | objective={best_score:.6f} | "
                f"unmet={m['unmet_energy_kwh']:.3f} kWh | "
                f"cost={m['cost_units']:.2f} | "
                f"battery proxy={m['battery_stress_proxy']:.4f}"
            )

        next_population = [
            individual.copy()
            for individual in population[:ELITE_COUNT]
        ]
        parent_pool = max(2, POPULATION_SIZE // 2)

        while len(next_population) < POPULATION_SIZE:
            p1 = population[int(rng.integers(0, parent_pool))]
            p2 = population[int(rng.integers(0, parent_pool))]
            child = crossover(p1, p2)
            if rng.random() < MUTATION_RATE:
                child = mutate(child)
            next_population.append(child)

        population = next_population

    # Preserve best-so-far; inspect final population too.
    for individual in population:
        score = evaluate(individual)["objective"]
        if score < best_score:
            best_score = float(score)
            best_schedule = individual.copy()

    metrics = evaluate(best_schedule)
    audit = constraint_audit(best_schedule)

    schedule_df = pd.DataFrame(
        best_schedule, columns=[f"slot_{t}" for t in range(N_SLOTS)]
    )
    schedule_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    schedule_path = RESULTS_DIR / "ga_battery_aware_best_schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    metrics_out = {
        key: value for key, value in metrics.items()
        if key not in {"ev_load", "grid_import"}
    }
    metrics_out.update(audit)
    metrics_out.update({
        "method": "Battery-aware GA",
        "seed": SEED,
        "weight_unmet": W_UNMET,
        "weight_cost": W_COST,
        "weight_peak": W_PEAK,
        "weight_battery": W_BATTERY,
        "tariff_type": "synthetic time-of-use",
        "solar_type": "synthetic illustrative profile",
    })

    metrics_path = RESULTS_DIR / "ga_battery_aware_metrics.csv"
    pd.DataFrame([metrics_out]).to_csv(metrics_path, index=False)

    plt.figure(figsize=(9, 4.5))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best normalized objective (lower is better)")
    plt.title("Battery-Aware GA Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    convergence_path = RESULTS_DIR / "ga_battery_aware_convergence.png"
    plt.savefig(convergence_path, dpi=200)
    plt.close()

    m = metrics
    time_hours = np.arange(N_SLOTS) * STEP_HOURS
    plt.figure(figsize=(11, 5))
    plt.plot(time_hours, m["ev_load"], label="Total EV charging load")
    plt.plot(time_hours, solar_kw, label="Solar available")
    plt.plot(time_hours, m["grid_import"], label="Estimated grid import")
    plt.axhline(MAX_GRID_KW, linestyle="--", label="Grid import limit")
    plt.xlabel("Hours from simulation start")
    plt.ylabel("Power (kW)")
    plt.title("Battery-Aware GA: EV Load, Solar and Grid Import")
    plt.xticks(np.arange(0, 37, 3))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    profile_path = RESULTS_DIR / "ga_battery_aware_load_profile.png"
    plt.savefig(profile_path, dpi=200)
    plt.close()

    print("\nBATTERY-AWARE GA RESULTS")
    print("=" * 60)
    for key, value in metrics_out.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSchedule saved to: {schedule_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Convergence plot saved to: {convergence_path}")
    print(f"Load profile saved to: {profile_path}")
    print("\nInterpretation: battery_stress_proxy is an illustrative relative-rate")
    print("indicator, not a validated battery degradation or lifespan prediction.")
    if not audit["feasible"]:
        print("\nWARNING: independent constraint audit failed; do not report as feasible.")


if __name__ == "__main__":
    main()
