from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0
POPULATION_SIZE = 30
GENERATIONS = 60
ELITE_COUNT = 2
MUTATION_RATE = 0.15
SEED = 42
rng = np.random.default_rng(SEED)

scenario = pd.read_csv(DATA_DIR / "ev_scenario.csv")
solar_df = pd.read_csv(DATA_DIR / "solar_profile_36h.csv")

if len(solar_df) != N_SLOTS:
    raise ValueError(f"Expected {N_SLOTS} solar slots, got {len(solar_df)}.")

n_ev = len(scenario)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = np.clip(scenario["arrival_slot"].to_numpy(dtype=int), 0, N_SLOTS)
departure = np.clip(scenario["departure_slot"].to_numpy(dtype=int), 0, N_SLOTS)
solar_kw = solar_df["solar_power_kw"].to_numpy(dtype=float)

# Same illustrative time-of-use tariff as the earlier cost-aware experiment.
slot_hours = np.arange(N_SLOTS) * STEP_HOURS
hour_of_day = slot_hours % 24.0
prices = np.select(
    [(hour_of_day < 6.0), (hour_of_day >= 17.0) & (hour_of_day < 22.0)],
    [4.0, 10.0],
    default=6.0,
).astype(float)

if (
    not np.isfinite(required).all()
    or not np.isfinite(max_power).all()
    or not np.isfinite(solar_kw).all()
    or np.any(required < 0)
    or np.any(max_power <= 0)
    or np.any(solar_kw < 0)
):
    raise ValueError("Scenario or solar data contains invalid values.")


def evaluate(schedule):
    """Evaluate unmet energy first, then grid-import cost and peak."""
    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)
    unmet = float(unmet_by_ev.sum())

    ev_load_kw = schedule.sum(axis=0)
    solar_used_kw = np.minimum(ev_load_kw, solar_kw)
    grid_kw = np.maximum(ev_load_kw - solar_kw, 0.0)

    # Grid-import energy is priced; solar energy is not billed as grid energy.
    grid_kwh_by_slot = grid_kw * STEP_HOURS
    cost_units = float(np.sum(grid_kwh_by_slot * prices))
    peak_grid_kw = float(grid_kw.max())
    solar_used_kwh = float(solar_used_kw.sum() * STEP_HOURS)
    total_ev_kwh = float(ev_load_kw.sum() * STEP_HOURS)

    # Unmet energy dominates the objective. The other terms break ties.
    # Scales are explicit so the objective is interpretable and stable.
    objective = (
        1_000_000.0 * unmet
        + 10.0 * cost_units
        + 100.0 * peak_grid_kw
        - 5.0 * solar_used_kwh
    )

    return {
        "objective": objective,
        "unmet_energy_kwh": unmet,
        "delivered_energy_kwh": total_ev_kwh,
        "cost_units": cost_units,
        "peak_grid_kw": peak_grid_kw,
        "solar_used_kwh": solar_used_kwh,
        "solar_available_kwh": float(solar_kw.sum() * STEP_HOURS),
        "fully_charged_evs": int((unmet_by_ev <= 0.05).sum()),
        "ev_load_kw": ev_load_kw,
        "grid_kw": grid_kw,
        "solar_used_kw": solar_used_kw,
    }


def repair_solar_schedule(candidate):
    """
    Rebuild a feasible schedule using candidate preferences.
    Solar availability reduces grid import, but EV power and charger limits
    still apply. The grid constraint is checked against EV load minus solar.
    """
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (n_ev, N_SLOTS):
        raise ValueError("Candidate schedule has an unexpected shape.")

    candidate = np.maximum(candidate, 0.0)
    candidate = np.minimum(candidate, max_power[:, None])

    for i in range(n_ev):
        candidate[i, :arrival[i]] = 0.0
        candidate[i, departure[i]:] = 0.0

    repaired = np.zeros((n_ev, N_SLOTS), dtype=float)
    delivered = np.zeros(n_ev, dtype=float)

    # Process high-solar and low-price slots earlier, while retaining a
    # small preference for slots selected by the candidate chromosome.
    slot_order = sorted(
        range(N_SLOTS),
        key=lambda t: (
            prices[t] - 0.04 * solar_kw[t],
            t,
        ),
    )

    for t in slot_order:
        eligible = [
            i for i in range(n_ev)
            if arrival[i] <= t < departure[i]
            and required[i] - delivered[i] > 1e-8
        ]

        eligible.sort(
            key=lambda i: (
                departure[i],
                -int(candidate[i, t] > 1e-9),
                -candidate[i, t],
                -(required[i] - delivered[i]),
                i,
            )
        )

        active_count = 0
        ev_load_now = 0.0

        for i in eligible:
            if active_count >= N_CHARGERS:
                break

            remaining = required[i] - delivered[i]
            if remaining <= 1e-8:
                continue

            # Available grid power plus the solar available in this slot.
            # The aggregate EV load cannot exceed solar + grid capacity.
            grid_room = max(0.0, MAX_GRID_KW + solar_kw[t] - ev_load_now)
            power = min(
                max_power[i],
                remaining / STEP_HOURS,
                grid_room,
            )

            if power <= 1e-9:
                continue

            repaired[i, t] = power
            delivered[i] += power * STEP_HOURS
            ev_load_now += power
            active_count += 1

    return repaired


def make_candidate():
    """Create a candidate by charging urgent EVs in preferred slots."""
    candidate = np.zeros((n_ev, N_SLOTS), dtype=float)
    for i in rng.permutation(n_ev):
        valid_slots = np.arange(arrival[i], departure[i])
        if len(valid_slots) == 0:
            continue
        # Bias slot preference toward lower tariff and higher solar output.
        weights = np.exp(
            -(prices[valid_slots] - prices[valid_slots].min()) / 3.0
            + solar_kw[valid_slots] / 50.0
        )
        weights = weights / weights.sum()
        ordered_slots = rng.choice(
            valid_slots, size=len(valid_slots), replace=False, p=weights
        )
        remaining = required[i]
        for t in ordered_slots:
            if remaining <= 1e-8:
                break
            candidate[i, t] = min(max_power[i], remaining / STEP_HOURS)
            remaining -= candidate[i, t] * STEP_HOURS
    return repair_solar_schedule(candidate)


def mutate_candidate(schedule):
    child = schedule.copy()
    i = int(rng.integers(0, n_ev))
    child[i, :] = 0.0

    slots = np.arange(arrival[i], departure[i])
    if len(slots):
        rng.shuffle(slots)
        remaining = required[i]
        for t in slots:
            if remaining <= 1e-8:
                break
            child[i, t] = min(max_power[i], remaining / STEP_HOURS)
            remaining -= child[i, t] * STEP_HOURS

    return repair_solar_schedule(child)


def crossover(parent1, parent2):
    choose_first = rng.random(n_ev) < 0.5
    child = np.where(choose_first[:, None], parent1, parent2)
    return repair_solar_schedule(child)


def main():
    population = [make_candidate() for _ in range(POPULATION_SIZE)]
    history = []
    best_schedule = None
    best_score = float("inf")

    for generation in range(GENERATIONS):
        scores = np.array([evaluate(ind)["objective"] for ind in population])
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = scores[order]

        if scores[0] < best_score:
            best_score = float(scores[0])
            best_schedule = population[0].copy()

        history.append(best_score)
        if generation % 10 == 0:
            m = evaluate(best_schedule)
            print(
                f"Generation {generation:3d} | objective={best_score:.3f} | "
                f"unmet={m['unmet_energy_kwh']:.3f} kWh | "
                f"grid peak={m['peak_grid_kw']:.3f} kW"
            )

        next_population = [
            individual.copy()
            for individual in population[:ELITE_COUNT]
        ]

        while len(next_population) < POPULATION_SIZE:
            pool_size = max(2, POPULATION_SIZE // 2)
            p1 = population[int(rng.integers(0, pool_size))]
            p2 = population[int(rng.integers(0, pool_size))]
            child = crossover(p1, p2)

            if rng.random() < MUTATION_RATE:
                child = mutate_candidate(child)

            next_population.append(child)

        population = next_population

    # Compare final population with retained best-so-far solution.
    for individual in population:
        score = evaluate(individual)["objective"]
        if score < best_score:
            best_score = float(score)
            best_schedule = individual.copy()

    metrics = evaluate(best_schedule)
    schedule_df = pd.DataFrame(
        best_schedule, columns=[f"slot_{t}" for t in range(N_SLOTS)]
    )
    schedule_df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    schedule_path = RESULTS_DIR / "ga_solar_best_schedule.csv"
    schedule_df.to_csv(schedule_path, index=False)

    # Validate all hard constraints.
    active_count = (best_schedule > 1e-9).sum(axis=0)
    ev_load_kw = best_schedule.sum(axis=0)
    grid_kw = np.maximum(ev_load_kw - solar_kw, 0.0)
    power_violations = int(
        (best_schedule > max_power[:, None] + 1e-8).sum()
    )
    time_violations = 0
    for i in range(n_ev):
        valid = np.zeros(N_SLOTS, dtype=bool)
        valid[arrival[i]:departure[i]] = True
        time_violations += int(
            ((best_schedule[i] > 1e-8) & ~valid).sum()
        )

    metrics_out = {
        key: value for key, value in metrics.items()
        if key not in {"ev_load_kw", "grid_kw", "solar_used_kw"}
    }
    metrics_out.update({
        "method": "Solar-aware GA",
        "grid_limit_violations": int((grid_kw > MAX_GRID_KW + 1e-8).sum()),
        "charger_limit_violations": int((active_count > N_CHARGERS).sum()),
        "individual_power_violations": power_violations,
        "time_window_violations": time_violations,
    })
    metrics_out["feasible"] = all(
        metrics_out[k] == 0 for k in (
            "grid_limit_violations",
            "charger_limit_violations",
            "individual_power_violations",
            "time_window_violations",
        )
    )

    metrics_path = RESULTS_DIR / "ga_solar_metrics.csv"
    pd.DataFrame([metrics_out]).to_csv(metrics_path, index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Best objective (lower is better)")
    plt.title("Solar-Aware GA Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    convergence_path = RESULTS_DIR / "ga_solar_convergence.png"
    plt.savefig(convergence_path, dpi=200)
    plt.close()

    time_hours = np.arange(N_SLOTS) * STEP_HOURS
    solar_used_kw = np.minimum(ev_load_kw, solar_kw)
    plt.figure(figsize=(11, 5))
    plt.plot(time_hours, ev_load_kw, label="EV charging demand")
    plt.plot(time_hours, solar_kw, label="Solar available")
    plt.plot(time_hours, grid_kw, label="Grid import")
    plt.axhline(MAX_GRID_KW, linestyle="--", label="Grid import limit")
    plt.xlabel("Hours from simulation start")
    plt.ylabel("Power (kW)")
    plt.title("Solar-Aware GA: Charging and Grid Import")
    plt.xticks(np.arange(0, 37, 3))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    profile_path = RESULTS_DIR / "ga_solar_load_profile.png"
    plt.savefig(profile_path, dpi=200)
    plt.close()

    print("\nSOLAR-AWARE GA RESULTS")
    print("=" * 60)
    for key, value in metrics_out.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSchedule saved to: {schedule_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Convergence plot saved to: {convergence_path}")
    print(f"Load profile saved to: {profile_path}")
    print(
        "\nNote: tariff and solar generation are synthetic assumptions. "
        "This is a heuristic GA, not a proof of global optimality."
    )


if __name__ == "__main__":
    main()
