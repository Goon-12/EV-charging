from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project paths and existing project loader.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))

from load_scenario import load_scenario

STEP = 0.25
N_SLOTS = 144
N_CHARGERS = 20
GRID_IMPORT_LIMIT_KW = 125.0
POPULATION_SIZE = 24
GENERATIONS = 60
ELITE_COUNT = 2
MUTATION_RATE = 0.18
SEED = int(os.environ.get("EV_OPT_SEED", "42"))
rng = np.random.default_rng(SEED)
TOL = 1e-9
FULL_CHARGE_TOL_KWH = 0.05

# Consistent illustrative time-of-use tariff.
def make_tariff():
    hours = (np.arange(N_SLOTS) * STEP) % 24
    return np.select(
        [(hours < 6), ((hours >= 17) & (hours < 22))],
        [4.0, 10.0],
        default=6.0,
    ).astype(float)


def load_solar():
    path = DATA / "solar_profile_36h.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/create_solar_profile.py first."
        )
    df = pd.read_csv(path)
    if "solar_power_kw" not in df.columns or len(df) != N_SLOTS:
        raise ValueError("Solar profile must contain 144 rows and solar_power_kw.")
    solar = df["solar_power_kw"].to_numpy(float)
    if not np.isfinite(solar).all() or np.any(solar < 0):
        raise ValueError("Solar profile contains invalid power values.")
    return solar


def evaluate(schedule, scenario, solar, prices):
    P = np.asarray(schedule, float)
    n_ev = len(scenario)
    if P.shape != (n_ev, N_SLOTS):
        raise ValueError(f"Expected schedule shape {(n_ev, N_SLOTS)}, got {P.shape}")
    if not np.isfinite(P).all():
        raise ValueError("Schedule contains NaN or infinity.")

    required = scenario["required_energy_kwh"].to_numpy(float)
    max_power = scenario["max_power_kw"].to_numpy(float)
    arrival = np.clip(scenario["arrival_slot"].to_numpy(int), 0, N_SLOTS)
    departure = np.clip(scenario["departure_slot"].to_numpy(int), 0, N_SLOTS)

    energy_ev = P.sum(axis=1) * STEP
    unmet_ev = np.maximum(required - energy_ev, 0.0)
    ev_load = P.sum(axis=0)
    grid_import = np.maximum(ev_load - solar, 0.0)
    solar_used = np.minimum(ev_load, solar)
    active = (P > TOL).sum(axis=0)

    grid_v = int((grid_import > GRID_IMPORT_LIMIT_KW + TOL).sum())
    charger_v = int((active > N_CHARGERS).sum())
    power_v = int((P > max_power[:, None] + TOL).sum())
    negative_v = int((P < -TOL).sum())
    overcharge_v = int((energy_ev > required + FULL_CHARGE_TOL_KWH).sum())
    time_v = 0
    for i in range(n_ev):
        valid = np.zeros(N_SLOTS, dtype=bool)
        valid[arrival[i]:departure[i]] = True
        time_v += int(((P[i] > TOL) & ~valid).sum())

    cost = float(np.sum(grid_import * STEP * prices))
    required_total = float(required.sum())
    delivered_total = float(energy_ev.sum())
    feasible = all(v == 0 for v in [
        grid_v, charger_v, power_v, negative_v, overcharge_v, time_v
    ])

    return {
        "required_energy_kwh": required_total,
        "delivered_energy_kwh": delivered_total,
        "unmet_energy_kwh": float(unmet_ev.sum()),
        "energy_demand_met_pct": 100.0 * delivered_total / max(required_total, 1e-12),
        "fully_charged_evs": int((unmet_ev <= FULL_CHARGE_TOL_KWH).sum()),
        "peak_ev_load_kw": float(ev_load.max()),
        "peak_grid_import_after_solar_kw": float(grid_import.max()),
        "solar_available_kwh": float(solar.sum() * STEP),
        "solar_used_kwh": float(solar_used.sum() * STEP),
        "solar_utilization_pct": float(100 * solar_used.sum() / max(solar.sum(), 1e-12)),
        "synthetic_grid_cost_units": cost,
        "grid_import_limit_violations": grid_v,
        "charger_limit_violations": charger_v,
        "individual_power_violations": power_v,
        "time_window_violations": time_v,
        "negative_power_violations": negative_v,
        "overcharge_violations": overcharge_v,
        "feasible": bool(feasible),
    }


def repair_energy_first(candidate, scenario, solar, slot_order=None):
    """
    Repair by rebuilding a schedule. Earlier departures are prioritized.
    At each slot, total EV load is limited to GRID_IMPORT_LIMIT_KW + solar[t],
    which guarantees grid import after solar does not exceed the cap.
    """
    candidate = np.asarray(candidate, float)
    n_ev = len(scenario)
    if candidate.shape != (n_ev, N_SLOTS):
        raise ValueError("Candidate has wrong shape.")

    required = scenario["required_energy_kwh"].to_numpy(float)
    max_power = scenario["max_power_kw"].to_numpy(float)
    arrival = np.clip(scenario["arrival_slot"].to_numpy(int), 0, N_SLOTS)
    departure = np.clip(scenario["departure_slot"].to_numpy(int), 0, N_SLOTS)

    # Candidate preferences are retained as a tie-breaker only.
    candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
    candidate = np.clip(candidate, 0.0, max_power[:, None])
    for i in range(n_ev):
        candidate[i, :arrival[i]] = 0.0
        candidate[i, departure[i]:] = 0.0

    if slot_order is None:
        slot_order = range(N_SLOTS)

    P = np.zeros((n_ev, N_SLOTS), float)
    delivered = np.zeros(n_ev, float)
    for t in slot_order:
        eligible = [
            i for i in range(n_ev)
            if arrival[i] <= t < departure[i] and required[i] - delivered[i] > 1e-8
        ]
        # Earliest departure first; then candidate slot preference; then more unmet.
        eligible.sort(key=lambda i: (
            departure[i],
            -int(candidate[i, t] > TOL),
            -candidate[i, t],
            -(required[i] - delivered[i]),
            i,
        ))

        active = 0
        ev_load_room = GRID_IMPORT_LIMIT_KW + solar[t]
        for i in eligible:
            if active >= N_CHARGERS or ev_load_room <= 1e-9:
                break
            remaining = required[i] - delivered[i]
            p = min(max_power[i], remaining / STEP, ev_load_room)
            if p > TOL:
                P[i, t] = p
                delivered[i] += p * STEP
                ev_load_room -= p
                active += 1
    return P


def seed_candidate(scenario, solar, prices, mode):
    n_ev = len(scenario)
    arrival = np.clip(scenario["arrival_slot"].to_numpy(int), 0, N_SLOTS)
    departure = np.clip(scenario["departure_slot"].to_numpy(int), 0, N_SLOTS)
    candidate = np.zeros((n_ev, N_SLOTS), float)

    # Use varied candidate preferences: cheap slots, early slots, randomized slot order.
    if mode == "cheap":
        slot_order = sorted(range(N_SLOTS), key=lambda t: (prices[t], t))
    elif mode == "late_solar":
        slot_order = sorted(range(N_SLOTS), key=lambda t: (-solar[t], prices[t], t))
    elif mode == "random":
        slot_order = rng.permutation(N_SLOTS)
    else:
        slot_order = range(N_SLOTS)

    # Random preference matrix provides genetic diversity; repair makes it feasible.
    candidate[:] = rng.random((n_ev, N_SLOTS)) * 5.0
    for i in range(n_ev):
        candidate[i, :arrival[i]] = 0.0
        candidate[i, departure[i]:] = 0.0
    return repair_energy_first(candidate, scenario, solar, slot_order=slot_order)


def objective(schedule, scenario, solar, prices):
    m = evaluate(schedule, scenario, solar, prices)
    # Main goal: minimize unmet energy. Secondary: fewer incomplete EVs,
    # then cost and peak grid import. Hard constraints are enforced by repair.
    incomplete = len(scenario) - m["fully_charged_evs"]
    return (
        1_000_000.0 * m["unmet_energy_kwh"]
        + 1_000.0 * incomplete
        + 10.0 * m["synthetic_grid_cost_units"]
        + 100.0 * m["peak_grid_import_after_solar_kw"]
    )


def mutate_candidate(schedule, scenario, solar, prices):
    candidate = schedule.copy()
    n_ev = len(scenario)
    i = int(rng.integers(n_ev))
    candidate[i, :] = 0.0
    a = int(np.clip(scenario.iloc[i]["arrival_slot"], 0, N_SLOTS))
    d = int(np.clip(scenario.iloc[i]["departure_slot"], 0, N_SLOTS))
    if d > a:
        slots = np.arange(a, d)
        rng.shuffle(slots)
        remaining = float(scenario.iloc[i]["required_energy_kwh"])
        max_p = float(scenario.iloc[i]["max_power_kw"])
        for t in slots:
            if remaining <= 1e-8:
                break
            candidate[i, t] = min(max_p, remaining / STEP)
            remaining -= candidate[i, t] * STEP
    return repair_energy_first(candidate, scenario, solar)


def load_saved_schedule(filename, n_ev):
    path = RESULTS / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    cols = [f"slot_{t}" for t in range(N_SLOTS)]
    if len(df) != n_ev or not all(c in df.columns for c in cols):
        print(f"Warning: skipping {filename}; unexpected format.")
        return None
    return df[cols].to_numpy(float)


def save_schedule(P, filename, scenario):
    df = pd.DataFrame(P, columns=[f"slot_{t}" for t in range(N_SLOTS)])
    df.insert(0, "ev_id", scenario["ev_id"].to_numpy())
    path = RESULTS / filename
    df.to_csv(path, index=False)
    return path


def main():
    scenario = load_scenario()
    solar = load_solar()
    prices = make_tariff()
    n_ev = len(scenario)
    print(f"Loaded {n_ev} EVs; seed={SEED}")
    print(f"Grid limit applies to import after solar: {GRID_IMPORT_LIMIT_KW:.1f} kW")

    # Construct initial population from several feasible heuristic schedules.
    population = []
    for mode in ["arrival", "cheap", "late_solar", "random"]:
        population.append(seed_candidate(scenario, solar, prices, mode))
    while len(population) < POPULATION_SIZE:
        mode = ["arrival", "cheap", "late_solar", "random"][len(population) % 4]
        population.append(seed_candidate(scenario, solar, prices, mode))

    history = []
    best = None
    best_score = float("inf")

    for gen in range(GENERATIONS):
        scores = np.array([objective(x, scenario, solar, prices) for x in population])
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = scores[order]
        if scores[0] < best_score:
            best_score = float(scores[0])
            best = population[0].copy()
        m = evaluate(best, scenario, solar, prices)
        history.append({"generation": gen, "objective": best_score,
                        "unmet_energy_kwh": m["unmet_energy_kwh"],
                        "delivered_energy_kwh": m["delivered_energy_kwh"],
                        "peak_ev_load_kw": m["peak_ev_load_kw"],
                        "peak_grid_import_after_solar_kw": m["peak_grid_import_after_solar_kw"]})
        if gen % 10 == 0:
            print(
                f"Generation {gen:3d} | delivered={m['delivered_energy_kwh']:.3f} kWh "
                f"| unmet={m['unmet_energy_kwh']:.3f} kWh "
                f"| peak EV load={m['peak_ev_load_kw']:.3f} kW "
                f"| peak grid import={m['peak_grid_import_after_solar_kw']:.3f} kW"
            )

        next_pop = [x.copy() for x in population[:ELITE_COUNT]]
        while len(next_pop) < POPULATION_SIZE:
            pool = max(1, POPULATION_SIZE // 2)
            p1 = population[int(rng.integers(pool))]
            p2 = population[int(rng.integers(pool))]
            mask = rng.random(n_ev) < 0.5
            child = np.where(mask[:, None], p1, p2)
            child = repair_energy_first(child, scenario, solar)
            if rng.random() < MUTATION_RATE:
                child = mutate_candidate(child, scenario, solar, prices)
            next_pop.append(child)
        population = next_pop

    # Save best GA result.
    best_path = save_schedule(best, "ga_peak_constrained_best_schedule.csv", scenario)
    metrics = evaluate(best, scenario, solar, prices)
    metrics.update({
        "method": "Peak-constrained energy-first GA",
        "seed": SEED,
        "grid_import_limit_kw": GRID_IMPORT_LIMIT_KW,
        "population_size": POPULATION_SIZE,
        "generations": GENERATIONS,
        "objective_definition": "unmet energy, incomplete EV count, synthetic grid cost, peak grid import",
        "tariff_type": "synthetic time-of-use",
        "solar_type": "synthetic illustrative profile",
    })
    metrics_path = RESULTS / "ga_peak_constrained_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    history_df = pd.DataFrame(history)
    history_path = RESULTS / "ga_peak_constrained_convergence.csv"
    history_df.to_csv(history_path, index=False)
    plt.figure(figsize=(9, 4.5))
    plt.plot(history_df["generation"], history_df["unmet_energy_kwh"])
    plt.xlabel("Generation")
    plt.ylabel("Unmet energy (kWh)")
    plt.title("Peak-Constrained Energy-First GA")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    convergence_path = RESULTS / "ga_peak_constrained_convergence.png"
    plt.savefig(convergence_path, dpi=180)
    plt.close()

    # Compare with available saved schedules using exactly the same evaluator.
    candidates = [
        ("Original GA", "ga_best_schedule.csv"),
        ("Feasibility-first", "feasibility_baseline_schedule.csv"),
        ("PSO", "pso_best_schedule.csv"),
        ("Battery-aware GA v3", "ga_battery_aware_v3_schedule.csv"),
        ("Solar-aware baseline", "solar_aware_schedule.csv"),
        ("Peak-constrained energy-first GA", "ga_peak_constrained_best_schedule.csv"),
    ]
    rows = []
    for name, filename in candidates:
        P = load_saved_schedule(filename, n_ev)
        if P is None:
            continue
        row = evaluate(P, scenario, solar, prices)
        row["method"] = name
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        ["feasible", "delivered_energy_kwh"], ascending=[False, False]
    )
    comparison_path = RESULTS / "ga_peak_constrained_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nPEAK-CONSTRAINED ENERGY-FIRST GA RESULTS")
    print("=" * 65)
    for key in [
        "required_energy_kwh", "delivered_energy_kwh", "unmet_energy_kwh",
        "energy_demand_met_pct", "fully_charged_evs", "peak_ev_load_kw",
        "peak_grid_import_after_solar_kw", "solar_available_kwh", "solar_used_kwh",
        "solar_utilization_pct", "synthetic_grid_cost_units",
        "grid_import_limit_violations", "charger_limit_violations",
        "individual_power_violations", "time_window_violations",
        "overcharge_violations", "feasible",
    ]:
        print(f"{key}: {metrics[key]}")
    print("\nSAME-EVALUATOR COMPARISON")
    print("=" * 95)
    cols = [
        "method", "delivered_energy_kwh", "unmet_energy_kwh",
        "energy_demand_met_pct", "fully_charged_evs",
        "peak_ev_load_kw", "peak_grid_import_after_solar_kw",
        "synthetic_grid_cost_units", "feasible",
    ]
    print(comparison[cols].round(3).to_string(index=False))
    print("\nSaved schedule:", best_path)
    print("Saved metrics:", metrics_path)
    print("Saved convergence:", convergence_path)
    print("Saved comparison:", comparison_path)
    print("\nNote: this is a heuristic GA, not a proof of global optimality.")
    print("Solar and tariff are synthetic; battery degradation is not modeled here.")


if __name__ == "__main__":
    main()
