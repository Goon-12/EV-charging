
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Common experimental configuration
STEP_HOURS = 0.25
N_SLOTS = 144
N_CHARGERS = 20
MAX_GRID_KW = 125.0
TOLERANCE_KWH = 0.05

scenario_path = DATA_DIR / "ev_scenario.csv"
scenario = pd.read_csv(scenario_path)

required = scenario["required_energy_kwh"].to_numpy(dtype=float)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
arrival = scenario["arrival_slot"].to_numpy(dtype=int)
departure = scenario["departure_slot"].to_numpy(dtype=int)
ev_ids = scenario["ev_id"].to_numpy()

schedule_files = {
    "Uncontrolled": "uncontrolled_schedule.csv",
    "FCFS": "fcfs_schedule.csv",
    "Original GA": "ga_best_schedule.csv",
    "PSO": "pso_best_schedule.csv",
    "Feasibility-first": "feasibility_baseline_schedule.csv",
    "Cost-aware GA": "ga_cost_best_schedule_latest.csv",
    "Solar-aware": "solar_aware_schedule.csv",
}

slot_columns = [f"slot_{t}" for t in range(N_SLOTS)]
rows = []

print("\nREPRODUCIBILITY AND FEASIBILITY AUDIT")
print("=" * 80)
print(f"EV count: {len(scenario)}")
print(f"Time slots: {N_SLOTS}")
print(f"Slot duration: {STEP_HOURS} hours")
print(f"Grid limit: {MAX_GRID_KW} kW")
print(f"Charger limit: {N_CHARGERS}")
print(f"Total requested energy: {required.sum():.3f} kWh")

for method, filename in schedule_files.items():
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"\n{method}: MISSING FILE — {filename}")
        rows.append({
            "method": method,
            "file_exists": False,
            "scenario_match": False,
            "peak_grid_kw": np.nan,
            "delivered_kwh": np.nan,
            "unmet_kwh": np.nan,
            "fully_charged_evs": np.nan,
            "grid_violations": np.nan,
            "charger_violations": np.nan,
            "power_violations": np.nan,
            "time_window_violations": np.nan,
            "feasible": False,
        })
        continue

    df = pd.read_csv(path)

    if len(df) != len(scenario):
        raise ValueError(
            f"{method}: EV count mismatch in {filename}. "
            "Do not compare this schedule until the mismatch is resolved."
        )

    if "ev_id" not in df.columns or not np.array_equal(
        df["ev_id"].to_numpy(), ev_ids
    ):
        raise ValueError(
            f"{method}: EV IDs/order differ from the saved scenario."
        )

    missing = [c for c in slot_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{method}: missing columns {missing[:5]}")

    schedule = df[slot_columns].to_numpy(dtype=float)

    if not np.all(np.isfinite(schedule)):
        raise ValueError(f"{method}: non-finite power values found.")

    if np.any(schedule < -1e-9):
        raise ValueError(f"{method}: negative power values found.")

    delivered_by_ev = schedule.sum(axis=1) * STEP_HOURS
    unmet_by_ev = np.maximum(required - delivered_by_ev, 0.0)
    delivered = float(delivered_by_ev.sum())
    unmet = float(unmet_by_ev.sum())

    load = schedule.sum(axis=0)
    active = (schedule > 1e-9).sum(axis=0)

    grid_violations = int(
        (load > MAX_GRID_KW + 1e-9).sum()
    )
    charger_violations = int(
        (active > N_CHARGERS).sum()
    )
    power_violations = int(
        (schedule > max_power[:, None] + 1e-9).sum()
    )

    time_violations = 0
    for i in range(len(scenario)):
        valid = np.zeros(N_SLOTS, dtype=bool)
        start = int(np.clip(arrival[i], 0, N_SLOTS))
        end = int(np.clip(departure[i], 0, N_SLOTS))
        valid[start:end] = True
        time_violations += int(
            ((schedule[i] > 1e-9) & ~valid).sum()
        )

    # Verify energy accounting independently.
    accounting_error = abs(
        required.sum() - delivered - unmet
    )

    feasible = (
        grid_violations == 0
        and charger_violations == 0
        and power_violations == 0
        and time_violations == 0
    )

    rows.append({
        "method": method,
        "file_exists": True,
        "scenario_match": True,
        "peak_grid_kw": float(load.max()),
        "delivered_kwh": delivered,
        "unmet_kwh": unmet,
        "fully_charged_evs": int(
            (unmet_by_ev <= TOLERANCE_KWH).sum()
        ),
        "grid_violations": grid_violations,
        "charger_violations": charger_violations,
        "power_violations": power_violations,
        "time_window_violations": time_violations,
        "energy_accounting_error_kwh": accounting_error,
        "feasible": feasible,
    })

    print(f"\n{method}")
    print(f"  File: {filename}")
    print(f"  Peak demand: {load.max():.3f} kW")
    print(f"  Energy delivered: {delivered:.3f} kWh")
    print(f"  Energy unmet: {unmet:.3f} kWh")
    print(f"  Fully charged EVs: {(unmet_by_ev <= TOLERANCE_KWH).sum()}")
    print(f"  Grid violations: {grid_violations}")
    print(f"  Charger violations: {charger_violations}")
    print(f"  Individual power violations: {power_violations}")
    print(f"  Time-window violations: {time_violations}")
    print(f"  Feasible under checked constraints: {feasible}")

audit = pd.DataFrame(rows)
output_path = RESULTS_DIR / "reproducibility_audit.csv"
audit.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("AUDIT SUMMARY")
print(
    audit[
        [
            "method",
            "file_exists",
            "scenario_match",
            "peak_grid_kw",
            "delivered_kwh",
            "unmet_kwh",
            "fully_charged_evs",
            "feasible",
        ]
    ].round(3).to_string(index=False)
)
print(f"\nAudit saved to: {output_path}")
print(
    "\nNote: this verifies saved schedules against the current scenario "
    "and configured limits. It cannot prove which random seed or settings "
    "were used historically when each schedule was generated."
)
