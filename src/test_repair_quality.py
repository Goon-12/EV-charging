
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from load_scenario import load_scenario
from run_ga import (
    repair_schedule,
    evaluate_fitness,
    STEP_HOURS,
    N_SLOTS,
    MAX_GRID_KW,
    N_CHARGERS,
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

scenario = load_scenario()
n_ev = len(scenario)

# Test candidate 1: all EVs request maximum power
# throughout their own available charging windows.
candidate = np.zeros((n_ev, N_SLOTS), dtype=float)

arrival = scenario["arrival_slot"].to_numpy(dtype=int)
departure = scenario["departure_slot"].to_numpy(dtype=int)
max_power = scenario["max_power_kw"].to_numpy(dtype=float)
required = scenario["required_energy_kwh"].to_numpy(dtype=float)

for i in range(n_ev):
    start = max(0, arrival[i])
    end = min(N_SLOTS, departure[i])
    candidate[i, start:end] = max_power[i]

repaired = repair_schedule(candidate, scenario)

delivered = repaired.sum(axis=1) * STEP_HOURS
unmet = np.maximum(required - delivered, 0)
grid_load = repaired.sum(axis=0)
active_count = (repaired > 1e-9).sum(axis=0)

print("\nREPAIR FUNCTION QUALITY TEST")
print("-" * 40)
print(f"EVs: {n_ev}")
print(f"Required energy: {required.sum():.3f} kWh")
print(f"Delivered energy: {delivered.sum():.3f} kWh")
print(f"Unmet energy: {unmet.sum():.3f} kWh")
print(f"Fully charged EVs (0.05 kWh tolerance): {(unmet <= 0.05).sum()}")
print(f"Peak grid demand: {grid_load.max():.3f} kW")
print(f"Grid-limit violations: {(grid_load > MAX_GRID_KW + 1e-8).sum()}")
print(f"Charger-limit violations: {(active_count > N_CHARGERS).sum()}")
print(f"Negative power entries: {(repaired < -1e-9).sum()}")
print(f"Fitness: {evaluate_fitness(repaired, scenario):.3f}")

output = pd.DataFrame(
    repaired,
    columns=[f"slot_{t}" for t in range(N_SLOTS)]
)
output.insert(0, "ev_id", scenario["ev_id"].to_numpy())

out_file = RESULTS_DIR / "repair_test_schedule.csv"
output.to_csv(out_file, index=False)
print(f"Test schedule saved to: {out_file}")
