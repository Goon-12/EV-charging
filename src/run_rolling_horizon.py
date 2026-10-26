"""Receding-horizon EV charging controller. Executes only the first 15-min slot, then replans."""
import numpy as np
from multiobjective_core import STEP_HOURS, repair_schedule
from multiobjective_optimizers import optimize_ga, optimize_pso

def optimize_rolling(ev, solar_kw, method="ga", window_slots=24, seed=46,
                     population_size=8, generations=5, particles=6, iterations=4):
    n_ev, n_slots = len(ev), len(solar_kw)
    arr=ev["arrival_slot"].to_numpy(int)
    dep=ev["departure_slot"].to_numpy(int)
    original_req=ev["required_energy_kwh"].to_numpy(float)
    remaining=original_req.copy()
    pmax=ev["max_power_kw"].to_numpy(float)
    final=np.zeros((n_ev,n_slots),dtype=float)
    for t in range(n_slots):
        end=min(n_slots,t+window_slots)
        width=end-t
        if width <= 0: break
        local=ev.copy()
        local["arrival_slot"]=np.clip(arr-t,0,width)
        local["departure_slot"]=np.clip(dep-t,0,width)
        # EVs whose departure has passed cannot receive any more energy.
        local_req=remaining.copy()
        local_req[dep <= t]=0.0
        local["required_energy_kwh"]=local_req
        local_solar=np.asarray(solar_kw[t:end],float)
        # Small optimization per replanning step keeps runtime manageable.
        if method.lower()=="pso":
            local_schedule,_=optimize_pso(local,local_solar,particles=particles,iterations=iterations,seed=seed+t)
        else:
            local_schedule,_=optimize_ga(local,local_solar,population_size=population_size,
                                         generations=generations,seed=seed+t)
        first=local_schedule[:,0]
        final[:,t]=first
        remaining=np.maximum(remaining-first*STEP_HOURS,0.0)
    return final
