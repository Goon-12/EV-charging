"""Run comparable one-shot and rolling-horizon experiments."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from multiobjective_core import load_inputs, evaluate_schedule, objective, STEP_HOURS, repair_schedule
from multiobjective_optimizers import optimize_ga, optimize_pso
from run_rolling_horizon import optimize_rolling

def make_baseline(ev, solar):
    n_ev=len(ev); n_slots=len(solar)
    # FCFS: earlier-arriving vehicles receive higher constant priority.
    arr=ev["arrival_slot"].to_numpy(float)
    rank=(arr.max()-arr)/(max(arr.max()-arr.min(),1.0))
    fcfs_priority=np.repeat(rank[:,None],n_slots,axis=1)
    fcfs=repair_schedule(fcfs_priority,ev,solar)
    # EDF/urgency baseline: priority increases as departure approaches.
    a=ev["arrival_slot"].to_numpy(int); d=ev["departure_slot"].to_numpy(int)
    urg=np.zeros((n_ev,n_slots))
    for t in range(n_slots):
        eligible=(a<=t)&(d>t)
        urg[eligible,t]=1/np.maximum(d[eligible]-t,1)
    edf=repair_schedule(urg,ev,solar)
    # Uncontrolled: each EV charges at max power whenever connected, ignoring site limits.
    uncontrolled=np.zeros((n_ev,n_slots))
    req=ev["required_energy_kwh"].to_numpy(float)
    pmax=ev["max_power_kw"].to_numpy(float)
    for i in range(n_ev):
        rem=req[i]
        for t in range(max(0,a[i]),min(n_slots,d[i])):
            p=min(pmax[i],rem/STEP_HOURS)
            uncontrolled[i,t]=p
            rem-=p*STEP_HOURS
            if rem<=1e-8: break
    return {"FCFS":fcfs,"EDF":edf,"Uncontrolled (infeasible reference)":uncontrolled}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scenario",default="data/processed/ev_scenario.csv")
    ap.add_argument("--solar",default="data/processed/solar_profile_36h.csv")
    ap.add_argument("--out",default="results/final_unified")
    ap.add_argument("--quick",action="store_true",help="Use smaller GA/PSO settings for a smoke test.")
    ap.add_argument("--skip-rolling",action="store_true",help="Skip rolling-horizon runs.")
    args=ap.parse_args()
    ev,solar=load_inputs(args.scenario,args.solar)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.quick:
        ga_kw=dict(population_size=8,generations=5,seed=46)
        pso_kw=dict(particles=6,iterations=4,seed=42)
    else:
        ga_kw=dict(population_size=20,generations=30,seed=46)
        pso_kw=dict(particles=12,iterations=15,seed=42)
    schedules={}; rows=[]
    def record(name,schedule,seconds):
        metrics=evaluate_schedule(schedule,ev,solar)
        metrics["method"]=name
        metrics["objective_score"]=objective(schedule,ev,solar)
        metrics["runtime_seconds"]=seconds
        rows.append(metrics); schedules[name]=schedule
    for name,fn,kw in [("GA",optimize_ga,ga_kw),("PSO",optimize_pso,pso_kw)]:
        t0=time.perf_counter(); schedule,score=fn(ev,solar,**kw); elapsed=time.perf_counter()-t0
        record(name,schedule,elapsed)
    if not args.skip_rolling:
        for method,name in [("ga","GA rolling horizon"),("pso","PSO rolling horizon")]:
            t0=time.perf_counter()
            if args.quick:
                schedule=optimize_rolling(ev,solar,method=method,window_slots=24,
                    population_size=6,generations=3,particles=4,iterations=2,
                    seed=46 if method=="ga" else 42)
            else:
                schedule=optimize_rolling(ev,solar,method=method,window_slots=24,
                    population_size=8,generations=5,particles=6,iterations=4,
                    seed=46 if method=="ga" else 42)
            record(name,schedule,time.perf_counter()-t0)
    for name,schedule in make_baseline(ev,solar).items():
        record(name,schedule,0.0)
    table=pd.DataFrame(rows)
    # Put method first, then key performance metrics.
    cols=["method","objective_score","required_energy_kwh","delivered_energy_kwh","unmet_energy_kwh",
          "demand_met_pct","fully_charged_evs","peak_ev_load_kw","peak_grid_import_kw",
          "grid_cost_units","solar_used_kwh","solar_utilization_pct","battery_stress_proxy",
          "constraint_violations","runtime_seconds"]
    table=table[cols].sort_values("method")
    table.to_csv(out/"comparison_metrics.csv",index=False)
    np.savez_compressed(out/"schedules.npz",**schedules)
    print("\nComparison results (all schedules use the same evaluator):")
    print(table.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    print(f"\nSaved: {out/'comparison_metrics.csv'}")
    print(f"Saved: {out/'schedules.npz'}")
    print("Interpretation: uncontrolled is intentionally allowed to violate site constraints; use it only as a reference.")

if __name__=="__main__": main()
