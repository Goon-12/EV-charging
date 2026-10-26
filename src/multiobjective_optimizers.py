"""GA/PSO implementations that share the same repair and objective."""
from __future__ import annotations
import numpy as np
from multiobjective_core import repair_schedule, objective

def optimize_ga(ev, solar_kw, population_size=20, generations=30, seed=46, weights=None):
    rng = np.random.default_rng(seed)
    n_ev, n_slots = len(ev), len(solar_kw)
    pop = [rng.random((n_ev,n_slots)) for _ in range(population_size)]
    cache = {}
    def assess(x):
        key = x.tobytes()
        if key not in cache:
            s = repair_schedule(x, ev, solar_kw)
            cache[key] = (objective(s,ev,solar_kw,weights), s)
        return cache[key]
    scored = [(assess(x)[0], x) for x in pop]
    for _ in range(generations):
        scored.sort(key=lambda z:z[0])
        next_pop = [scored[i][1].copy() for i in range(min(2,len(scored)))]
        while len(next_pop) < population_size:
            # tournament selection
            a,b = rng.choice(len(scored),2,replace=False)
            p1 = scored[a][1] if scored[a][0] < scored[b][0] else scored[b][1]
            a,b = rng.choice(len(scored),2,replace=False)
            p2 = scored[a][1] if scored[a][0] < scored[b][0] else scored[b][1]
            mask = rng.random((n_ev,n_slots)) < 0.5
            child = np.where(mask,p1,p2)
            mut = rng.random((n_ev,n_slots)) < 0.04
            child[mut] = rng.random(np.sum(mut))
            next_pop.append(child)
        pop = next_pop
        scored = [(assess(x)[0],x) for x in pop]
    scored.sort(key=lambda z:z[0])
    return repair_schedule(scored[0][1],ev,solar_kw), float(scored[0][0])

def optimize_pso(ev, solar_kw, particles=12, iterations=15, seed=42, weights=None):
    rng = np.random.default_rng(seed)
    shape=(len(ev),len(solar_kw))
    pos=rng.random((particles,*shape))
    vel=np.zeros_like(pos)
    pbest=pos.copy()
    pscore=np.full(particles,np.inf)
    cache={}
    def assess(x):
        key=x.tobytes()
        if key not in cache:
            s=repair_schedule(x,ev,solar_kw)
            cache[key]=(objective(s,ev,solar_kw,weights),s)
        return cache[key]
    for p in range(particles):
        pscore[p]=assess(pos[p])[0]
    gi=int(np.argmin(pscore)); gbest=pbest[gi].copy(); gscore=float(pscore[gi])
    for _ in range(iterations):
        for p in range(particles):
            r1=rng.random(shape); r2=rng.random(shape)
            vel[p]=0.7*vel[p]+1.5*r1*(pbest[p]-pos[p])+1.5*r2*(gbest-pos[p])
            vel[p]=np.clip(vel[p],-0.25,0.25)
            pos[p]=np.clip(pos[p]+vel[p],0,1)
            score,_=assess(pos[p])
            if score < pscore[p]:
                pscore[p]=score; pbest[p]=pos[p].copy()
                if score < gscore:
                    gscore=float(score); gbest=pos[p].copy()
    return repair_schedule(gbest,ev,solar_kw), gscore
