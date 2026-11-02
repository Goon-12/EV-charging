# Demo web app

A single-page dashboard for the Tier 2 results, plus a live optimizer you can run
from the browser. Standard library only — it adds nothing to
`requirements.txt`.

```bash
cd EV-charging
.venv/Scripts/python webapp/app.py          # Windows
.venv/bin/python webapp/app.py              # Linux / macOS
```

The page opens at <http://127.0.0.1:8000/>. Useful flags:

```bash
.venv/Scripts/python webapp/app.py --port 9000 --no-open
.venv/Scripts/python webapp/app.py --results results/final_unified
```

## What is on the page

| Tab | Content |
| --- | --- |
| **Results** | Scenario facts, port-hours supply vs demand, KPI cards, delivered energy per method against the best conventional policy and the MILP optimum, cost-per-kWh vs service, and the EDF/PSO trade-off table. |
| **Load profiles** | Interactive overlay of the committed schedules with solar availability, the 125 kW contract and the evening tariff band. Computes delivered energy, peak and port utilisation from the schedules themselves. |
| **Schedule** | EV × 15-minute heatmap for any committed method (or a live run), port occupancy per slot, and the six least-served EVs. Hover any cell for the EV's window, requirement and delivery. |
| **Live run** | Choose a budget, seed, optimizer and objective weights, then run GA/PSO on the real scenario. Shows the verdict, a comparison table, live load profiles and convergence. Weights are normalised server-side. |
| **Method & evidence** | Genome/decoder and MILP-referee explanation, the reference-optimum table with MIP gaps and input hashes, seed stability, the run manifest, reproduction commands and the limitations. |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/overview` | Scenario, committed tables, reference optima, manifest, test count |
| GET | `/api/schedule?method=PSO` | One schedule: load profile, occupancy, per-EV summary, base64 heatmap cells |
| POST | `/api/optimise` | Run baselines and GA/PSO live at a small budget |
| GET | `/artifacts/<name>` | Download a committed CSV or PNG |
| GET | `/api/health` | Liveness |

`POST /api/optimise` body:

```json
{
  "budget": "fast",
  "seed": 46,
  "algorithms": ["GA", "PSO"],
  "weights": {"unmet": 0.55, "cost": 0.30, "peak": 0.08, "battery": 0.02, "solar": 0.05}
}
```

Budgets (evaluations, exact):

| Key | GA | PSO | Typical wall time (both) |
| --- | --- | --- | --- |
| `fast` | 178 | 250 | a few seconds |
| `standard` | 464 | 584 | ~10 s |
| `deep` | 1,018 | 956 | ~20 s |

The committed run used 12,680 (GA) and 13,030 (PSO) evaluations. A live result is
a demonstration of the mechanism at a smaller budget; the dashboard labels it as
such rather than presenting it as the reported number.

## Honesty notes

- The denominator for "% of optimum" is `Max service (MILP)` from
  `results/final_unified/reference_optimum.csv` (1,293.7 kWh, MIP gap 0.76%). It
  is solved for the committed scenario, which is also the one a live run uses.
- Live runs go through `multiobjective_core` and `multiobjective_optimizers`, the
  same evaluator, decoder and objective as `src/run_final_experiments.py`.
- Live budgets are computed by `engine.count_*_evaluations`, and
  `tests/test_webapp.py` asserts those formulas against the experiment runner so
  the displayed counts cannot drift.
- Nothing in the app writes to `results/`; it reads committed artifacts only.

## Tests

```bash
.venv/Scripts/python -m pytest tests/test_webapp.py -q
```

The tests start the server on an ephemeral port and check that the API numbers
match the CSVs, that a schedule payload agrees with its own metrics, that only
CSV/PNG artifacts are downloadable, and that a fast live run still beats EDF with
both optimizers.

## Layout

```
webapp/
  app.py              HTTP server, four endpoints, static files
  engine.py           artifact loading, schedule encoding, live experiments
  static/index.html   page shell
  static/styles.css   styling
  static/app.js       charts, schedule explorer, live-run client
```
