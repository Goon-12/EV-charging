"""
Tests for the demo web app in ``webapp/``.

The app is a presentation layer, but two things still have to be true: the
numbers it shows are the committed numbers, and a live run actually runs the
project's optimizers. Both are checked here against the same artifacts and the
same evaluation code the experiments use.

The server is started on port 0 in a daemon thread, so these tests never fight
over 8000 or leave a process behind.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pandas as pd
import pytest

import config as cfg
from webapp.app import create_server
from webapp.engine import BUDGETS, count_ga_evaluations, count_pso_evaluations


@pytest.fixture(scope="module")
def live_server():
    httpd, engine = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", engine
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def request_json(url, payload=None, timeout=120):
    """Return ``(status, decoded body)`` for success and HTTP errors alike."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def get_text(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, response.read().decode("utf-8")


# --------------------------------------------------------------------------
# Committed results
# --------------------------------------------------------------------------
def test_overview_reports_the_committed_numbers(live_server):
    base, engine = live_server
    status, overview = request_json(base + "/api/overview")
    assert status == 200

    committed = pd.read_csv(engine.results_dir / "service_vs_optimal.csv").set_index(
        "method"
    )
    rows = {row["method"]: row for row in overview["methods"]}
    assert rows["PSO"]["service_delivered_kwh"] == pytest.approx(
        committed.loc["PSO", "service_delivered_kwh"]
    )
    assert rows["GA"]["service_delivered_kwh"] == pytest.approx(
        committed.loc["GA", "service_delivered_kwh"]
    )
    # The headline value is rounded to 0.01 kWh for display.
    assert overview["headline"]["reference_kwh"] == pytest.approx(
        pd.read_csv(engine.results_dir / "reference_optimum.csv")[
            "delivered_energy_kwh"
        ].max(),
        abs=0.01,
    )
    assert rows["PSO"]["family"] == "metaheuristic"
    assert rows["EDF (chronological)"]["family"] == "conventional"
    assert rows["EDF-priority (repair-based)"]["family"] == "priority"
    assert overview["site"]["n_ev"] == 100
    assert overview["site"]["n_chargers"] == 20
    assert overview["headline"]["violations"] == 0
    assert (
        overview["port_pressure"]["port_hours_needed"]
        > overview["port_pressure"]["port_hours_available"]
    )


def test_schedule_payload_matches_its_metrics(live_server):
    base, _ = live_server
    status, payload = request_json(base + "/api/schedule?method=PSO")
    assert status == 200

    n_ev, n_slots = payload["shape"]
    assert [n_ev, n_slots] == [100, cfg.N_SLOTS]
    assert len(payload["load_kw"]) == n_slots
    assert len(payload["occupancy"]) == n_slots
    # base64 of n_ev * n_slots bytes is 4/3 of the raw size.
    assert len(payload["cells"]) == (n_ev * n_slots * 4) // 3

    delivered = sum(payload["load_kw"]) * cfg.STEP_HOURS
    assert delivered == pytest.approx(
        payload["metrics"]["service_delivered_kwh"], rel=1e-3
    )
    assert max(payload["occupancy"]) <= cfg.N_CHARGERS


def test_page_and_assets_are_served(live_server):
    base, _ = live_server
    status, html = get_text(base + "/")
    assert status == 200
    assert "Charging scheduler lab" in html

    status, script = get_text(base + "/static/app.js")
    assert status == 200
    assert "renderServiceChart" in script

    status, stylesheet = get_text(base + "/static/styles.css")
    assert status == 200
    assert "--accent" in stylesheet


def test_artifacts_are_whitelisted(live_server):
    base, _ = live_server
    status, csv_text = get_text(base + "/artifacts/service_vs_optimal.csv")
    assert status == 200
    assert csv_text.startswith("method,")

    # The schedule archive is a binary artifact; the app must not serve it.
    status, body = request_json(base + "/artifacts/schedules.npz")
    assert status == 403
    assert "error" in body

    status, _ = request_json(base + "/artifacts/not-a-file.csv")
    assert status == 404


# --------------------------------------------------------------------------
# Live runs
# --------------------------------------------------------------------------
def test_live_run_uses_the_real_optimizers(live_server):
    base, _ = live_server
    spec = BUDGETS["fast"]
    status, result = request_json(
        base + "/api/optimise",
        {
            "budget": "fast",
            "seed": 46,
            "algorithms": ["GA", "PSO"],
            "weights": {"unmet": 0.55, "cost": 0.30, "peak": 0.08,
                        "battery": 0.02, "solar": 0.05},
        },
    )
    assert status == 200, result
    rows = {row["method"]: row for row in result["rows"]}

    for name in ("GA", "PSO", "EDF (chronological)"):
        assert name in rows
        assert rows[name]["feasible"] is True
        assert rows[name]["violations"] == 0

    assert rows["GA"]["evaluations"] == count_ga_evaluations(
        spec["ga"]["population_size"],
        spec["ga"]["generations"],
        spec["ga"]["polish_evaluations"],
    )
    assert rows["PSO"]["evaluations"] == count_pso_evaluations(
        spec["pso"]["particles"],
        spec["pso"]["iterations"],
        spec["pso"]["polish_evaluations"],
    )

    # The headline claim, re-checked with a small live budget.
    assert rows["PSO"]["delivered_kwh"] > rows["EDF (chronological)"]["delivered_kwh"]
    assert rows["GA"]["delivered_kwh"] > rows["EDF (chronological)"]["delivered_kwh"]
    assert rows["PSO"]["pct_of_optimum"] > 90

    assert result["schedules"]["PSO"]["shape"] == [100, cfg.N_SLOTS]
    assert len(result["schedules"]["PSO"]["history"]) > 0
    assert result["schedules"]["PSO"]["cells"]


def test_live_weights_are_normalised(live_server):
    base, _ = live_server
    status, result = request_json(
        base + "/api/optimise",
        {
            "budget": "fast",
            "seed": 42,
            "algorithms": ["PSO"],
            "weights": {"unmet": 5.5, "cost": 3.0, "peak": 0.8,
                        "battery": 0.02, "solar": 0.05},
        },
    )
    assert status == 200, result
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-5)


def test_bad_requests_are_rejected(live_server):
    base, _ = live_server

    status, _ = request_json(base + "/api/schedule?method=NotAMethod")
    assert status == 404

    status, _ = request_json(base + "/api/optimise", {"budget": "enormous"})
    assert status == 400

    status, _ = request_json(
        base + "/api/optimise", {"weights": {"unmet": -1.0}, "algorithms": ["GA"]}
    )
    assert status == 400

    status, _ = request_json(base + "/api/optimise", {"algorithms": []})
    assert status == 400


# --------------------------------------------------------------------------
# Budget arithmetic must not drift from the experiment runner
# --------------------------------------------------------------------------
def test_budget_counts_match_the_experiment_runner():
    from run_final_experiments import (
        count_ga_evaluations as runner_ga,
        count_pso_evaluations as runner_pso,
    )

    for key, spec in BUDGETS.items():
        assert count_ga_evaluations(
            spec["ga"]["population_size"],
            spec["ga"]["generations"],
            spec["ga"]["polish_evaluations"],
        ) == runner_ga(
            spec["ga"]["population_size"],
            spec["ga"]["generations"],
            polish_evaluations=spec["ga"]["polish_evaluations"],
        ), key
        assert count_pso_evaluations(
            spec["pso"]["particles"],
            spec["pso"]["iterations"],
            spec["pso"]["polish_evaluations"],
        ) == runner_pso(
            spec["pso"]["particles"],
            spec["pso"]["iterations"],
            polish_evaluations=spec["pso"]["polish_evaluations"],
        ), key
