"""
Tiny HTTP server for the EV charging demo.

Deliberately standard library only: the project's requirements are pinned and a
demonstration app should not add a web framework to them. It serves one page,
four JSON endpoints and the static assets next to this file.

    .venv/Scripts/python webapp/app.py
    .venv/Scripts/python webapp/app.py --port 9000 --no-open

Endpoints
---------
GET  /api/overview          committed results, scenario facts, reference optima
GET  /api/schedule?method=X one committed schedule + load profile
POST /api/optimise          run baselines and GA/PSO live at a small budget
GET  /artifacts/<name>      download a committed result file (csv/png)
GET  /static/<path>         page assets
GET  /                      the dashboard
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:  # imported as a package (python -m webapp.app)
    from .engine import DemoEngine, RESULTS_DIR, WEBAPP_DIR
except ImportError:  # run as a script (python webapp/app.py)
    from engine import DemoEngine, RESULTS_DIR, WEBAPP_DIR

STATIC_DIR = WEBAPP_DIR / "static"
MAX_BODY_BYTES = 1_000_000

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def _content_type(path):
    return CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


def make_handler(engine: DemoEngine):
    """Bind one engine into a request handler class."""

    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "EVChargingDemo/1.0"
        protocol_version = "HTTP/1.1"

        # -- plumbing ---------------------------------------------------
        def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
            if args and str(args[0]).startswith(("GET /static", "GET /favicon")):
                return
            super().log_message(fmt, *args)

        def _send(self, status, body=b"", content_type="application/octet-stream"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body and self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, payload, status=200):
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _error(self, status, message):
            self._json({"error": message}, status)

        def _file(self, path):
            path = Path(path)
            if not path.is_file():
                self._error(404, "Not found")
                return
            self._send(200, path.read_bytes(), _content_type(path))

        # -- routes -----------------------------------------------------
        def do_GET(self):  # noqa: N802 - stdlib signature
            parsed = urlparse(self.path)
            route = parsed.path

            try:
                if route == "/":
                    self._file(STATIC_DIR / "index.html")
                elif route.startswith("/static/"):
                    self._serve_static(route[len("/static/"):])
                elif route == "/api/overview":
                    self._json(engine.overview())
                elif route == "/api/schedule":
                    method = parse_qs(parsed.query).get("method", [""])[0]
                    if not method:
                        self._error(400, "Missing 'method' query parameter.")
                        return
                    self._json(engine.schedule(method))
                elif route == "/api/health":
                    self._json({"ok": True})
                elif route.startswith("/artifacts/"):
                    self._serve_artifact(route[len("/artifacts/"):])
                elif route == "/favicon.ico":
                    self._send(204)
                else:
                    self._error(404, "Not found")
            except KeyError as exc:
                self._error(404, str(exc).strip("'"))
            except Exception as exc:  # pragma: no cover - last-resort reporting
                self._error(500, f"{type(exc).__name__}: {exc}")

        def do_POST(self):  # noqa: N802 - stdlib signature
            parsed = urlparse(self.path)
            if parsed.path != "/api/optimise":
                self._error(404, "Not found")
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(400, "Invalid Content-Length.")
                return
            if length <= 0:
                self._error(400, "Empty request body.")
                return
            if length > MAX_BODY_BYTES:
                self._error(413, "Request body too large.")
                return

            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(400, "Body must be valid JSON.")
                return

            try:
                self._json(engine.run(payload))
            except ValueError as exc:
                self._error(400, str(exc))
            except RuntimeError as exc:
                self._error(429, str(exc))
            except Exception as exc:  # pragma: no cover - surfaced to the UI
                self._error(500, f"{type(exc).__name__}: {exc}")

        # -- helpers ----------------------------------------------------
        def _serve_static(self, relative):
            target = (STATIC_DIR / unquote(relative)).resolve()
            if STATIC_DIR.resolve() not in target.parents:
                self._error(403, "Forbidden")
                return
            self._file(target)

        def _serve_artifact(self, name):
            name = unquote(name)
            target = (RESULTS_DIR / name).resolve()
            if RESULTS_DIR.resolve() not in target.parents:
                self._error(403, "Forbidden")
                return
            if target.suffix.lower() not in (".csv", ".png"):
                self._error(403, "Only committed CSV and PNG artifacts are served.")
                return
            self._file(target)

    return DemoHandler


def create_server(host="127.0.0.1", port=8000, engine=None):
    """Create (but do not start) the demo server. Useful for tests."""
    engine = engine or DemoEngine()
    httpd = ThreadingHTTPServer((host, port), make_handler(engine))
    httpd.daemon_threads = True
    return httpd, engine


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--results",
        default=None,
        help="Artifact directory (default: results/final_unified).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser window on startup.",
    )
    args = parser.parse_args(argv)

    engine = DemoEngine(args.results)
    httpd, _ = create_server(args.host, args.port, engine)
    url = f"http://{args.host}:{httpd.server_port}/"

    print("EV charging demo")
    print(f"  scenario : {len(engine.ev)} EVs, {len(engine.solar)} slots, "
          f"{int(engine.ev['required_energy_kwh'].sum())} kWh requested")
    print(f"  artifacts: {engine.results_dir}")
    print(f"  serving  : {url}")
    print("  stop     : Ctrl+C")

    if not args.no_open:
        threading.Timer(0.8, webbrowser.open, [url]).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
