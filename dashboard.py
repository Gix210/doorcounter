"""Dashboard for the office network: serves dashboard.html and the counters' data.

  python dashboard.py              http://<this PC>:8790
  python dashboard.py --port 8791

GET /api/data?from=YYYY-MM-DDTHH:MM&to=YYYY-MM-DDTHH:MM returns every event
in that range from each camera's log, plus whether each counter is running.
"""
import argparse
import csv
import json
import socket
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from common import HERE, camera_settings, load_settings, log

PAGE = HERE / "dashboard.html"
TIME_FMT = "%Y-%m-%d %H:%M:%S"
RUNNING_WITHIN = 180  # a live counter writes a status line every minute


def read_rows(path):
    if not path.exists():
        return []
    for attempt in range(3):
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                return list(csv.DictReader(f))
        except PermissionError:  # the counter is appending this instant
            time.sleep(0.2)
    return []


def counter_status(channel):
    """Running if the counter's own log was written in the last few minutes."""
    path = HERE / f"counter_ch{channel:02d}.log"
    if not path.exists():
        return {"running": False, "last_seen": None}
    mtime = path.stat().st_mtime
    return {"running": time.time() - mtime < RUNNING_WITHIN,
            "last_seen": datetime.fromtimestamp(mtime).strftime(TIME_FMT)}


def data(start, end):
    settings = load_settings()
    lo, hi = start.strftime(TIME_FMT), end.strftime(TIME_FMT)
    cameras, first_day = [], None
    for entry in settings["cameras"]:
        cam = camera_settings(settings, entry["channel"])
        rows = read_rows(HERE / cam["log"])
        key = "entered_at" if cam["mode"] == "stay" else "time"
        if rows:
            day = min(r[key] for r in rows)[:10]
            first_day = min(first_day or day, day)
        out = {"channel": cam["channel"], "name": cam["name"], "mode": cam["mode"],
               "status": counter_status(cam["channel"])}
        picked = [r for r in rows if lo <= r[key] < hi]
        if cam["mode"] == "stay":
            out["visits"] = [[r["entered_at"], r["left_at"], int(r["stay_seconds"])] for r in picked]
        else:
            out.update(in_label=cam.get("in_label", "In"), out_label=cam.get("out_label", "Out"),
                       events=[[r["time"], r["door"], r["direction"]] for r in picked])
        cameras.append(out)
    return {"generated": datetime.now().strftime(TIME_FMT), "first_day": first_day, "cameras": cameras}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type, headers=()):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/data":
            q = parse_qs(url.query)
            try:
                end = datetime.fromisoformat(q["to"][0]) if "to" in q else datetime.now()
                start = datetime.fromisoformat(q["from"][0]) if "from" in q else end - timedelta(days=1)
            except ValueError:
                self._send(400, b'{"error": "from/to must look like 2026-10-22T08:00"}', "application/json")
                return
            self._send(200, json.dumps(data(start, end)).encode(), "application/json")
        else:
            self._send(404, b"Not found", "text/plain")

    def log_message(self, *args):
        pass  # keep the console for real news


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "this-pc"
    log(f"dashboard on http://{ip}:{args.port} (and http://localhost:{args.port} on this PC)")
    server.serve_forever()


if __name__ == "__main__":
    main()
