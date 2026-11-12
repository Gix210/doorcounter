"""Shared helpers for the door counter: NVR login, RTSP access, settings."""
import hashlib
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from urllib.parse import quote

# Pull RTSP over TCP (UDP drops packets and smears frames), and keep FFmpeg
# quiet so it never prints a stream URL with the password in it.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / "lorex.env"
SETTINGS_FILE = HERE / "settings.json"
SNAPSHOT_DIR = HERE / "snapshots"

# Shared defaults; each entry in "cameras" can override any of them.
DEFAULT_SETTINGS = {
    "stream": "sub",           # "sub" (lighter) or "main" (full resolution)
    "model": "yolo26n.pt",
    "device": None,            # None: CPU; "intel:gpu" with an OpenVINO model runs on the Intel graphics chip
    "imgsz": 640,
    "confidence": 0.1,         # low on purpose: ByteTrack uses weak detections to keep tracks alive
    "track_activation_threshold": 0.5,
    "high_conf_threshold": 0.5,
    "track_buffer_seconds": 2.0,
    "stitch_seconds": 1.0,     # rejoin a track that reappears where another vanished this recently
    "process_fps": None,       # analyse at most this many frames a second (None: every frame)
    "threads": None,           # CPU threads for this camera's counter (None: all)
    # mode "doors": count crossings of door lines
    "line_margin": 0.03,       # dead zone each side of a door line, as a fraction of frame height
    "min_crossing_frames": 2,  # frames a person must stay on the new side before it counts
    "settle_seconds": 1.5,     # ...and seconds before the crossing is final (stops flip-flops in a doorway)
    "stay_scope": "none",      # pair OUTs with INs for a stay time: "door", "room" (any door) or "none"
    "max_stay_minutes": 480,   # INs older than this are dropped from stay matching
    "doors": [],               # {"name", "line": [[x, y], [x, y]], "inside": [x, y]}, 0..1 coords
    # mode "stay": time each person spends in view
    "rejoin_seconds": 10,      # gone this long = left; back sooner and nearby = same visit
    "min_stay_seconds": 5,     # shorter visits are people passing the edge of the view
    "cameras": [],             # {"channel", "name", "mode": "doors" | "stay", "log", overrides...}
}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def load_settings():
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        settings.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    return settings


def camera_settings(settings, channel=None):
    """One camera's settings: the shared defaults overlaid with its own entry."""
    cameras = settings["cameras"]
    if channel is None:
        if len(cameras) != 1:
            raise SystemExit(f"Pass --channel; settings.json has cameras {[c['channel'] for c in cameras]}")
        cam = cameras[0]
    else:
        cam = next((c for c in cameras if c["channel"] == channel), None)
        if cam is None:
            raise SystemExit(f"Channel {channel} is not in settings.json")
    merged = {k: v for k, v in settings.items() if k != "cameras"}
    merged.update(cam)
    return merged


def save_settings(settings):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ NVR login

def lorex_config():
    if not ENV_FILE.exists():
        raise SystemExit(f"Missing {ENV_FILE}")
    env = dotenv_values(ENV_FILE)
    cfg = {
        "host": (env.get("LOREX_HOST") or "").strip(),
        "port": int(env.get("LOREX_PORT") or 554),
        "user": (env.get("LOREX_USER") or "admin").strip(),
        "password": env.get("LOREX_PASS") or "",
    }
    if not cfg["host"] or not cfg["password"]:
        raise SystemExit(f"Fill in LOREX_HOST and LOREX_PASS in {ENV_FILE}")
    return cfg


def live_path(channel, main=False):
    # Lorex NVRs are Dahua inside, so they use Dahua's RTSP paths.
    return f"/cam/realmonitor?channel={channel}&subtype={0 if main else 1}"


def playback_path(channel, start):
    # No endtime: on cameras that record on motion, a fixed window with no
    # recording in it is a 404, while a bare start plays from the next recording.
    return f"/cam/playback?channel={channel}&starttime={start.strftime('%Y_%m_%d_%H_%M_%S')}"


def rtsp_url(cfg, path):
    """Stream URL with the login in it. Never print it; print masked_url()."""
    user, password = quote(cfg["user"], safe=""), quote(cfg["password"], safe="")
    return f"rtsp://{user}:{password}@{cfg['host']}:{cfg['port']}{path}"


def masked_url(cfg, path):
    return f"rtsp://{cfg['user']}:****@{cfg['host']}:{cfg['port']}{path}"


def _rtsp_request(sock, method, uri, cseq, auth=None):
    req = f"{method} {uri} RTSP/1.0\r\nCSeq: {cseq}\r\nUser-Agent: door-counter\r\nAccept: application/sdp\r\n"
    if auth:
        req += f"Authorization: {auth}\r\n"
    sock.sendall((req + "\r\n").encode())
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("the NVR closed the connection")
        data += chunk
    head, body = data.split(b"\r\n\r\n", 1)
    lines = head.decode(errors="replace").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers.setdefault(name.strip().lower(), []).append(value.strip())
    length = int(headers.get("content-length", ["0"])[0])
    while len(body) < length:
        chunk = sock.recv(4096)
        if not chunk:
            break
        body += chunk
    return status, headers, body.decode(errors="replace")


def _digest_auth(headers, user, password, method, uri):
    challenge = next((h for h in headers.get("www-authenticate", []) if h.lower().startswith("digest")), None)
    if challenge is None:
        return None
    fields = {k: a or b for k, a, b in re.findall(r'(\w+)=(?:"([^"]*)"|([^,\s]*))', challenge[6:])}
    md5 = lambda s: hashlib.md5(s.encode()).hexdigest()  # noqa: E731
    realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
    ha1, ha2 = md5(f"{user}:{realm}:{password}"), md5(f"{method}:{uri}")
    auth = f'Digest username="{user}", realm="{realm}", nonce="{nonce}", uri="{uri}"'
    if "auth" in fields.get("qop", "").split(","):
        nc, cnonce = "00000001", os.urandom(8).hex()
        auth += f', qop=auth, nc={nc}, cnonce="{cnonce}", response="{md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}")}"'
    else:
        auth += f', response="{md5(f"{ha1}:{nonce}:{ha2}")}"'
    if "opaque" in fields:
        auth += f', opaque="{fields["opaque"]}"'
    return auth


def rtsp_describe(cfg, path, timeout=5):
    """One RTSP DESCRIBE with digest login. Returns (status_code, sdp_text).

    Talks RTSP directly instead of going through OpenCV so a wrong password
    (401) can be told apart from a missing channel (400/404)."""
    uri = f"rtsp://{cfg['host']}:{cfg['port']}{path}"
    with socket.create_connection((cfg["host"], cfg["port"]), timeout=timeout) as sock:
        status, headers, body = _rtsp_request(sock, "DESCRIBE", uri, 1)
        if status == 401:
            auth = _digest_auth(headers, cfg["user"], cfg["password"], "DESCRIBE", uri)
            if auth:
                status, headers, body = _rtsp_request(sock, "DESCRIBE", uri, 2, auth)
    return status, body


# ---------------------------------------------------------------- video input

def open_capture(source, timeout_ms=10000):
    return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms])


def grab_frame(url, timeout_ms=8000, warmup=3):
    """Open a stream, read a few frames and return the last good one (or None)."""
    cap = open_capture(url, timeout_ms)
    frame = None
    try:
        for _ in range(warmup):
            ok, f = cap.read()
            if not ok:
                break
            frame = f
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return frame, fps


class FrameReader:
    """Reads a live stream on a background thread and keeps only the newest
    frame, so a slow detector skips frames instead of falling behind.
    Reconnects on its own when the stream drops."""

    def __init__(self, url, label):
        self.url, self.label = url, label
        self.fps = None
        self._frame, self._ts, self._seq = None, 0.0, 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        backoff = 2
        while not self._stop.is_set():
            cap = open_capture(self.url)
            if not cap.isOpened():
                log(f"cannot open {self.label}; retrying in {backoff} s")
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60)
                continue
            self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            log(f"connected to {self.label} ({self.fps:.0f} fps)")
            backoff = 2
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log("stream lost; reconnecting")
                    break
                with self._cond:
                    self._frame, self._ts, self._seq = frame, time.time(), self._seq + 1
                    self._cond.notify_all()
            cap.release()

    def latest(self):
        """The newest frame without waiting: (seq, timestamp, frame); frame is None before the first."""
        with self._cond:
            return self._seq, self._ts, self._frame

    def read(self, after_seq, timeout=30):
        """Wait for a frame newer than after_seq. Returns (seq, timestamp, frame) or None."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq, timeout):
                return None
            return self._seq, self._ts, self._frame

    def stop(self):
        self._stop.set()
