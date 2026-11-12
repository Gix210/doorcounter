"""Count people through doors, or time how long people stay in a room.

YOLO finds people and ByteTrack follows them from frame to frame. Each camera
in settings.json has a mode:

  doors  one virtual line per door; every crossing is an IN or OUT row in the
         camera's log. Optionally (stay_scope) each OUT is paired with the same
         person's IN, else the earliest unmatched IN, for a stay time.
  stay   times each person from when they appear in view until they leave it,
         whichever way they go; one row per visit.

  python door_counter.py --channel 16                  live
  python door_counter.py --channel 16 --show           live, with a preview window
  python door_counter.py --source clips/ch12_x.mp4     test on a recorded clip
"""
import argparse
import csv
import math
import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from trackers import ByteTrackTracker
from ultralytics import YOLO

from common import (HERE, FrameReader, camera_settings, live_path, load_settings, log, lorex_config, masked_url,
                    open_capture, rtsp_url)

TIME_FMT = "%Y-%m-%d %H:%M:%S"
ANCHOR = sv.Position.BOTTOM_CENTER  # a person's feet are what crosses the threshold


def full_speed():
    """Ask Windows not to power-throttle this process. Started by the scheduled
    task it has no window, and Windows 11 would otherwise run it in efficiency
    mode and analyse fewer frames than the camera sends."""
    if sys.platform != "win32":
        return
    import ctypes

    class PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    state = PowerThrottling(1, 0x1, 0)  # control EXECUTION_SPEED, state off = never throttle
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))


def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime(TIME_FMT)


def parse_time(text):
    return datetime.strptime(text, TIME_FMT).timestamp()


# ---------------------------------------------------------------------- doors

class Door:
    """A virtual line across a doorway, with a dead zone of `margin` (fraction
    of frame height) on each side. A person's side changes when their feet go
    from clearly on one side to clearly on the other for min_frames frames, and
    a crossing counts once they have stayed on the new side for `settle`
    seconds. Someone standing half hidden in the doorway, whose box keeps
    jumping between the sides, then counts once for where they end up, or not
    at all if they end up back where they started."""

    def __init__(self, cfg, width, height, margin, min_frames, settle):
        self.name = cfg["name"]
        self.start, self.end = [np.array([x * width, y * height]) for x, y in cfg["line"]]
        self.inside = np.array([cfg["inside"][0] * width, cfg["inside"][1] * height])
        self.length = float(np.linalg.norm(self.end - self.start))
        self.along = (self.end - self.start) / self.length
        self.normal = np.array([-self.along[1], self.along[0]])
        if (self.inside - self.start) @ self.normal < 0:
            self.normal = -self.normal  # normal points inside
        self.margin = margin * height
        self.min_frames = min_frames
        self.settle = settle
        # track_id -> {"side", "counted": side last counted, "since": ts side changed, "cand", "n"}
        self.state = {}
        self.counts = {"IN": 0, "OUT": 0}

    def update(self, detections, ts):
        """Returns [(direction, track_id, ts of the crossing)] that settled by ts."""
        for foot, tid in zip(detections.get_anchors_coordinates(ANCHOR), detections.tracker_id):
            rel = foot - self.start
            if not 0 <= rel @ self.along <= self.length:
                continue  # beside the doorway, not in front of it
            dist = rel @ self.normal
            if abs(dist) < self.margin:
                continue
            side = 1 if dist > 0 else -1
            st = self.state.pop(tid, None)
            if st is None:
                st = {"side": side, "counted": side, "since": ts, "cand": None, "n": 0}
            elif side == st["side"]:
                st["cand"], st["n"] = None, 0
            else:
                st["n"] = st["n"] + 1 if st["cand"] == side else 1
                st["cand"] = side
                if st["n"] >= self.min_frames:
                    st.update(side=side, since=ts, cand=None, n=0)
            self.state[tid] = st  # re-insert so the dict stays in last-seen order
        events = self._settled(lambda st: ts - st["since"] >= self.settle)
        while len(self.state) > 500:
            del self.state[next(iter(self.state))]  # forget the longest-gone tracks
        return events

    def flush(self):
        """End of a recorded clip: count crossings still settling."""
        return self._settled(lambda st: True)

    def _settled(self, ready):
        events = []
        for tid, st in self.state.items():
            if st["side"] != st["counted"] and ready(st):
                st["counted"] = st["side"]
                direction = "IN" if st["side"] > 0 else "OUT"
                self.counts[direction] += 1
                events.append((direction, int(tid), st["since"]))
        return events

    def draw(self, img):
        s = max(1, img.shape[1] / 800)  # keep lines and text readable on the main stream
        p1, p2 = tuple(map(int, self.start)), tuple(map(int, self.end))
        cv2.line(img, p1, p2, (0, 255, 255), round(2 * s))
        mid = (self.start + self.end) / 2
        tip = mid + (self.inside - mid) * 30 * s / max(np.linalg.norm(self.inside - mid), 1)
        cv2.arrowedLine(img, tuple(map(int, mid)), tuple(map(int, tip)), (0, 255, 255), round(2 * s), tipLength=0.4)
        label = f"{self.name}  IN {self.counts['IN']}  OUT {self.counts['OUT']}"
        org = (min(p1[0], p2[0]), max(min(p1[1], p2[1]) - round(8 * s), round(14 * s)))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 0), round(3 * s))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 255), round(s))


def inside_share(a, b):
    """Share of box a's area that lies inside box b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(w, 0) * max(h, 0) / area if area > 0 else 0


def drop_duplicates(detections, overlap=0.8):
    """YOLO26 skips non-max suppression and now and then returns a second,
    partial box on one person (just the upper body, say). When one box lies
    mostly inside another, keep only the more confident one."""
    if len(detections) < 2:
        return detections
    keep = []
    for i in np.argsort(-detections.confidence):
        box = detections.xyxy[i]
        if all(inside_share(box, detections.xyxy[k]) < overlap and inside_share(detections.xyxy[k], box) < overlap
               for k in keep):
            keep.append(i)
    return detections[np.sort(keep)]


class TrackStitcher:
    """Gives a new track the id of one that vanished in the same spot moments
    before. ByteTrack splits a person in two when their box changes shape
    suddenly, e.g. legs appearing as they step out from behind something; the
    new half would be born past the door line and its crossing lost."""

    def __init__(self, max_gap_seconds):
        self.max_gap = max_gap_seconds
        self.alias = {}  # tracker id -> id we report
        self.born = {}   # tracker id -> ts first seen
        self.last = {}   # reported id -> (ts, box) when last seen

    def update(self, detections, ts):
        raw = [int(t) for t in detections.tracker_id]
        for tid in raw:
            if tid not in self.alias:
                self.alias[tid], self.born[tid] = tid, ts
        in_frame = {self.alias[t] for t in raw}
        ids = []
        for box, tid in zip(detections.xyxy, raw):
            # Keep trying during a new track's first moments: the old track can
            # linger a frame or two alongside it (one partial, one full box).
            if self.alias[tid] == tid and ts - self.born[tid] <= self.max_gap:
                gone = [(ts - t, cid) for cid, (t, old) in self.last.items()
                        if cid != tid and cid not in in_frame and ts - t <= self.max_gap
                        and inside_share(old, box) >= 0.6]
                if gone:
                    self.alias[tid] = min(gone)[1]
                    in_frame.add(self.alias[tid])
            cid = self.alias[tid]
            if cid in ids:  # two boxes claim the same person: keep them apart
                cid = self.alias[tid] = tid
            ids.append(cid)
            self.last[cid] = (ts, box)
        for cid in [c for c, (t, _) in self.last.items() if ts - t > 60]:
            del self.last[cid]
        for tid in [t for t, b in self.born.items() if ts - b > 60 and self.alias[t] not in self.last]:
            del self.born[tid], self.alias[tid]
        detections.tracker_id = np.array(ids, dtype=int)
        return detections


# ------------------------------------------------------------- stay matching

class StayMatcher:
    """Pairs each OUT with an open IN to get a stay time.

    If the tracker followed the same person from IN to OUT, that pairing is
    exact ("track"). Otherwise the OUT closes the oldest open IN ("fifo").
    scope "door" keeps one queue per door; "room" shares one queue across all
    doors, for rooms people enter by one door and leave by another."""

    def __init__(self, max_stay_seconds, scope="door"):
        self.max_stay = max_stay_seconds
        self.scope = scope
        self.open = defaultdict(deque)  # queue key -> deque of (ts, track_id)

    def _queue(self, door):
        return self.open[door if self.scope == "door" else "room"]

    def entry(self, door, ts, track_id):
        self._queue(door).append((ts, track_id))

    def exit(self, door, ts, track_id):
        """Returns (entry_ts, method) or (None, "")."""
        queue = self._queue(door)
        while queue and ts - queue[0][0] > self.max_stay:
            queue.popleft()  # its OUT was missed; don't let it skew later matches
        for i, (entered, tid) in enumerate(queue):
            if tid == track_id and tid is not None:
                del queue[i]
                return entered, "track"
        return (queue.popleft()[0], "fifo") if queue else (None, "")

    def replay(self, path, now):
        """Rebuild the open INs from the log after a restart. Track ids start
        over on restart, so replayed INs can only be matched first-in-first-out."""
        if not path.exists():
            return 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ts = parse_time(row["time"])
                if now - ts > self.max_stay:
                    continue
                if row["direction"] == "IN":
                    self.entry(row["door"], ts, None)
                else:
                    self.exit(row["door"], ts, None)
        return sum(len(q) for q in self.open.values())


class EventLog:
    """Appends rows to the CSV, holding them in memory while the file is locked
    (Excel locks a CSV it has open)."""

    def __init__(self, path, fields):
        self.path = path
        self.fields = fields
        self.pending = []
        self._locked = False

    def write(self, row):
        self.pending.append(row)
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, self.fields)
                if f.tell() == 0:
                    f.write("﻿")  # BOM so Excel reads accented door names right
                    writer.writeheader()
                writer.writerows(self.pending)
            self.pending.clear()
            self._locked = False
        except PermissionError:
            if not self._locked:
                log(f"{self.path.name} is locked (open in Excel?); holding events until it's free")
            self._locked = True


# ---------------------------------------------------------------------- modes

class DoorCounting:
    """Mode "doors": IN/OUT rows for every door-line crossing."""

    fields = ["time", "door", "direction", "track_id", "stay_seconds", "entered_at", "matched_by"]

    def __init__(self, cam, width, height, event_log, replay_from=None):
        self.doors = [Door(d, width, height, cam["line_margin"], cam["min_crossing_frames"], cam["settle_seconds"])
                      for d in cam["doors"]]
        self.log = event_log
        self.matcher = None
        if cam["stay_scope"] != "none":
            self.matcher = StayMatcher(cam["max_stay_minutes"] * 60, cam["stay_scope"])
            if replay_from and (n := self.matcher.replay(replay_from, time.time())):
                log(f"{n} person(s) still inside from earlier in {replay_from.name}")

    def update(self, detections, ts, frame):
        return self._record([(door, e) for door in self.doors for e in door.update(detections, ts)])

    def finish(self):
        return self._record([(door, e) for door in self.doors for e in door.flush()])

    def _record(self, crossings):
        events = []
        for door, (direction, track_id, ts) in crossings:
            row = {"time": fmt_time(ts), "door": door.name, "direction": direction, "track_id": track_id}
            msg = ""
            if self.matcher and direction == "IN":
                self.matcher.entry(door.name, ts, track_id)
            elif self.matcher:
                entered, method = self.matcher.exit(door.name, ts, track_id)
                if entered is not None:
                    row.update(stay_seconds=round(ts - entered), entered_at=fmt_time(entered), matched_by=method)
                msg = f" (stayed {(ts - entered) / 60:.1f} min, {method})" if entered is not None else " (no open IN)"
            self.log.write(row)
            events.append({"msg": f"{door.name}: {direction} #{track_id}{msg}", "image": None,
                           "tag": f"{re.sub(r'\W+', '-', door.name)}_{direction}_{track_id}"})
        return events

    def label(self, track_id, ts):
        return f"#{track_id}"

    def draw(self, img):
        for door in self.doors:
            door.draw(img)

    def status(self):
        return "  ".join(f"{d.name} in {d.counts['IN']} / out {d.counts['OUT']}" for d in self.doors)


class VisitTiming:
    """Mode "stay": how long each person stays in view, whichever way they leave.

    A visit starts when a person is first seen and ends once they have been
    gone for rejoin_seconds. A new track that appears near where a visit's
    person vanished, within that time, continues the visit: people disappear
    behind things or bend out of view."""

    fields = ["entered_at", "left_at", "stay_seconds", "visit_id"]

    def __init__(self, cam, event_log):
        self.rejoin = cam["rejoin_seconds"]
        self.min_stay = cam["min_stay_seconds"]
        self.log = event_log
        self.visits = {}    # visit id -> {"start", "last", "box", "frame"}
        self.of_track = {}  # track id -> visit id
        self.next_id = 1
        self.finished = 0

    @staticmethod
    def _near(a, b):
        """Feet of the two boxes within one body height of each other."""
        dist = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, a[3] - b[3])
        return dist <= max(a[3] - a[1], b[3] - b[1])

    def update(self, detections, ts, frame):
        events = []
        people = [(box, int(tid)) for box, tid in zip(detections.xyxy, detections.tracker_id)]
        seen = {self.of_track[t] for _, t in people if self.of_track.get(t) in self.visits}
        for box, tid in people:
            vid = self.of_track.get(tid)
            if vid not in self.visits:
                back = [v for v, d in self.visits.items() if v not in seen and self._near(d["box"], box)]
                if back:
                    vid = max(back, key=lambda v: self.visits[v]["last"])  # the most recent to vanish
                else:
                    vid, self.next_id = self.next_id, self.next_id + 1
                    self.visits[vid] = {"start": ts}
                    events.append({"msg": f"visit {vid} started", "image": None, "tag": f"visit{vid}_start"})
                self.of_track[tid] = vid
                seen.add(vid)
            self.visits[vid].update(last=ts, box=box, frame=frame)
        for vid in [v for v, d in self.visits.items() if ts - d["last"] > self.rejoin]:
            events += self._close(vid)
        return events

    def _close(self, vid):
        v = self.visits.pop(vid)
        self.of_track = {t: x for t, x in self.of_track.items() if x != vid}
        stay = v["last"] - v["start"]
        if stay < self.min_stay:
            return []
        self.finished += 1
        self.log.write({"entered_at": fmt_time(v["start"]), "left_at": fmt_time(v["last"]),
                        "stay_seconds": round(stay), "visit_id": vid})
        img = v["frame"].copy()
        x1, y1, x2, y2 = v["box"].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return [{"msg": f"visit {vid} ended: stayed {stay / 60:.1f} min", "image": img, "tag": f"visit{vid}_end"}]

    def finish(self):
        """End of a recorded clip: close the visits still open."""
        events = []
        for vid in list(self.visits):
            events += self._close(vid)
        return events

    def label(self, track_id, ts):
        vid = self.of_track.get(track_id)
        if vid not in self.visits:
            return f"#{track_id}"
        return f"visit {vid} {(ts - self.visits[vid]['start']) / 60:.1f} min"

    def draw(self, img):
        pass

    def status(self):
        return f"{len(self.visits)} in view, {self.finished} visits logged"


# -------------------------------------------------------------------- sources

class LiveSource:
    def __init__(self, cam):
        cfg = lorex_config()
        path = live_path(cam["channel"], main=cam["stream"] == "main")
        self.reader = FrameReader(rtsp_url(cfg, path), masked_url(cfg, path))
        self.fps = None
        self.dropped = 0

    def frames(self):
        seq = 0
        while True:
            got = self.reader.read(seq)
            if got is None:
                log("no video for 30 s; still waiting for the NVR")
                continue
            if seq:
                self.dropped += got[0] - seq - 1
            seq, ts, frame = got
            self.fps = self.reader.fps or 15
            yield ts, frame


class FileSource:
    def __init__(self, path, start):
        self.cap = open_capture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
        self.start = start
        self.dropped = 0

    def frames(self):
        n = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield self.start + n / self.fps, frame
            n += 1


def clip_start_time(path, given):
    """Wall-clock time of a clip's first frame: --clip-start, else the
    YYYYMMDD-HHMMSS that record_clip.py puts in the name, else the file time."""
    if given:
        return datetime.strptime(given, "%Y-%m-%d %H:%M").timestamp()
    m = re.search(r"(\d{8}-\d{6})", path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp()
    return path.stat().st_mtime


def clip_channel(path):
    """The chNN that record_clip.py puts at the start of a clip's name."""
    m = re.match(r"ch(\d+)_", path.name)
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------- main

def annotate(frame, detections, mode, ts):
    img = frame.copy()
    for (x1, y1, x2, y2), tid in zip(detections.xyxy.astype(int), detections.tracker_id):
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 200, 80), 2)
        cv2.circle(img, ((x1 + x2) // 2, y2), 4, (80, 200, 80), -1)
        cv2.putText(img, mode.label(int(tid), ts), (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 2)
    mode.draw(img)
    return img


def save_picture(event, frame, detections, mode, ts):
    out = HERE / "events" / datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    img = event["image"] if event["image"] is not None else annotate(frame, detections, mode, ts)
    cv2.imwrite(str(out / f"{datetime.fromtimestamp(ts):%H%M%S}_{event['tag']}.jpg"), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", type=int, help="camera from settings.json (default: from the clip name)")
    ap.add_argument("--source", type=Path, help="video file to process instead of the live stream")
    ap.add_argument("--clip-start", help='wall-clock time of the clip\'s first frame, "YYYY-MM-DD HH:MM"')
    ap.add_argument("--log", type=Path, help="CSV to append to (default: the camera's log, or <clip>.csv)")
    ap.add_argument("--show", action="store_true", help="show a preview window (q to quit)")
    ap.add_argument("--save-video", type=Path, help="write the annotated video here")
    ap.add_argument("--save-events", action="store_true", help="save a picture of every event in events/")
    args = ap.parse_args()

    channel = args.channel or (clip_channel(args.source) if args.source else None)
    cam = camera_settings(load_settings(), channel)
    if cam["mode"] == "doors" and not cam["doors"]:
        raise SystemExit(f"No doors for channel {cam['channel']} in settings.json yet: run set_doors.py first.")
    if args.source:
        source = FileSource(args.source, clip_start_time(args.source, args.clip_start))
        log_path = args.log or args.source.with_suffix(".csv")
        if not args.log and log_path.exists():
            log_path.unlink()  # a clip run starts its own log fresh
    else:
        source = LiveSource(cam)
        log_path = args.log or HERE / cam["log"]

    full_speed()
    if cam["threads"]:
        torch.set_num_threads(cam["threads"])  # several counters on one PC: don't fight over every core
        cv2.setNumThreads(cam["threads"])
    model_path = HERE / cam["model"]
    model = YOLO(str(model_path) if model_path.exists() else cam["model"], task="detect")
    min_gap = 0.95 / cam["process_fps"] if cam["process_fps"] else 0
    log(f"{cam['name']}: mode {cam['mode']}, logging to {log_path}")

    tracker = mode = writer = None
    stitcher = TrackStitcher(cam["stitch_seconds"])
    frames_done, frames_at_status, last_ts, last_status, started = 0, 0, -math.inf, time.time(), time.time()
    try:
        for ts, frame in source.frames():
            if ts - last_ts < min_gap:
                continue  # this camera only needs process_fps frames a second
            last_ts = ts
            if mode is None:
                h, w = frame.shape[:2]
                fps = min(source.fps, cam["process_fps"] or math.inf)
                if cam["mode"] == "doors":
                    event_log = EventLog(log_path, DoorCounting.fields)
                    mode = DoorCounting(cam, w, h, event_log, None if args.source else log_path)
                else:
                    mode = VisitTiming(cam, EventLog(log_path, VisitTiming.fields))
                tracker = ByteTrackTracker(
                    lost_track_buffer=round(cam["track_buffer_seconds"] * 30),
                    frame_rate=fps,
                    track_activation_threshold=cam["track_activation_threshold"],
                    high_conf_det_threshold=cam["high_conf_threshold"])
                log(f"{w}x{h}, analysing {fps:.0f} frames a second")
                if args.save_video:
                    writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            result = model.predict(frame, classes=[0], conf=cam["confidence"], imgsz=cam["imgsz"], device=cam["device"],
                                   verbose=False)[0]
            detections = drop_duplicates(sv.Detections.from_ultralytics(result))
            detections = tracker.update(detections, timestamp=ts)
            detections = detections[detections.tracker_id >= 0]  # -1 = not tracked yet
            detections = stitcher.update(detections, ts)

            for event in mode.update(detections, ts, frame):
                log(event["msg"])
                if args.save_events:
                    save_picture(event, frame, detections, mode, ts)

            if args.show or writer:
                img = annotate(frame, detections, mode, ts)
                if writer:
                    writer.write(img)
                if args.show:
                    cv2.imshow(f"{cam['name']} (q to quit)", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frames_done += 1
            if time.time() - last_status >= 60:
                rate = (frames_done - frames_at_status) / (time.time() - last_status)
                log(f"{rate:.1f} fps analysed, {source.dropped} frames skipped | {mode.status()}")
                last_status, frames_at_status = time.time(), frames_done
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if mode:
        if args.source:
            for event in mode.finish():
                log(event["msg"])
                if args.save_events:
                    save_picture(event, None, None, mode, last_ts)
        elapsed = time.time() - started
        log(f"done: {frames_done} frames in {elapsed:.0f} s ({frames_done / max(elapsed, 1e-9):.1f} fps) | {mode.status()}")


if __name__ == "__main__":
    main()
