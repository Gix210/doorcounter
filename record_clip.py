"""Record a clip from the door camera to test the counter on.

  python record_clip.py --minutes 15                             live, starting now
  python record_clip.py --from "2026-10-22 06:45" --minutes 30   from the NVR's recording

Clips go to clips/chNN_YYYYMMDD-HHMMSS.mp4; door_counter.py reads the start
time back from the name.
"""
import argparse
import time
from datetime import datetime

import cv2

from common import HERE, camera_settings, live_path, load_settings, log, lorex_config, open_capture, playback_path, rtsp_url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--from", dest="start", help='start of a recorded stretch, "YYYY-MM-DD HH:MM"')
    ap.add_argument("--channel", type=int, help="default: the only camera in settings.json")
    ap.add_argument("--main", action="store_true", help="record the full-resolution stream")
    args = ap.parse_args()

    channel = args.channel or camera_settings(load_settings())["channel"]
    cfg = lorex_config()
    if args.start:
        # Plays the NVR's recording, which skips the gaps when the camera
        # records on motion only, so clip time isn't wall-clock time.
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
        path = playback_path(channel, start)
    else:
        start = datetime.now()
        path = live_path(channel, main=args.main)

    cap = open_capture(rtsp_url(cfg, path), timeout_ms=15000)
    if not cap.isOpened():
        raise SystemExit("Could not open the stream.")
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("The stream opened but sent no video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    h, w = frame.shape[:2]
    out = HERE / "clips" / f"ch{channel:02d}_{start:%Y%m%d-%H%M%S}.mp4"
    out.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    log(f"recording {'playback' if args.start else 'live'} ch{channel} {w}x{h} at {fps:.0f} fps to {out.name}")

    total = int(args.minutes * 60 * fps)
    frames, began = 0, time.time()
    try:
        while frames < total:
            writer.write(frame)
            frames += 1
            if frames % int(fps * 60) == 0:
                log(f"{frames / fps / 60:.0f} of {args.minutes:g} min")
            ok, frame = cap.read()
            if not ok:
                log("stream ended")
                break
    except KeyboardInterrupt:
        pass
    finally:
        writer.release()
        cap.release()
    log(f"saved {frames / fps / 60:.1f} min of video in {(time.time() - began) / 60:.1f} min: {out}")


if __name__ == "__main__":
    main()
