"""Check the RTSP login to the Lorex NVR and snapshot its channels.

  python rtsp_check.py               find every channel, save a snapshot of each
  python rtsp_check.py --channel 3   check one channel, sub and main stream
"""
import argparse
import re
import socket

import cv2
import numpy as np

from common import SNAPSHOT_DIR, grab_frame, live_path, lorex_config, rtsp_describe, rtsp_url


def codec(sdp):
    m = re.search(r"a=rtpmap:\d+ ([\w.-]+)/", sdp)
    return m.group(1) if m else "?"


def contact_sheet(tiles, path, cols=4, size=(400, 225)):
    cells = []
    for label, frame in tiles:
        cell = np.zeros((size[1], size[0], 3), np.uint8) if frame is None else cv2.resize(frame, size)
        cv2.rectangle(cell, (0, 0), (size[0], 26), (0, 0, 0), -1)
        cv2.putText(cell, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cells.append(cell)
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", type=int, help="check just this channel")
    ap.add_argument("--max-channels", type=int, default=32)
    args = ap.parse_args()

    cfg = lorex_config()
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    print(f"NVR {cfg['host']}:{cfg['port']}, user {cfg['user']}")

    # One login attempt first. Lorex locks the account after a few wrong
    # passwords, so stop right away if this one is refused.
    try:
        status, _ = rtsp_describe(cfg, live_path(args.channel or 1))
    except (OSError, socket.timeout) as e:
        raise SystemExit(f"Cannot reach the NVR on port {cfg['port']}: {e}")
    if status == 401:
        raise SystemExit("Login refused (401): check LOREX_USER / LOREX_PASS in lorex.env.\n"
                         "Stopped after one try so the NVR doesn't lock the account.")
    if status == 403:
        raise SystemExit("Refused (403): the account may be locked after failed logins (wait ~30 min), "
                         "or this user isn't allowed live view.")
    print("Login OK\n")

    channels = [args.channel] if args.channel else range(1, args.max_channels + 1)
    tiles, found = [], []
    for ch in channels:
        try:
            status, sdp = rtsp_describe(cfg, live_path(ch))
        except OSError:
            print(f"ch{ch:>2}: no answer (camera offline?)")
            continue
        if status != 200:
            print(f"ch{ch:>2}: no stream (RTSP {status})")
            continue
        frame, fps = grab_frame(rtsp_url(cfg, live_path(ch)))
        if frame is None:
            print(f"ch{ch:>2}: stream answers but sends no video ({codec(sdp)}) - empty channel?")
            tiles.append((f"ch{ch}  no video", None))
            continue
        h, w = frame.shape[:2]
        print(f"ch{ch:>2}: OK  sub stream {w}x{h} {codec(sdp)} {fps:.0f} fps")
        cv2.imwrite(str(SNAPSHOT_DIR / f"ch{ch:02d}.jpg"), frame)
        tiles.append((f"ch{ch}  {w}x{h}", frame))
        found.append(ch)

    if args.channel and found:
        status, sdp = rtsp_describe(cfg, live_path(args.channel, main=True))
        frame, fps = grab_frame(rtsp_url(cfg, live_path(args.channel, main=True)), timeout_ms=15000)
        if frame is not None:
            h, w = frame.shape[:2]
            print(f"ch{args.channel:>2}: OK  main stream {w}x{h} {codec(sdp)} {fps:.0f} fps")
            cv2.imwrite(str(SNAPSHOT_DIR / f"ch{args.channel:02d}_main.jpg"), frame)
    elif tiles:
        sheet = SNAPSHOT_DIR / "all_channels.jpg"
        contact_sheet(tiles, sheet)
        print(f"\n{len(found)} channel(s) with video. Overview: {sheet}")
        print("Check one in detail with: python rtsp_check.py --channel N")
    if not found:
        raise SystemExit("No channel sent video.")


if __name__ == "__main__":
    main()
