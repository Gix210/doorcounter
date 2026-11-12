"""Draw the door lines on a snapshot of the door camera.

  python set_doors.py --channel 16              grab a fresh frame from the NVR and draw
  python set_doors.py --channel 16 --image X    draw on an existing snapshot
  python set_doors.py --channel 16 --preview    only render the saved lines to snapshots/doors_chNN.jpg

In the window, for each door: click the two ends of a line across the doorway
(where people's feet cross), then click once on the INSIDE side. The console
asks for the door's name. Keys: u = undo last door, s = save and quit,
q = quit without saving.
"""
import argparse
import math

import cv2

from common import (SNAPSHOT_DIR, camera_settings, grab_frame, live_path, load_settings, lorex_config, rtsp_url,
                    save_settings)

COLOR = (0, 255, 255)


def draw_doors(img, doors, pending=()):
    h, w = img.shape[:2]
    out = img.copy()
    for d in doors:
        (x1, y1), (x2, y2) = [(int(x * w), int(y * h)) for x, y in d["line"]]
        ix, iy = d["inside"][0] * w, d["inside"][1] * h
        cv2.line(out, (x1, y1), (x2, y2), COLOR, 2)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        scale = 35 / max(math.hypot(ix - mx, iy - my), 1)
        cv2.arrowedLine(out, (mx, my), (int(mx + (ix - mx) * scale), int(my + (iy - my) * scale)), COLOR, 2, tipLength=0.4)
        for color, thick in (((0, 0, 0), 3), (COLOR, 1)):
            cv2.putText(out, f"{d['name']} (arrow = IN)", (min(x1, x2), max(min(y1, y2) - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thick)
    for p in pending:
        cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
    if len(pending) == 2:
        cv2.line(out, tuple(map(int, pending[0])), tuple(map(int, pending[1])), (0, 0, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", type=int, help="camera from settings.json (default: the only one)")
    ap.add_argument("--image", help="snapshot to draw on (default: a fresh frame from the NVR)")
    ap.add_argument("--preview", action="store_true", help="only render the saved lines")
    args = ap.parse_args()

    settings = load_settings()
    cam = camera_settings(settings, args.channel)
    entry = next(c for c in settings["cameras"] if c["channel"] == cam["channel"])
    if args.image:
        img = cv2.imread(args.image)
    else:
        img, _ = grab_frame(rtsp_url(lorex_config(), live_path(cam["channel"], main=cam["stream"] == "main")))
    if img is None:
        raise SystemExit("No image to draw on.")
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    if args.preview:
        out = SNAPSHOT_DIR / f"doors_ch{cam['channel']:02d}.jpg"
        cv2.imwrite(str(out), draw_doors(img, cam["doors"]))
        print(out)
        return

    h, w = img.shape[:2]
    doors, clicks = list(cam["doors"]), []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    win = "set doors: 2 clicks = line, 3rd click = inside | u undo, s save, q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        if len(clicks) == 3:
            (x1, y1), (x2, y2), (ix, iy) = clicks
            name = input(f"Name for door {len(doors) + 1}: ").strip() or f"Door {len(doors) + 1}"
            doors.append({"name": name, "line": [[x1 / w, y1 / h], [x2 / w, y2 / h]], "inside": [ix / w, iy / h]})
            clicks.clear()
        cv2.imshow(win, draw_doors(img, doors, clicks))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u"):
            if clicks:
                clicks.clear()
            elif doors:
                doors.pop()
        elif key == ord("s"):
            entry["doors"] = [{k: ([[round(v, 4) for v in p] for p in d[k]] if k == "line" else
                                   [round(v, 4) for v in d[k]] if k == "inside" else d[k]) for k in d} for d in doors]
            save_settings(settings)
            print(f"Saved {len(doors)} door(s) for channel {cam['channel']} to settings.json")
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
