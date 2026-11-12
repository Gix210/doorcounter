"""Daily report: one self-contained HTML page per day, one section per camera.

  python daily_report.py                         today (or the last day with data)
  python daily_report.py --date 2026-10-22
  python daily_report.py --log clips/ch12_x.csv  one camera's section, from a test clip's log

Writes reports/report_YYYY-MM-DD.html and prints a short summary.
"""
import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from common import HERE, camera_settings, load_settings

TIME_FMT = "%Y-%m-%d %H:%M:%S"
STAY_BUCKETS = [(0, 60, "< 1"), (60, 120, "1–2"), (120, 300, "2–5"), (300, 600, "5–10"),
                (600, 1200, "10–20"), (1200, 1800, "20–30"), (1800, 3600, "30–60"), (3600, math.inf, "60 +")]


def read_log(path):
    """Rows with their times parsed; "at" is when the row happened."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("time", "entered_at", "left_at"):
            if r.get(key):
                r[key + "_dt"] = datetime.strptime(r[key], TIME_FMT)
        r["at"] = r.get("time_dt") or r["entered_at_dt"]
    return sorted(rows, key=lambda r: r["at"])


def fmt_duration(seconds):
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s" if m < 10 else f"{m} min"
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m"


def nice_scale(v, target_ticks=4):
    """(axis max, tick step) with whole-number steps of 1, 2 or 5 x 10^k."""
    raw = max(v, 1) / target_ticks
    mag = 10 ** math.floor(math.log10(raw)) if raw >= 1 else 1
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    return step * math.ceil(max(v, 1) / step), step


def esc(s):
    return html.escape(str(s), quote=True)


# ------------------------------------------------------------------- charts

W, H, ML, MR, MT, MB = 720, 240, 40, 12, 14, 30
PW, PH = W - ML - MR, H - MT - MB


def y_axis(ymax, step):
    out = []
    for i in range(round(ymax / step) + 1):
        v = step * i
        y = MT + PH - PH * v / ymax
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{ML - 6}" y="{y + 4:.1f}" text-anchor="end">{v:g}</text>')
    return "".join(out)


def column(x, w, value, ymax, cls):
    """Column with a 4px rounded top, square at the baseline."""
    if value <= 0:
        return ""
    h = PH * value / ymax
    r = min(4, h, w / 2)
    x2, y, base = x + w, MT + PH - h, MT + PH
    return (f'<path class="{cls}" d="M{x:.1f},{base} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
            f'H{x2 - r:.1f} Q{x2:.1f},{y:.1f} {x2:.1f},{y + r:.1f} V{base} Z"/>')


def column_chart(labels, series, tips, aria, value_labels=False):
    """series: [(css class, values)], drawn side by side in each band."""
    ymax, step = nice_scale(max(max(vals) for _, vals in series))
    band = PW / len(labels)
    n = len(series)
    bw = min(24 if n == 1 else 12, (band * 0.7 - 2 * (n - 1)) / n)
    parts = [y_axis(ymax, step)]
    for i, label in enumerate(labels):
        cx = ML + band * i + band / 2
        x0 = cx - (n * bw + 2 * (n - 1)) / 2
        for j, (cls, vals) in enumerate(series):
            parts.append(column(x0 + j * (bw + 2), bw, vals[i], ymax, cls))
        if value_labels and series[0][1][i]:
            y = MT + PH - PH * series[0][1][i] / ymax - 6
            parts.append(f'<text class="val" x="{cx:.1f}" y="{y:.1f}" text-anchor="middle">{series[0][1][i]}</text>')
        if len(labels) <= 14 or i % 2 == 0:
            parts.append(f'<text class="tick" x="{cx:.1f}" y="{H - 10}" text-anchor="middle">{esc(label)}</text>')
        parts.append(f'<rect class="hit" x="{ML + band * i:.1f}" y="{MT}" width="{band:.1f}" height="{PH}" '
                     f'data-tip="{esc(tips[i])}"/>')
    parts.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + PH}" y2="{MT + PH}"/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}">{"".join(parts)}</svg>'


def occupancy_chart(points, t0, t1):
    """points: [(datetime, people in the room)] as a step line."""
    ymax, ystep = nice_scale(max(p for _, p in points))
    span = (t1 - t0).total_seconds() or 1
    X = lambda t: ML + PW * (t - t0).total_seconds() / span  # noqa: E731
    Y = lambda v: MT + PH - PH * v / ymax  # noqa: E731
    d = f"M{X(t0):.1f},{Y(0):.1f}"
    for t, v in points:
        d += f" H{X(t):.1f} V{Y(v):.1f}"
    d += f" H{X(t1):.1f}"
    parts = [y_axis(ymax, ystep), f'<path class="area" d="{d} V{Y(0):.1f} Z"/>', f'<path class="line" d="{d}"/>']
    hour = t0.replace(minute=0, second=0) + timedelta(hours=1)
    step = max(1, math.ceil(span / 3600 / 12))
    while hour < t1:
        if hour.hour % step == 0:
            parts.append(f'<text class="tick" x="{X(hour):.1f}" y="{H - 10}" text-anchor="middle">{hour:%H}h</text>')
        hour += timedelta(hours=1)
    parts.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{MT + PH}" y2="{MT + PH}"/>'
                 f'<line class="cross" x1="0" x2="0" y1="{MT}" y2="{MT + PH}" visibility="hidden"/>'
                 f'<circle class="dot" r="5" visibility="hidden"/>')
    data = json.dumps([[round(X(t), 1), round(Y(v), 1), f"{t:%H:%M:%S}", v] for t, v in points])
    parts.append(f'<rect class="hit occ" x="{ML}" y="{MT}" width="{PW}" height="{PH}" data-points=\'{esc(data)}\'/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="People in the room through the day">{"".join(parts)}</svg>'


def hour_span(times):
    first, last = min(times).hour, max(times).hour
    return list(range(first, last + 1))


def tiles(items):
    return '<div class="tiles">' + "".join(
        f'<div class="tile{" hero" if i == 0 else ""}"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for i, (label, value) in enumerate(items)) + "</div>"


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var(--{cls})"></i>{esc(label)}</span>' for cls, label in items) + "</div>"


# ----------------------------------------------------------------- sections

def door_section(cam, rows):
    in_label, out_label = cam.get("in_label", "In"), cam.get("out_label", "Out")
    ins = [r for r in rows if r["direction"] == "IN"]
    outs = [r for r in rows if r["direction"] == "OUT"]
    summary = f"{cam['name']}: {len(ins)} {in_label.lower()}, {len(outs)} {out_label.lower()}"
    if not rows:
        return f'<h2 class="section">{esc(cam["name"])}</h2><div class="card"><p>No one went through today.</p></div>', summary
    hours = hour_span([r["at"] for r in rows])
    in_h, out_h = Counter(r["at"].hour for r in ins), Counter(r["at"].hour for r in outs)
    busiest = max(hours, key=lambda h: in_h[h] + out_h[h])
    chart = column_chart([f"{h:02d}h" for h in hours],
                         [("s1", [in_h[h] for h in hours]), ("s2", [out_h[h] for h in hours])],
                         [f"{h:02d}:00–{h + 1:02d}:00|{in_label} {in_h[h]}|{out_label} {out_h[h]}" for h in hours],
                         "People through the door per hour")
    table = "".join(f"<tr><td>{h:02d}:00–{h + 1:02d}:00</td><td>{in_h[h]}</td><td>{out_h[h]}</td></tr>"
                    for h in hours if in_h[h] or out_h[h])
    body = f"""<h2 class="section">{esc(cam['name'])}</h2>
{tiles([(in_label, len(ins)), (out_label, len(outs)), ("Busiest hour", f"{busiest:02d}:00")])}
<div class="card"><h3>Through the door per hour</h3>
{legend([("s1", in_label), ("s2", out_label)])}{chart}
<details><summary>Show the numbers</summary><table><tr><th>Hour</th><th>{esc(in_label)}</th><th>{esc(out_label)}</th></tr>
{table}</table></details></div>"""
    return body, summary


def stay_section(cam, rows):
    stays = [int(r["stay_seconds"]) for r in rows]
    summary = f"{cam['name']}: {len(rows)} visits"
    if not rows:
        return f'<h2 class="section">{esc(cam["name"])}</h2><div class="card"><p>No visits today.</p></div>', summary
    summary += f", typical stay {fmt_duration(statistics.median(stays))}, average {fmt_duration(statistics.mean(stays))}"

    changes = sorted([(r["entered_at_dt"], 1) for r in rows] + [(r["left_at_dt"], -1) for r in rows])
    points, inside, peak, peak_at = [], 0, 0, None
    for t, delta in changes:
        inside += delta
        points.append((t, inside))
        if inside > peak:
            peak, peak_at = inside, t
    hours = hour_span([r["entered_at_dt"] for r in rows] + [r["left_at_dt"] for r in rows])
    arrivals = Counter(r["entered_at_dt"].hour for r in rows)
    t0 = min(t for t, _ in changes).replace(minute=0, second=0)
    t1 = max(t for t, _ in changes).replace(minute=0, second=0) + timedelta(hours=1)
    buckets = [sum(lo <= s < hi for s in stays) for lo, hi, _ in STAY_BUCKETS]

    visits_chart = column_chart([f"{h:02d}h" for h in hours], [("s1", [arrivals[h] for h in hours])],
                                [f"{h:02d}:00–{h + 1:02d}:00|{arrivals[h]} came in" for h in hours],
                                "People coming in per hour", value_labels=True)
    stay_chart = column_chart([label for _, _, label in STAY_BUCKETS], [("s1", buckets)],
                              [f"Stayed {label} min|{n} visits" for (_, _, label), n in zip(STAY_BUCKETS, buckets)],
                              "How long people stayed", value_labels=True)
    visit_rows = "".join(f"<tr><td>{r['entered_at_dt']:%H:%M:%S}</td><td>{r['left_at_dt']:%H:%M:%S}</td>"
                         f"<td>{fmt_duration(int(r['stay_seconds']))}</td></tr>" for r in rows)
    body = f"""<h2 class="section">{esc(cam['name'])}</h2>
{tiles([("Visits", len(rows)), ("Typical stay", fmt_duration(statistics.median(stays))),
        ("Average stay", fmt_duration(statistics.mean(stays))), ("Longest stay", fmt_duration(max(stays))),
        ("Most at once", f"{peak} <span class='note'>at {peak_at:%H:%M}</span>")])}
<div class="card"><h3>How long people stayed (minutes)</h3>
<p class="note">Typical = half the visits were shorter, half longer. Timed from when a person appears on the camera until they leave it.</p>
{stay_chart}</div>
<div class="card"><h3>People coming in per hour</h3>{visits_chart}</div>
<div class="card"><h3>People in the room through the day</h3>{occupancy_chart(points, t0, t1)}</div>
<div class="card"><details><summary>Every visit ({len(rows)})</summary>
<table><tr><th>Came in</th><th>Left</th><th>Stayed</th></tr>{visit_rows}</table></details></div>"""
    return body, summary


# --------------------------------------------------------------------- page

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:800px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 2px}
h2.section{font-size:18px;margin:28px 0 10px}h3{font-size:15px;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 8px}.note{color:var(--muted);font-size:13px;margin:2px 0 6px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:12px}
.tile,.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.tile .label{color:var(--ink2);font-size:13px}.tile .value{font-size:24px;font-weight:600;margin-top:2px;white-space:nowrap}
.tile.hero .value{font-size:48px;line-height:1.1}.card{margin-bottom:12px}
.legend{display:flex;gap:16px;color:var(--ink2);font-size:13px;margin:6px 0 4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
svg{width:100%;height:auto;display:block;overflow:visible;margin-top:6px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.val{fill:var(--ink2);font-size:12px}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.line{fill:none;stroke:var(--s1);stroke-width:2;stroke-linejoin:round}
.area{fill:var(--s1);opacity:.1}.cross{stroke:var(--axis);stroke-width:1}.dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}
.hit{fill:transparent}.hit:hover{fill:var(--ink);fill-opacity:.04}.hit.occ:hover{fill:transparent}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-radius:8px;padding:6px 10px;font-size:13px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9}
#tip b{display:block;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;margin-top:8px}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}summary{cursor:pointer;color:var(--ink2);margin-top:8px}
@media (max-width:560px){.tick{font-size:19px}.val{font-size:21px}.tile.hero .value{font-size:40px}}
"""

JS = """
const tip=document.getElementById('tip');
function show(e,lines){tip.innerHTML='<b>'+lines[0]+'</b>'+lines.slice(1).join('<br>');tip.style.display='block';
 const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px';}
document.querySelectorAll('.hit[data-tip]').forEach(r=>{r.addEventListener('pointermove',e=>show(e,r.dataset.tip.split('|')));
 r.addEventListener('pointerleave',()=>tip.style.display='none');});
document.querySelectorAll('.hit.occ').forEach(r=>{const pts=JSON.parse(r.dataset.points),svg=r.ownerSVGElement,
 cross=svg.querySelector('.cross'),dot=svg.querySelector('.dot');
 r.addEventListener('pointermove',e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
  const x=p.matrixTransform(svg.getScreenCTM().inverse()).x;let best=null;
  for(const q of pts){if(q[0]<=x)best=q;else break;}if(!best){tip.style.display='none';return;}
  cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
  dot.setAttribute('cx',x);dot.setAttribute('cy',best[1]);dot.setAttribute('visibility','visible');
  show(e,['Since '+best[2],best[3]+' in the room']);});
 r.addEventListener('pointerleave',()=>{tip.style.display='none';cross.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden');});});
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, or the last day with data)")
    ap.add_argument("--log", type=Path, help="report only this log (e.g. a test clip's), for its camera")
    ap.add_argument("--channel", type=int, help="camera the --log belongs to (default: from its name)")
    ap.add_argument("--out", type=Path, help="HTML file to write (default: reports/report_<date>.html)")
    args = ap.parse_args()

    settings = load_settings()
    if args.log:
        m = re.match(r"ch(\d+)_", args.log.name)
        cams = [(camera_settings(settings, args.channel or (int(m.group(1)) if m else None)), args.log)]
    else:
        cams = [(camera_settings(settings, c["channel"]), HERE / c["log"]) for c in settings["cameras"]]
    logs = [(cam, read_log(path) if path.exists() else []) for cam, path in cams]

    days = sorted({r["at"].strftime("%Y-%m-%d") for _, rows in logs for r in rows})
    if not days:
        raise SystemExit("No data in the logs yet.")
    today = datetime.now().strftime("%Y-%m-%d")
    day = args.date or (today if today in days else days[-1])

    sections, summaries = [], []
    for cam, rows in logs:
        rows = [r for r in rows if r["at"].strftime("%Y-%m-%d") == day]
        body, summary = (stay_section if cam["mode"] == "stay" else door_section)(cam, rows)
        sections.append(body)
        summaries.append(summary)

    date = datetime.strptime(day, "%Y-%m-%d")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Daily report {day}</title><style>{CSS}</style></head>
<body><main><h1>Daily report</h1><p class="sub">{date:%A %B %d, %Y}</p>{"".join(sections)}</main>
<div id="tip"></div><script>{JS}</script></body></html>"""
    out = args.out or HERE / "reports" / f"report_{day}.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(day)
    for s in summaries:
        print(" ", s)
    print(out)


if __name__ == "__main__":
    main()
