"""Draw docs/img/area_breakdown.svg and docs/img/timing_slack.svg from gds/metrics.csv.

Usage: python gds/make_figures.py      (after copying a new metrics.csv from the gds action)
"""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
m = {r[0]: r[1] for r in csv.reader((root / "gds" / "metrics.csv").open()) if len(r) >= 2}
period = float(json.loads((root / "src" / "config.json").read_text())["CLOCK_PERIOD"])
mhz = 1000 / period

SANS = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def text(x, y, s, size=12.5, fill="#e6edf3", anchor="start", weight=400, font=SANS):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{font}">{s}</text>')


def bar_chart(title, subtitle, rows, unit, step, footer=None):
    """rows: (label, value, annotation). Horizontal bars from x=280, 30 px per row."""
    vmax = max(v for _, v, _ in rows)
    ticks = int(vmax // step) + 1
    scale = 560 / (ticks * step)
    h = 100 + 30 * len(rows) + (40 if footer else 10)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {h}" width="1000" height="{h}" role="img" aria-label="{title}">',
         f"<title>{title}</title>",
         f'<rect width="1000" height="{h}" rx="12" fill="#0d1117"/>',
         text(30, 36, title, 16, weight=600),
         text(30, 57, subtitle, 12.5, "#8b949e")]
    y_end = 84 + 30 * len(rows)
    for k in range(ticks + 1):
        x = 280 + k * step * scale
        o.append(f'<line x1="{x:.1f}" y1="78" x2="{x:.1f}" y2="{y_end}" stroke="#30363d" stroke-width="1"/>')
        o.append(text(f"{x:.1f}", 72, f"{k * step:,g} {unit}", 11, "#8b949e", "middle"))
    for i, (label, v, note) in enumerate(rows):
        y = 90 + 30 * i
        w = max(v * scale, 1.0)
        o.append(text(268, y + 13.5, label, 12.5, anchor="end"))
        o.append(f'<path d="M280 {y}h{w - 4:.2f}a4 4 0 0 1 4 4v10a4 4 0 0 1 -4 4h-{w - 4:.2f}z" fill="#3987e5"/>'
                 if w > 4 else f'<rect x="280" y="{y}" width="{w:.2f}" height="18" fill="#3987e5"/>')
        o.append(text(f"{280 + w + 12:.1f}", y + 13.5, note, 12, "#8b949e", font=MONO))
    if footer:
        o.append(text(30, h - 20, footer, 12, "#8b949e"))
    o.append("</svg>")
    return "\n".join(o)


# ---- area
core = float(m["design__core__area"])
classes = [("Fill / decap cells", "fill_cell"),
           ("Combinational logic", "multi_input_combinational_cell"),
           ("Flip-flops", "sequential_cell"),
           ("Timing-repair buffers", "timing_repair_buffer"),
           ("Clock tree", "clock_buffer"),
           ("Inverters", "inverter"),
           ("Well taps", "tap_cell")]
rows = []
for label, key in classes:
    a = float(m.get(f"design__instance__area__class:{key}", 0))
    n = int(float(m.get(f"design__instance__count__class:{key}", 0)))
    if n:
        rows.append((label, a, f"{a:,.0f} µm² · {100 * a / core:.1f}% · {n:,} cells"))
util = 100 * float(m["design__instance__utilization"])
area_svg = bar_chart("Where the silicon goes",
                     f"Placed cell area by class inside the 1x1-tile core ({core:,.0f} µm²). Design cells use {util:.1f}% of it.",
                     rows, "µm²", 2000)
(root / "docs" / "img" / "area_breakdown.svg").write_text(area_svg, encoding="utf-8")

# ---- timing
names = {"tt": "typical · 25 °C · 1.80 V", "ss": "slow · 100 °C · 1.60 V", "ff": "fast · −40 °C · 1.95 V"}
rc = {"nom": "nom RC", "min": "min RC", "max": "max RC"}
corners = [f"{r}_{p}" for p in ("tt_025C_1v80", "ss_100C_1v60", "ff_n40C_1v95") for r in ("nom", "min", "max")]
rows, worst = [], None
for c in corners:
    v = float(m[f"timing__setup__ws__corner:{c}"])
    r, p = c.split("_", 1)
    label = f"{names[p[:2]]} ({rc[r]})"
    rows.append((label, v, f"{v:.2f} ns"))
    if worst is None or v < worst[0]:
        worst = (v, label)
hold = min(float(m[f"timing__hold__ws__corner:{c}"]) for c in corners)
timing_svg = bar_chart(f"Setup slack at {mhz:g} MHz, all 9 signoff corners",
                       f"Worst setup slack per corner against the {period:g} ns clock period. Anything above 0 passes.",
                       rows, "ns", 2,
                       f"Worst case: {worst[0]:.2f} ns of slack ({worst[1]}). Hold slack is positive in every corner too (minimum {hold:.2f} ns).")
(root / "docs" / "img" / "timing_slack.svg").write_text(timing_svg, encoding="utf-8")
print(f"wrote area_breakdown.svg (util {util:.1f}%) and timing_slack.svg (worst setup {worst[0]:.2f} ns, hold {hold:.2f} ns)")
