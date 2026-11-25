"""Draw ../docs/img/spikes.svg: test image 0 run through the exact integer model the chip implements.

Usage: python model/make_figures.py [image_index]
"""
import sys
from pathlib import Path

import numpy as np

from export import rate_encode_np

here = Path(__file__).resolve().parent
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rows = [l.split() for l in (here / "test_vectors.txt").read_text().splitlines() if not l.startswith("#")]
lv = np.array([int(c) for c in rows[idx][0]])
label, gold = int(rows[idx][1]), int(rows[idx][2])
w = np.load(here / "weights_ternary.npz")["arr_0"]  # (10, pixels)
T, P = 16, w.shape[1]
S = int(round(P ** 0.5))  # image side
NB = (P + 7) // 8         # spike bytes per timestep

# integer model, keeping every output spike
s = rate_encode_np(lv[None, :], T)[:, 0, :]  # (T, pixels)
v = np.zeros(10, dtype=np.int32)
out = np.zeros((T, 10), dtype=np.int32)
for t in range(T):
    v = v - (v >> 4)
    v = np.clip(v + s[t] @ w.T, -128, 127)
    z = (v >= 8).astype(np.int32)
    v = v - 8 * z
    out[t] = z
counts = out.sum(0)
assert list(counts) == [int(x) for x in rows[idx][3:13]], "figure model disagrees with golden vectors"

W, H = 960, 545
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="13">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     '<style>.t{font-weight:bold;fill:#111;font-size:14px}.s{fill:#555;font-size:11.5px}.a{fill:#888;font-size:10.5px}</style>']

# 1. the input image
grey = ["#ffffff", "#c7cbd1", "#7b8390", "#1f2937"]
x0, y0, c = 24, 58, 130 // S
o.append(f'<text class="t" x="{x0}" y="30">Input: {S}×{S}, 2-bit levels</text>')
o.append(f'<text class="s" x="{x0}" y="46">test image {idx}, label {label}</text>')
for p in range(P):
    r, k = divmod(p, S)
    o.append(f'<rect x="{x0 + k * c}" y="{y0 + r * c}" width="{c}" height="{c}" fill="{grey[lv[p]]}" stroke="#e5e7eb" stroke-width="0.5"/>')
for q in range(4):
    o.append(f'<rect x="{x0 + q * 33}" y="{y0 + S * c + 14}" width="11" height="11" fill="{grey[q]}" stroke="#9ca3af" stroke-width="0.6"/>')
    o.append(f'<text class="a" x="{x0 + q * 33 + 14}" y="{y0 + S * c + 23}">{q}</text>')

# 2. the 16 input spike frames
fx, fy, fc, gap = 200, 58, 40 // S, 6
o.append(f'<text class="t" x="{fx}" y="30">Input spikes, one frame per timestep (sent as {NB} bytes each)</text>')
o.append(f'<text class="s" x="{fx}" y="46">a pixel at level q spikes q times every 4 steps</text>')
for t in range(T):
    gx = fx + t * (S * fc + gap)
    o.append(f'<rect x="{gx - 1}" y="{fy - 1}" width="{S * fc + 2}" height="{S * fc + 2}" fill="#f9fafb" stroke="#d1d5db" stroke-width="0.8"/>')
    for p in np.nonzero(s[t])[0]:
        r, k = divmod(int(p), S)
        o.append(f'<rect x="{gx + k * fc}" y="{fy + r * fc}" width="{fc}" height="{fc}" fill="#2563eb"/>')
    o.append(f'<text class="a" x="{gx + S * fc / 2}" y="{fy + S * fc + 14}" text-anchor="middle">{t}</text>')
    o.append(f'<text class="a" x="{gx + S * fc / 2}" y="{fy + S * fc + 27}" text-anchor="middle">{int(s[t].sum())}</text>')
o.append(f'<text class="a" x="{fx - 6}" y="{fy + S * fc + 14}" text-anchor="end">step</text>')
o.append(f'<text class="a" x="{fx - 6}" y="{fy + S * fc + 27}" text-anchor="end">spikes</text>')

# 3. output neuron raster + spike counts
rx, ry, cw, rh = 200, 272, 30, 24
o.append(f'<text class="t" x="24" y="{ry - 22}">Output: 10 LIF neurons, one per digit (θ = 8)</text>')
for n in range(10):
    y = ry + n * rh
    win = n == gold
    o.append(f'<text x="{rx - 14}" y="{y + 16}" text-anchor="end" font-weight="{"bold" if win else "normal"}" fill="#111">digit {n}</text>')
    o.append(f'<line x1="{rx}" y1="{y + 12}" x2="{rx + T * cw}" y2="{y + 12}" stroke="#e5e7eb"/>')
    for t in range(T):
        if out[t, n]:
            o.append(f'<rect x="{rx + t * cw + 11}" y="{y + 3}" width="8" height="18" rx="2" fill="{"#dc2626" if win else "#6b7280"}"/>')
    bx = rx + T * cw + 30
    o.append(f'<rect x="{bx}" y="{y + 4}" width="{counts[n] * 14}" height="16" fill="{"#dc2626" if win else "#9ca3af"}"/>')
    o.append(f'<text x="{bx + counts[n] * 14 + 6}" y="{y + 16}" fill="#111">{counts[n]}</text>')
for t in range(T):
    o.append(f'<text class="a" x="{rx + t * cw + 15}" y="{ry + 10 * rh + 14}" text-anchor="middle">{t}</text>')
o.append(f'<text class="s" x="{rx + T * cw / 2}" y="{ry + 10 * rh + 30}" text-anchor="middle">timestep</text>')
o.append(f'<text class="s" x="{rx + T * cw + 30}" y="{ry - 6}">spike count</text>')
o.append(f'<text class="t" x="{rx + T * cw + 30}" y="{ry + 10 * rh + 30}" fill="#dc2626">predicted: {gold}</text>')

o.append("</svg>")
dst = here.parent / "docs" / "img" / "spikes.svg"
dst.write_text("\n".join(o), encoding="utf-8")
print(f"wrote {dst}  counts {list(counts)}  predicted {gold}")
