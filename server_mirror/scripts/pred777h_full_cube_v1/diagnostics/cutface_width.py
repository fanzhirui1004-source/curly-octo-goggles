#!/usr/bin/env python3
"""How thin is the cut face in the actual population?  For every cut cell, the polygon of the plane inside the unit
cube, its area and its minimum caliper width (the smallest distance between two parallel supporting lines).  A width
below one carrier spacing h = 1/32 means the cut-face carrier is a needle strip whatever its node count."""
import json
from pathlib import Path
import numpy as np

H = 1.0 / 32
P = Path("/root/autodl-tmp/_claude_diag/population/cells")
rows = [json.loads(l) for l in (P / "POPULATION.jsonl").read_text().splitlines() if l.strip()]
CORN = np.array([(x, y, z) for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def polygon(n, d):
    pts = []
    for i, j in EDGES:
        a, b = CORN[i], CORN[j]; da, db = n @ a - d, n @ b - d
        if da * db < 0: pts.append(a + (b - a) * (da / (da - db)))
        elif abs(da) < 1e-12: pts.append(a)
    if len(pts) < 3: return None
    Pp = np.unique(np.round(np.asarray(pts), 12), axis=0)
    if len(Pp) < 3: return None
    c = Pp.mean(axis=0); u = Pp[0] - c; u /= np.linalg.norm(u); v = np.cross(n, u)
    ang = np.arctan2((Pp - c) @ v, (Pp - c) @ u)
    return Pp[np.argsort(ang)]


def width_and_area(Pp, n):
    c = Pp.mean(axis=0); u = Pp[1] - Pp[0]; u /= np.linalg.norm(u); v = np.cross(n, u)
    Q = np.column_stack([(Pp - c) @ u, (Pp - c) @ v])
    area = 0.5 * abs(sum(Q[i, 0] * Q[(i + 1) % len(Q), 1] - Q[(i + 1) % len(Q), 0] * Q[i, 1] for i in range(len(Q))))
    w = np.inf
    for i in range(len(Q)):
        e = Q[(i + 1) % len(Q)] - Q[i]; L = np.linalg.norm(e)
        if L < 1e-15: continue
        nrm = np.array([-e[1], e[0]]) / L
        proj = (Q - Q[i]) @ nrm; w = min(w, proj.max() - proj.min())
    return float(w), float(area)


out = []
for r in rows:
    if not r.get("cut_plane"): continue
    p = np.asarray(r["cut_plane"], float); nn = np.linalg.norm(p[:3]); n = p[:3] / nn; d = p[3] / nn
    Pp = polygon(n, d)
    if Pp is None: out.append({"case": r["case"], "width": 0.0, "area": 0.0}); continue
    w, a = width_and_area(Pp, n); out.append({"case": r["case"], "width": w, "area": a, "verts": len(Pp)})
W = np.asarray([o["width"] for o in out]); A = np.asarray([o["area"] for o in out])
print(f"cut cells {len(out)};  carrier spacing h = {H:.5f}")
print(f"cut-face minimum width:  min {W.min():.2e}  p01 {np.percentile(W,1):.4f}  p05 {np.percentile(W,5):.4f}  med {np.median(W):.4f}  max {W.max():.4f}")
for thr, name in ((0.25, "h/4"), (0.5, "h/2"), (1.0, "h"), (2.0, "2h"), (3.0, "3h")):
    k = int((W < thr * H).sum()); print(f"   width < {name:<4} ({thr*H:.5f}): {k:>4} cells ({100*k/len(out):.1f} %),  cut-face triangle aspect ratio then above ~{4/thr:.0f}")
print(f"cut-face area: min {A.min():.2e} med {np.median(A):.4f} max {A.max():.4f}; area < 0.01: {int((A<0.01).sum())}, < 0.001: {int((A<0.001).sum())}")
worst = sorted(out, key=lambda o: o["width"])[:8]
print("thinnest cut faces:"); [print(f"   {o['case']:<20} width {o['width']:.3e} ({o['width']/H:.4f} h)  area {o['area']:.3e}") for o in worst]
json.dump(out, open("/root/autodl-tmp/_claude_diag/tonight/CUTFACE_WIDTH.json", "w"), indent=1)
