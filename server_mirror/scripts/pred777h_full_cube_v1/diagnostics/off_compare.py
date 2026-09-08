#!/usr/bin/env python3
"""Old vs new surface of a cell that the volume mesher now refuses: cap needles, shortest edges, smallest triangles."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
def tri_min_angle(P):
    a = np.linalg.norm(P[:, 1] - P[:, 2], axis=1); b = np.linalg.norm(P[:, 0] - P[:, 2], axis=1); c = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    ang = lambda x, y, z: np.degrees(np.arccos(np.clip((y * y + z * z - x * x) / np.maximum(2 * y * z, 1e-300), -1, 1)))
    return np.minimum(np.minimum(ang(a, b, c), ang(b, a, c)), ang(c, a, b))
def load(p):
    L = Path(p).read_text().split("\n"); nv, nf, _ = map(int, L[1].split())
    V = np.asarray([[float(x) for x in l.split()[:3]] for l in L[2:2 + nv]]); F = np.asarray([[int(x) for x in l.split()[1:4]] for l in L[2 + nv:2 + nv + nf]]); return V, F
for label, p in zip(sys.argv[1::2], sys.argv[2::2]):
    V, F = load(p)
    onplane = np.zeros(len(V), bool)
    for k in range(3): onplane |= (np.abs(V[:, k]) < 1e-9) | (np.abs(V[:, k] - 1) < 1e-9)
    cap = onplane[F].all(axis=1); a = tri_min_angle(V[F])
    E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), axis=1), axis=0); L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    P = V[F]; area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    print(f"{label}: V {len(V)} F {len(F)} | cap min angle {a[cap].min():.4f}, cap < 1 deg {(a[cap] < 1).sum()}, < 2 deg {(a[cap] < 2).sum()} | sheet min angle {a[~cap].min():.3f} | shortest edge {L.min():.2e}, edges < 1e-3: {(L < 1e-3).sum()}, < 3e-3: {(L < 3e-3).sum()} | smallest area {area.min():.2e}, areas < 1e-7: {(area < 1e-7).sum()}")
    i = int(np.argmin(L)); print(f"     shortest edge at {V[E[i, 0]].round(5).tolist()} - {V[E[i, 1]].round(5).tolist()}; smallest triangle centroid {P[int(np.argmin(area))].mean(axis=0).round(4).tolist()}")
