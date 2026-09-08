#!/usr/bin/env python3
"""Classify every cap needle in the surface output: what is its long edge (chain edge shared with a sheet triangle,
carrier lattice segment, or a free triangulation edge) and what is its apex (crease vertex, carrier lattice node, or a
Steiner point of the cap)?"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cap_needles import tri_min_angle
ROOTD = Path("/root/autodl-tmp/_claude_diag/tonight")
for tag in sys.argv[1:]:
    lines = (ROOTD / tag / "SHEET_SOLID_SURFACE.off").read_text().split("\n")
    nv, nf, _ = map(int, lines[1].split()); V = np.asarray([[float(x) for x in l.split()[:3]] for l in lines[2:2 + nv]])
    F = np.asarray([[int(x) for x in l.split()[1:4]] for l in lines[2 + nv:2 + nv + nf]])
    onplane = np.zeros(len(V), bool)
    for k in range(3): onplane |= (np.abs(V[:, k]) < 1e-9) | (np.abs(V[:, k] - 1) < 1e-9)
    cap = onplane[F].all(axis=1)
    crease = np.zeros(len(V), bool); crease[np.unique(F[cap])] = True; crease &= np.isin(np.arange(len(V)), np.unique(F[~cap]))
    lattice = np.all(np.abs(V * 32 - np.round(V * 32)) < 1e-9, axis=1)
    sheet_edges = set()
    for f in F[~cap]:
        for a, b in ((0, 1), (1, 2), (0, 2)): sheet_edges.add((min(f[a], f[b]), max(f[a], f[b])))
    cap_edge_count = defaultdict(int)
    for f in F[cap]:
        for a, b in ((0, 1), (1, 2), (0, 2)): cap_edge_count[(min(f[a], f[b]), max(f[a], f[b]))] += 1
    Fc = F[cap]; ang = tri_min_angle(V[Fc]); needles = np.flatnonzero(ang < 1.0)
    kinds = defaultdict(int)
    print(f"=== {tag} === cap needles < 1 deg: {len(needles)}")
    for i in needles[np.argsort(ang[needles])]:
        f = Fc[i]; P = V[f]
        e = np.asarray([np.linalg.norm(P[1] - P[2]), np.linalg.norm(P[0] - P[2]), np.linalg.norm(P[0] - P[1])]); m = int(np.argmax(e))
        a, b = [k for k in range(3) if k != m]; A, B, Pp = int(f[a]), int(f[b]), int(f[m])
        key = (min(A, B), max(A, B))
        long_kind = "chain edge" if key in sheet_edges else ("lattice segment" if lattice[A] and lattice[B] else ("cap boundary" if cap_edge_count[key] == 1 else "free cap edge"))
        apex_kind = "carrier node" if lattice[Pp] else ("crease vertex" if crease[Pp] else "cap Steiner")
        ends = f"A {'node' if lattice[A] else ('crease' if crease[A] else 'steiner')}, B {'node' if lattice[B] else ('crease' if crease[B] else 'steiner')}"
        d = np.linalg.norm(np.cross(P[b] - P[a], P[m] - P[a])) / e[m]
        kinds[(long_kind, apex_kind)] += 1
        if ang[i] < 0.6: print(f"  {ang[i]:.4f} deg: long edge = {long_kind} ({ends}); apex = {apex_kind}, {d:.1e} off; |AB| {e[m]:.4f}")
    print("  summary:", dict(kinds))
