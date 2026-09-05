#!/usr/bin/env python3
"""What the slivers ARE: where the worst tetrahedra sit relative to the port faces.

For the remedy it matters whether a sliver has (a) three vertices on a port face and one interior vertex barely off
it (then smoothing that interior vertex, or a flip, removes it), or (b) all four vertices on the surface (then only a
flip or a Steiner point can), or (c) is interior."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOTD = Path("/root/autodl-tmp/_claude_diag/tonight")


def read_medit(path):
    V = []; Tt = []
    with open(path) as fh:
        it = iter(fh)
        for line in it:
            s = line.strip()
            if s == "Vertices":
                n = int(next(it)); V = np.asarray([[float(x) for x in next(it).split()[:3]] for _ in range(n)])
            elif s == "Tetrahedra":
                n = int(next(it)); Tt = np.asarray([[int(x) for x in next(it).split()[:4]] for _ in range(n)]) - 1
    return V, Tt


def dihedrals(V, T):
    P = V[T]
    out = np.full(len(T), 180.0)
    pairs = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)), ((1, 2), (0, 3)), ((1, 3), (0, 2)), ((2, 3), (0, 1))]
    faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    N = []
    for f in faces:
        n = np.cross(P[:, f[1]] - P[:, f[0]], P[:, f[2]] - P[:, f[0]]); N.append(n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None])
    # dihedral at edge shared by faces i, j = pi - angle(N_i, N_j)
    for i in range(4):
        for j in range(i + 1, 4):
            c = np.clip(np.einsum("ij,ij->i", N[i], N[j]), -1, 1)
            out = np.minimum(out, 180.0 - np.degrees(np.arccos(c)))
    return out


for tag in sys.argv[1:]:
    V, T = read_medit(ROOTD / tag / "mesh.mesh")
    faces = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), axis=1)
    u, cnt = np.unique(faces, axis=0, return_counts=True)
    bfaces = u[cnt == 1]; on_surface = np.zeros(len(V), bool); on_surface[np.unique(bfaces)] = True
    onplane = np.zeros(len(V), bool)
    for k in range(3):
        onplane |= (np.abs(V[:, k]) < 1e-9) | (np.abs(V[:, k] - 1.0) < 1e-9)
    d = dihedrals(V, T)
    print(f"\n=== {tag} ===  tets {len(T)}  min dihedral {d.min():.4f} deg")
    for thr in (0.1, 0.5, 1.0, 2.0, 5.0):
        sel = np.flatnonzero(d < thr)
        if not len(sel): print(f"  < {thr:>4} deg: 0"); continue
        ns = on_surface[T[sel]].sum(axis=1); npl = onplane[T[sel]].sum(axis=1)
        kinds = {"4 on surface": int((ns == 4).sum()), "3 on surface": int((ns == 3).sum()),
                 "<=2 on surface": int((ns <= 2).sum())}
        print(f"  < {thr:>4} deg: {len(sel):>4}   {kinds}   on a box plane: 4:{int((npl==4).sum())} 3:{int((npl==3).sum())} 2:{int((npl==2).sum())}")
    # the worst ten
    worst = np.argsort(d)[:8]
    for t in worst:
        vs = T[t]; print(f"    {d[t]:8.4f} deg  surface {on_surface[vs].astype(int).tolist()}  plane {onplane[vs].astype(int).tolist()}  "
                         f"vol {abs(np.linalg.det(np.vstack([V[vs[1]]-V[vs[0]], V[vs[2]]-V[vs[0]], V[vs[3]]-V[vs[0]]]))/6):.2e}")
