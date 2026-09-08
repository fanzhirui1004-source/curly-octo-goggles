#!/usr/bin/env python3
"""Which tetrahedra can inject a stiffness bomb?  For Tet4 shape functions |grad N_k| = area(face opposite k) / (3 V);
the element energy of a unit nodal displacement scales with V |grad N|^2 = area^2 / (9 V).  Rank tetrahedra by
max_k area_k^2 / V (dimension length; compare with the mesh size ~0.02) and locate the worst."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sliver_smooth import read_medit, min_dihedral
ROOTD = Path("/root/autodl-tmp/_claude_diag/tonight")
FACES = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
for tag in sys.argv[1:]:
    V, T = read_medit(ROOTD / tag / "mesh.mesh"); P = V[T]
    vol = np.abs(np.einsum("ij,ij->i", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))) / 6
    bomb = np.zeros(len(T))
    for k, f in enumerate(FACES):
        area = 0.5 * np.linalg.norm(np.cross(P[:, f[1]] - P[:, f[0]], P[:, f[2]] - P[:, f[0]]), axis=1)
        bomb = np.maximum(bomb, area ** 2 / np.maximum(vol, 1e-300))
    d = min_dihedral(P)
    onplane = np.zeros(len(V), bool)
    for k in range(3): onplane |= (np.abs(V[:, k]) < 1e-9) | (np.abs(V[:, k] - 1) < 1e-9)
    order = np.argsort(-bomb)
    print(f"=== {tag} === tets {len(T)}; bomb metric (area^2/V): median {np.median(bomb):.4f}, p99 {np.percentile(bomb, 99):.4f}, max {bomb.max():.4f}")
    for t in order[:6]:
        c = P[t].mean(axis=0); npl = int(onplane[T[t]].sum())
        print(f"   {bomb[t]:9.4f}  min dihedral {d[t]:7.3f} deg  vol {vol[t]:.2e}  centroid {c.round(4).tolist()}  vertices on box planes {npl}")
