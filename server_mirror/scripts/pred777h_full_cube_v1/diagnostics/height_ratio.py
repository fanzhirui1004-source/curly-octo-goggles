#!/usr/bin/env python3
"""Distribution of the flatness measure h_min / L_max (min altitude over max edge; regular tet 0.816) and of the
stiffness-bomb metric max_k area_k^2 / V = 3 max_k area_k / h_k."""
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
    areas = np.stack([0.5 * np.linalg.norm(np.cross(P[:, f[1]] - P[:, f[0]], P[:, f[2]] - P[:, f[0]]), axis=1) for f in FACES], axis=1)
    h = 3 * vol[:, None] / np.maximum(areas, 1e-300); hmin = h.min(axis=1)
    L = np.max([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)], axis=0)
    ratio = hmin / L; bomb = (areas ** 2 / np.maximum(vol, 1e-300)[:, None]).max(axis=1); d = min_dihedral(P)
    print(f"=== {tag} === h_min/L_max: p01 {np.percentile(ratio, 1):.4f} p05 {np.percentile(ratio, 5):.4f} median {np.median(ratio):.3f}; below 0.02: {(ratio < 0.02).sum()}, 0.05: {(ratio < 0.05).sum()}, 0.1: {(ratio < 0.1).sum()}, 0.2: {(ratio < 0.2).sum()}")
    o = np.argsort(ratio)[:5]
    print("   flattest:", [(round(float(ratio[t]), 4), round(float(d[t]), 2), round(float(bomb[t]), 3)) for t in o], "(ratio, min dihedral deg, bomb)")
    print(f"   bomb max {bomb.max():.3f}; tets with bomb > 0.5: {(bomb > 0.5).sum()}, > 1: {(bomb > 1).sum()}; of tets with dihedral >= 5 deg the worst bomb is {bomb[d >= 5].max():.3f} (ratio {ratio[d >= 5][np.argmax(bomb[d >= 5])]:.4f})")
