#!/usr/bin/env python3
"""Are the slivers inherited from needle-shaped CAP triangles, and are those needles a thin cap STRIP (the material
touching the face along a hairline) rather than a bad triangulation of a fat region?"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sliver_smooth import read_medit, min_dihedral
ROOTD = Path("/root/autodl-tmp/_claude_diag/tonight")


def tri_min_angle(P):
    a = np.linalg.norm(P[:, 1] - P[:, 2], axis=1); b = np.linalg.norm(P[:, 0] - P[:, 2], axis=1); c = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    ang = lambda x, y, z: np.degrees(np.arccos(np.clip((y * y + z * z - x * x) / np.maximum(2 * y * z, 1e-300), -1, 1)))
    return np.minimum(np.minimum(ang(a, b, c), ang(b, a, c)), ang(c, a, b))


for tag in sys.argv[1:]:
    V, T = read_medit(ROOTD / tag / "mesh.mesh")
    faces = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), axis=1)
    u, cnt = np.unique(faces, axis=0, return_counts=True); B = u[cnt == 1]
    onplane = np.zeros(len(V), bool)
    for k in range(3): onplane |= (np.abs(V[:, k]) < 1e-9) | (np.abs(V[:, k] - 1) < 1e-9)
    cap = onplane[B].all(axis=1); sheet = ~cap
    crease = np.zeros(len(V), bool); crease[np.unique(B[cap])] = True; crease &= np.isin(np.arange(len(V)), np.unique(B[sheet]))
    ang_cap = tri_min_angle(V[B[cap]]); ang_sheet = tri_min_angle(V[B[sheet]])
    Bc = B[cap]; all_crease = crease[Bc].all(axis=1)
    d = min_dihedral(V[T]); sl = np.flatnonzero(d < 1.0)
    # does each sliver sit on a needle cap triangle?
    capset = {tuple(f): i for i, f in enumerate(Bc)}
    inherited = 0; on_needle_all_crease = 0
    for t in sl:
        vs = T[t]
        for k in range(4):
            f = tuple(sorted(int(vs[j]) for j in range(4) if j != k))
            if f in capset:
                i = capset[f]
                if ang_cap[i] < 1.0: inherited += 1; on_needle_all_crease += int(all_crease[i])
                break
    print(f"=== {tag} ===")
    print(f"  cap triangles {len(Bc)}: min angle < 0.1 deg {(ang_cap < 0.1).sum()}, < 1 deg {(ang_cap < 1).sum()}, < 5 deg {(ang_cap < 5).sum()}, p01 {np.percentile(ang_cap, 1):.3f}, min {ang_cap.min():.4f}")
    print(f"  sheet triangles {len(B[sheet])}: min angle < 1 deg {(ang_sheet < 1).sum()}, < 5 deg {(ang_sheet < 5).sum()}, min {ang_sheet.min():.3f}")
    print(f"  needle cap triangles (< 1 deg) with ALL three vertices on the cap/sheet crease (a hairline strip): {(all_crease & (ang_cap < 1)).sum()} of {(ang_cap < 1).sum()}")
    print(f"  tets below 1 deg: {len(sl)}, of which sitting on a needle cap triangle: {inherited}, on a needle that is a crease strip: {on_needle_all_crease}")
    for i in np.argsort(ang_cap)[:4]:
        P = V[Bc[i]]; e = [np.linalg.norm(P[a] - P[b]) for a, b in ((0, 1), (1, 2), (0, 2))]
        print(f"    needle {ang_cap[i]:.4f} deg at {P.mean(axis=0).round(4).tolist()}, edges {np.round(e, 5).tolist()}, crease vertices {int(crease[Bc[i]].sum())}/3")
