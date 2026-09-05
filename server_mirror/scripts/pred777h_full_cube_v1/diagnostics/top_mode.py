#!/usr/bin/env python3
"""Is lambda_max a physical port stiffness or one bad element?

A sliver tetrahedron has vanishing volume and gradients like 1/thickness, so its element stiffness diverges and it
injects ONE very stiff, very LOCAL mode.  A physical port stiffness is spread over the material's footprint.  So:
look at the top eigenvector's participation.  If almost all of its mass sits on a handful of carrier nodes,
lambda_max measures the mesh, not the cell.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import load_label

T = Path("/root/autodl-tmp/_claude_diag/tonight")
print(f"{'label':<34}{'lambda_max':>11}{'nodes>1%':>10}{'top1':>8}{'top4':>8}{'top16':>8}{'minDihed':>10}{'<5deg':>7}")
for tag in sys.argv[1:]:
    d = T / tag
    lab = load_label(d); S = lab["S"]; n = S.shape[0] // 3
    w, U = np.linalg.eigh(S); v = U[:, -1]
    part = np.linalg.norm(v.reshape(n, 3), axis=1) ** 2          # sums to 1
    order = np.argsort(-part)
    m = lab["receipt"]["mesh"]
    print(f"{tag:<34}{w[-1]:>11.5g}{int((part > 0.01).sum()):>10}"
          f"{part[order[0]]:>8.3f}{part[order[:4]].sum():>8.3f}{part[order[:16]].sum():>8.3f}"
          f"{m['min_dihedral_degrees']:>10.3f}{m['tets_below_5deg']:>7}")
    xyz = lab["xyz"]
    print("      top carrier nodes: " + ", ".join(f"{np.round(xyz[i], 4).tolist()} ({part[i]:.2f})" for i in order[:4]))
    sys.stdout.flush()
