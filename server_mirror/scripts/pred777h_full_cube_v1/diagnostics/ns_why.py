#!/usr/bin/env python3
"""WHY the operator has more null modes than 6*components + 3*unsupported.

The label operator is  S = P' S_bb P  with  P = scalar (x) I3, where `scalar` is the P1 prolongation from the
active carrier scalar nodes to the fine boundary-trace vertices (fixed_port_adapter._compile_scalar_map).  Hence

    Null(S) = null(scalar) (x) R3   +   P^-1( Null(S_bb) )   =   3 * dim null(scalar)  +  6 * components.

null(scalar) contains one indicator per carrier column the mesh trace never reaches (the "zero rows"), but it is
LARGER whenever several carrier columns are linearly dependent on the trace: that happens where the material band
is narrower than the carrier patch, so the patch holds fewer trace vertices than the carrier basis functions
covering it.  Each such dependency costs exactly THREE null modes, not one.

This script tests that decomposition WITHOUT re-running the pipeline, from the operator alone:

  * V = an orthonormal basis of Null(S) (eigenvalues below tol * lambda_max).
  * H = sum_d V_d V_d'  with V_d = V[d::3, :]  (n x n, eigenvalues in [0, 3]).
    A scalar pattern w has w (x) e_d in Null(S) for EVERY direction d  <=>  H w = 3 w.
    So dim W := #{mu = 3} is dim null(scalar) whenever the decomposition above holds, and then
        dim Null(S)  ==  3 * dim W + 3        (the +3 are the rotations, which are not of the form w (x) e_d).
  * The extra scalar patterns (W minus the unsupported-node indicators minus the constant) are then located: if
    they sit on a handful of NEIGHBOURING carrier nodes, they are dependent columns, which is the claim.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_error import load_label

T = Path("/root/autodl-tmp/_claude_diag/tonight")
TOL = 1e-14                     # relative to lambda_max, the "exact" null space
MU_TOL = 1e-8                   # mu >= 3 - MU_TOL counts as an exact scalar pattern

cases = sys.argv[1:] or ["pop_cut_0136", "pop_cut_0236", "pop_cut_0424", "pop_cut_0181"]
out = []
for case in cases:
    d = T / "B" / case
    if not (d / "FIXED_PORT_SCHUR_TET10.npz").exists():
        print(f"{case}: missing"); continue
    lab = load_label(d); S = lab["S"]; xyz = lab["xyz"]; ids = lab["ids"]
    r = lab["receipt"]; n = S.shape[0] // 3
    comps = int(((r.get("surface") or {}).get("components") or {}).get("components", 1))
    dropped = int(((r.get("surface") or {}).get("components") or {}).get("dropped", 0))
    kept = max(comps - dropped, 1)

    w, U = np.linalg.eigh(S); wmax = w[-1]
    k = int((w <= TOL * wmax).sum()); V = U[:, :k]
    rown = np.linalg.norm(S.reshape(n, 3, -1), axis=(1, 2))
    zero = rown <= 1e-12 * rown.max(); nz = int(zero.sum())

    H = sum(V[dd::3, :] @ V[dd::3, :].T for dd in range(3))
    mu, Q = np.linalg.eigh(H)
    dimW = int((mu >= 3.0 - MU_TOL).sum())
    Wb = Q[:, mu >= 3.0 - MU_TOL]                              # n x dimW, orthonormal
    predicted_from_W = 3 * dimW + 3 * kept                     # 3 rotations per kept component

    # strip the unsupported-node indicators: restrict to supported nodes and take the column space there
    sup = ~zero
    Ws = Wb[sup, :]
    if Ws.size:
        s = np.linalg.svd(Ws, compute_uv=False)
        dim_sup = int((s > 1e-8 * max(s[0], 1e-300)).sum())
    else:
        dim_sup = 0
    # an orthonormal basis of that supported column space, with the constant removed
    extra_patterns = []
    if dim_sup:
        Uw, sw, _ = np.linalg.svd(Ws, full_matrices=False)
        B = Uw[:, :dim_sup]                                    # n_sup x dim_sup
        one = np.ones(B.shape[0]); one /= np.linalg.norm(one)
        c = B.T @ one
        # basis of B's span orthogonal to the constant
        Bc = B - np.outer(one, c)
        Ub, sb, _ = np.linalg.svd(Bc, full_matrices=False)
        m = int((sb > 1e-8 * max(sb[0], 1e-300)).sum())
        idx_sup = np.flatnonzero(sup)
        for j in range(min(m, 12)):
            v = Ub[:, j]; a = np.abs(v)
            car = np.flatnonzero(a > 0.02 * a.max())
            order = car[np.argsort(-a[car])]
            extra_patterns.append({
                "nodes": int(len(car)),
                "top": [{"id": ids[int(idx_sup[t])], "xyz": [round(float(x), 6) for x in xyz[int(idx_sup[t])]],
                         "w": round(float(v[t]), 4)} for t in order[:8]],
                "mass_on_top4": float((a[order[:4]] ** 2).sum() / (a ** 2).sum()),
            })
        n_extra = m
    else:
        n_extra = 0

    row = {"case": case, "n": n, "q": 3 * n, "kept_components": kept, "zero_rows": nz,
           "null_dim": k, "combinatorial_prediction": 6 * kept + 3 * nz,
           "dim_W": dimW, "prediction_from_W": predicted_from_W,
           "W_matches_null": bool(predicted_from_W == k),
           "dim_W_on_supported_nodes": dim_sup, "extra_scalar_patterns": n_extra,
           "extra_patterns": extra_patterns,
           "mu_near_3": [float(x) for x in mu[-min(dimW + 6, n):][:0]] or None}
    out.append(row)
    print(f"\n=== {case} ===")
    print(f"  carrier scalar nodes n = {n}, q = {3*n}, kept components {kept}, unsupported (zero) columns {nz}")
    print(f"  dim Null(S) measured                 {k}")
    print(f"  6*components + 3*unsupported         {6*kept + 3*nz}      (difference {k - 6*kept - 3*nz:+d})")
    print(f"  dim W  (scalar patterns, H w = 3w)   {dimW}")
    print(f"  3*dim W + 3*components               {predicted_from_W}      -> {'MATCH' if predicted_from_W == k else 'MISMATCH'}")
    print(f"  of those scalar patterns: {nz} are unsupported-node indicators, 1 is the constant, {n_extra} are")
    print(f"  genuine linear dependencies between SUPPORTED carrier columns")
    for j, p in enumerate(extra_patterns[:6]):
        print(f"    pattern {j}: {p['nodes']} nodes carry it, {p['mass_on_top4']*100:.1f}% of its mass on the top 4:")
        for t in p["top"][:4]:
            print(f"        w {t['w']:+.3f}  at {t['xyz']}   {t['id']}")
    sys.stdout.flush()

json.dump(out, open(T / "NULLSPACE_WHY.json", "w"), indent=1, default=float)
print("\nwritten", T / "NULLSPACE_WHY.json")
