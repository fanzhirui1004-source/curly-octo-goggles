"""A1, fourth version: the error on SMOOTH port fields, defined by the carrier itself.

v2 (energy floor from the top) is dominated by single-node modes whose value is the local mesh around one carrier
node; v3 (softest modes in the mass norm) is dominated by hairline modes one mesh resolves and the other does not.
Neither is what a neighbouring cell transmits.  The carrier Laplace-Beltrami operator L and mass matrix M are shipped
with every label and depend on the geometry only, so the k lowest eigenvectors of  L w = nu M w  are a mesh-independent
basis of smooth port fields.  Each scalar mode gives three vector modes; the six rigid modes are projected out in the
M inner product; the bound is max |lambda - 1| over the generalized eigenvalues of (Q' S_h Q, Q' S_ref Q).
"""
from __future__ import annotations
import numpy as np, scipy.linalg
from pathlib import Path
from label_error import load_label, uniform_strain_stiffness

KS_SCALAR = (4, 10, 30, 100, 300)


def load_norms(directory: Path):
    z = np.load(Path(directory) / "FIXED_PORT_SCHUR_TET10.npz"); n = int(z["carrier_scalar_count"][0]); out = []
    for name in ("carrier_mass", "carrier_laplacian"):
        rc = z[f"{name}_coo"]; A = np.zeros((n, n)); np.add.at(A, (rc[0], rc[1]), z[f"{name}_values"]); out.append(A)
    return out[0], out[1], np.asarray(z["rigid_basis"], dtype=np.float64)


def compare_v4(dir_h: Path, dir_ref: Path) -> dict:
    a = load_label(dir_h); b = load_label(dir_ref)
    assert a["ids"] == b["ids"], "the active carrier set is geometric and must coincide"
    S_h, S_r = a["S"], b["S"]; M, L, R = load_norms(dir_ref); n = M.shape[0]
    nu, W = scipy.linalg.eigh(L, M)                          # ascending: constants first
    M3 = np.kron(M, np.eye(3))
    # rigid modes, M-orthonormalised, to be projected out
    G = R.T @ M3 @ R; Rn = R @ np.linalg.inv(np.linalg.cholesky(G)).T
    out = {"case": a["receipt"]["case_id"], "resolution_h": a["receipt"]["resolution"]["preset"], "ladder": []}
    for ks in KS_SCALAR:
        ks = min(ks, n); Q = np.kron(W[:, :ks], np.eye(3))    # 3 ks vector modes
        Q = Q - Rn @ (Rn.T @ (M3 @ Q))                         # remove rigid content
        # M-orthonormal basis of what is left (drop the (near) dependent directions)
        Gq = Q.T @ M3 @ Q; e, U = np.linalg.eigh(Gq); keep = e > 1e-8 * e.max(); Q = Q @ (U[:, keep] / np.sqrt(e[keep]))
        A = Q.T @ S_h @ Q; B = Q.T @ S_r @ Q
        lam = scipy.linalg.eigvalsh(0.5 * (A + A.T), 0.5 * (B + B.T))
        out["ladder"].append({"k_scalar": ks, "modes": int(Q.shape[1]), "nu_k": float(nu[ks - 1]), "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
                              "bound": float(max(abs(lam.min() - 1), abs(lam.max() - 1))), "lambda_median": float(np.median(lam))})
    Ch = uniform_strain_stiffness(S_h, a["xyz"]); Cr = uniform_strain_stiffness(S_r, b["xyz"])
    out["uniform_strain_max_abs_relative"] = float(np.nanmax(np.abs(Ch / np.where(np.abs(Cr) > 0, Cr, np.nan) - 1)))
    return out
