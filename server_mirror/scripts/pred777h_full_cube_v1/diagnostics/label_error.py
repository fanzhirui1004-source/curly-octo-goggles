"""Single-cell label error: how far is one cell's fixed-port Schur operator from a reference, without choosing a load.

The carrier is fixed by geometry alone (a Cartesian background grid intersected with the cell's boundary planes) and
never depends on the tet mesh, so two labels of the same cell at different mesh resolutions live in the SAME port
space and can be compared entry by entry.  The metric of record is therefore the generalized eigenvalue problem

    S_h v = lambda S_ref v      on the range of S_ref (its six rigid modes and its unsupported nodes removed),

whose extreme eigenvalues bound the relative strain-energy error over EVERY port displacement:

    lambda_min <= (v' S_h v) / (v' S_ref v) <= lambda_max     for all v in that subspace.

No load is chosen and the bound is sharp.  Reported alongside, for interpretation only, are the six uniform-strain
apparent stiffnesses, which are the standard homogenisation load cases.

Comparing across CARRIER resolutions is a different question, because the space itself changes; there the operators
are compared through the uniform-strain stiffnesses (the affine fields belong to every carrier space exactly).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RIGID_TOL = 1.0e-9          # a generalized eigenvalue is kept when the reference mode carries this much energy


def load_label(directory: Path) -> dict:
    """Operator, carrier coordinates and identities of one produced label."""
    z = np.load(Path(directory) / "FIXED_PORT_SCHUR_TET10.npz")
    q = int(z["q"][0]); S = np.zeros((q, q)); iu = np.triu_indices(q); S[iu] = z["schur_upper_f32"]
    S = S + S.T - np.diag(np.diag(S))
    receipt = json.loads((Path(directory) / "LABEL_RECEIPT.json").read_text())
    return {"S": S, "q": q, "xyz": np.asarray(z["carrier_coordinates"], dtype=np.float64),
            "ids": [str(v) for v in z["carrier_ids"]], "receipt": receipt,
            "volume": float(z["material_volume"][0]) if "material_volume" in z.files else float("nan")}


def align(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Common carrier nodes of two labels of the same cell, by carrier identity (the identity is exact and geometric,
    so the sets agree unless the geometric active-set rule saw a different band)."""
    ia = {k: i for i, k in enumerate(a["ids"])}; ib = {k: i for i, k in enumerate(b["ids"])}
    common = [k for k in a["ids"] if k in ib]
    ka = np.asarray([ia[k] for k in common], dtype=np.int64); kb = np.asarray([ib[k] for k in common], dtype=np.int64)
    da = np.repeat(3 * ka, 3) + np.tile(np.arange(3), len(ka)); db = np.repeat(3 * kb, 3) + np.tile(np.arange(3), len(kb))
    report = {"nodes_a": len(a["ids"]), "nodes_b": len(b["ids"]), "common": len(common),
              "only_a": len(a["ids"]) - len(common), "only_b": len(b["ids"]) - len(common),
              "max_coordinate_mismatch": float(np.abs(a["xyz"][ka] - b["xyz"][kb]).max()) if len(ka) else 0.0}
    return da, db, report


def generalized_spectrum(S_h: np.ndarray, S_ref: np.ndarray, *, tol: float = RIGID_TOL) -> dict:
    """Eigenvalues of S_h relative to S_ref on the range of S_ref.

    S_ref is symmetric positive semidefinite with a six-dimensional rigid null space and one zero row per carrier node
    the material does not reach.  Project both onto the reference's range (eigenvalues above tol * lambda_max), where
    the pencil is a genuine similarity, and return the extreme relative energies."""
    w, U = np.linalg.eigh(S_ref)
    keep = w > tol * w[-1]
    if not keep.any():
        return {"error": "reference operator is numerically zero"}
    Q = U[:, keep] / np.sqrt(w[keep])                       # S_ref-orthonormal basis of the range
    M = Q.T @ S_h @ Q
    lam = np.linalg.eigvalsh(0.5 * (M + M.T))
    return {"kept_modes": int(keep.sum()), "removed_modes": int((~keep).sum()),
            "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
            "lambda_median": float(np.median(lam)), "lambda_p01": float(np.percentile(lam, 1)), "lambda_p99": float(np.percentile(lam, 99)),
            "energy_error_bound": float(max(abs(lam.min() - 1.0), abs(lam.max() - 1.0))),
            "reference_condition": float(w[-1] / w[keep].min())}


def uniform_strain_stiffness(S: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apparent stiffness in the six uniform strains: C[i] = u_i' S u_i with u_i the affine port displacement of the
    i-th unit strain (xx, yy, zz, yz, xz, xy).  Affine fields lie in every carrier space exactly, so this is
    comparable across carrier resolutions as well as across meshes."""
    x = xyz - xyz.mean(axis=0)
    out = []
    for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
        u = np.zeros_like(x)
        if i == j: u[:, i] = x[:, i]
        else: u[:, i] += 0.5 * x[:, j]; u[:, j] += 0.5 * x[:, i]
        v = u.ravel(); out.append(float(v @ S @ v))
    return np.asarray(out)


def compare(dir_h: Path, dir_ref: Path) -> dict:
    """One label against a reference label of the same cell."""
    a = load_label(dir_h); b = load_label(dir_ref)
    da, db, rep = align(a, b)
    out = {"alignment": rep, "case": a["receipt"]["case_id"],
           "resolution_h": a["receipt"]["resolution"], "resolution_ref": b["receipt"]["resolution"],
           "q_h": a["q"], "q_ref": b["q"], "volume_h": a["volume"], "volume_ref": b["volume"]}
    if rep["common"] < 12:
        out["error"] = "too few common carrier nodes"; return out
    Sh = a["S"][np.ix_(da, da)]; Sr = b["S"][np.ix_(db, db)]
    out["spectrum"] = generalized_spectrum(Sh, Sr)
    Ch = uniform_strain_stiffness(Sh, a["xyz"][da[::3] // 3]); Cr = uniform_strain_stiffness(Sr, b["xyz"][db[::3] // 3])
    out["uniform_strain_h"] = Ch.tolist(); out["uniform_strain_ref"] = Cr.tolist()
    out["uniform_strain_relative"] = (Ch / np.where(np.abs(Cr) > 0, Cr, np.nan) - 1.0).tolist()
    out["uniform_strain_max_abs_relative"] = float(np.nanmax(np.abs(Ch / np.where(np.abs(Cr) > 0, Cr, np.nan) - 1.0)))
    # the whitened Frobenius the route has been quoting, for continuity with earlier reports
    out["frobenius_relative"] = float(np.linalg.norm(Sh - Sr) / max(np.linalg.norm(Sr), 1e-300))
    return out


def compare_across_carriers(dir_h: Path, dir_ref: Path) -> dict:
    """Two labels of the same cell on DIFFERENT carrier resolutions: the port spaces differ, so only quantities whose
    fields belong to both spaces are comparable.  The six uniform strains are affine and belong to every P1 carrier."""
    a = load_label(dir_h); b = load_label(dir_ref)
    Ch = uniform_strain_stiffness(a["S"], a["xyz"]); Cr = uniform_strain_stiffness(b["S"], b["xyz"])
    rel = Ch / np.where(np.abs(Cr) > 0, Cr, np.nan) - 1.0
    return {"case": a["receipt"]["case_id"], "carrier_h": a["receipt"]["resolution"]["carrier_n"], "carrier_ref": b["receipt"]["resolution"]["carrier_n"],
            "q_h": a["q"], "q_ref": b["q"], "uniform_strain_h": Ch.tolist(), "uniform_strain_ref": Cr.tolist(),
            "uniform_strain_relative": rel.tolist(), "uniform_strain_max_abs_relative": float(np.nanmax(np.abs(rel))),
            "volume_h": a["volume"], "volume_ref": b["volume"]}


__all__ = ["load_label", "align", "generalized_spectrum", "uniform_strain_stiffness", "compare", "compare_across_carriers"]
