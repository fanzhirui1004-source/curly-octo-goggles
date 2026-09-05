"""A1, third version: the error on the INTERFACE modes a neighbour can excite.

Version 2 bounded the relative energy over every reference mode above an energy floor, and the bound is 2 to 30 even
between repaired labels whose uniform-strain stiffnesses agree to 0.1 %: the top of the spectrum is hundreds of
single-node modes whose value is the local mesh around one carrier node, and those a neighbouring cell never
excites.  Assembly couples cells through the SOFT interface modes, so the metric of record is taken there:

    S_ref v = mu M v,   M = carrier mass matrix (x) I3 on the common support,   the k lowest mu (rigid modes removed),
    bound_k = max |lambda - 1| over the energies of S_h on span(v_1..v_k), for a ladder of k.

The mass matrix is the label's own normalisation (decision of record), so the ranking of modes is mesh independent.
"""
from __future__ import annotations
import numpy as np, scipy.linalg
from label_error import load_label, uniform_strain_stiffness
from label_error2 import common_support
from pathlib import Path

KS = (12, 30, 100, 300, 1000)


def load_mass(directory: Path, ids_order: list[str]) -> np.ndarray:
    z = np.load(Path(directory) / "FIXED_PORT_SCHUR_TET10.npz"); n = int(z["carrier_scalar_count"][0])
    M = np.zeros((n, n)); rc = z["carrier_mass_coo"]; np.add.at(M, (rc[0], rc[1]), z["carrier_mass_values"])
    return M


def compare_v3(dir_h: Path, dir_ref: Path) -> dict:
    a = load_label(dir_h); b = load_label(dir_ref)
    da, db, rep = common_support(a, b)
    Sh = a["S"][np.ix_(da, da)]; Sr = b["S"][np.ix_(db, db)]
    Mn = load_mass(dir_ref, b["ids"])[np.ix_(db[::3] // 3, db[::3] // 3)]; M = np.kron(Mn, np.eye(3))
    # M-orthonormal soft modes of the reference (M is SPD on the support)
    mu, V = scipy.linalg.eigh(Sr, M)                       # ascending mu; V' M V = I
    keep = mu > 1e-9 * mu[-1]                              # drop the six rigid modes (and nothing else on a supported set)
    V = V[:, keep]; mu = mu[keep]
    out = {"case": a["receipt"]["case_id"], "resolution_h": a["receipt"]["resolution"]["preset"], "support": rep, "ladder": []}
    for k in KS:
        k = min(k, V.shape[1]); Q = V[:, :k]
        A = Q.T @ Sh @ Q; B = Q.T @ Sr @ Q
        lam = scipy.linalg.eigvalsh(0.5 * (A + A.T), 0.5 * (B + B.T))
        out["ladder"].append({"k": k, "mu_k_over_mu_max": float(mu[k - 1] / mu[-1]), "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
                              "bound": float(max(abs(lam.min() - 1), abs(lam.max() - 1)))})
    xa = a["xyz"][da[::3] // 3]; Ch = uniform_strain_stiffness(Sh, xa); Cr = uniform_strain_stiffness(Sr, xa)
    out["uniform_strain_max_abs_relative"] = float(np.nanmax(np.abs(Ch / np.where(np.abs(Cr) > 0, Cr, np.nan) - 1)))
    return out
