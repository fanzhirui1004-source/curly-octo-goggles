"""A1 revisited: the single-cell mesh error on a subspace where the question is well posed.

The first version solved  S_h v = lambda S_ref v  on the range of S_ref, keeping every reference mode above
1e-9 * lambda_max.  That is not a usable metric here, because the operator has no spectral gap: below about
1e-6 * lambda_max sit hundreds of modes that are slivers of material or carrier columns one mesh reaches and the
other does not, and dividing by their energy produced lambda in [0, 510].

Two changes make it well posed.

  COMMON SUPPORT.  Restrict to carrier nodes both labels actually load.  A node one mesh reaches and the other does
  not is not a discretization error of the operator, it is a different active support; it is counted and reported
  separately instead of being divided by.

  AN ENERGY FLOOR THAT IS DECLARED.  Report the bound as a function of the retained reference energy: keep the
  reference modes with lambda_ref >= eps * lambda_max and give max|lambda - 1| for a ladder of eps.  There is no
  gap to find, so no single eps is "the" answer; the curve is the answer, and the reader picks the operating point.
  Alongside it, the share of trace(S_ref) the retained modes carry, so "eps = 1e-6" can be read as "this covers
  99.4 % of the operator's energy".
"""
from __future__ import annotations

import numpy as np

from label_error import load_label, uniform_strain_stiffness

FLOORS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8)


def supported_nodes(S: np.ndarray, *, relative: float = 1e-12) -> np.ndarray:
    n = S.shape[0] // 3
    rn = np.linalg.norm(S.reshape(n, 3, -1), axis=(1, 2))
    return rn > relative * rn.max()


def common_support(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Vector dof index arrays into a and b for the carrier nodes both operators load."""
    ia = {k: i for i, k in enumerate(a["ids"])}; ib = {k: i for i, k in enumerate(b["ids"])}
    sa = supported_nodes(a["S"]); sb = supported_nodes(b["S"])
    keys = [k for k in a["ids"] if k in ib and sa[ia[k]] and sb[ib[k]]]
    ka = np.asarray([ia[k] for k in keys], dtype=np.int64); kb = np.asarray([ib[k] for k in keys], dtype=np.int64)
    rep = {"nodes_a": len(a["ids"]), "nodes_b": len(b["ids"]),
           "supported_a": int(sa.sum()), "supported_b": int(sb.sum()), "common_supported": len(keys),
           "supported_in_ref_only": int(sb.sum()) - len(keys), "supported_in_h_only": int(sa.sum()) - len(keys)}
    da = np.repeat(3 * ka, 3) + np.tile(np.arange(3), len(ka))
    db = np.repeat(3 * kb, 3) + np.tile(np.arange(3), len(kb))
    return da, db, rep


def spectrum_by_floor(S_h: np.ndarray, S_ref: np.ndarray, floors=FLOORS) -> list[dict]:
    """max|lambda - 1| over the reference modes above each energy floor, with the trace share they carry."""
    w, U = np.linalg.eigh(S_ref); wmax = w[-1]; total = float(w[w > 0].sum())
    M_full = U.T @ S_h @ U
    out = []
    for eps in floors:
        keep = w >= eps * wmax
        if keep.sum() < 6:
            out.append({"floor": eps, "modes": int(keep.sum()), "error": "fewer than six modes above the floor"}); continue
        idx = np.flatnonzero(keep)
        Q = U[:, idx] / np.sqrt(w[idx])
        M = Q.T @ S_h @ Q
        lam = np.linalg.eigvalsh(0.5 * (M + M.T))
        out.append({"floor": eps, "modes": int(keep.sum()),
                    "energy_share": float(w[idx].sum() / max(total, 1e-300)),
                    "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
                    "bound": float(max(abs(lam.min() - 1.0), abs(lam.max() - 1.0))),
                    "lambda_p05": float(np.percentile(lam, 5)), "lambda_p95": float(np.percentile(lam, 95))})
    _ = M_full
    return out


def compare_v2(dir_h, dir_ref) -> dict:
    a = load_label(dir_h); b = load_label(dir_ref)
    da, db, rep = common_support(a, b)
    out = {"case": a["receipt"]["case_id"], "support": rep,
           "resolution_h": a["receipt"]["resolution"]["preset"], "resolution_ref": b["receipt"]["resolution"]["preset"],
           "q_h": a["q"], "q_ref": b["q"], "tets_h": int(a["receipt"]["mesh"]["tet_count"]),
           "tets_ref": int(b["receipt"]["mesh"]["tet_count"])}
    if rep["common_supported"] < 12:
        out["error"] = "too few commonly supported nodes"; return out
    Sh = a["S"][np.ix_(da, da)]; Sr = b["S"][np.ix_(db, db)]
    out["ladder"] = spectrum_by_floor(Sh, Sr)
    xa = a["xyz"][da[::3] // 3]
    Ch = uniform_strain_stiffness(Sh, xa); Cr = uniform_strain_stiffness(Sr, xa)
    out["uniform_strain_relative"] = (Ch / np.where(np.abs(Cr) > 0, Cr, np.nan) - 1.0).tolist()
    out["uniform_strain_max_abs_relative"] = float(np.nanmax(np.abs(np.asarray(out["uniform_strain_relative"]))))
    out["frobenius_relative_on_common_support"] = float(np.linalg.norm(Sh - Sr) / max(np.linalg.norm(Sr), 1e-300))
    return out


__all__ = ["supported_nodes", "common_support", "spectrum_by_floor", "compare_v2", "FLOORS"]
