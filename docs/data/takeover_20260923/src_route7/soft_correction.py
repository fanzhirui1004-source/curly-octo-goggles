"""Route 7: single-precision Schur operator made double-precision accurate where it matters.

T32 (fp32 partial factorization) has relative errors ~1e-6 of |T|; they matter only in soft directions, where
x^T T x is ~1e-5 of the largest eigenvalue (precision_study: whitened spectrum up to 2.2% off, 2-cell lattice cell
energies 3.8% off). Correction on the soft subspace:
  V = the k softest non-rigid eigenvectors of P T32 P (orthonormal, orthogonal to the rigid modes R),
  W = T V computed in double precision (here T64 V; in production k refined solves, i.e. k/512 panels of the panel route),
  E = W - T32p V,  T_c = T32p + E V^T + V E^T - V (V^T E) V^T,
so that T_c V = W = T V exactly, T_c R = 0, and the remaining error lives on the stiffer complement.
Reports, for k in a list: whitened spectrum of T_c against T64 and the 2-cell lattice (config x) with T_c for both cells.
Also: the eigenvalue distribution of T64 (counts below fractions of the largest eigenvalue).
Usage: soft_correction.py <body_dir> <out_dir> <cut_case> <full_case> <k,k,...> [<extra_cut_case> ...]
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import precision_study as PS

dev, dt = PS.dev, PS.dt


def whitened(T64h, Tch, R, alpha):
    """max |mu - 1| and counts of mu = eig(T64^-1/2 Tc T64^-1/2) on the rigid complement (both rigid-free)."""
    A = T64h.to(dev, dt); PS.project_(A, R); A += alpha * (R @ R.T)
    L = torch.linalg.cholesky(A); del A
    B = Tch.to(dev, dt); B += alpha * (R @ R.T)
    X = torch.linalg.solve_triangular(L, B, upper=False); del B
    Xt = X.T.contiguous(); del X
    B = torch.linalg.solve_triangular(L, Xt, upper=False); del Xt, L
    C = B + B.T; del B
    C.mul_(0.5)
    mu = torch.linalg.eigvalsh(C); del C
    d = (mu - 1).abs()
    out = {'min': float(mu.min()), 'max': float(mu.max()), 'dev_max': float(d.max()),
           'above_1e-4': int((d > 1e-4).sum()), 'above_1e-3': int((d > 1e-3).sum()), 'above_1e-2': int((d > 1e-2).sum())}
    del mu, d; gc.collect(); torch.cuda.empty_cache()
    return out


def main(body_dir, out, cut, full, ks, extra):
    Path(out).mkdir(parents=True, exist_ok=True)
    rec = dict(cut=cut, full=full, ks=ks, cells={})
    lat = {}
    kmax = max(ks)
    for case in [cut] + extra + [full]:
        is_full = case == full
        t0 = time.perf_counter()
        T64, ids, n, _ = PS.schur_T(case, body_dir, 'fp64', host=is_full)
        T32, _, _, _ = PS.schur_T(case, body_dir, 'fp32', host=False)
        R = PS.rigid_basis(ids, n)
        # eigen-structure of T64 (rigid-projected) and the soft subspace of T32p
        A = T64.to(dev, dt); PS.project_(A, R)
        ev64 = torch.linalg.eigvalsh(A).sort().values; del A
        gc.collect(); torch.cuda.empty_cache()
        lam_max = float(ev64[-1])
        dist = {f'below_{f:g}': int((ev64[6:] < f * lam_max).sum()) for f in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)}
        del ev64
        B = T32.to(dev, dt); PS.project_(B, R)
        T32p = B.cpu()
        ev, vec = torch.linalg.eigh(B); del B
        order = torch.argsort(ev)
        V_all = vec[:, order[6:6 + kmax]].contiguous()                                # skip the 6 rigid (zero) ones
        lam_k = {k: float(ev[order[6 + k]] / ev[order[-1]]) for k in ks if 6 + k < len(ev)}
        del ev, vec, order; gc.collect(); torch.cuda.empty_cache()
        W_all = torch.cat([T64[r0:r0 + 4096].to(dev, dt) @ V_all for r0 in range(0, T64.shape[0], 4096)])   # = T V
        row = dict(box=int(T64.shape[0]), T64_eig_distribution=dist, T32_lambda_k_over_max=lam_k, corrections={})
        row['uncorrected'] = whitened(T64, T32p, R, lam_max)
        print(json.dumps(dict(case=case, k=0, **row['uncorrected'], dist=dist)), flush=True)
        for k in ks:
            V, W = V_all[:, :k], W_all[:, :k]
            Tc = T32p.to(dev, dt)
            E = W - Tc @ V
            Tc += E @ V.T
            Tc += V @ E.T
            Tc -= V @ ((V.T @ E) @ V.T)
            Tc = 0.5 * (Tc + Tc.T)
            Tch = Tc.cpu(); del Tc, E
            gc.collect(); torch.cuda.empty_cache()
            row['corrections'][k] = whitened(T64, Tch, R, lam_max)
            print(json.dumps(dict(case=case, k=k, **row['corrections'][k])), flush=True)
            if case in (cut, full) and k == ks[len(ks) // 2]:
                lat.setdefault(case, (ids, {'T64': T64}))[1][f'Tc{k}'] = Tch.to(torch.float64)
                lat[case][1]['T32p'] = T32p
            del Tch
        row['seconds'] = time.perf_counter() - t0
        rec['cells'][case] = row
        del R, V_all, W_all, T32
        gc.collect(); torch.cuda.empty_cache()
    cells = [('cut', (0, 0, 0), lat[cut][0], lat[cut][1]), ('full', (-1, 0, 0), lat[full][0], lat[full][1])]
    rec['lattice_x'] = PS.lattice_check(cells, log=print)
    (Path(out) / 'SOFT_CORRECTION.json').write_text(json.dumps(rec, indent=2, default=float))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], [int(x) for x in sys.argv[5].split(',')], sys.argv[6:])
