#!/usr/bin/env python3
"""Put the banded sweep on the eps_op axis, which it was never measured on.

PARAM_ECONOMY reported only floor(T) for T = R* truncated to a spatial band.  E1-D then showed on
this same teacher that a floor of 1.452e-01 corresponds to eps_op = 65.4, so a floor of 2.95e-02
means nothing on its own.  Every construction has to be compared on the gate, not on a divergence.

For T banded and D at its closed-form optimum the whitened operator is available without forming
A_hat at all:

    H = R*^-T (T^T D T) R*^-1 = G^T D G,     G = T R*^-1,     D_b = ([G G^T]_bb)^-1

so with D_b = L_b L_b^T, H = M^T M for M stacked from M_b = L_b^T G_b, and mu = eig(H) exactly.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
from backend import spatial_blocks

NEC3, NEC10 = 4.5921e-4, 5.3605e-3


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
    order = g['data'].quotient.order.cpu().numpy(); posn = order[6:]
    pts = label.ijk[posn // 3].astype(float) / 64.0
    P = torch.as_tensor(pts, dtype=torch.float64, device=V.DEV)
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    blocks = [torch.as_tensor(b, device=V.DEV) for b in spatial_blocks(pts, 64)]
    dense_params = d * (d + 1) // 2
    print('seat 328  d %d\n' % d, flush=True)
    print('%-8s %-11s %-9s %-12s %-8s %-8s %-11s %-11s %s'
          % ('radius', 'params', 'fracdense', 'floor D/d', 'n>3%', 'n>10%', 'mu_min', 'mu_max', 'eps_op'), flush=True)
    rows = []
    for radius in (0.0, 0.03, 0.06, 0.10, 0.15, 0.25):
        ts = time.time()
        if radius == 0.0:
            T = torch.diag(R.diagonal()).clone(); nnz = d
        else:
            keep = torch.zeros(d, d, dtype=torch.bool, device=V.DEV)
            for lo in range(0, d, 2048):
                hi = min(lo + 2048, d)
                keep[lo:hi] = torch.cdist(P[lo:hi], P) <= radius
            keep &= torch.triu(torch.ones(d, d, dtype=torch.bool, device=V.DEV))
            keep.fill_diagonal_(True)
            T = torch.where(keep, R, torch.zeros((), dtype=torch.float64, device=V.DEV))
            nnz = int(keep.sum()); del keep
        G = T @ Rinv
        del T; torch.cuda.empty_cache()
        M = torch.empty_like(G)
        floor = 0.0
        for idx in blocks:
            Gb = G[idx]
            Cbb = Gb @ Gb.T
            floor += float(torch.linalg.slogdet(Cbb)[1])
            Lb = torch.linalg.cholesky(torch.linalg.inv(0.5 * (Cbb + Cbb.T)), upper=True)  # D_b = Lb^T Lb
            M[idx] = Lb @ Gb
        floor = (floor - 2.0 * float(torch.linalg.slogdet(G)[1])) / d
        del G; torch.cuda.empty_cache()
        H = M.T @ M
        del M; torch.cuda.empty_cache()
        mu = torch.linalg.eigvalsh(0.5 * (H + H.T))
        del H; torch.cuda.empty_cache()
        eps = float((mu - 1).abs().max())
        o3 = int(((mu - 1).abs() > 0.03).sum()); o10 = int(((mu - 1).abs() > 0.10).sum())
        params = nnz + sum(len(b) * (len(b) + 1) // 2 for b in blocks)
        print('%-8.2f %-11d %-9.4f %-12.4e %-8d %-8d %-11.4e %-11.4e %.4e %s'
              % (radius, params, params/dense_params, floor, o3, o10, float(mu[0]), float(mu[-1]), eps,
                 'PASSES +-3%' if eps <= 0.03 else 'PASSES +-10%' if eps <= 0.10 else 'fails'), flush=True)
        rows.append(dict(radius=radius, params=params, frac_of_dense=params/dense_params, floor=floor,
                         eps_op=eps, n_outside_3pct=o3, n_outside_10pct=o10,
                         mu_min=float(mu[0]), mu_max=float(mu[-1]), seconds=time.time()-ts))
        del mu; torch.cuda.empty_cache()
    Path('/root/autodl-tmp/NEURAL_SCHUR/BANDED_AUDIT.json').write_text(json.dumps(
        dict(seat=328, d=int(d), dense_params=int(dense_params), block=64,
             nec3=NEC3, nec10=NEC10, rows=rows, seconds=time.time()-t0), indent=1))
    print('\nwritten BANDED_AUDIT.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
