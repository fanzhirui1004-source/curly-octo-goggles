#!/usr/bin/env python3
"""Parameter economy: how many predictable numbers does it take to determine S to the gate?

The structure's job is not to make the apply fast -- a dense apply at d = 12792 is about 1 ms.
Its job is to cut what the network must emit from d(d+1)/2 = 8.2e7 independent entries down to
something a decoder can produce.  So the axis that matters is parameter count against achievable
accuracy, and the oracle floor measures exactly that: for a given T and block partition,

    floor(T) = sum_b logdet C_bb - logdet C,      C = T A^-1 T^T = (T R*^-1)(T R*^-1)^T

is the divergence no D in that class can beat.  Turning the gates into necessary conditions:

    all mu within +-3%   requires  D/d <= 4.5921e-4
    all mu within +-10%  requires  D/d <= 5.3605e-3

T here is the teacher's own Cholesky factor truncated to a spatial band.  That is a concrete,
free, non-trivial T, and floor(T_trunc) is ACHIEVABLE, so it upper-bounds the best floor over
the banded class at that radius.  A small value is therefore informative; a large one says this
particular candidate is bad, not that the class is.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

NEC3, NEC10 = 4.5921e-4, 5.3605e-3


def spatial_blocks(points, block_size):
    d = len(points); out = []
    def rec(idx):
        if len(idx) <= block_size:
            out.append(np.sort(idx)); return
        axis = int(np.argmax(points[idx].max(0) - points[idx].min(0)))
        order = idx[np.argsort(points[idx, axis], kind='stable')]
        h = len(order)//2; rec(order[:h]); rec(order[h:])
    rec(np.arange(d, dtype=np.int64))
    return out


def floor_from_G(G, blocks):
    sign, ldG = torch.linalg.slogdet(G)
    total = 0.0
    for idx in blocks:
        Gb = G[idx]
        total += float(torch.linalg.slogdet(Gb @ Gb.T)[1])
    return (total - 2.0 * float(ldG)) / G.shape[0]


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
    dense_params = d * (d + 1) // 2
    print('seat 328  d %d   dense S has %.2e independent entries' % (d, dense_params), flush=True)
    print('necessary conditions on the floor:  +-3%%  D/d <= %.4e |  +-10%%  D/d <= %.4e\n' % (NEC3, NEC10), flush=True)
    print('%-8s %-11s %-12s %-12s %-11s %-9s' %
          ('radius', 'band nnz', 'block', 'params', 'floor D/d', 'sec'), flush=True)
    rows = []
    blocks_cache = {}
    for radius in (0.0, 0.03, 0.06, 0.10, 0.15, 0.25):
        ts = time.time()
        if radius == 0.0:
            T = torch.diag(R.diagonal()).clone()
            nnz = d
        else:
            keep = torch.zeros(d, d, dtype=torch.bool, device=V.DEV)
            for lo in range(0, d, 2048):
                hi = min(lo + 2048, d)
                dist = torch.cdist(P[lo:hi], P)
                keep[lo:hi] = dist <= radius
            iu = torch.triu(torch.ones(d, d, dtype=torch.bool, device=V.DEV))
            keep &= iu
            keep.fill_diagonal_(True)
            T = torch.where(keep, R, torch.zeros((), dtype=torch.float64, device=V.DEV))
            nnz = int(keep.sum()); del keep, iu
        G = T @ Rinv
        del T; torch.cuda.empty_cache()
        for block in (1, 8, 24, 64):
            if block not in blocks_cache:
                blocks_cache[block] = [torch.as_tensor(b, device=V.DEV) for b in spatial_blocks(pts, block)]
            bl = blocks_cache[block]
            f = floor_from_G(G, bl)
            nblk = sum(len(b) * (len(b) + 1) // 2 for b in bl)
            params = nnz + nblk
            mark = 'PASSES +-3%' if f <= NEC3 else ('passes +-10%' if f <= NEC10 else '')
            print('%-8.2f %-11d %-12d %-12d %-11.4e %-9.0f %s'
                  % (radius, nnz, block, params, f, time.time()-ts, mark), flush=True)
            rows.append(dict(radius=radius, band_nnz=nnz, block=block, params=params,
                             floor=f, frac_of_dense=params / dense_params))
        del G; torch.cuda.empty_cache()
    Path('/root/autodl-tmp/NEURAL_SCHUR/PARAM_ECONOMY.json').write_text(
        json.dumps(dict(seat=328, d=int(d), dense_params=int(dense_params),
                        nec3=NEC3, nec10=NEC10, rows=rows, seconds=time.time()-t0), indent=1))
    print('\nwritten PARAM_ECONOMY.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
