#!/usr/bin/env python3
"""E2 - Schaefer/Owhadi made concrete: does a maximin-ordered radius pattern beat
magnitude truncation, and at what number of entries?

Schaefer & Owhadi (arXiv 2110.05351; Schaefer/Katzfuss/Owhadi SISC 43(3) 2021) prove
that the Cholesky factor of an elliptic operator becomes sparse in a maximin
(coarse-to-fine) ordering, with operator-norm accuracy eps at O(N log N log^d(N/eps))
nonzeros. Operator norm is exactly the norm eps_op reads, so if the constants are
tolerable at our N the compression half needs no learning at all.

This measures the constants on OUR operator instead of estimating them. For each
ordering and radius rho, keep entry (i,j) of the reordered Cholesky factor when
dist(x_i, x_j) <= rho * ell_j (ell = the maximin length scale), and report eps_op and
the nonzero count.

Baselines to beat, both on seat 0328: hierarchical low rank reaches eps_op 2.6e-2 at
12.23e6 numbers (14.95% of dense); magnitude truncation at 12.2% of dense gives 1.15.
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
DEV = V.DEV; F64 = torch.float64
OUT = Path('/root/autodl-tmp/CLAUDE_MAXIMIN_20260917'); OUT.mkdir(exist_ok=True)


def maximin_nodes(P):
    """Greedy maximin (farthest-point) ordering; returns order and length scales."""
    n = len(P); order = np.empty(n, dtype=np.int64); ell = np.empty(n)
    Pt = torch.as_tensor(P, dtype=F64, device=DEV)
    start = int(torch.cdist(Pt, Pt.mean(0, keepdim=True)).squeeze(1).argmin())
    order[0] = start; ell[0] = float('inf')
    dist = torch.cdist(Pt, Pt[start:start + 1]).squeeze(1)
    for k in range(1, n):
        j = int(dist.argmax()); order[k] = j; ell[k] = float(dist[j])
        dist = torch.minimum(dist, torch.cdist(Pt, Pt[j:j + 1]).squeeze(1))
    ell[0] = ell[1] * 2 if n > 1 else 1.0
    return order, ell


def eps_op(Rh, Rs, d):
    Mt = torch.linalg.solve_triangular(Rs.T, Rh.T, upper=False); M = Mt.T.contiguous(); del Mt
    W = M.T @ M; del M; gc.collect(); torch.cuda.empty_cache()
    mu = torch.linalg.eigvalsh((W + W.T) * .5); del W; gc.collect(); torch.cuda.empty_cache()
    mn, mx = float(mu.min()), float(mu.max()); del mu
    return max(abs(mn - 1), abs(mx - 1)), mn, mx


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    order_q = g['data'].quotient.order.cpu().numpy()
    node_of = order_q[6:] // 3                      # quotient coord -> trace node
    pts = label.ijk[node_of].astype(float) / 64.0   # (d,3)
    R0 = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, DEV)
    A = R0.T @ R0
    dense = d * (d + 1) // 2
    print(f'seat 328  d={d}  dense entries={dense}', flush=True)

    nodes = np.unique(node_of)
    Pn = np.stack([pts[node_of == nd][0] for nd in nodes])
    print(f'{len(nodes)} unique trace nodes; computing maximin ordering...', flush=True)
    mm_order, mm_ell = maximin_nodes(Pn)
    print(f'  maximin done ({time.time()-t0:.0f} s), ell from {mm_ell[1]:.4g} to {mm_ell[-1]:.4g}', flush=True)
    groups = {int(nd): np.flatnonzero(node_of == nd) for nd in nodes}

    out = dict(seat=328, d=int(d), dense_entries=int(dense),
               reference=dict(hmatrix_numbers=12229217, hmatrix_eps_op=2.597e-2,
                              magnitude_truncation_at_12pct=1.15), orderings={})
    for tag, node_seq, ell_seq in (
            ('maximin', mm_order, mm_ell),
            ('reverse_maximin', mm_order[::-1], mm_ell[::-1])):
        perm = np.concatenate([groups[int(nodes[k])] for k in node_seq])
        ell = np.concatenate([np.full(len(groups[int(nodes[k])]), max(ell_seq[i], 1e-12))
                              for i, k in enumerate(node_seq)])
        pi = torch.as_tensor(perm, device=DEV)
        Ap = A[pi.unsqueeze(1), pi.unsqueeze(0)]
        Ap = 0.5 * (Ap + Ap.T)
        Rs = torch.linalg.cholesky(Ap).mH.contiguous(); del Ap; gc.collect(); torch.cuda.empty_cache()
        Pp = torch.as_tensor(pts[perm], dtype=F64, device=DEV)
        Lp = torch.as_tensor(ell, dtype=F64, device=DEV)
        rows = []
        for rho in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            Dm = torch.cdist(Pp, Pp)
            keep = torch.triu(Dm <= rho * Lp.unsqueeze(0)); del Dm
            keep |= torch.eye(d, dtype=torch.bool, device=DEV)
            Rh = torch.where(keep, Rs, torch.zeros((), dtype=F64, device=DEV))
            nnz = int(keep.sum()); del keep; gc.collect(); torch.cuda.empty_cache()
            v, mn, mx = eps_op(Rh, Rs, d); del Rh; gc.collect(); torch.cuda.empty_cache()
            rows.append(dict(rho=rho, nnz=nnz, fraction_of_dense=nnz / dense, eps_op=v,
                             mu_min=mn, mu_max=mx))
            print(f'  {tag:<16} rho={rho:<5} nnz={nnz:>10} ({nnz/dense:7.3%})  '
                  f'eps_op={v:12.5g}  mu=[{mn:.4g},{mx:.4g}]', flush=True)
            out['orderings'][tag] = rows
            (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1))
        del Rs, Pp, Lp; gc.collect(); torch.cuda.empty_cache()
    print(f'\ndone ({time.time()-t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
