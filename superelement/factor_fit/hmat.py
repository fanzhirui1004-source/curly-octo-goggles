#!/usr/bin/env python3
"""Parameter economy of the HIERARCHICALLY LOW-RANK class, on the same axes as the banded curve.

The banded sweep (PARAM_ECONOMY.json) truncated the teacher's own Cholesky factor to a spatial
band.  It flattens out far above the gate.  The structural reason to suspect is that an interface
Schur complement of an elliptic operator is not banded -- it is a Dirichlet-to-Neumann-like
nonlocal operator whose FAR-FIELD BLOCKS ARE LOW RANK.  That is the H-matrix / FMM structure, and
it is a different parameterisation class, not a bigger band.

So: partition A hierarchically, approximate every admissible (far-field) block by a truncated SVD,
keep near-field leaf blocks dense, and place the result on the same (params, D/d) plane.  Unlike a
floor, every point here is ACHIEVED by an explicit Ahat, so a point that passes is a real pass and
a point that fails only rejects this construction at that tolerance.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

NEC3, NEC10 = 4.5921e-4, 5.3605e-3
LEAF, ETA, RANK_CAP = 64, 2.0, 320
TOLS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3)


class Node:
    __slots__ = ('idx', 'lo', 'hi', 'kids', 'nid')
    def __init__(self, idx, pts, nid):
        self.idx, self.nid, self.kids = idx, nid, []
        self.lo, self.hi = pts[idx].min(0), pts[idx].max(0)


def build_tree(pts):
    nodes, root = [], None
    def rec(idx):
        n = Node(idx, pts, len(nodes)); nodes.append(n)
        if len(idx) > LEAF:
            ax = int(np.argmax(n.hi - n.lo))
            o = idx[np.argsort(pts[idx, ax], kind='stable')]
            h = len(o) // 2
            n.kids = [rec(o[:h]), rec(o[h:])]
        return n
    root = rec(np.arange(len(pts), dtype=np.int64))
    return root, nodes


def diam(n):  return float(np.linalg.norm(n.hi - n.lo))
def dist(a, b):
    g = np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))
    return float(np.linalg.norm(g))


def block_partition(root):
    """Standard H-matrix admissibility: min(diam) <= ETA * dist."""
    adm, near = [], []
    def rec(s, t):
        if s.nid > t.nid:   # symmetric: keep the upper half only
            return
        if min(diam(s), diam(t)) <= ETA * dist(s, t):
            adm.append((s, t)); return
        if not s.kids and not t.kids:
            near.append((s, t)); return
        ss = s.kids if s.kids else [s]
        tt = t.kids if t.kids else [t]
        for a in ss:
            for b in tt:
                rec(a, b) if a.nid <= b.nid else rec(b, a)
    rec(root, root)
    return adm, near


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
    A = R.T @ R                                  # the teacher label, exactly
    print('seat 328  d %d   ||A|| %.3e   dense %.3e entries' % (d, float(A.norm()), d*(d+1)/2), flush=True)

    root, nodes = build_tree(pts)
    adm, near = block_partition(root)
    print('tree: %d nodes, leaf %d | partition: %d admissible + %d near-field (eta=%.1f)'
          % (len(nodes), LEAF, len(adm), len(near), ETA), flush=True)

    near_params = 0
    for s, t in near:
        near_params += len(s.idx)*(len(s.idx)+1)//2 if s.nid == t.nid else len(s.idx)*len(t.idx)

    # one SVD pass per admissible block, kept at the tightest tolerance
    tmin = min(TOLS); store = []; capped = 0
    for s, t in adm:
        I = torch.as_tensor(s.idx, device=V.DEV); J = torch.as_tensor(t.idx, device=V.DEV)
        M = A[I][:, J]
        mn = min(M.shape)
        q = min(mn, RANK_CAP)
        if mn <= 2*RANK_CAP:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        else:
            U, S, Vt = torch.svd_lowrank(M, q=min(mn, q + 24)); Vh = Vt.T
        nrm = float(M.norm())
        tail = torch.sqrt(torch.clamp(torch.flip(torch.cumsum(torch.flip(S, [0])**2, 0), [0]), min=0))
        k = int(torch.searchsorted(-tail, torch.tensor(-tmin*nrm, device=V.DEV)).item()) if nrm > 0 else 0
        k = max(1, min(k + 1, q))
        if k >= q: capped += 1
        store.append((I, J, U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k].contiguous(), nrm, tail[:k].contiguous()))
        del M, U, S, Vh
    torch.cuda.empty_cache()
    print('svd pass done (%.0f s), %d/%d blocks hit the rank cap %d\n' % (time.time()-t0, capped, len(adm), RANK_CAP), flush=True)

    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    dense_params = d*(d+1)//2
    print('%-8s %-12s %-11s %-11s %-11s %-9s' % ('tol', 'params', 'frac dense', 'eps_op', 'D/d', 'sec'), flush=True)
    rows = []
    first = True
    for tol in (0.0,) + TOLS:          # tol 0 keeps every stored rank: a self-test that must return ~0
        ts = time.time()
        Ah = torch.zeros(d, d, dtype=torch.float64, device=V.DEV)
        cover = torch.zeros(d, d, dtype=torch.uint8, device=V.DEV) if first else None
        for s, t in near:
            I = torch.as_tensor(s.idx, device=V.DEV); J = torch.as_tensor(t.idx, device=V.DEV)
            M = A[I][:, J]
            Ah[I.unsqueeze(1), J.unsqueeze(0)] = M
            if s.nid != t.nid:
                Ah[J.unsqueeze(1), I.unsqueeze(0)] = M.T
            cover[I.unsqueeze(1), J.unsqueeze(0)] += 1
            if s.nid != t.nid:
                cover[J.unsqueeze(1), I.unsqueeze(0)] += 1
        lr_params = 0
        for I, J, U, S, Vh, nrm, tail in store:
            if tol == 0.0:
                k = len(S)
            else:
                k = int(torch.searchsorted(-tail, torch.tensor(-tol*nrm, device=V.DEV)).item()) if nrm > 0 else 0
                k = max(1, min(k + 1, len(S)))
            M = (U[:, :k] * S[:k]) @ Vh[:k]
            Ah[I.unsqueeze(1), J.unsqueeze(0)] = M
            if not torch.equal(I, J):
                Ah[J.unsqueeze(1), I.unsqueeze(0)] = M.T
            cover[I.unsqueeze(1), J.unsqueeze(0)] += 1
            if not torch.equal(I, J):
                cover[J.unsqueeze(1), I.unsqueeze(0)] += 1
            lr_params += k * (len(I) + len(J))
        Ah = 0.5 * (Ah + Ah.T)            # blocks are already transposes of each other; this is a no-op guard
        if first:
            miss = int((cover == 0).sum()); dup = int((cover > 1).sum())
            print('COVERAGE: %d entries uncovered, %d entries written more than once' % (miss, dup), flush=True)
            if miss:
                raise SystemExit('PARTITION_DOES_NOT_COVER')
        H = Rinv.T @ Ah @ Rinv
        del Ah; torch.cuda.empty_cache()
        H = 0.5*(H + H.T)
        mu = torch.linalg.eigvalsh(H)
        del H; torch.cuda.empty_cache()
        neg = int((mu <= 0).sum())
        eps_op = float((mu - 1).abs().max())
        Dd = float('inf') if neg else float((mu - torch.log(mu) - 1).sum()) / d
        params = near_params + lr_params
        mark = 'PASSES +-3%' if Dd <= NEC3 else ('passes +-10% nec' if Dd <= NEC10 else '')
        if neg: mark = '%d non-positive mu' % neg
        print('%-8.0e %-12d %-11.4f %-11.4e %-11.4e %-9.0f %s'
              % (tol, params, params/dense_params, eps_op, Dd, time.time()-ts, mark), flush=True)
        if first:
            print('  ^ self-test row (tol=0, full stored rank): eps_op must be ~1e-10, not %s' % ('ok' if eps_op < 1e-6 else 'BROKEN'), flush=True)
            del cover; torch.cuda.empty_cache()
        first = False
        rows.append(dict(tol=tol, params=params, near_params=near_params, lr_params=lr_params,
                         frac_of_dense=params/dense_params, eps_op=eps_op,
                         divergence_per_d=None if neg else Dd, nonpositive_mu=neg))
    Path('/root/autodl-tmp/NEURAL_SCHUR/HMAT_ECONOMY2.json').write_text(json.dumps(
        dict(seat=328, d=int(d), dense_params=int(dense_params), leaf=LEAF, eta=ETA,
             n_admissible=len(adm), n_near=len(near), near_params=int(near_params),
             nec3=NEC3, nec10=NEC10, rows=rows, seconds=time.time()-t0), indent=1))
    print('\nwritten HMAT_ECONOMY2.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
