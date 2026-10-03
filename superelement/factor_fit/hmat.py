#!/usr/bin/env python3
"""Hierarchical low-rank parameter economy -- rewritten, with the invariants asserted.

Two earlier versions of this script were wrong in ways that produced confident numbers:
  v1  filled one orientation per block and then called triu.  The block index sets come from a
      spatial bisection, so they are scattered, not contiguous ranges: triu discarded whichever
      half of each off-diagonal block fell below the diagonal.  Symptom, and the reason it was
      caught: eps_op sat at ~778 across four decades of tolerance instead of decreasing.
  v2  fixed that but (a) double-counted parameters, because the recursion visits each
      off-diagonal node pair twice, and (b) had a "self-test" that could not pass, because the
      SVD store only keeps the rank needed at the tightest tolerance.

So this version asserts three things before any number is read:
  EXACT   fill every block straight from A -> eps_op must be < 1e-9.  This is the real test of
          partition coverage and block orientation, and it needs no SVD store.
  COVER   every one of the d^2 entries written exactly once.
  DEDUP   each unordered node pair appears once in the block list.

Each row is an ACHIEVED operator with a real full-spectrum audit, so a pass is a pass and a
failure rejects this construction at that tolerance -- not the hierarchical class.  See
docs/CLAIM_SCOPE_20260916.md.
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
    nodes = []
    def rec(idx):
        n = Node(idx, pts, len(nodes)); nodes.append(n)
        if len(idx) > LEAF:
            ax = int(np.argmax(n.hi - n.lo))
            o = idx[np.argsort(pts[idx, ax], kind='stable')]
            h = len(o) // 2
            n.kids = [rec(o[:h]), rec(o[h:])]
        return n
    return rec(np.arange(len(pts), dtype=np.int64)), nodes


def diam(n):  return float(np.linalg.norm(n.hi - n.lo))
def dist(a, b):
    g = np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))
    return float(np.linalg.norm(g))


def block_partition(root):
    """Admissibility min(diam) <= ETA*dist.  `seen` dedupes: the recursion reaches each
    off-diagonal node pair from both orderings."""
    adm, near, seen = [], [], set()
    def rec(s, t):
        key = (s.nid, t.nid) if s.nid <= t.nid else (t.nid, s.nid)
        if key in seen:
            return
        seen.add(key)
        if min(diam(s), diam(t)) <= ETA * dist(s, t):
            adm.append((s, t)); return
        if not s.kids and not t.kids:
            near.append((s, t)); return
        for a in (s.kids or [s]):
            for b in (t.kids or [t]):
                rec(a, b)
    rec(root, root)
    return adm, near


def place(Ah, cover, s, t, I, J, M):
    """Write a block and its transpose.  Diagonal blocks are written once."""
    Ah[I.unsqueeze(1), J.unsqueeze(0)] = M
    if cover is not None:
        cover[I.unsqueeze(1), J.unsqueeze(0)] += 1
    if s.nid != t.nid:
        Ah[J.unsqueeze(1), I.unsqueeze(0)] = M.T
        if cover is not None:
            cover[J.unsqueeze(1), I.unsqueeze(0)] += 1


def spectrum(Ah, Rinv, d):
    H = Rinv.T @ Ah @ Rinv
    H = 0.5 * (H + H.T)
    mu = torch.linalg.eigvalsh(H)
    del H; torch.cuda.empty_cache()
    neg = int((mu <= 0).sum())
    eps = float((mu - 1).abs().max())
    div = float('inf') if neg else float((mu - torch.log(mu) - 1).sum()) / d
    return eps, div, neg


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
    A = R.T @ R
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    dense_params = d * (d + 1) // 2
    print('seat 328  d %d   dense %.3e independent entries' % (d, dense_params), flush=True)

    root, nodes = build_tree(pts)
    adm, near = block_partition(root)
    idxs = {}
    for s, t in adm + near:
        for n in (s, t):
            if n.nid not in idxs:
                idxs[n.nid] = torch.as_tensor(n.idx, device=V.DEV)
    print('tree %d nodes (leaf %d) | %d admissible + %d near-field (eta %.1f)'
          % (len(nodes), LEAF, len(adm), len(near), ETA), flush=True)

    near_params = sum(len(s.idx)*(len(s.idx)+1)//2 if s.nid == t.nid else len(s.idx)*len(t.idx)
                      for s, t in near)

    # ---- EXACT: coverage + orientation, with no SVD in the way -------------------------------
    Ah = torch.zeros(d, d, dtype=torch.float64, device=V.DEV)
    cover = torch.zeros(d, d, dtype=torch.uint8, device=V.DEV)
    for s, t in adm + near:
        I, J = idxs[s.nid], idxs[t.nid]
        place(Ah, cover, s, t, I, J, A[I][:, J])
    miss, dup = int((cover == 0).sum()), int((cover > 1).sum())
    eps, div, neg = spectrum(Ah, Rinv, d)
    print('\nEXACT fill: %d uncovered, %d written twice, eps_op %.3e  ->  %s'
          % (miss, dup, eps, 'OK' if (miss == 0 and dup == 0 and eps < 1e-9) else 'BROKEN'), flush=True)
    assert miss == 0 and dup == 0 and eps < 1e-9, 'ASSEMBLY_SELF_TEST_FAILED'
    del cover, Ah; torch.cuda.empty_cache()

    # ---- one SVD pass, kept at the tightest tolerance ----------------------------------------
    tmin = min(TOLS); store = []; capped = 0
    for s, t in adm:
        I, J = idxs[s.nid], idxs[t.nid]
        M = A[I][:, J]; mn = min(M.shape); q = min(mn, RANK_CAP)
        if mn <= 2 * RANK_CAP:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        else:
            U, S, Vt = torch.svd_lowrank(M, q=min(mn, q + 24)); Vh = Vt.T
        nrm = float(M.norm())
        tail = torch.sqrt(torch.clamp(torch.flip(torch.cumsum(torch.flip(S, [0])**2, 0), [0]), min=0))
        k = int(torch.searchsorted(-tail, torch.tensor(-tmin*nrm, device=V.DEV)).item()) if nrm > 0 else 0
        k = max(1, min(k + 1, q))
        capped += k >= q
        store.append((s, t, U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k].contiguous(), nrm,
                      tail[:k].contiguous()))
        del M, U, S, Vh
    torch.cuda.empty_cache()
    print('svd pass %.0f s, %d/%d blocks at the rank cap %d\n' % (time.time()-t0, capped, len(adm), RANK_CAP), flush=True)

    print('%-8s %-12s %-11s %-12s %-12s %-6s %s'
          % ('tol', 'params', 'frac dense', 'eps_op', 'D/d', 'sec', 'verdict'), flush=True)
    rows = []
    for tol in TOLS:
        ts = time.time()
        Ah = torch.zeros(d, d, dtype=torch.float64, device=V.DEV)
        for s, t in near:
            I, J = idxs[s.nid], idxs[t.nid]
            place(Ah, None, s, t, I, J, A[I][:, J])
        lr_params = 0
        for s, t, U, S, Vh, nrm, tail in store:
            k = int(torch.searchsorted(-tail, torch.tensor(-tol*nrm, device=V.DEV)).item()) if nrm > 0 else 0
            k = max(1, min(k + 1, len(S)))
            I, J = idxs[s.nid], idxs[t.nid]
            place(Ah, None, s, t, I, J, (U[:, :k] * S[:k]) @ Vh[:k])
            lr_params += k * (len(I) + len(J)) if s.nid != t.nid else k * len(I)
        eps, div, neg = spectrum(Ah, Rinv, d)
        del Ah; torch.cuda.empty_cache()
        params = near_params + lr_params
        if neg:
            v = '%d non-positive mu' % neg
        elif eps <= 0.03:
            v = 'PASSES +-3%'
        elif eps <= 0.10:
            v = 'PASSES +-10%'
        else:
            v = 'fails (nec. cond. %s)' % ('met' if div <= NEC3 else 'met@10%' if div <= NEC10 else 'also fails')
        print('%-8.0e %-12d %-11.4f %-12.4e %-12.4e %-6.0f %s'
              % (tol, params, params/dense_params, eps, div, time.time()-ts, v), flush=True)
        rows.append(dict(tol=tol, params=params, near_params=near_params, lr_params=lr_params,
                         frac_of_dense=params/dense_params, eps_op=eps,
                         divergence_per_d=None if neg else div, nonpositive_mu=neg))
    Path('/root/autodl-tmp/NEURAL_SCHUR/HMAT_ECONOMY3.json').write_text(json.dumps(
        dict(seat=328, d=int(d), dense_params=int(dense_params), leaf=LEAF, eta=ETA,
             n_admissible=len(adm), n_near=len(near), near_params=int(near_params),
             nec3=NEC3, nec10=NEC10, rows=rows, seconds=time.time()-t0), indent=1))
    print('\nwritten HMAT_ECONOMY3.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
