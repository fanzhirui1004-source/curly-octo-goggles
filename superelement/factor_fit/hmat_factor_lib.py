#!/usr/bin/env python3
"""The hierarchical low-rank approximation of R*, as a reusable generator.

Extracted verbatim from hmat_factor.py (the version whose EXACT-fill self-test returns 7.1e-14 and
whose parameter counts are in HMAT_FACTOR.json), so the operators the assembly test consumes are
the same ones the module-level gate was measured on.  The self-test runs here too: if the partition
does not cover every entry exactly once, or filling every block straight from R* does not reproduce
it, nothing is yielded.
"""
import numpy as np, torch

LEAF, ETA, RANK_CAP = 64, 2.0, 768


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


def _diam(n): return float(np.linalg.norm(n.hi - n.lo))
def _dist(a, b):
    g = np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))
    return float(np.linalg.norm(g))


def block_partition(root):
    adm, near, seen = [], [], set()
    def rec(s, t):
        key = (s.nid, t.nid) if s.nid <= t.nid else (t.nid, s.nid)
        if key in seen: return
        seen.add(key)
        if min(_diam(s), _diam(t)) <= ETA * _dist(s, t):
            adm.append((s, t)); return
        if not s.kids and not t.kids:
            near.append((s, t)); return
        for a in (s.kids or [s]):
            for b in (t.kids or [t]):
                rec(a, b)
    rec(root, root)
    # R* is not symmetric: each ORDERED pair is its own block
    return ([(s, t) for s, t in adm] + [(t, s) for s, t in adm if s.nid != t.nid],
            [(s, t) for s, t in near] + [(t, s) for s, t in near if s.nid != t.nid])


def build_factor_approximations(label, R, dev, tols=(3e-2, 1e-2, 3e-3, 1e-3), verbose=True):
    """Yield (tol, Rhat) for each tolerance, tightest rank stored once."""
    d = R.shape[0]
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    order = g['data'].quotient.order.cpu().numpy()
    pts = label.ijk[order[6:] // 3].astype(float) / 64.0
    root, _ = build_tree(pts)
    adm, near = block_partition(root)
    idxs = {}
    for s, t in adm + near:
        for n in (s, t):
            idxs.setdefault(n.nid, torch.as_tensor(n.idx, device=dev))

    near_store, near_params = [], 0
    cover = torch.zeros(d, d, dtype=torch.uint8, device=dev)
    Rhat = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for s, t in near:
        I, J = idxs[s.nid], idxs[t.nid]
        M = R[I][:, J].contiguous()
        near_store.append((I, J, M)); near_params += int((M != 0).sum())
        Rhat[I.unsqueeze(1), J.unsqueeze(0)] = M
        cover[I.unsqueeze(1), J.unsqueeze(0)] += 1
    for s, t in adm:
        I, J = idxs[s.nid], idxs[t.nid]
        Rhat[I.unsqueeze(1), J.unsqueeze(0)] = R[I][:, J]
        cover[I.unsqueeze(1), J.unsqueeze(0)] += 1
    miss, dup = int((cover == 0).sum()), int((cover > 1).sum())
    err = float((Rhat - R).abs().max() / R.abs().max())
    del cover, Rhat; torch.cuda.empty_cache()
    if verbose:
        print('   [hmat] self-test: %d uncovered, %d duplicated, max rel entry error %.2e'
              % (miss, dup, err), flush=True)
    assert miss == 0 and dup == 0 and err < 1e-13, 'ASSEMBLY_SELF_TEST_FAILED'

    tmin = min(tols); store = []
    for s, t in adm:
        I, J = idxs[s.nid], idxs[t.nid]
        M = R[I][:, J]
        if float(M.abs().max()) == 0.0:
            del M; continue
        mn = min(M.shape); qq = min(mn, RANK_CAP)
        if mn <= 2 * RANK_CAP:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        else:
            U, S, Vt = torch.svd_lowrank(M, q=min(mn, qq + 24)); Vh = Vt.T
        nrm = float(M.norm())
        tail = torch.sqrt(torch.clamp(torch.flip(torch.cumsum(torch.flip(S, [0])**2, 0), [0]), min=0))
        k = int(torch.searchsorted(-tail, torch.tensor(-tmin*nrm, device=dev)).item()) if nrm > 0 else 0
        k = max(1, min(k + 1, qq))
        store.append((I, J, U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k].contiguous(), nrm, tail[:k].contiguous()))
        del M, U, S, Vh
    torch.cuda.empty_cache()

    for tol in tols:
        Rhat = torch.zeros(d, d, dtype=torch.float64, device=dev)
        for I, J, M in near_store:
            Rhat[I.unsqueeze(1), J.unsqueeze(0)] = M
        params = near_params
        for I, J, U, S, Vh, nrm, tail in store:
            k = int(torch.searchsorted(-tail, torch.tensor(-tol*nrm, device=dev)).item()) if nrm > 0 else 0
            k = max(1, min(k + 1, len(S)))
            Rhat[I.unsqueeze(1), J.unsqueeze(0)] = (U[:, :k] * S[:k]) @ Vh[:k]
            params += k * (len(I) + len(J))
        yield tol, Rhat, params
