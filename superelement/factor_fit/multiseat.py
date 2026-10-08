#!/usr/bin/env python3
"""Does "+-3% at ~15% of the dense entry count" hold across geometries, or was 0328 lucky?

Same hierarchical low-rank construction of the FACTOR R* that reached +-3% at 12.23e6 parameters
(14.95% of dense) on seat 0328, run on several seats spanning a range of d.  Each row is an
achieved operator with a full-spectrum audit, and every seat asserts the same three invariants
before any number is read (partition covers every entry exactly once; filling every block straight
from R* reproduces it to ~1e-13).

Memory note: at these d a dense d x d float64 buffer is 1.3-2.3 GB, so the near-field blocks are
extracted into small tensors and the full target is freed before the sweep.  Nothing else changes.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

LEAF, ETA, RANK_CAP = 64, 2.0, 768
TOLS = (3e-2, 1e-2, 3e-3, 1e-3)
SEATS = [int(s) for s in sys.argv[1:]] or [328]


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


def diam(n): return float(np.linalg.norm(n.hi - n.lo))
def dist(a, b):
    g = np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))
    return float(np.linalg.norm(g))


def block_partition(root):
    adm, near, seen = [], [], set()
    def rec(s, t):
        key = (s.nid, t.nid) if s.nid <= t.nid else (t.nid, s.nid)
        if key in seen: return
        seen.add(key)
        if min(diam(s), diam(t)) <= ETA * dist(s, t):
            adm.append((s, t)); return
        if not s.kids and not t.kids:
            near.append((s, t)); return
        for a in (s.kids or [s]):
            for b in (t.kids or [t]):
                rec(a, b)
    rec(root, root)
    # R* is not symmetric: every ORDERED pair is its own block
    return ([(s, t) for s, t in adm] + [(t, s) for s, t in adm if s.nid != t.nid],
            [(s, t) for s, t in near] + [(t, s) for s, t in near if s.nid != t.nid])


def run_seat(seat, out):
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == seat][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_%04d' % seat)
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
    order = g['data'].quotient.order.cpu().numpy(); posn = order[6:]
    pts = label.ijk[posn // 3].astype(float) / 64.0
    dense = d * (d + 1) // 2
    print('\n================  seat %04d   d = %d   dense = %.3e  ================' % (seat, d, dense), flush=True)

    root, nodes = build_tree(pts)
    adm, near = block_partition(root)
    idxs = {}
    for s, t in adm + near:
        for n in (s, t):
            idxs.setdefault(n.nid, torch.as_tensor(n.idx, device=V.DEV))
    print('tree %d nodes | %d admissible + %d near-field ordered blocks' % (len(nodes), len(adm), len(near)), flush=True)

    # invariants, then keep the near field as small tensors and free the dense target
    cover = torch.zeros(d, d, dtype=torch.uint8, device=V.DEV)
    near_store, near_params = [], 0
    for s, t in near:
        I, J = idxs[s.nid], idxs[t.nid]
        M = R[I][:, J].contiguous()
        near_store.append((I, J, M))
        near_params += int((M != 0).sum())
        cover[I.unsqueeze(1), J.unsqueeze(0)] += 1
    Rhat = torch.zeros(d, d, dtype=torch.float64, device=V.DEV)
    for I, J, M in near_store:
        Rhat[I.unsqueeze(1), J.unsqueeze(0)] = M
    for s, t in adm:
        I, J = idxs[s.nid], idxs[t.nid]
        Rhat[I.unsqueeze(1), J.unsqueeze(0)] = R[I][:, J]
        cover[I.unsqueeze(1), J.unsqueeze(0)] += 1
    miss, dup = int((cover == 0).sum()), int((cover > 1).sum())
    del cover; torch.cuda.empty_cache()
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    X = Rhat @ Rinv; H = X.T @ X; del X
    mu = torch.linalg.eigvalsh(0.5 * (H + H.T)); del H, Rhat; torch.cuda.empty_cache()
    e0 = float((mu - 1).abs().max()); del mu
    print('EXACT fill: %d uncovered, %d duplicated, eps_op %.3e -> %s'
          % (miss, dup, e0, 'OK' if (miss == 0 and dup == 0 and e0 < 1e-9) else 'BROKEN'), flush=True)
    assert miss == 0 and dup == 0 and e0 < 1e-9, 'ASSEMBLY_SELF_TEST_FAILED seat %d' % seat

    tmin = min(TOLS); store = []; zero = 0; capped = 0
    for s, t in adm:
        I, J = idxs[s.nid], idxs[t.nid]
        M = R[I][:, J]
        if float(M.abs().max()) == 0.0:
            zero += 1; del M; continue
        mn = min(M.shape); q = min(mn, RANK_CAP)
        if mn <= 2 * RANK_CAP:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        else:
            U, S, Vt = torch.svd_lowrank(M, q=min(mn, q + 24)); Vh = Vt.T
        nrm = float(M.norm())
        tail = torch.sqrt(torch.clamp(torch.flip(torch.cumsum(torch.flip(S, [0])**2, 0), [0]), min=0))
        k = int(torch.searchsorted(-tail, torch.tensor(-tmin*nrm, device=V.DEV)).item()) if nrm > 0 else 0
        k = max(1, min(k + 1, q)); capped += k >= q
        store.append((I, J, U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k].contiguous(), nrm, tail[:k].contiguous()))
        del M, U, S, Vh
    del R; torch.cuda.empty_cache()
    print('svd pass %.0f s | %d zero blocks | %d at the rank cap %d\n' % (time.time()-t0, zero, capped, RANK_CAP), flush=True)

    print('%-8s %-12s %-10s %-11s %-7s %-7s %s' % ('tol', 'params', 'fracdense', 'eps_op', 'n>3%', 'n>10%', 'verdict'), flush=True)
    rows = []
    for tol in TOLS:
        Rhat = torch.zeros(d, d, dtype=torch.float64, device=V.DEV)
        for I, J, M in near_store:
            Rhat[I.unsqueeze(1), J.unsqueeze(0)] = M
        lr = 0
        for I, J, U, S, Vh, nrm, tail in store:
            k = int(torch.searchsorted(-tail, torch.tensor(-tol*nrm, device=V.DEV)).item()) if nrm > 0 else 0
            k = max(1, min(k + 1, len(S)))
            Rhat[I.unsqueeze(1), J.unsqueeze(0)] = (U[:, :k] * S[:k]) @ Vh[:k]
            lr += k * (len(I) + len(J))
        X = Rhat @ Rinv; del Rhat; torch.cuda.empty_cache()
        H = X.T @ X; del X; torch.cuda.empty_cache()
        mu = torch.linalg.eigvalsh(0.5 * (H + H.T)); del H; torch.cuda.empty_cache()
        eps = float((mu - 1).abs().max())
        o3 = int(((mu - 1).abs() > 0.03).sum()); o10 = int(((mu - 1).abs() > 0.10).sum())
        del mu; torch.cuda.empty_cache()
        p = near_params + lr
        v = 'PASSES +-3%' if eps <= 0.03 else 'PASSES +-10%' if eps <= 0.10 else 'fails'
        print('%-8.0e %-12d %-10.4f %-11.4e %-7d %-7d %s' % (tol, p, p/dense, eps, o3, o10, v), flush=True)
        rows.append(dict(tol=tol, params=p, frac_of_dense=p/dense, eps_op=eps,
                         n_outside_3pct=o3, n_outside_10pct=o10))
    best3 = min((r for r in rows if r['eps_op'] <= 0.03), key=lambda r: r['params'], default=None)
    best10 = min((r for r in rows if r['eps_op'] <= 0.10), key=lambda r: r['params'], default=None)
    print('  -> +-3%%  %s   |   +-10%%  %s'
          % ('%d params (%.2f%% of dense)' % (best3['params'], 100*best3['frac_of_dense']) if best3 else 'not reached in this tolerance range',
             '%d params (%.2f%% of dense)' % (best10['params'], 100*best10['frac_of_dense']) if best10 else 'not reached'), flush=True)
    out[str(seat)] = dict(d=int(d), dense_params=int(dense), near_params=int(near_params),
                          n_admissible=len(adm), n_near=len(near), zero_blocks=zero,
                          rows=rows, at_3pct=best3, at_10pct=best10, seconds=time.time()-t0)
    del store, near_store, Rinv, idxs; torch.cuda.empty_cache()


def main():
    out = {}
    for seat in SEATS:
        try:
            run_seat(seat, out)
        except torch.cuda.OutOfMemoryError as e:
            print('\nseat %04d: OUT OF MEMORY, skipped (%s)' % (seat, str(e)[:120]), flush=True)
            out[str(seat)] = dict(error='oom')
            torch.cuda.empty_cache()
        Path('/root/autodl-tmp/NEURAL_SCHUR/MULTISEAT.json').write_text(json.dumps(out, indent=1))
    print('\n\n================  summary  ================', flush=True)
    print('%-8s %-8s %-14s %-14s' % ('seat', 'd', '+-10%', '+-3%'), flush=True)
    for s, r in out.items():
        if r.get('error'):
            print('%-8s %-8s %s' % (s, '-', r['error'])); continue
        f10 = '%.2f%% (%.2fe6)' % (100*r['at_10pct']['frac_of_dense'], r['at_10pct']['params']/1e6) if r['at_10pct'] else '-'
        f3 = '%.2f%% (%.2fe6)' % (100*r['at_3pct']['frac_of_dense'], r['at_3pct']['params']/1e6) if r['at_3pct'] else '-'
        print('%-8s %-8d %-14s %-14s' % (s, r['d'], f10, f3), flush=True)
    print('\nwritten MULTISEAT.json', flush=True)


if __name__ == '__main__':
    main()
