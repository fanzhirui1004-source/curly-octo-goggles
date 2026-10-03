#!/usr/bin/env python3
"""Route 3 (A^{1/2}) - the two numbers nobody has ever measured.

FIRST_PRINCIPLES judged the H^2 square-root route "conditionally alive" and listed two
untested premises (S 6, path 1):
  - "A^{1/2} 的 H2 可压缩性从没测过（我们测的是 Cholesky 因子）"
  - "每胞 ~1e5 个光滑耦合数要 1e-4 相对精度"
Its main objection was the coherence requirement of 3e-6..3e-5, which
WHERE_THE_ERROR_IS measured today at 2.5x, not 1e3..1e4 - so the objection is gone and
these two premises are what is left.

A) COMPRESSIBILITY. Apply the same hierarchical partition that compresses the Cholesky
   factor to 12,229,217 numbers at eps_op 2.597e-2, to A^{1/2} instead. Same tree, same
   admissibility, same per-block relative-Frobenius SVD truncation, same seat.
B) AMPLIFICATION. Perturb A^{1/2} entrywise by relative eps and measure eps_op(eps).
   The Cholesky factor gives eps_op = 104 * eps.

Self-tests: (i) ||A^{1/2} A^{1/2} - A|| / ||A|| must be ~1e-15; (ii) filling every block
exactly (no truncation) must return eps_op ~1e-13; (iii) eps = 0 must return ~1e-13.
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
from hmat_factor_lib import build_tree, block_partition
DEV = V.DEV; F64 = torch.float64
OUT = Path('/root/autodl-tmp/CLAUDE_SQRT_20260917'); OUT.mkdir(exist_ok=True)


def eps_op(Ahat, Rs, d):
    C = torch.linalg.solve_triangular(Rs.T, Ahat, upper=False)
    C = torch.linalg.solve_triangular(Rs.T, C.T.contiguous(), upper=False)
    mu = torch.linalg.eigvalsh(0.5 * (C + C.T)); del C
    mn, mx = float(mu.min()), float(mu.max()); del mu
    gc.collect(); torch.cuda.empty_cache()
    return max(abs(mn - 1), abs(mx - 1)), mn, mx


def symmetric_partition(root):
    """Unordered blocks: a symmetric matrix stores each off-diagonal pair once."""
    adm, near, seen = [], [], set()
    from hmat_factor_lib import _diam, _dist
    def rec(s, t):
        key = (s.nid, t.nid) if s.nid <= t.nid else (t.nid, s.nid)
        if key in seen: return
        seen.add(key)
        if min(_diam(s), _diam(t)) <= 2.0 * _dist(s, t):
            adm.append((s, t)); return
        if not s.kids and not t.kids:
            near.append((s, t)); return
        for a in (s.kids or [s]):
            for b in (t.kids or [t]):
                rec(a, b)
    rec(root, root)
    return adm, near


def compress(M, adm, near, idxs, tol, d):
    """Rebuild M under the hierarchical partition; return (Mhat, numbers stored)."""
    H = torch.zeros_like(M); numbers = 0
    for s, t in near:
        I, J = idxs[s.nid], idxs[t.nid]
        blk = M[I][:, J].contiguous()
        H[I.unsqueeze(1), J.unsqueeze(0)] = blk
        if s.nid == t.nid:
            numbers += len(I) * (len(I) + 1) // 2
        else:
            H[J.unsqueeze(1), I.unsqueeze(0)] = blk.T
            numbers += blk.numel()
    for s, t in adm:
        I, J = idxs[s.nid], idxs[t.nid]
        blk = M[I][:, J].contiguous()
        if tol is None:
            H[I.unsqueeze(1), J.unsqueeze(0)] = blk
            r = min(len(I), len(J))
        else:
            U, S, Vh = torch.linalg.svd(blk, full_matrices=False)
            keep = int((torch.cumsum(S.flip(0) ** 2, 0).flip(0) > (tol * S.norm()) ** 2).sum())
            r = max(1, min(keep, 768))
            H[I.unsqueeze(1), J.unsqueeze(0)] = (U[:, :r] * S[:r]) @ Vh[:r]
        numbers += r * (len(I) + len(J))
        if s.nid != t.nid:
            H[J.unsqueeze(1), I.unsqueeze(0)] = H[I][:, J].T
        del blk
    return 0.5 * (H + H.T), numbers


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    order = g['data'].quotient.order.cpu().numpy()
    pts = label.ijk[order[6:] // 3].astype(float) / 64.0
    Rs = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, DEV)
    A = Rs.T @ Rs; A = 0.5 * (A + A.T)
    dense = d * (d + 1) // 2
    print(f'seat 328  d={d}  dense entries={dense}', flush=True)

    t = time.time()
    lam, Vv = torch.linalg.eigh(A)
    print(f'  eig lambda in [{float(lam.min()):.4e}, {float(lam.max()):.4e}]  '
          f'cond {float(lam.max()/lam.min()):.4e}  ({time.time()-t:.0f}s)', flush=True)
    M = (Vv * lam.clamp_min(0).sqrt()) @ Vv.T; M = 0.5 * (M + M.T)
    del Vv, lam; gc.collect(); torch.cuda.empty_cache()
    rel = float((M @ M - A).norm() / A.norm())
    print(f'  SELF-TEST ||A^1/2 A^1/2 - A||/||A|| = {rel:.3e} {"OK" if rel < 1e-12 else "FAIL"}',
          flush=True)
    out = dict(seat=328, d=int(d), dense_entries=int(dense), sqrt_residual=rel,
               reference=dict(cholesky_numbers=12229217, cholesky_eps_op=2.597e-2,
                              cholesky_amplification_per_entry=104.0),
               compressibility=[], amplification=[])
    if rel >= 1e-12:
        (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1)); return

    root, nodes = build_tree(pts)
    adm, near = symmetric_partition(root)
    idxs = {}
    for s, t in adm + near:
        for n in (s, t):
            idxs.setdefault(n.nid, torch.as_tensor(n.idx, device=DEV))
    print(f'  partition: {len(adm)} admissible + {len(near)} near blocks', flush=True)

    print('\nA) compressibility of A^{1/2}', flush=True)
    for tol in (None, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
        t = time.time()
        H, numbers = compress(M, adm, near, idxs, tol, d)
        v, mn, mx = eps_op(H @ H, Rs, d); del H; gc.collect(); torch.cuda.empty_cache()
        tag = 'EXACT-fill' if tol is None else f'{tol:.0e}'
        out['compressibility'].append(dict(tol=tol, numbers=int(numbers),
                                           fraction_of_dense=numbers / dense, eps_op=v,
                                           mu_min=mn, mu_max=mx, seconds=time.time() - t))
        print(f'  tol {tag:>10}  numbers {numbers:>10} ({numbers/dense:7.3%})  '
              f'eps_op {v:12.5g}  ({time.time()-t:.0f}s)', flush=True)
        if tol is None:
            ok = v < 1e-10
            print(f'    SELF-TEST exact fill: {"PASS" if ok else "FAIL"}', flush=True)
            out['exact_fill_pass'] = bool(ok)
            if not ok:
                (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1)); return
        (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1))

    print('\nB) per-entry amplification of A^{1/2}', flush=True)
    gen = torch.Generator(device=DEV).manual_seed(20260917)
    for e in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        noise = torch.randn(M.shape, generator=gen, device=DEV, dtype=F64)
        noise = 0.5 * (noise + noise.T)
        P = M * (1 + e * noise); del noise
        v, mn, mx = eps_op(P @ P, Rs, d); del P; gc.collect(); torch.cuda.empty_cache()
        amp = '' if e == 0 else f'{v/e:.1f}'
        out['amplification'].append(dict(entry_relative_noise=e, eps_op=v, mu_min=mn,
                                         mu_max=mx, amplification=(None if e == 0 else v / e)))
        print(f'  eps {e:.0e}  eps_op {v:12.5g}  amplification {amp:>10}'
              f'   (Cholesky factor: 104)', flush=True)
        (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1))
    print(f'\ndone ({time.time()-t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
