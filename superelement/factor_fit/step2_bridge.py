#!/usr/bin/env python3
"""Does physical dof compliance substitute for the quotient column scale?

The only factor parameterization that trains scales column j of the raw upper-triangular factor
by 1/sqrt(s_j) with s_j = (A^-1)_jj, A = B S B^T the d x d quotient operator.  A network has no
A at prediction time, so step 2 regressed instead the compliance of a PHYSICAL trace dof,
c_k = (S^+)_kk, because that is the quantity that indexes nodes and so can carry geometric
features.  Whether that substitution is legitimate has not been measured.

The quotient map is B = E_{6:} H_6 ... H_1 P, with P the stored permutation `order` and H_i six
dense Householder reflectors.  So quotient coordinate j is the image of physical dof
order[j+6], and row j of B is that coordinate vector minus a rank-6 correction lying in the span
of the reflectors.  Since S^+ annihilates rigid modes, the correction is harmless exactly to the
extent the reflectors span the rigid space.

This measures, for one seat:
  - s_j and c_k separately, and their raw spreads
  - how close rows of B are to coordinate vectors
  - the decisive number: spread of s_j / c_{order[j+6]}, which is the curvature spread that
    survives preconditioning with physical compliance instead of the true quotient diagonal.
A spread near 1 means the substitution is free.  Large means the network needs another source.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
from stage_cutfem_neural_a.dense_fit import FactorSample
from stage_cutfem_neural_a.elimination_reference import load_upper_factor


def stats(name, v, out=None):
    v = np.asarray(v, dtype=float); lo, hi = v.min(), v.max()
    qq = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99])
    print('  %-28s min %.4e  max %.4e  spread %.4e   q01 %.3e  q50 %.3e  q99 %.3e'
          % (name, lo, hi, hi/max(lo, 1e-300), qq[0], qq[2], qq[4]), flush=True)
    return dict(min=float(lo), max=float(hi), spread=float(hi/max(lo, 1e-300)),
                q01=float(qq[0]), q25=float(qq[1]), q50=float(qq[2]), q75=float(qq[3]), q99=float(qq[4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seat', type=int, default=328)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--labels', default='/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time(); dev = torch.device(a.device)

    L = json.load(open(a.labels))
    rec = [r for r in (L if isinstance(L, list) else L.values()) if int(r['seat']) == a.seat][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_%04d' % a.seat)
    sample = FactorSample(rec, 2026091197); data = sample.data
    d = data.dimension
    Q = data.quotient
    order = Q.order.cpu().numpy(); q = len(order)
    refl = Q.reflectors.cpu().numpy()
    print('seat %d  d %d  q %d  reflectors %s  device %s  threads %d  (%.1f s)'
          % (a.seat, d, q, refl.shape, dev, a.threads, time.time()-t0), flush=True)

    Rstar = load_upper_factor(Path(rec['reference'])/'R_UPPER.npy', d, dev)
    dq = data.cuda() if dev.type == 'cuda' else data

    # s_j = (A^-1)_jj = squared norm of column j of R*^-T
    Zq = torch.linalg.solve_triangular(Rstar.T, torch.eye(d, dtype=torch.float64, device=dev), upper=False)
    s = Zq.square().sum(0).cpu().numpy(); del Zq
    print('quotient diagonal done (%.1f s)' % (time.time()-t0), flush=True)

    # c_k = (S^+)_kk = squared norm of column k of R*^-T B
    B = dq.quotient(torch.eye(q, dtype=torch.float64, device=dev))            # (d, q)
    orth = float((B @ B.T - torch.eye(d, dtype=torch.float64, device=dev)).abs().max())
    Zp = torch.linalg.solve_triangular(Rstar.T, B, upper=False)
    c = Zp.square().sum(0).cpu().numpy(); del Zp, Rstar
    Bnp = B.cpu().numpy(); del B
    print('physical diagonal done, ||BB^T - I||_max %.3e (%.1f s)' % (orth, time.time()-t0), flush=True)

    out = dict(schema='STEP2_BRIDGE_V1', seat=a.seat, d=int(d), q=int(q),
               device=str(dev), B_orthogonality_defect=orth)

    print('\nraw quantities:', flush=True)
    out['s_quotient'] = stats('s_j = (A^-1)_jj', s)
    out['c_physical'] = stats('c_k = (S^+)_kk', c)

    print('\nrows of B against coordinate vectors:', flush=True)
    pos = order[6:]                                       # quotient j  <->  physical pos[j]
    peak = np.abs(Bnp[np.arange(d), pos])
    rown = (Bnp**2).sum(1)
    out['B_peak_at_order'] = stats('|B_{j,order[j+6]}|', peak)
    out['B_offpeak_energy'] = stats('||b_j||^2 - peak^2', np.maximum(rown - peak**2, 1e-300))
    argm = np.abs(Bnp).argmax(axis=1)
    agree = float((argm == pos).mean())
    print('  argmax of row j equals order[j+6] for %.4f of rows' % agree, flush=True)
    out['argmax_agrees_with_order'] = agree

    print('\nsubstitution: divide the true quotient curvature by the physical compliance', flush=True)
    ratio = s / c[pos]
    out['ratio_aligned'] = stats('s_j / c_{order[j+6]}', ratio)
    ratio_pull = s / np.maximum((Bnp**2) @ c, 1e-300)
    out['ratio_diagonal_pullback'] = stats('s_j / (B.^2 c)_j', ratio_pull)
    # index-free lower bound: the best any relabelling of c could do, by matching sorted order
    out['ratio_sorted'] = stats('sort(s) / sort(c)[6:]', np.sort(s) / np.sort(c)[6:])

    print('\nverdict:', flush=True)
    print('  raw curvature spread                    %.4e' % (s.max()/s.min()), flush=True)
    print('  after dividing by aligned c             %.4e' % out['ratio_aligned']['spread'], flush=True)
    print('  after dividing by diagonal pullback     %.4e' % out['ratio_diagonal_pullback']['spread'], flush=True)
    print('  best possible (sorted, index-free)      %.4e' % out['ratio_sorted']['spread'], flush=True)
    out['raw_spread_s'] = float(s.max()/s.min())
    out['seconds'] = time.time()-t0
    np.savez_compressed(Path(a.out).with_suffix('.npz'), s=s, c=c, pos=pos, peak=peak, row_norm2=rown)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print('\nwritten', a.out, '(%.1f s)' % out['seconds'], flush=True)


if __name__ == '__main__': main()
