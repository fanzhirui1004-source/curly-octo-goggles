#!/usr/bin/env python3
"""Can the CURRENT L pass the gate, for ANY M? A necessary condition, computed exactly.

The backend is A_hat = W^T (I + M M^T)^-1 W with W = L pi(B^T), q x d.  Since
(I + M M^T)^-1 = I - M (I + M^T M)^-1 M^T, we have

    A_hat = W^T W  -  (positive semidefinite, rank <= k),     k = columns of M

so in teacher-whitened coordinates H = R*^-T A_hat R*^-1 obeys

    H = H_0 - P,     H_0 = R*^-T W^T W R*^-1,     P >= 0,   rank(P) <= k.

The low-rank term can only SOFTEN.  Two necessary conditions follow, both exact:

  (a) H <= H_0, so lam_min(H) <= lam_min(H_0).  Passing needs lam_min(H_0) >= 1 - tau.
  (b) if H_0 has m eigenvalues above 1 + tau, then the m-dimensional top eigenspace meets
      ker(P) (dimension >= d - k) in dimension >= m - k, and on that intersection
      x^T H x = x^T H_0 x > (1 + tau)|x|^2.  So H keeps at least m - k eigenvalues above the
      bound, and passing needs m <= k.

m > k therefore PROVES that no M whatsoever lets this L pass, which is the discriminating
direction.  The converse needs the softening to be realisable as M (I + M^T M)^-1 M^T, whose
eigenvalues are strictly below 1, and that step is not checked here.

This is a diagnostic on a frozen checkpoint: one d x d symmetric eigendecomposition.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='omit for the initial (untrained) state')
    ap.add_argument('--seat', type=int, default=328)
    ap.add_argument('--rank', type=int, default=1500)
    ap.add_argument('--r-near', type=float, default=0.2)
    ap.add_argument('--decay', type=float, default=0.03)
    ap.add_argument('--off-scale', type=float, default=10.0)
    ap.add_argument('--labels', default='/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); t0 = time.time()

    recs = json.load(open(a.labels)); rec = [r for r in recs if int(r['seat']) == a.seat][0]
    label = V.Label(rec, a.r_near, a.decay, a.off_scale)
    g = label.to_gpu(need_A=False, need_Z=True, z_dtype=torch.float32)
    net = V.FreeScaledModel(label, rank=a.rank).to(V.DEV)
    step = -1
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location=V.DEV)
        net.load_state_dict(ck['net']); step = int(ck.get('step', -1))
    d, q, k = label.d, label.q, a.rank
    print('seat %d  step %d  d %d  q %d  k(M columns) %d' % (a.seat, step, d, q, k), flush=True)

    with torch.no_grad():
        values, M = net(g, g['w'].log()); op = V.Operator(g, values, M)
        W = torch.sparse.mm(torch.sparse_coo_tensor(op.index, op.values, (q, q), is_coalesced=True),
                            op.pi(V.quotient_transpose(g)))                       # (q, d)
        R = g['Rstar']
        Vt = torch.linalg.solve_triangular(R.T, W.T, upper=False)                 # (d, q) = R*^-T W^T
        del W
        H0 = Vt @ Vt.T; del Vt                                                    # (d, d)
        H0 = 0.5 * (H0 + H0.T)
        lam = torch.linalg.eigvalsh(H0).cpu().numpy(); del H0
        torch.cuda.empty_cache()
        # the realised H, for reference
        Mm = op.M
        C = op.C
        # P = (R*^-T W^T) Mstuff ... reuse: recompute V then the correction
        W = torch.sparse.mm(torch.sparse_coo_tensor(op.index, op.values, (q, q), is_coalesced=True),
                            op.pi(V.quotient_transpose(g)))
        Y = torch.linalg.solve_triangular(C, Mm.T @ W, upper=False)               # (r, d)
        Ahat = W.T @ W - Y.T @ Y; del W, Y
        Hh = torch.linalg.solve_triangular(R.T, Ahat, upper=False)
        Hh = torch.linalg.solve_triangular(R.T, Hh.T, upper=False).T; del Ahat
        Hh = 0.5 * (Hh + Hh.T)
        mu = torch.linalg.eigvalsh(Hh).cpu().numpy(); del Hh
        torch.cuda.empty_cache()

    out = dict(schema='FIXED_L_REACHABILITY_V1', seat=a.seat, step=step, d=int(d), q=int(q), k=int(k),
               checkpoint=a.checkpoint, seconds=None)
    print('\nH_0 (unsoftened baseline, M set to zero):', flush=True)
    print('  lam_min %.6e   lam_max %.6e   spread %.3e' % (lam.min(), lam.max(), lam.max()/max(lam.min(), 1e-300)), flush=True)
    print('\nH   (realised, current M):', flush=True)
    print('  mu_min  %.6e   mu_max  %.6e' % (mu.min(), mu.max()), flush=True)
    Dv = float(np.sum(mu - np.log(np.maximum(mu, 1e-300)) - 1.0))
    print('  D/d %.6e' % (Dv/d), flush=True)
    out['H0'] = dict(min=float(lam.min()), max=float(lam.max()))
    out['H'] = dict(min=float(mu.min()), max=float(mu.max()), D_per_mode=Dv/d)

    print('\nnecessary condition, for any M with %d columns:' % k, flush=True)
    print('  tau    lam_min(H_0)>=1-tau   m=#{lam(H_0)>1+tau}   m<=k    verdict', flush=True)
    for tau in (0.03, 0.10):
        soft_ok = bool(lam.min() >= 1 - tau)
        m = int((lam > 1 + tau).sum())
        rank_ok = bool(m <= k)
        verdict = 'possible' if (soft_ok and rank_ok) else 'IMPOSSIBLE for this L'
        print('  %.2f   %-19s   %-20d %-7s %s'
              % (tau, '%s (%.4f)' % ('yes' if soft_ok else 'NO', lam.min()), m, 'yes' if rank_ok else 'NO', verdict), flush=True)
        out['tau_%g' % tau] = dict(lam_min=float(lam.min()), soft_side_ok=soft_ok,
                                   m_above=m, k=int(k), rank_side_ok=rank_ok, possible=bool(soft_ok and rank_ok))
    # how much of the spectrum is where
    for lo, hi in ((0.9, 1.1), (0.97, 1.03)):
        inside = int(((lam >= lo) & (lam <= hi)).sum())
        print('  H_0: %d of %d (%.2f%%) inside [%.2f, %.2f];  below %d, above %d'
              % (inside, d, 100*inside/d, lo, hi, int((lam < lo).sum()), int((lam > hi).sum())), flush=True)
    out['seconds'] = time.time()-t0
    Path(a.out).write_text(json.dumps(out, indent=1))
    np.save(Path(a.out).with_suffix('.H0.npy'), lam)
    np.save(Path(a.out).with_suffix('.H.npy'), mu)
    print('\nwritten %s (%.0f s)' % (a.out, out['seconds']), flush=True)


if __name__ == '__main__': main()
