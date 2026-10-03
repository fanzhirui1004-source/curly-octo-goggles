#!/usr/bin/env python3
"""Score a checkpoint on A_hat directly, with no rigid solve anywhere.

The training objective and the `divergence_per_mode` reported at evaluation both come from

    logdet(B K0 B^T) = logdet(K0) + logdet(N^T K0^-1 N) - logdet(N^T N)

which inverts the auxiliary matrix K0 = L^T (I + M M^T)^-1 L.  R7 drove cond(K0) to about 1e52
and sigma_min(L) to 1.9e-26, so that route cannot be assumed accurate.  Nothing about the
supervised object requires K0 to be near-singular: A_hat = B K0 B^T is unchanged by adding
alpha U U^T to K0 for the rigid basis U, so the rigid block of K0 is free.

This materializes A_hat and computes the divergence from its own Cholesky:

    D = tr(A^-1 A_hat) - logdet(A_hat) + logdet(A) - d

and the exact whitened spectrum mu = eig(R*^-T A_hat R*^-1).  Neither touches K0^-1.  The
comparison against the identity-based number says whether the loss was distorted, which is a
different question from whether K0 is in a dangerous numerical state.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--seat', type=int, default=328)
    ap.add_argument('--rank', type=int, default=1500)
    ap.add_argument('--r-near', type=float, default=0.2)
    ap.add_argument('--decay', type=float, default=0.03)
    ap.add_argument('--off-scale', type=float, default=10.0)
    ap.add_argument('--labels', default='/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.time(); DEV = V1.DEV

    recs = json.load(open(a.labels))
    rec = [r for r in recs if int(r['seat']) == a.seat][0]
    label = V1.Label(rec, a.r_near, a.decay, a.off_scale)
    g = label.to_gpu(need_A=True, need_Z=True, z_dtype=torch.float64)
    net = V1.FreeScaledModel(label, rank=a.rank).to(DEV)
    ck = torch.load(a.checkpoint, map_location=DEV)
    net.load_state_dict(ck['net']); step = ck.get('step', -1)
    print('seat %d  step %d  d %d  q %d  (%.0f s)' % (a.seat, step, label.d, label.q, time.time()-t0), flush=True)

    with torch.no_grad():
        values, M = net(g, g['w'].log()); op = V1.Operator(g, values, M)
        D_id, trace_id, gap_id, resid = V1.divergence_terms(g, op, trace_dtype=torch.float64, refine=4)
        print('identity route : D/d %.6f  trace/d %.6f  gap %.3f  rigid residual %.3e'
              % (float(D_id)/label.d, float(trace_id)/label.d, float(gap_id), resid), flush=True)

        K0 = op.materialize(); Ahat, _ = V1.quotient_dense(K0, g['data'].quotient); del K0
        torch.cuda.empty_cache()
        A, R = g['A'], g['Rstar']
        Lc, info = torch.linalg.cholesky_ex(Ahat, upper=True)
        print('A_hat Cholesky info %d  (%.0f s)' % (int(info), time.time()-t0), flush=True)
        logdet_Ahat = 2.0 * Lc.diagonal().log().sum(); del Lc
        # W = R*^-T A_hat R*^-1 ; trace(A^-1 A_hat) = trace(W)
        W = torch.linalg.solve_triangular(R.T, Ahat, upper=False)
        W = torch.linalg.solve_triangular(R.T, W.T, upper=False).T
        trace_dir = W.diagonal().sum()
        logdet_A = 2.0 * R.diagonal().log().sum()
        D_dir = trace_dir - (logdet_Ahat - logdet_A) - label.d
        print('direct route   : D/d %.6f  trace/d %.6f  logdet gap %.3f'
              % (float(D_dir)/label.d, float(trace_dir)/label.d, float(logdet_Ahat - logdet_A)), flush=True)

        W = 0.5 * (W + W.T)
        mu = torch.linalg.eigvalsh(W); del W
        mu = mu.cpu().numpy()
        D_spec = float(np.sum(mu - np.log(np.maximum(mu, 1e-300)) - 1.0))
        print('spectrum       : D/d %.6f  mu in [%.6f, %.6f]' % (D_spec/label.d, mu.min(), mu.max()), flush=True)
        gate10 = int(((mu < 0.9) | (mu > 1.1)).sum()); gate3 = int(((mu < 0.97) | (mu > 1.03)).sum())
        err = float(torch.linalg.vector_norm(Ahat - A) / torch.linalg.vector_norm(A))
        print('work gate +-10%%: %d of %d outside (%.4f%%)' % (gate10, label.d, 100*gate10/label.d), flush=True)
        print('target   +-3%% : %d of %d outside (%.4f%%)' % (gate3, label.d, 100*gate3/label.d), flush=True)
        print('||A_hat - A||/||A|| = %.6f' % err, flush=True)

    out = dict(schema='AHAT_AUDIT_V1', seat=a.seat, step=int(step), d=int(label.d), q=int(label.q),
               identity=dict(D_per_mode=float(D_id)/label.d, trace_per_mode=float(trace_id)/label.d,
                             logdet_gap=float(gap_id), rigid_residual=float(resid)),
               direct=dict(D_per_mode=float(D_dir)/label.d, trace_per_mode=float(trace_dir)/label.d),
               spectrum=dict(D_per_mode=D_spec/label.d, mu_min=float(mu.min()), mu_max=float(mu.max()),
                             outside_10pct=gate10, outside_3pct=gate3,
                             q01=float(np.quantile(mu, .01)), q50=float(np.quantile(mu, .5)), q99=float(np.quantile(mu, .99))),
               schur_relative_error=err, seconds=time.time()-t0)
    Path(a.out).write_text(json.dumps(out, indent=1))
    np.save(Path(a.out).with_suffix('.mu.npy'), mu)
    print('written', a.out, '(%.0f s)' % out['seconds'], flush=True)


if __name__ == '__main__': main()
