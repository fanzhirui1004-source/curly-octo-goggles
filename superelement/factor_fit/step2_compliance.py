#!/usr/bin/env python3
"""Step 2's ruler: can a trace dof's compliance be predicted from local geometry?

The only factor parameterization that trains scales column j by 1/sqrt(c_j), c_j = (A^-1)_jj the
compliance of dof j.  A network has no A at prediction time, so c must come from geometry.  This
regresses log c on local features and reports the number that matters: the curvature spread left
after dividing by the prediction (raw spread ~2.7e3; ~10 is enough for one learning rate).

Compliance is computed per physical trace dof, c_k = ||column k of R*^-T B||^2, so it indexes nodes
directly instead of quotient coordinates.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56'); sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v0_superelement as V0
from stage_cutfem_neural_a.dense_fit import FactorSample
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
N = 65


def features(cache, ijk, w, grid, face, deg, cut_dist):
    nn = len(ijk); xyz = ijk/(N-1)
    g = grid.astype(np.float32); loc = np.zeros((nn, g.shape[0]))
    for n, (i, j, k) in enumerate(ijk):                       # 5^3 neighbourhood mean of every grid channel
        sl = (slice(max(i-2, 0), i+3), slice(max(j-2, 0), j+3), slice(max(k-2, 0), k+3))
        loc[n] = g[(slice(None),) + sl].reshape(g.shape[0], -1).mean(1)
    F = [np.log(w)[:, None], xyz, np.eye(7)[np.clip(face, 0, 6)] if face.max() < 7 else face[:, None].astype(float),
         np.log1p(deg)[:, None], cut_dist[:, None], np.abs(cut_dist)[:, None], loc]
    return np.concatenate(F, axis=1)


def ridge(X, y, lam=1e-3):
    Xm, Xs = X.mean(0), X.std(0) + 1e-9; Xn = (X-Xm)/Xs; Xa = np.c_[Xn, np.ones(len(X))]
    beta = np.linalg.solve(Xa.T @ Xa + lam*np.eye(Xa.shape[1]), Xa.T @ y)
    return lambda Z: np.c_[(Z-Xm)/Xs, np.ones(len(Z))] @ beta


def quad(X):
    iu = np.triu_indices(X.shape[1]); return np.c_[X, (X[:, :, None]*X[:, None, :])[:, iu[0], iu[1]]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seat', type=int, default=328); ap.add_argument('--out', required=True)
    ap.add_argument('--labels', default='/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'); a = ap.parse_args()
    t0 = time.time(); dev = torch.device('cuda')
    L = json.load(open(a.labels)); rec = [r for r in (L if isinstance(L, list) else L.values()) if int(r['seat']) == a.seat][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_%04d' % a.seat)
    sample = FactorSample(rec, 2026091197); data = sample.data; cache = sample.cache
    ijk = np.stack(np.unravel_index(cache['background_nodes'][cache['indices']], (N,)*3), axis=1); nn = len(ijk); q = 3*nn; d = data.dimension
    w = np.clip(V0.support_weights(ijk, N, cache['tau_corners'], cache['cut_plane']), 1e-4, 1.0)
    face = np.asarray(V0.face_ids(ijk, N)); grid = data.grid[0].numpy()
    xyz_t = torch.from_numpy(ijk/(N-1)).to(dev).double(); ni, nj = V0.pairs_within(xyz_t, xyz_t, 0.2, upper=True)
    deg = np.bincount(np.r_[ni.cpu().numpy(), nj.cpu().numpy()], minlength=nn).astype(float)
    cp = np.asarray(cache['cut_plane'], dtype=float).ravel()
    cut_dist = (ijk/(N-1)) @ cp[:3] - cp[3] if cp.size >= 4 else np.zeros(nn)
    print('seat %d  nodes %d  q %d  d %d  features built %.1f s' % (a.seat, nn, q, d, time.time()-t0), flush=True)

    # compliance per physical trace dof: c_k = ||column k of R*^-T B||^2
    Rstar = load_upper_factor(Path(rec['reference'])/'R_UPPER.npy', d, dev)
    dq = data.cuda() if dev.type == 'cuda' else data
    B = dq.quotient(torch.eye(q, dtype=torch.float64, device=dev))                      # (d, q)
    Z = torch.linalg.solve_triangular(Rstar.T, B, upper=False); del B
    c = Z.square().sum(0).cpu().numpy(); del Z, Rstar
    c_node = c.reshape(nn, 3)
    print('compliance: min %.3e max %.3e spread %.3e   (sqrt spread %.1f)' % (c.min(), c.max(), c.max()/c.min(), math.sqrt(c.max()/c.min())), flush=True)

    Xn = features(cache, ijk, w, grid, face, deg, cut_dist)
    X = np.repeat(Xn, 3, axis=0); X = np.c_[X, np.tile(np.eye(3), (nn, 1))]; y = np.log(c)
    rng = np.random.default_rng(0); test_nodes = rng.random(nn) < 0.2; test = np.repeat(test_nodes, 3)
    out = dict(schema='STEP2_COMPLIANCE_PROXY_V1', seat=a.seat, nodes=nn, q=q, d=d, n_features=int(X.shape[1]),
               compliance=dict(min=float(c.min()), max=float(c.max()), spread=float(c.max()/c.min())), models={})
    def report(name, pred):
        r = y - pred; res_spread = float(np.exp(r[test].max() - r[test].min()))
        r2 = 1 - r[test].var()/y[test].var()
        q10, q90 = np.quantile(r[test], [0.05, 0.95]); core = float(np.exp(q90 - q10))
        m = dict(R2_test=float(r2), residual_spread_test=res_spread, residual_spread_5_95=core,
                 residual_sqrt_spread_test=math.sqrt(res_spread), rms_log_residual_test=float(np.sqrt((r[test]**2).mean())))
        print('  %-18s R2 %.4f   residual spread %.3e (5-95%%: %.2e)   as sqrt %.1f   rms log-res %.3f'
              % (name, r2, res_spread, core, math.sqrt(res_spread), m['rms_log_residual_test']), flush=True)
        return m
    print('\nmodels (test = 20%% of nodes, all 3 components held out together):', flush=True)
    out['models']['constant'] = report('constant', np.full_like(y, y[~test].mean()))
    out['models']['log_w_only'] = report('log w only', ridge(X[:, :1][~test], y[~test])(X[:, :1]))
    out['models']['ridge_linear'] = report('ridge linear', ridge(X[~test], y[~test])(X))
    Xq = quad(X); out['models']['ridge_quadratic'] = report('ridge quadratic', ridge(Xq[~test], y[~test], 1e-2)(Xq))
    # per-node neighbour baseline: how much is simply spatial smoothness?
    from scipy.spatial import cKDTree
    tree = cKDTree(ijk[~test_nodes]); _, idx = tree.query(ijk, k=4)
    ytr = np.log(c_node[~test_nodes]); knn = ytr[idx].mean(1).reshape(-1)
    out['models']['knn4_spatial'] = report('4-NN spatial', np.repeat(knn.reshape(nn, 3).mean(1), 3) if False else knn)
    out['seconds'] = time.time()-t0
    Path(a.out).write_text(json.dumps(out, indent=1)); print('\nwritten', a.out, '(%.1f s)' % out['seconds'])


if __name__ == '__main__': main()
