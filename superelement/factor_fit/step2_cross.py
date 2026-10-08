#!/usr/bin/env python3
"""Does the column-scale predictor transfer to a geometry it has never seen?

step2_quotient_scale showed that within one body, holding out 20% of nodes, geometry predicts
the quotient curvature well enough to flatten it from spread 2.7e3 to about 11.  That is
interpolation inside a single cut cell.  A network at prediction time faces a body it has never
seen, which is strictly harder.

This runs leave-one-seat-out over several seats.  Pass 1 caches per-seat features and both
targets (s on quotient coordinates, c on physical dofs).  Pass 2 fits on all other seats and
scores on the held-out one, reporting the spread of s_j / s-hat_j -- the curvature spread a
network would actually face on a new geometry.

Two routes are scored: regressing log s directly, and regressing log c then applying the exact
(B.^2) pullback, whose map is pure geometry and so is available at prediction time.
"""
import argparse, json, math, sys, time, gc
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v0_superelement as V0
from stage_cutfem_neural_a.dense_fit import FactorSample
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
N = 65
LAB = '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'


def node_features(ijk, w, grid, face, deg, cut_dist):
    nn = len(ijk); xyz = ijk/(N-1); g = grid.astype(np.float32)
    loc = np.zeros((nn, g.shape[0]))
    for n, (i, j, k) in enumerate(ijk):
        sl = (slice(max(i-2, 0), i+3), slice(max(j-2, 0), j+3), slice(max(k-2, 0), k+3))
        loc[n] = g[(slice(None),) + sl].reshape(g.shape[0], -1).mean(1)
    return np.concatenate([np.log(w)[:, None], xyz,
                           np.eye(7)[np.clip(face, 0, 6)], np.log1p(deg)[:, None],
                           cut_dist[:, None], np.abs(cut_dist)[:, None], loc], axis=1)


def load_rec(seat):
    L = json.load(open(LAB)); R = L if isinstance(L, list) else list(L.values())
    rec = [r for r in R if int(r['seat']) == seat][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_%04d' % seat)
    return rec


def build_cache(seat, cache_dir, threads):
    p = Path(cache_dir)/('seat%04d.npz' % seat)
    if p.exists(): print('  seat %04d cached' % seat, flush=True); return
    torch.set_num_threads(threads); t0 = time.time()
    rec = load_rec(seat); sample = FactorSample(rec, 2026091197)
    data = sample.data; ch = sample.cache; d = data.dimension
    order = data.quotient.order.cpu().numpy(); q = len(order)
    ijk = np.stack(np.unravel_index(ch['background_nodes'][ch['indices']], (N,)*3), axis=1)
    w = np.clip(V0.support_weights(ijk, N, ch['tau_corners'], ch['cut_plane']), 1e-4, 1.0)
    face = np.asarray(V0.face_ids(ijk, N)); grid = data.grid[0].numpy()
    xyz_t = torch.from_numpy(ijk/(N-1)).double()
    ni, nj = V0.pairs_within(xyz_t, xyz_t, 0.2, upper=True)
    deg = np.bincount(np.r_[ni.numpy(), nj.numpy()], minlength=len(ijk)).astype(float)
    cp = np.asarray(ch['cut_plane'], dtype=float).ravel()
    cut_dist = (ijk/(N-1)) @ cp[:3] - cp[3] if cp.size >= 4 else np.zeros(len(ijk))
    X = node_features(ijk, w, grid, face, deg, cut_dist)
    Rstar = load_upper_factor(Path(rec['reference'])/'R_UPPER.npy', d, torch.device('cpu'))
    Z = torch.linalg.solve_triangular(Rstar.T, torch.eye(d, dtype=torch.float64), upper=False)
    s = Z.square().sum(0).numpy(); del Z; gc.collect()
    B = data.quotient(torch.eye(q, dtype=torch.float64))
    Zp = torch.linalg.solve_triangular(Rstar.T, B, upper=False)
    c = Zp.square().sum(0).numpy(); del Zp, B, Rstar; gc.collect()
    np.savez_compressed(p, X=X, s=s, c=c, order=order, d=d, q=q, nodes=len(ijk))
    print('  seat %04d  nodes %d  d %d  q %d  s-spread %.2e  c-spread %.2e  (%.0f s)'
          % (seat, len(ijk), d, q, s.max()/s.min(), c.max()/c.min(), time.time()-t0), flush=True)


def B2_for(seat):
    rec = load_rec(seat); data = FactorSample(rec, 2026091197).data
    q = len(data.quotient.order)
    B = data.quotient(torch.eye(q, dtype=torch.float64)).numpy()
    return B*B


def ridge_fit(X, y, lam):
    Xm, Xs = X.mean(0), X.std(0) + 1e-9
    Xa = np.c_[(X-Xm)/Xs, np.ones(len(X))]
    beta = np.linalg.solve(Xa.T @ Xa + lam*np.eye(Xa.shape[1]), Xa.T @ y)
    return lambda Z: np.c_[(Z-Xm)/Xs, np.ones(len(Z))] @ beta


def quad(X):
    iu = np.triu_indices(X.shape[1]); return np.c_[X, (X[:, :, None]*X[:, None, :])[:, iu[0], iu[1]]]


def rows_quotient(D):
    pos = D['order'][6:]; return np.c_[D['X'][pos//3], np.eye(3)[pos % 3]], np.log(D['s']), pos


def rows_physical(D):
    q = int(D['q']); return np.c_[D['X'][np.arange(q)//3], np.eye(3)[np.arange(q) % 3]], np.log(D['c'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seats', default='328,253,403,347,974,575')
    ap.add_argument('--cache-dir', default='/root/autodl-tmp/STEP2_BRIDGE/cache')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--stage', default='both', choices=['cache', 'fit', 'both'])
    ap.add_argument('--out', default='/root/autodl-tmp/STEP2_BRIDGE/cross.json')
    a = ap.parse_args()
    seats = [int(x) for x in a.seats.split(',')]
    Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)

    if a.stage in ('cache', 'both'):
        print('pass 1: caching %d seats' % len(seats), flush=True)
        for s in seats: build_cache(s, a.cache_dir, a.threads)
    if a.stage == 'cache': return

    D = {s: dict(np.load(Path(a.cache_dir)/('seat%04d.npz' % s))) for s in seats}
    print('\npass 2: leave-one-seat-out over %s' % seats, flush=True)
    out = dict(schema='STEP2_CROSS_V1', seats=seats, results={})
    for held in seats:
        tr = [s for s in seats if s != held]
        row = dict(train_seats=tr)
        Dh = D[held]
        # route A: regress log s on quotient rows
        Xtr = np.vstack([rows_quotient(D[s])[0] for s in tr]); ytr = np.concatenate([rows_quotient(D[s])[1] for s in tr])
        Xte, yte, _ = rows_quotient(Dh)
        for nm, f, lam in (('linear', lambda Z: Z, 1e-3), ('quadratic', quad, 1e-2)):
            pred = ridge_fit(f(Xtr), ytr, lam)(f(Xte))
            r = yte - pred; sp = float(np.exp(r.max()-r.min()))
            p5, p95 = np.quantile(r, [0.05, 0.95])
            row['direct_'+nm] = dict(spread=sp, spread_5_95=float(np.exp(p95-p5)),
                                     R2=float(1-r.var()/yte.var()), rms=float(np.sqrt((r**2).mean())))
        # route B: regress log c on physical rows, then the exact pullback
        Xtr = np.vstack([rows_physical(D[s])[0] for s in tr]); ytr = np.concatenate([rows_physical(D[s])[1] for s in tr])
        Xte_p, _ = rows_physical(Dh)
        B2 = B2_for(held)
        for nm, f, lam in (('linear', lambda Z: Z, 1e-3), ('quadratic', quad, 1e-2)):
            chat = np.exp(ridge_fit(f(Xtr), ytr, lam)(f(Xte_p)))
            shat = np.log(np.maximum(B2 @ chat, 1e-300))
            r = yte - shat; sp = float(np.exp(r.max()-r.min()))
            p5, p95 = np.quantile(r, [0.05, 0.95])
            row['pullback_'+nm] = dict(spread=sp, spread_5_95=float(np.exp(p95-p5)),
                                       R2=float(1-r.var()/yte.var()), rms=float(np.sqrt((r**2).mean())))
        orc = Dh['s']/np.maximum(B2 @ Dh['c'], 1e-300)
        row['oracle_pullback_spread'] = float(orc.max()/orc.min())
        row['raw_spread'] = float(Dh['s'].max()/Dh['s'].min())
        del B2; gc.collect()
        print('  held %04d  raw %.2e | direct quad %.2e (5-95 %.2f) | pullback quad %.2e (5-95 %.2f) | oracle %.3f'
              % (held, row['raw_spread'], row['direct_quadratic']['spread'], row['direct_quadratic']['spread_5_95'],
                 row['pullback_quadratic']['spread'], row['pullback_quadratic']['spread_5_95'],
                 row['oracle_pullback_spread']), flush=True)
        out['results']['%04d' % held] = row
    Path(a.out).write_text(json.dumps(out, indent=1))
    print('\nwritten', a.out, flush=True)


if __name__ == '__main__': main()
