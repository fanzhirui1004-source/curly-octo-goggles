#!/usr/bin/env python3
"""Regress the quotient column scale itself, not physical dof compliance.

Step 2 regressed c_k = (S^+)_kk, the compliance of physical trace dof k, on the stated ground
that compliance "indexes nodes directly instead of quotient coordinates".  That premise is
wrong.  The quotient map is B = E_{6:} H_6 ... H_1 P with P the stored permutation `order`, so
quotient coordinate j IS physical dof order[j+6]: the bijection is explicit and the same
per-node features apply.  Only the six anchor dofs order[0:6] have no partner.

The bridge measurement showed the values differ even though the indices match: dividing the
true curvature s_j = (A^-1)_jj by the aligned c leaves spread 4.1, because rows of B are
coordinate vectors only to ~1e-3 of energy and that small rigid-mode admixture lands on
directions whose compliance is a thousand times larger.  The diagonal pullback (B.^2 c) fixes
it exactly (spread 1.17), but it is a detour.

This regresses log s directly and reports the number that decides whether a network can
preconditionitself: the spread of s_j / s-hat_j on held-out nodes.  For comparison it also runs
the two-stage route, regressing c and then applying the known (B.^2) map.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v0_superelement as V0
from stage_cutfem_neural_a.dense_fit import FactorSample
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
N = 65


def features(ijk, w, grid, face, deg, cut_dist):
    nn = len(ijk); xyz = ijk/(N-1)
    g = grid.astype(np.float32); loc = np.zeros((nn, g.shape[0]))
    for n, (i, j, k) in enumerate(ijk):
        sl = (slice(max(i-2, 0), i+3), slice(max(j-2, 0), j+3), slice(max(k-2, 0), k+3))
        loc[n] = g[(slice(None),) + sl].reshape(g.shape[0], -1).mean(1)
    F = [np.log(w)[:, None], xyz,
         np.eye(7)[np.clip(face, 0, 6)] if face.max() < 7 else face[:, None].astype(float),
         np.log1p(deg)[:, None], cut_dist[:, None], np.abs(cut_dist)[:, None], loc]
    return np.concatenate(F, axis=1)


def ridge(X, y, lam=1e-3):
    Xm, Xs = X.mean(0), X.std(0) + 1e-9; Xn = (X-Xm)/Xs; Xa = np.c_[Xn, np.ones(len(X))]
    beta = np.linalg.solve(Xa.T @ Xa + lam*np.eye(Xa.shape[1]), Xa.T @ y)
    return lambda Z: np.c_[(Z-Xm)/Xs, np.ones(len(Z))] @ beta


def quad(X):
    iu = np.triu_indices(X.shape[1]); return np.c_[X, (X[:, :, None]*X[:, None, :])[:, iu[0], iu[1]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seat', type=int, default=328)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--labels', default='/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); t0 = time.time(); dev = torch.device(a.device)

    L = json.load(open(a.labels))
    rec = [r for r in (L if isinstance(L, list) else L.values()) if int(r['seat']) == a.seat][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_%04d' % a.seat)
    sample = FactorSample(rec, 2026091197); data = sample.data; cache = sample.cache
    d = data.dimension; order = data.quotient.order.cpu().numpy(); q = len(order)
    ijk = np.stack(np.unravel_index(cache['background_nodes'][cache['indices']], (N,)*3), axis=1)
    nn = len(ijk)
    print('seat %d  nodes %d  q %d  d %d' % (a.seat, nn, q, d), flush=True)

    Rstar = load_upper_factor(Path(rec['reference'])/'R_UPPER.npy', d, dev)
    Zq = torch.linalg.solve_triangular(Rstar.T, torch.eye(d, dtype=torch.float64, device=dev), upper=False)
    s = Zq.square().sum(0).cpu().numpy(); del Zq
    B = data.quotient(torch.eye(q, dtype=torch.float64, device=dev))
    Zp = torch.linalg.solve_triangular(Rstar.T, B, upper=False)
    c = Zp.square().sum(0).cpu().numpy(); del Zp, Rstar
    B2 = (B.cpu().numpy())**2; del B
    print('targets built (%.1f s)  s spread %.3e  c spread %.3e'
          % (time.time()-t0, s.max()/s.min(), c.max()/c.min()), flush=True)

    w = np.clip(V0.support_weights(ijk, N, cache['tau_corners'], cache['cut_plane']), 1e-4, 1.0)
    face = np.asarray(V0.face_ids(ijk, N)); grid = data.grid[0].numpy()
    xyz_t = torch.from_numpy(ijk/(N-1)).double()
    ni, nj = V0.pairs_within(xyz_t, xyz_t, 0.2, upper=True)
    deg = np.bincount(np.r_[ni.cpu().numpy(), nj.cpu().numpy()], minlength=nn).astype(float)
    cp = np.asarray(cache['cut_plane'], dtype=float).ravel()
    cut_dist = (ijk/(N-1)) @ cp[:3] - cp[3] if cp.size >= 4 else np.zeros(nn)
    Xn = features(ijk, w, grid, face, deg, cut_dist)
    print('features %s built (%.1f s)' % (Xn.shape, time.time()-t0), flush=True)

    # quotient coordinate j  <->  physical dof pos[j] = order[j+6]  <->  node pos[j]//3, comp pos[j]%3
    pos = order[6:]; node_of = pos // 3; comp_of = pos % 3
    Xq_rows = np.c_[Xn[node_of], np.eye(3)[comp_of]]
    rng = np.random.default_rng(0); test_nodes = rng.random(nn) < 0.2
    test = test_nodes[node_of]
    print('held out %d of %d nodes -> %d of %d quotient coordinates'
          % (test_nodes.sum(), nn, test.sum(), d), flush=True)

    out = dict(schema='STEP2_QUOTIENT_SCALE_V1', seat=a.seat, nodes=int(nn), q=int(q), d=int(d),
               n_features=int(Xq_rows.shape[1]), test_nodes=int(test_nodes.sum()),
               raw_spread_s=float(s.max()/s.min()), raw_spread_c=float(c.max()/c.min()), models={})

    def report(tag, name, y_true_log, pred_log):
        r = y_true_log - pred_log
        rt = r[test]
        full = float(np.exp(rt.max()-rt.min()))
        p5, p95 = np.quantile(rt, [0.05, 0.95]); core = float(np.exp(p95-p5))
        r2 = float(1 - rt.var()/y_true_log[test].var())
        m = dict(R2_test=r2, precond_spread_test=full, precond_spread_5_95=core,
                 scale_spread_test=math.sqrt(full), rms_log_residual_test=float(np.sqrt((rt**2).mean())))
        print('  %-20s R2 %7.4f   precond spread %.3e (5-95%%: %.2e)   as scale %.1f   rms %.3f'
              % (name, r2, full, core, math.sqrt(full), m['rms_log_residual_test']), flush=True)
        out['models'].setdefault(tag, {})[name] = m
        return m

    ys = np.log(s)
    print('\nA. regress log s directly on quotient coordinates (test = 20%% of nodes):', flush=True)
    report('direct_s', 'constant', ys, np.full_like(ys, ys[~test].mean()))
    report('direct_s', 'ridge linear', ys, ridge(Xq_rows[~test], ys[~test])(Xq_rows))
    Xqq = quad(Xq_rows)
    report('direct_s', 'ridge quadratic', ys, ridge(Xqq[~test], ys[~test], 1e-2)(Xqq))

    print('\nB. two-stage: regress log c on physical dofs, then apply the known (B.^2) map:', flush=True)
    Xp_rows = np.c_[Xn[np.arange(q)//3], np.eye(3)[np.arange(q) % 3]]
    testp = test_nodes[np.arange(q)//3]
    yc = np.log(c)
    for nm, Xa_, lam in (('ridge linear', Xp_rows, 1e-3), ('ridge quadratic', quad(Xp_rows), 1e-2)):
        pc = ridge(Xa_[~testp], yc[~testp], lam)(Xa_)
        chat = np.exp(pc)
        shat = B2 @ chat
        report('pullback_c', nm, ys, np.log(np.maximum(shat, 1e-300)))
    print('\n  (oracle) true c through the pullback: spread %.4e'
          % (lambda v: float(v.max()/v.min()))(s/np.maximum(B2 @ c, 1e-300)), flush=True)
    out['oracle_pullback_spread'] = float((s/np.maximum(B2 @ c, 1e-300)).max()/(s/np.maximum(B2 @ c, 1e-300)).min())
    out['seconds'] = time.time()-t0
    Path(a.out).write_text(json.dumps(out, indent=1))
    print('\nwritten %s (%.1f s)' % (a.out, out['seconds']), flush=True)


if __name__ == '__main__': main()
