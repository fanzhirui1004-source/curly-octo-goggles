"""A8: can the fictitious fringe be continued locally? An oracle on exact fields (no network involved).

W = interior weak nodes (NETDATA weak & ~is_port; weak = 3x3 diagonal block norm < 1% of the geometry median). The exact
extension u* of a val direction has (K u*)_I = 0, so replacing u*_W by v_W (delta = v - u* on W, zero elsewhere) costs
exactly  (u* + delta)^T K (u* + delta) - u*^T K u* = delta^T K_WW delta   (relative: the banks have unit exact energy).
Candidates for v_W (linear maps of the non-weak values built from the mesh alone - no K - except (iii) and (iv)):
  zero    v_W = 0 (the scale of what is at stake)
  root    (i) Q2 interpolant of the nearest element with volume fraction >= vf_min and no W node (AgFEM root), evaluated
          (extrapolated) at the node; nodes without such a root within R element rings take (ii)
  affine  (ii) least squares u(x) = a + B (x - x_w) (12 parameters) on the non-weak nodes of the union of the node's elements
          (two element rings when that gives fewer than 4 points)
  ridge   (iii) best local linear map fitted on TRAIN-bank exact fields: per group (a connected cluster of W under element
          adjacency; clusters above max_group nodes split into patch^3-element blocks) ridge regression of u_group on the
          non-weak values of the elements touching the group; the ridge per group from a 20% hold-out of the train fields
  exact   (iv) K_WW v_W = -K_W,rest u*_rest (one sparse solve = one per connected component of K_WW, ghost-penalty couplings
          included): must reproduce u* (sanity)
Per candidate and val class: excess energy mean / p90 / max; 8-corner sensitivity error |s(u* + delta) - s(u*)| / |s(u*)|
(trainlib's sens_hat operator (_Sens with the slot's dM, Tm) evaluated in fp64: the fp32 evaluation alone differs from the
exact labels by up to 0.2% on force / support directions, which would floor the measure); coverage. Decision (review A8), on the worst class-mean excess:
  min((i), (ii)) <= hard (0.3%) -> hard continuation; <= gated (2%) -> gated continuation;
  else (iii) <= gated -> learned local map (P3); else non-local: stop.
Usage: diag_fringe.py <out.json> [--cases a,b] [--split SPLIT.json] [--val-max 20] [--n-train 256] ... (--help)
Remote: the exact fields come from the interior factor (teacher.Cell.factor(neumann=False, fp32=True) + fp64 refinement).
"""
import json, sys, time, argparse, itertools
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import diag_cert as DC

dev, dt = TL.dev, TL.dt
CANDS = ('zero', 'root', 'affine', 'ridge', 'exact')


def q2_1d(o, x):
    """Q2 Lagrange basis on the nodes 0, 1/2, 1 (index o) at local coordinate x (as lattice3.face_traction_weights)."""
    return np.where(o == 0, 2 * (x - .5) * (x - 1), np.where(o == 1, -4 * x * (x - 1), 2 * x * (x - .5)))


def _gather(ptr, val, rows):
    """CSR rows -> (position in rows, value) pairs."""
    cnt = ptr[rows + 1] - ptr[rows]
    rep = np.repeat(np.arange(len(rows)), cnt)
    off = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return rep, val[np.repeat(ptr[rows], cnt) + off]


def _upairs(i, j, nj):
    k = np.unique(i.astype(np.int64) * nj + j)
    return k // nj, k % nj


class Topo:
    """Mesh data of one geometry from NETDATA (node order = teacher NODES; DOF 3 i + c of node i)."""

    def __init__(self, nd):
        self.g = nd['grid'].astype(np.int64)                                        # N x 3, 0 .. 2n
        self.en = nd['elem_nodes'].astype(np.int64)                                 # E x 27
        self.ec = nd['elem_cells'].astype(np.int64)                                 # E x 3
        self.n = int(np.asarray(nd['n']).reshape(-1)[0])
        self.N, self.E = len(self.g), len(self.en)
        self.vf = np.clip(np.asarray(nd['moments'])[:, 0] * self.n ** 3, 0, 1)      # as models.MGCache
        self.weak = np.asarray(nd['weak'], bool); self.port = np.asarray(nd['is_port'], bool)
        self.W = np.flatnonzero(self.weak & ~self.port)
        self.isW = np.zeros(self.N, bool); self.isW[self.W] = True
        o = np.argsort(self.en.reshape(-1), kind='stable')
        self.ne_el = o // 27
        self.ne_ptr = np.concatenate([[0], np.cumsum(np.bincount(self.en.reshape(-1), minlength=self.N))])

    def elems(self, nodes, rings=1):
        """(i, e) pairs: elements within `rings` element rings of nodes[i]."""
        i, e = _gather(self.ne_ptr, self.ne_el, nodes)
        for _ in range(rings - 1):
            i2 = np.repeat(i, 27); nn = self.en[e].reshape(-1)
            i3, e3 = _gather(self.ne_ptr, self.ne_el, nn)
            i, e = _upairs(i2[i3], e3, self.E)
        return i, e

    def stencil(self, nodes, rings=1):
        """(i, j) pairs: non-weak nodes j of the elements within `rings` rings of nodes[i]."""
        i, e = self.elems(nodes, rings)
        i2, j = np.repeat(i, 27), self.en[e].reshape(-1)
        k = ~self.weak[j]
        return _upairs(i2[k], j[k], self.N)

    # ------------------------------------------------------------------ (ii) affine
    def affine(self, nodes):
        """(i, j, w): v(nodes[i]) = sum w u(j), the least-squares affine fit evaluated at the node."""
        i, j = self.stencil(nodes, 1)
        cnt = np.bincount(i, minlength=len(nodes))
        few = np.flatnonzero(cnt < 4)
        if few.size:
            i2, j2 = self.stencil(nodes[few], 2)
            keep = ~np.isin(i, few)
            i, j = np.concatenate([i[keep], few[i2]]), np.concatenate([j[keep], j2])
        phi = np.concatenate([np.ones((len(i), 1)), (self.g[j] - self.g[nodes[i]]) / 2.0], 1)       # element units
        A = torch.zeros((len(nodes), 4, 4), dtype=dt)
        ph = torch.as_tensor(phi)
        A.index_add_(0, torch.as_tensor(i), ph[:, :, None] * ph[:, None, :])
        c = torch.linalg.pinv(A, rtol=1e-10, hermitian=True)[:, 0, :]              # e_1^T A^+ (A symmetric)
        w = (c[torch.as_tensor(i)] * ph).sum(1).numpy()
        return i, j, w, dict(nodes=len(nodes), two_rings=int(few.size), empty=int((np.bincount(i, minlength=len(nodes)) == 0).sum()),
                             points_median=float(np.median(np.bincount(i, minlength=len(nodes)))))

    # ------------------------------------------------------------------ (i) root element
    def root(self, nodes, vf_min=0.5, R=2):
        """(i, j, w, has): Q2 interpolant of the nearest clean element with vf >= vf_min (centre distance, ties -> larger vf)."""
        n = self.n
        cid = np.full((n, n, n), -1, np.int64); cid[self.ec[:, 0], self.ec[:, 1], self.ec[:, 2]] = np.arange(self.E)
        good = (self.vf >= vf_min) & ~self.isW[self.en].any(1)
        gw = self.g[nodes]; base = np.clip(gw // 2, 0, n - 1)
        best = np.full(len(nodes), -1); bd = np.full(len(nodes), np.inf)
        for d in itertools.product(range(-R, R + 1), repeat=3):
            c = base + np.asarray(d)
            ok = (c >= 0).all(1) & (c < n).all(1)
            e = np.full(len(nodes), -1); e[ok] = cid[c[ok, 0], c[ok, 1], c[ok, 2]]
            ok &= e >= 0
            ok[ok] = good[e[ok]]
            dist = np.full(len(nodes), np.inf)
            dist[ok] = np.linalg.norm(gw[ok] / 2.0 - (c[ok] + .5), axis=1) - 1e-6 * self.vf[e[ok]]
            upd = dist < bd
            best[upd], bd[upd] = e[upd], dist[upd]
        has = best >= 0
        r = best[has]
        xi = gw[has] / 2.0 - self.ec[r]                                             # local coordinates in the root (extrapolation)
        o = self.g[self.en[r]] - 2 * self.ec[r][:, None, :]                           # slot offsets in {0, 1, 2}^3
        Nb = q2_1d(o, xi[:, None, :]).prod(2)
        i = np.repeat(np.flatnonzero(has), 27)
        return i, self.en[r].reshape(-1), Nb.reshape(-1), has, dict(nodes=len(nodes), rooted=int(has.sum()),
                                                                    dist_median=float(np.median(bd[has])) if has.any() else None,
                                                                    dist_max=float(bd[has].max()) if has.any() else None,
                                                                    xi_absmax=float(np.abs(xi - .5).max() + .5) if has.any() else None)

    # ------------------------------------------------------------------ (iii) groups
    def groups(self, patch=1, max_group=32):
        """Group id per W node: connected clusters of W (element adjacency), big ones split into patch^3-element blocks;
        and the (group, input node) pairs: non-weak nodes of the elements touching the group."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        nW = len(self.W)
        i, e = _gather(self.ne_ptr, self.ne_el, self.W)
        A = csr_matrix((np.ones(len(i)), (i, nW + e)), shape=(nW + self.E, nW + self.E))
        _, lab = connected_components(A, directed=False)
        lab = np.unique(lab[:nW], return_inverse=True)[1]
        big = np.bincount(lab)[lab] > max_group
        nbk = self.n // patch + 1
        b = self.g[self.W] // (2 * patch)
        blk = (b[:, 0] * nbk + b[:, 1]) * nbk + b[:, 2]
        key = lab * (nbk ** 3 + 1) + np.where(big, blk, nbk ** 3)
        grp = np.unique(key, return_inverse=True)[1]
        gi, ge = _upairs(grp[i], e, self.E)
        g2, j = np.repeat(gi, 27), self.en[ge].reshape(-1)
        k = ~self.weak[j]
        gi, gj = _upairs(g2[k], j[k], self.N)
        return grp, gi, gj, dict(clusters=int(lab.max() + 1) if nW else 0, groups=int(grp.max() + 1) if nW else 0,
                                 split_nodes=int(big.sum()), group_size_max=int(np.bincount(grp).max()) if nW else 0,
                                 inputs_max=int(np.bincount(gi).max()) if len(gi) else 0)


def apply_map(i, j, w, u, rows):
    """v[rows[i]] = sum w u[j] per component: u (nb, B) DOF field -> (len(rows) * 3, B) values on the rows' DOFs."""
    un = u.reshape(-1, 3, u.shape[1])
    ii, jj = torch.as_tensor(i, device=u.device), torch.as_tensor(j, device=u.device)
    ww = torch.as_tensor(w, dtype=u.dtype, device=u.device)
    v = torch.zeros((len(rows), 3, u.shape[1]), dtype=u.dtype, device=u.device)
    v.index_add_(0, ii, ww[:, None, None] * un[jj])
    return v.reshape(-1, u.shape[1])


def dofs(nodes, device=None):
    nodes = torch.as_tensor(np.asarray(nodes), device=device)
    return (3 * nodes[:, None] + torch.arange(3, device=device)[None, :]).reshape(-1)


def upper_scipy(C):
    import scipy.sparse as sp
    return sp.csr_matrix((C.vals.double().cpu().numpy(), C.cu.long().cpu().numpy(), C.crow.long().cpu().numpy()), shape=(C.nb, C.nb))


class Exact:
    """(iv): v_W = -K_WW^-1 K_W,rest u_rest with one sparse LU of K_WW (fp64, host)."""

    def __init__(self, C, Wd):
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        wd = Wd.cpu().numpy()
        Uw = upper_scipy(C)[wd][:, wd]                                                  # K_WW = U_WW + U_WW^T - diag
        self.lu = splu((Uw + Uw.T - sp.diags(Uw.diagonal())).tocsc())
        self.C, self.Wd = C, Wd

    def __call__(self, u):
        x = u.clone(); x[self.Wd] = 0
        r = -(self.C @ x)[self.Wd]
        return torch.as_tensor(self.lu.solve(r.cpu().numpy()), dtype=dt, device=u.device)


def ridge_fit(X, Y, lams=(1e-8, 1e-6, 1e-4, 1e-2), hold=0.2):
    """M (p, t) minimizing |Y - X M|^2 + lam s |M|^2, s = mean diag X^T X; lam from a hold-out of the last rows."""
    def fit(X_, Y_, lam):
        s = (X_ * X_).sum() / X_.shape[1]
        if X_.shape[1] <= X_.shape[0]:
            A = X_.T @ X_; A.diagonal().add_(lam * s)
            return torch.linalg.solve(A, X_.T @ Y_)
        A = X_ @ X_.T; A.diagonal().add_(lam * s)
        return X_.T @ torch.linalg.solve(A, Y_)
    m = X.shape[0]; k = max(int(round(m * (1 - hold))), 1)
    if len(lams) > 1 and m - k >= 2:
        err = [float(((Y[k:] - X[k:] @ fit(X[:k], Y[:k], lam)) ** 2).sum()) for lam in lams]
        lam = lams[int(np.argmin(err))]
    else:
        lam = lams[0]
    return fit(X, Y, lam), lam


class Ridge:
    """(iii): per group g, v_g = M_g^T x_g with x_g the non-weak input values; fitted on train exact fields."""

    def __init__(self, topo, extend, Qtr, patch=1, max_group=32, lams=(1e-8, 1e-6, 1e-4, 1e-2), chunk=64, device=dev):
        grp, gi, gj, self.info = topo.groups(patch, max_group)
        W, G = topo.W, int(grp.max()) + 1 if len(topo.W) else 0
        split = lambda key, val: np.split(val[np.argsort(key, kind='stable')], np.cumsum(np.bincount(key, minlength=G))[:-1])
        self.gd = [dofs(x, device) for x in split(grp, W)] if G else []                                     # target DOFs
        self.xd = [dofs(x, device) for x in split(gi, gj)] if G else []                                     # input DOFs
        used = torch.unique(torch.cat(self.gd + self.xd)) if self.gd else torch.zeros(0, dtype=torch.long, device=device)
        pos = torch.full((topo.N * 3,), -1, dtype=torch.long, device=device); pos[used] = torch.arange(len(used), device=device)
        self.gd_u, self.xd_u = [pos[d] for d in self.gd], [pos[d] for d in self.xd]
        U = torch.cat([extend(Qtr[:, j:j + chunk].to(dt))[used] for j in range(0, Qtr.shape[1], chunk)], 1).T.contiguous()   # m x used
        self.M, lam = [], []
        for gd, xd in zip(self.gd_u, self.xd_u):
            M_, l_ = ridge_fit(U[:, xd], U[:, gd], lams) if len(xd) else (torch.zeros((0, len(gd)), dtype=dt, device=device), None)
            self.M.append(M_); lam.append(l_)
        self.info.update(train=int(U.shape[0]), used_dofs=int(len(used)), no_inputs=lam.count(None),
                         lam_counts={str(l_): lam.count(l_) for l_ in lams})

    def __call__(self, u, Wd):
        """Values on Wd (the W DOFs in W order) of the fitted maps applied to u (nb, B)."""
        out = torch.zeros((len(Wd), u.shape[1]), dtype=u.dtype, device=u.device)
        pos = torch.full((u.shape[0],), -1, dtype=torch.long, device=u.device); pos[Wd] = torch.arange(len(Wd), device=u.device)
        for gd, xd, M_ in zip(self.gd, self.xd, self.M):
            out[pos[gd]] = (u[xd].T @ M_).T
        return out


def sens_fn(geo, chunk=128):
    """u (nb, B) -> -u^T dK/dtau_c u (8, B): trainlib.Geo.sens_hat's operator (_Sens, the slot's dM32 / Tm32) in fp64."""
    Tm, dM = geo.Tm32.to(dt), geo.dM32.to(dt)
    return lambda u: TL._Sens.apply(u.to(dt), geo.C.dofs, Tm, dM, chunk)


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), p90=float(np.quantile(x, .9)), max=float(x.max()))


def decide(worst, hard=0.003, gated=0.02):
    local = min(worst['root'], worst['affine'])
    if local <= hard:
        return 'hard_continuation'
    if local <= gated:
        return 'gated_continuation'
    if worst.get('ridge', np.inf) <= gated:
        return 'learned_local_map'
    return 'non_local_stop'


@torch.no_grad()
def run_geo(geo, extend, n_train=256, vf_min=0.5, R=2, patch=1, max_group=32, lams=(1e-8, 1e-6, 1e-4, 1e-2), chunk=16,
            hard=0.003, gated=0.02, ustar=None, log=print):
    """A8 on one geometry. extend(Q) -> exact fields (nb, k) fp64 (interior solve); ustar: optional dict cls -> exact val
    fields (else extend(val bank)). Returns the per-geometry record."""
    t0 = time.perf_counter()
    topo = Topo(geo.nd)
    W = topo.W; Wd = dofs(W, dev)
    rec = dict(case=geo.case, nodes=topo.N, weak_nodes=int(topo.weak.sum()), W_nodes=len(W), W_dof_fraction=3 * len(W) / geo.nb)
    if not len(W):
        rec['decision'] = 'no_interior_weak_nodes'
        return rec
    iA, jA, wA, rec['affine'] = topo.affine(W)
    iR, jR, wR, has, rec['root'] = topo.root(W, vf_min, R)
    fb = np.isin(iA, np.flatnonzero(~has))                                           # rootless nodes: the affine fit
    iR, jR, wR = np.concatenate([iR, iA[fb]]), np.concatenate([jR, jA[fb]]), np.concatenate([wR, wA[fb]])
    rec['root']['fallback_affine'] = int((~has).sum())
    Qtr = torch.cat([geo.banks['train'][c][:, :n_train] for c in geo.classes], 1)
    t1 = time.perf_counter()
    ridge = Ridge(topo, extend, Qtr, patch, max_group, lams, device=dev)
    rec['ridge'] = ridge.info; rec['ridge']['seconds'] = time.perf_counter() - t1
    t1 = time.perf_counter()
    exact = Exact(geo.C, Wd); rec['exact_factor_seconds'] = time.perf_counter() - t1
    sens = sens_fn(geo)
    maps = dict(zero=lambda u: torch.zeros((len(Wd), u.shape[1]), dtype=dt, device=u.device),
                root=lambda u: apply_map(iR, jR, wR, u, W), affine=lambda u: apply_map(iA, jA, wA, u, W),
                ridge=lambda u: ridge(u, Wd), exact=exact)
    rec['classes'] = {}
    for c in geo.classes:
        Q = geo.banks['val'][c]
        acc = {k: dict(excess=[], sens=[]) for k in CANDS}
        chk = dict(energy=[], resid=[], sens_label=[])
        for j in range(0, Q.shape[1], chunk):
            us = ustar[c][:, j:j + chunk].to(dev, dt) if ustar is not None else extend(Q[:, j:j + chunk].to(dt))
            Ku = geo.C @ us
            chk['energy'].append(((us * Ku).sum(0) - 1).cpu())
            chk['resid'].append((Ku[Wd].abs().amax(0) / Ku[geo.P].abs().amax(0)).cpu())    # (K u*)_W = 0 check
            s0 = sens(us)
            if geo.sens is not None and c in geo.sens['val']:
                lab = geo.sens['val'][c][:, j:j + chunk]
                chk['sens_label'].append(((s0 - lab).norm(dim=0) / lab.norm(dim=0)).cpu())
            for k in CANDS:
                d = torch.zeros_like(us); d[Wd] = maps[k](us) - us[Wd]
                acc[k]['excess'].append((d * (geo.C @ d)).sum(0).cpu())
                s1 = sens(us + d)
                acc[k]['sens'].append(((s1 - s0).norm(dim=0) / s0.norm(dim=0)).cpu())
        r = {k: dict(excess=stats(torch.cat(v['excess'])), sens=stats(torch.cat(v['sens']))) for k, v in acc.items()}
        r['check'] = dict(energy_minus_1_absmax=float(torch.cat(chk['energy']).abs().max()),
                          resid_W_rel_max=float(torch.cat(chk['resid']).max()),
                          sens_hat_vs_label=stats(torch.cat(chk['sens_label'])) if chk['sens_label'] else None)
        rec['classes'][c] = r
        log(json.dumps(dict(case=geo.case, cls=c, **{k: round(r[k]['excess']['mean'], 6) for k in CANDS},
                            **{f'sens_{k}': round(r[k]['sens']['mean'], 5) for k in CANDS})))
    rec['worst_class_mean'] = {k: max(rec['classes'][c][k]['excess']['mean'] for c in rec['classes']) for k in CANDS}
    rec['decision'] = decide(rec['worst_class_mean'], hard, gated)
    rec['seconds'] = time.perf_counter() - t0
    return rec


def summary(recs, hard=0.003, gated=0.02):
    ok = [r for r in recs if 'worst_class_mean' in r]
    if not ok:
        return {}
    wc = {k: np.asarray([r['worst_class_mean'][k] for r in ok]) for k in CANDS}
    worst = {k: float(v.max()) for k, v in wc.items()}; med = {k: float(np.median(v)) for k, v in wc.items()}
    return dict(geometries=len(ok), worst=worst, median=med, decision_worst=decide(worst, hard, gated),
                decision_median=decide(med, hard, gated), decisions={r['case']: r['decision'] for r in ok})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out')
    ap.add_argument('--cases', default=''); ap.add_argument('--split', default=DC.SPLIT); ap.add_argument('--val-max', type=int, default=20)
    ap.add_argument('--body', default=DC.BODY); ap.add_argument('--data', default=DC.DATA); ap.add_argument('--slots', default=DC.SLOTS)
    ap.add_argument('--n-train', type=int, default=256); ap.add_argument('--vf-min', type=float, default=0.5)
    ap.add_argument('--rings', type=int, default=2); ap.add_argument('--patch', type=int, default=1)
    ap.add_argument('--max-group', type=int, default=32); ap.add_argument('--lams', default='1e-8,1e-6,1e-4,1e-2')
    ap.add_argument('--hard', type=float, default=0.003); ap.add_argument('--gated', type=float, default=0.02)
    args = ap.parse_args(argv)
    cases = args.cases.split(',') if args.cases else json.loads(Path(args.split).read_text())['val'][:args.val_max]
    lams = tuple(float(x) for x in args.lams.split(','))
    rec = dict(args=vars(args), geometries=[])
    for case in cases:
        geo = C = None
        try:
            geo = DC.load_geo(case, args.body, args.data, args.slots, log=lambda d: print(json.dumps(d), flush=True))
            C = geo.C
            C.factor(neumann=False, fp32=True)                                            # exact fields: fp32 factor + fp64 refinement
            r = run_geo(geo, C.extend, args.n_train, args.vf_min, args.rings, args.patch, args.max_group, lams,
                        hard=args.hard, gated=args.gated)
        except Exception as e:
            r = dict(case=case, error=repr(e)[:300])
        finally:
            if C is not None and getattr(C, 'sol_I', None) is not None:
                C._free()                                                                 # cuDSS memory is released only by free()
        print(json.dumps(dict(event='GEO_DONE', case=case, decision=r.get('decision'), error=r.get('error'))), flush=True)
        rec['geometries'].append(r)
        rec['summary'] = summary(rec['geometries'], args.hard, args.gated)
        Path(args.out).write_text(json.dumps(rec, indent=1))
        geo = C = None; DC.free_mem()
    print(json.dumps(dict(event='SUMMARY', **rec.get('summary', {}))), flush=True)
    return rec


if __name__ == '__main__':
    main()
