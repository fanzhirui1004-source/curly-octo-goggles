"""E3: can a coarse Galerkin correction fix what the smoothing tail cannot? On the actual network error of a checkpoint.
For validation directions q: exact u, network u0 = geo.field(model, q) (tail off). Coarse spaces V (interior-restricted
prolongations from tensor grids over the unit cell): Q1 on 33^3 / 17^3 / 9^3 vertices, Q2 with 8 elements per axis
(17^3 nodes), and PU-linear on 17^3 / 9^3 (each vertex: translation + 3x3 slopes, u = sum_v N_v (a_v + B_v (x - x_v))).
Deployable two-grid corrections use only the residual r_I = -(K u)_I:  c = (V^T K_II V)^-1 V^T r_I,  u <- u + V c.
Reports the energy excess ||u - u*||_K^2 / ||u*||_K^2 (mean, p90) for: net, net+tail8, net+coarse, net+coarse+tail8,
net+tail8+coarse+tail8 (V-cycle-like), and zero-interior start + coarse + tail8 (what the network contributes).
Usage: e3_coarse.py <ckpt> <out.json> <case>[,...] [--classes force_c,force] [--m 32] [--spaces Q1_33,Q1_17,...]
(OPL_DEV=cpu: exact solve by PARDISO/SuperLU, GPU hidden)"""
import diag_sens as DS                                                   # first: CPU env
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD
import trainlib as TL

dev, dt = TL.dev, TL.dt


def prolong(xyz, I, order, ne, pu=False):
    nn = order * ne + 1
    e = np.minimum(np.floor(xyz * ne).astype(int), ne - 1); t = xyz * ne - e
    sh = (lambda s: np.stack([1 - s, s], -1)) if order == 1 else \
         (lambda s: np.stack([(2 * s - 1) * (2 * s - 2) / 2, 1 - (2 * s - 1) ** 2, (2 * s - 1) * (2 * s) / 2], -1))
    W = [sh(t[:, d]) for d in range(3)]
    rows, cols, vals = [], [], []
    N = len(xyz)
    for a in range(order + 1):
        for b in range(order + 1):
            for c in range(order + 1):
                vid = ((order * e[:, 0] + a) * nn + order * e[:, 1] + b) * nn + order * e[:, 2] + c
                w = W[0][:, a] * W[1][:, b] * W[2][:, c]
                if not pu:
                    for comp in range(3):
                        rows.append(3 * np.arange(N) + comp); cols.append(3 * vid + comp); vals.append(w)
                else:                                                    # 12 dofs per vertex: a (3) + B (3x3)
                    xv = np.stack([(order * e[:, 0] + a), (order * e[:, 1] + b), (order * e[:, 2] + c)], 1) / (order * ne)
                    dx = xyz - xv
                    for comp in range(3):
                        rows.append(3 * np.arange(N) + comp); cols.append(12 * vid + comp); vals.append(w)
                        for j in range(3):
                            rows.append(3 * np.arange(N) + comp); cols.append(12 * vid + 3 + 3 * comp + j); vals.append(w * dx[:, j])
    import scipy.sparse as sp
    ncol = (12 if pu else 3) * nn ** 3
    Pd = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * N, ncol))
    PI = Pd[I]
    used = np.flatnonzero(np.abs(PI).sum(0).A1 > 1e-14)
    return PI[:, used].tocsc()


SPACES = {'Q1_33': (1, 32, False), 'Q1_17': (1, 16, False), 'Q1_9': (1, 8, False), 'Q2_17': (2, 8, False),
          'PU_17': (1, 16, True), 'PU_9': (1, 8, True)}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--classes', default='force_c,force'); ap.add_argument('--m', type=int, default=32)
    ap.add_argument('--spaces', default='Q1_9,PU_9,Q1_17,Q2_17,PU_17,Q1_33'); ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data_v2')
    a = ap.parse_args(argv)
    import scipy.sparse as sp
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    rec = dict(ckpt=a.ckpt, k=a.k, per_case={})
    model = None
    for case in a.cases.split(','):
        t0 = time.perf_counter()
        geo = TL.Geo(case, a.body, a.data, neumann=False, log=lambda s_: None)
        C = geo.C
        DS.cpu_factor(C) if dev.type == 'cpu' else C.factor(neumann=False)
        if model is None:
            model = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=dev.type != 'cpu', smooth_k=0)).to(dev)
            MD.load_compat(model, ck['model']); model.eval()
        else:
            model.add_geo(geo)
        n2 = 2 * C.n + 1
        xyz = np.stack(np.unravel_index(np.asarray(C.nodes), (n2,) * 3), 1).astype(float) / (n2 - 1)
        I = C.I.cpu().numpy()
        # K_II as scipy (symmetric) for the coarse Galerkin matrices
        pm = np.zeros(C.nb, bool); pm[C.P.cpu().numpy()] = True
        new = -np.ones(C.nb, np.int64); new[I] = np.arange(len(I))
        ru, cu, vv = C.ru.long().numpy(), C.cu.long().numpy(), C.vals.numpy()
        sel = ~pm[ru] & ~pm[cu]
        U = sp.csr_matrix((vv[sel], (new[ru[sel]], new[cu[sel]])), shape=(len(I), len(I)))
        KII = (U + sp.triu(U, 1).T).tocsr()
        r = dict(interior=int(len(I)))
        Vs = {}
        for s in a.spaces.split(','):
            order, ne, pu = SPACES[s]
            V = prolong(xyz, I, order, ne, pu)
            A = (V.T @ (KII @ V)).tocsc(); A = (A + A.T) / 2
            dg = A.diagonal(); keep = np.flatnonzero(dg > 1e-12 * dg.max())
            V = V[:, keep]; A = A[keep][:, keep].tocsc()
            try:
                import pypardiso
                ps = pypardiso.PyPardisoSolver(); Acsr = A.tocsr(); ps.factorize(Acsr)
                solve = (lambda Acsr, ps: (lambda b: ps.solve(Acsr, np.ascontiguousarray(b))))(Acsr, ps)
            except ImportError:
                import scipy.sparse.linalg as sla
                lu = sla.splu(A); solve = lu.solve
            Vs[s] = (V, solve)
            r.setdefault('coarse_dofs', {})[s] = int(V.shape[1])

        def coarse(u, s):
            V, solve = Vs[s]
            x = u.clone()
            rI = -(C.K @ x)[C.I].numpy()
            c = solve(V.T @ rI).reshape(V.shape[1], -1)
            x[C.I] += torch.as_tensor(V @ c, dtype=dt)
            return x
        tail = lambda u: TL.smooth_tail(C, u, a.k, 30.0)
        for cls in [c for c in a.classes.split(',') if c in geo.classes]:
            Q = geo.banks['val'][cls][:, :a.m].to(dt)
            with torch.no_grad():
                u = C.extend(Q); u0 = geo.field(model, Q).to(dt)
                z = torch.zeros_like(u); z[C.P] = Q
                eu = TL.energy(u, C.K)
                ex = lambda x: (TL.energy(x, C.K) / eu - 1).numpy()
                st = lambda v: dict(mean=float(v.mean()), p90=float(np.quantile(v, .9)))
                out = dict(net=st(ex(u0)), net_tail=st(ex(tail(u0))))
                for s in Vs:
                    out[f'net+{s}'] = st(ex(coarse(u0, s)))
                    out[f'net+{s}+tail'] = st(ex(tail(coarse(u0, s))))
                    out[f'net+tail+{s}+tail'] = st(ex(tail(coarse(tail(u0), s))))
                    out[f'zero+tail+{s}+tail'] = st(ex(tail(coarse(tail(z), s))))
            r[cls] = out
            print(json.dumps(dict(case=case, cls=cls, **{k: round(v['mean'], 4) for k, v in out.items()})), flush=True)
        r['seconds'] = time.perf_counter() - t0
        rec['per_case'][case] = r
        model.caches.pop(case, None); C._free(); del geo, C
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
