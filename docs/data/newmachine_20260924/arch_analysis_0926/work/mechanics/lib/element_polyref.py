"""Fast geometric reference for cut-element stiffness: piecewise-linear level sets on an s^3 sub-grid.

Region of one cell: |f(x)| <= tau(x) (Schwarz-P sheet, f = cos2pi x + cos2pi y + cos2pi z, tau trilinear from the
eight lexicographic corners) and n.x <= offset (retained side of the macro plane, when present).
On every sub-cube the three functions psi1 = tau - f, psi2 = tau + f, psi3 = offset - n.x are replaced by their
linear interpolants on the six Kuhn tetrahedra; each tetrahedron is clipped by the three half-spaces
(marching-tetrahedra cases) and the 125 tensor moments int xi^a eta^b zeta^c dx (a,b,c <= 4) are integrated exactly
on the pieces (collapsed Gauss-Jacobi, exact to total degree 13). Fully-inside sub-cubes use tensor Gauss.
The plane is linear, so its cut is exact; only the curved sheet surface is approximated.
Output: moments aligned with the archived element order, K_ref = sum M T (PSD, exact rigid kernel by
construction), and per-element Loewner deviation from the exact element matrix.
"""
import argparse, json, time
from itertools import product
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi, roots_legendre

import element_moments as EM

ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
KUHN = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])
CUBE = np.array(list(product((0, 1), repeat=3)))  # index = 4x + 2y + z


def tet_rule(m=7):
    a, wa = roots_jacobi(m, 2, 0); b, wb = roots_jacobi(m, 1, 0); c, wc = roots_legendre(m)
    a, b, c = (a + 1) / 2, (b + 1) / 2, (c + 1) / 2
    wa, wb, wc = wa / 8, wb / 4, wc / 2
    A, Bb, Cc = np.meshgrid(a, b, c, indexing='ij')
    W = (wa[:, None, None] * wb[None, :, None] * wc[None, None, :]).ravel()
    u = A.ravel(); v = (Bb * (1 - A)).ravel(); w = (Cc * (1 - A) * (1 - Bb)).ravel()
    return np.stack([u, v, w], 1), W  # reference tet volume 1/6 = W.sum()


def cube_rule():
    g, w = roots_legendre(3)
    P = np.array(list(product(g, g, g))); W = np.prod(np.array(list(product(w, w, w))), axis=1)
    return (P + 1) / 2, W / 8  # unit cube


def clip(tets, attr, k):
    """Keep {attr[..., k] >= 0} inside every tet (linear within the tet)."""
    v = attr[:, :, k]
    inside = v >= 0
    nin = inside.sum(1)
    order = np.argsort(~inside, axis=1, kind='stable')
    T = np.take_along_axis(tets, order[:, :, None], 1); A = np.take_along_axis(attr, order[:, :, None], 1)
    vv = np.take_along_axis(v, order, 1)
    out_t, out_a = [T[nin == 4]], [A[nin == 4]]

    def edge(Ts, As, vs, i, j):
        t = vs[:, i] / (vs[:, i] - vs[:, j])
        return Ts[:, i] + t[:, None] * (Ts[:, j] - Ts[:, i]), As[:, i] + t[:, None] * (As[:, j] - As[:, i])

    def prism(p, q):
        (p0, p1, p2), (q0, q1, q2) = p, q
        for quad in ((p0, p1, p2, q0), (p1, p2, q0, q1), (p2, q0, q1, q2)):
            out_t.append(np.stack([x[0] for x in quad], 1)); out_a.append(np.stack([x[1] for x in quad], 1))

    s = nin == 1
    if s.any():
        Ts, As, vs = T[s], A[s], vv[s]
        pts = [(Ts[:, 0], As[:, 0])] + [edge(Ts, As, vs, 0, j) for j in (1, 2, 3)]
        out_t.append(np.stack([p[0] for p in pts], 1)); out_a.append(np.stack([p[1] for p in pts], 1))
    s = nin == 2
    if s.any():
        Ts, As, vs = T[s], A[s], vv[s]
        prism(((Ts[:, 0], As[:, 0]), edge(Ts, As, vs, 0, 2), edge(Ts, As, vs, 0, 3)),
              ((Ts[:, 1], As[:, 1]), edge(Ts, As, vs, 1, 2), edge(Ts, As, vs, 1, 3)))
    s = nin == 3
    if s.any():
        Ts, As, vs = T[s], A[s], vv[s]
        prism(((Ts[:, 0], As[:, 0]), (Ts[:, 1], As[:, 1]), (Ts[:, 2], As[:, 2])),
              (edge(Ts, As, vs, 0, 3), edge(Ts, As, vs, 1, 3), edge(Ts, As, vs, 2, 3)))
    return np.concatenate(out_t), np.concatenate(out_a)


def monomial_sum(P, W):
    pw = [np.stack([P[:, d] ** k for k in range(5)], 1) for d in range(3)]
    return np.einsum('pa,pb,pc,p->abc', pw[0], pw[1], pw[2], W, optimize=True).ravel()


def element_moments(cell, n, taus, normal, offset, s, trule, crule):
    """125 moments in local xi = 2(n x - cell) - 1, physical measure."""
    g = np.linspace(-1, 1, s + 1)
    X = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1)  # (s+1)^3 x 3 local
    phys = (cell + (X + 1) / 2) / n
    f = np.cos(2 * np.pi * phys).sum(-1)
    tau = sum(taus[c] * np.prod(np.where(np.array(corner), phys, 1 - phys), axis=-1) for c, corner in enumerate(CUBE))
    psi = np.stack([tau - f, tau + f, (offset - phys @ normal) if normal is not None else np.ones_like(f)], -1)
    idx = np.array(list(product(range(s), repeat=3)))
    corner_idx = idx[:, None, :] + CUBE[None]  # (s^3, 8, 3)
    cx = X[corner_idx[..., 0], corner_idx[..., 1], corner_idx[..., 2]]
    cp = psi[corner_idx[..., 0], corner_idx[..., 1], corner_idx[..., 2]]
    full = (cp >= 0).all(axis=(1, 2)); empty = (cp < 0).all(axis=1).any(axis=1)
    part = ~full & ~empty
    M = np.zeros(125)
    if full.any():
        h = 2 / s; lo = cx[full][:, 0]  # (F,3) lower corner in xi
        P = (lo[:, None, :] + h * crule[0][None]).reshape(-1, 3)
        W = np.tile(crule[1] * h ** 3, full.sum())
        M += monomial_sum(P, W)
    if part.any():
        tets = cx[part][:, KUHN].reshape(-1, 4, 3); attr = cp[part][:, KUHN].reshape(-1, 4, 3)
        for k in range(3):
            tets, attr = clip(tets, attr, k)
            if len(tets) == 0:
                break
        if len(tets):
            J = np.abs(np.linalg.det(np.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]], 1)))
            ref, w = trule
            E = np.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]], 1)  # (T,3,3)
            P = (tets[:, None, 0, :] + np.einsum('qk,tkd->tqd', ref, E)).reshape(-1, 3)
            W = (J[:, None] * w[None]).ravel()
            M += monomial_sum(P, W)
    return M * (1 / (2 * n)) ** 3


def main(a):
    t0 = time.perf_counter()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    ctx = json.loads((ROOT / 'packets' / a.case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n']); E = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu_ = E / (2 * (1 + nu))
    taus = np.array([float(v) for v in ctx['case']['tau_corners']])
    normal = None if ctx['case'].get('normal') is None else np.array([float(v) for v in ctx['case']['normal']])
    offset = None if normal is None else float(ctx['case']['offset'])
    truth = np.load(Path(a.moments) / 'MOMENTS125.npz')
    Mt, cells, ordering, keys = truth['moments'], truth['cells'], truth['ordering'], truth['orderings']
    Tm = {k: EM.pattern_operators(keys[k].reshape(27, 3), lam, mu_, n)[1] for k in range(len(keys))}
    trule, crule = tet_rule(), cube_rule()
    rec = dict(case=a.case, elements=int(len(cells)), subgrids={})
    for s in a.subgrids:
        t = time.perf_counter()
        Mr = np.stack([element_moments(c, n, taus, normal, offset, s, trule, crule) for c in cells])
        vol_rel = np.abs(Mr[:, 0] - Mt[:, 0]) / Mt[:, 0]
        dev = np.empty(len(cells)); psd_min = np.empty(len(cells))
        for e in range(len(cells)):
            K = np.einsum('m,mij->ij', Mt[e], Tm[ordering[e]]); Kr = np.einsum('m,mij->ij', Mr[e], Tm[ordering[e]])
            ev, U = np.linalg.eigh(K); keep = ev > ev.max() * 1e-12; Wh = U[:, keep] / np.sqrt(ev[keep])
            g = np.linalg.eigvalsh(Wh.T @ Kr @ Wh)
            dev[e] = np.abs(g - 1).max(); psd_min[e] = g.min()
        vf = Mt[:, 0] * n ** 3
        bins = [(0, 1e-6), (1e-6, 1e-3), (1e-3, 0.1), (0.1, 0.999999), (0.999999, 2)]
        by = {f'{lo:g}-{hi:g}': dict(count=int(((vf >= lo) & (vf < hi)).sum()),
                                      dev_median=float(np.median(dev[(vf >= lo) & (vf < hi)])) if ((vf >= lo) & (vf < hi)).any() else None,
                                      dev_max=float(dev[(vf >= lo) & (vf < hi)].max()) if ((vf >= lo) & (vf < hi)).any() else None)
              for lo, hi in bins}
        r = dict(volume_rel_median=float(np.median(vol_rel)), volume_rel_max=float(vol_rel.max()),
                 loewner_dev_median=float(np.median(dev)), loewner_dev_q90=float(np.quantile(dev, .9)),
                 loewner_dev_q99=float(np.quantile(dev, .99)), loewner_dev_max=float(dev.max()),
                 within_3pct=float(np.mean(dev <= .03)), within_10pct=float(np.mean(dev <= .1)),
                 by_volume_fraction=by, seconds=time.perf_counter() - t)
        rec['subgrids'][s] = r
        print(json.dumps(dict(event='POLYREF', s=s, **r)), flush=True)
        np.savez(out / f'POLYREF_S{s}.npz', moments=Mr, loewner_dev=dev, psd_min=psd_min)
    rec['seconds'] = time.perf_counter() - t0
    (out / 'RESULT.json').write_text(json.dumps(rec, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True); p.add_argument('--moments', required=True); p.add_argument('--output', required=True)
    p.add_argument('--subgrids', type=int, nargs='+', default=[2, 4, 8])
    main(p.parse_args())
