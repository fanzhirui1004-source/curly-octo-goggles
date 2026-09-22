"""Batched (GPU) version of element_polyref.element_moments: all cells, all sub-cubes, all tetrahedra at once.

Same geometry and same arithmetic as element_polyref.py (piecewise-linear psi on Kuhn tetrahedra of an s^3
sub-grid, clipped by psi1 = tau - f, psi2 = tau + f, psi3 = offset - n.x; tensor Gauss on fully-inside sub-cubes,
collapsed Gauss-Jacobi on clipped pieces). Returns the 125 tensor moments per cell in local xi coordinates with
the physical measure, in the order of `cells`.
"""
import time
from itertools import product
import numpy as np
import torch

from element_polyref import KUHN, CUBE, tet_rule, cube_rule


def _clip(tets, attr, owner, k):
    v = attr[:, :, k]
    inside = v >= 0
    nin = inside.sum(1)
    order = torch.argsort((~inside).to(torch.int8), dim=1, stable=True)
    T = torch.gather(tets, 1, order[:, :, None].expand(-1, -1, 3))
    A = torch.gather(attr, 1, order[:, :, None].expand(-1, -1, 3))
    vv = torch.gather(v, 1, order)
    out_t, out_a, out_o = [T[nin == 4]], [A[nin == 4]], [owner[nin == 4]]

    def edge(Ts, As, vs, i, j):
        t = (vs[:, i] / (vs[:, i] - vs[:, j]))[:, None]
        return Ts[:, i] + t * (Ts[:, j] - Ts[:, i]), As[:, i] + t * (As[:, j] - As[:, i])

    def emit(pts, o):
        out_t.append(torch.stack([p[0] for p in pts], 1)); out_a.append(torch.stack([p[1] for p in pts], 1)); out_o.append(o)

    def prism(p, q, o):
        (p0, p1, p2), (q0, q1, q2) = p, q
        for quad in ((p0, p1, p2, q0), (p1, p2, q0, q1), (p2, q0, q1, q2)):
            emit(quad, o)

    s = nin == 1
    if s.any():
        Ts, As, vs, o = T[s], A[s], vv[s], owner[s]
        emit([(Ts[:, 0], As[:, 0])] + [edge(Ts, As, vs, 0, j) for j in (1, 2, 3)], o)
    s = nin == 2
    if s.any():
        Ts, As, vs, o = T[s], A[s], vv[s], owner[s]
        prism(((Ts[:, 0], As[:, 0]), edge(Ts, As, vs, 0, 2), edge(Ts, As, vs, 0, 3)),
              ((Ts[:, 1], As[:, 1]), edge(Ts, As, vs, 1, 2), edge(Ts, As, vs, 1, 3)), o)
    s = nin == 3
    if s.any():
        Ts, As, vs, o = T[s], A[s], vv[s], owner[s]
        prism(((Ts[:, 0], As[:, 0]), (Ts[:, 1], As[:, 1]), (Ts[:, 2], As[:, 2])),
              (edge(Ts, As, vs, 0, 3), edge(Ts, As, vs, 1, 3), edge(Ts, As, vs, 2, 3)), o)
    return torch.cat(out_t), torch.cat(out_a), torch.cat(out_o)


def _accumulate(M, P, W, owner, chunk=1 << 20):
    for lo in range(0, len(P), chunk):
        p, w, o = P[lo:lo + chunk], W[lo:lo + chunk], owner[lo:lo + chunk]
        pw = [torch.stack([p[:, d] ** k for k in range(5)], 1) for d in range(3)]
        mono = (pw[0][:, :, None, None] * pw[1][:, None, :, None] * pw[2][:, None, None, :]).reshape(len(p), 125)
        M.index_add_(0, o, mono * w[:, None])


def cell_moments(cells, n, taus, normal, offset, s, device='cuda', batch=2048, rule_order=7):
    dt = torch.float64
    trule = tet_rule(rule_order); crule = cube_rule()
    tref = torch.tensor(trule[0], dtype=dt, device=device); tw = torch.tensor(trule[1], dtype=dt, device=device)
    cref = torch.tensor(crule[0], dtype=dt, device=device); cw = torch.tensor(crule[1], dtype=dt, device=device)
    kuhn = torch.tensor(KUHN, device=device); cube = torch.tensor(CUBE, dtype=dt, device=device)
    taus_t = torch.tensor(np.asarray(taus, float), dtype=dt, device=device)
    nrm = None if normal is None else torch.tensor(np.asarray(normal, float), dtype=dt, device=device)
    g = torch.linspace(-1, 1, s + 1, dtype=dt, device=device)
    X = torch.stack(torch.meshgrid(g, g, g, indexing='ij'), -1)  # (s+1)^3 x 3
    idx = torch.tensor(list(product(range(s), repeat=3)), device=device)
    corner_idx = idx[:, None, :] + torch.tensor(CUBE, device=device)[None]  # s^3 x 8 x 3
    cx = X[corner_idx[..., 0], corner_idx[..., 1], corner_idx[..., 2]]  # s^3 x 8 x 3
    out = torch.zeros((len(cells), 125), dtype=dt, device=device)
    cells_t = torch.as_tensor(np.asarray(cells), dtype=dt, device=device)
    h = 2 / s
    for b0 in range(0, len(cells), batch):
        cb = cells_t[b0:b0 + batch]; nb = len(cb)
        phys = (cb[:, None, None, None, :] + (X[None] + 1) / 2) / n  # nb x (s+1)^3 x 3
        f = torch.cos(2 * torch.pi * phys).sum(-1)
        w8 = torch.where(cube.bool()[:, None, None, None, None, :], phys[None], 1 - phys[None]).prod(-1)  # 8 x nb x ...
        tau = torch.einsum('c,c...->...', taus_t, w8)
        p3 = (offset - phys @ nrm) if nrm is not None else torch.ones_like(f)
        psi = torch.stack([tau - f, tau + f, p3], -1)
        cp = psi[:, corner_idx[..., 0], corner_idx[..., 1], corner_idx[..., 2]]  # nb x s^3 x 8 x 3
        full = (cp >= 0).all(-1).all(-1); empty = (cp < 0).all(2).any(-1)
        part = ~full & ~empty
        M = torch.zeros((nb, 125), dtype=dt, device=device)
        e_full, c_full = torch.nonzero(full, as_tuple=True)
        if len(e_full):
            lo = cx[c_full][:, 0]
            P = (lo[:, None, :] + h * cref[None]).reshape(-1, 3)
            W = (cw * h ** 3).repeat(len(e_full))
            _accumulate(M, P, W, e_full.repeat_interleave(len(cw)))
        e_part, c_part = torch.nonzero(part, as_tuple=True)
        if len(e_part):
            tets = cx[c_part][:, kuhn].reshape(-1, 4, 3)
            attr = cp[e_part, c_part][:, kuhn].reshape(-1, 4, 3)
            owner = e_part.repeat_interleave(6)
            for k in range(3):
                tets, attr, owner = _clip(tets, attr, owner, k)
            if len(tets):
                E = torch.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]], 1)
                J = torch.linalg.det(E).abs()
                keep = J > 0
                tets, E, J, owner = tets[keep], E[keep], J[keep], owner[keep]
                for lo_t in range(0, len(tets), 8192):
                    P = (tets[lo_t:lo_t + 8192, None, 0, :] + torch.einsum('qk,tkd->tqd', tref, E[lo_t:lo_t + 8192])).reshape(-1, 3)
                    W = (J[lo_t:lo_t + 8192, None] * tw[None]).reshape(-1)
                    _accumulate(M, P, W, owner[lo_t:lo_t + 8192].repeat_interleave(len(tw)))
        out[b0:b0 + nb] = M
    return (out * (1 / (2 * n)) ** 3).cpu().numpy()


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True); ap.add_argument('--s', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--compare', default=None, help='POLYREF_01/<case> directory with numpy results')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
    ctx = json.loads((ROOT / 'packets' / a.case / 'FRESH_CONTEXT.json').read_text())
    n = int(ctx['n']); taus = [float(v) for v in ctx['case']['tau_corners']]
    normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
    offset = None if normal is None else float(ctx['case']['offset'])
    import element_moments as EM
    cells = EM.members(a.case, ['CELL_INDICES.npy'])['CELL_INDICES.npy']
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for s in a.s:
        torch.cuda.synchronize(); t = time.perf_counter()
        M = cell_moments(cells, n, taus, normal, offset, s)
        torch.cuda.synchronize(); sec = time.perf_counter() - t
        rec = dict(case=a.case, s=s, elements=int(len(cells)), seconds=sec)
        if a.compare:
            ref = np.load(Path(a.compare) / f'POLYREF_S{s}.npz')['moments']
            rec['max_rel_vs_numpy'] = float((np.abs(M - ref).max(1) / np.abs(ref).max(1).clip(1e-300)).max())
        np.savez(out / f'POLYREF_TORCH_S{s}.npz', moments=M)
        print(json.dumps(rec), flush=True)
