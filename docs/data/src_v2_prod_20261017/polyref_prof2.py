"""Batched (GPU) version of element_polyref.element_moments: all cells, all sub-cubes, all tetrahedra at once.

Route 7 speed-ups (same integrals, different summation order): sub-cubes entirely inside the material get their
moments in closed form (product of the 1-D integrals (hi^(k+1) - lo^(k+1))/(k+1)), which the tensor Gauss rule
reproduces exactly; clipped tetrahedra are reduced over their quadrature points first (tensor-product form, one
batched matmul) and added once per tetrahedron instead of once per point and monomial.

Same geometry and same arithmetic as element_polyref.py (piecewise-linear psi on Kuhn tetrahedra of an s^3
sub-grid, clipped by psi1 = tau - f, psi2 = tau + f, psi3 = offset - n.x; tensor Gauss on fully-inside sub-cubes,
collapsed Gauss-Jacobi on clipped pieces). Returns the 125 tensor moments per cell in local xi coordinates with
the physical measure, in the order of `cells`.
"""
import time
from itertools import product
import numpy as np
import torch
PROF = {}
def _tick(name, t0):
    torch.cuda.synchronize(); t1 = time.perf_counter(); PROF[name] = PROF.get(name, 0.0) + t1 - t0; return t1

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


def cell_moments(cells, n, taus, normal, offset, s, device='cuda', batch=2048, rule_order=4, levels=0):
    """125 moments per cell. Base s^3 sub-cubes; partial sub-cubes are refined 2x2x2 up to `levels` times
    (octree near the material boundary only), and clipped as Kuhn tetrahedra at the finest level."""
    dt = torch.float64
    trule = tet_rule(rule_order); crule = cube_rule()
    tref = torch.tensor(trule[0], dtype=dt, device=device); tw = torch.tensor(trule[1], dtype=dt, device=device)
    cref = torch.tensor(crule[0], dtype=dt, device=device); cw = torch.tensor(crule[1], dtype=dt, device=device)
    kuhn = torch.tensor(KUHN, device=device); cube = torch.tensor(CUBE, dtype=dt, device=device)
    taus_t = torch.tensor(np.asarray(taus, float), dtype=dt, device=device)
    nrm = None if normal is None else torch.tensor(np.asarray(normal, float), dtype=dt, device=device)
    idx = torch.tensor(list(product(range(s), repeat=3)), dtype=dt, device=device)
    out = torch.zeros((len(cells), 125), dtype=dt, device=device)
    cells_t = torch.as_tensor(np.asarray(cells), dtype=dt, device=device)
    kp1 = torch.arange(1, 6, dtype=dt, device=device)
    kpow = torch.arange(5, dtype=dt, device=device)

    def psi_at(cell_xyz, xi):  # xi local in [-1,1]; cell_xyz integer cell index per row
        phys = (cell_xyz + (xi + 1) / 2) / n
        f = torch.cos(2 * torch.pi * phys).sum(-1)
        w8 = torch.where(cube.bool().view(8, *([1] * (phys.dim() - 1)), 3), phys[None], 1 - phys[None]).prod(-1)
        tau = torch.einsum('c,c...->...', taus_t, w8)
        p3 = (offset - phys @ nrm) if nrm is not None else torch.ones_like(f)
        return torch.stack([tau - f, tau + f, p3], -1)

    for b0 in range(0, len(cells), batch):
        cb = cells_t[b0:b0 + batch]; nb = len(cb)
        M = torch.zeros((nb, 125), dtype=dt, device=device)
        h = 2 / s
        owner = torch.arange(nb, device=device).repeat_interleave(len(idx))
        lo = (-1 + h * idx).repeat(nb, 1)
        torch.cuda.synchronize(); tp = time.perf_counter()
        for level in range(levels + 1):
            cx = lo[:, None, :] + h * cube[None]                     # C x 8 x 3
            cp = psi_at(cb[owner][:, None, :], cx)                   # C x 8 x 3
            tp = _tick('classify', tp)
            full = (cp >= 0).all(-1).all(-1); empty = (cp < 0).all(1).any(-1)
            part = ~full & ~empty
            if full.any():
                lf = lo[full]; hf = lf + h
                m1 = (hf[:, :, None] ** kp1 - lf[:, :, None] ** kp1) / kp1          # F x 3 x 5 exact 1-D moments
                mono = (m1[:, 0, :, None, None] * m1[:, 1, None, :, None] * m1[:, 2, None, None, :]).reshape(-1, 125)
                M.index_add_(0, owner[full], mono)
            tp = _tick('full_subcubes', tp)
            if not part.any():
                break
            if level < levels:  # refine partial cubes
                lo = (lo[part][:, None, :] + (h / 2) * cube[None]).reshape(-1, 3)
                owner = owner[part].repeat_interleave(8)
                h = h / 2
                continue
            tets = cx[part][:, kuhn].reshape(-1, 4, 3)
            attr = cp[part][:, kuhn].reshape(-1, 4, 3)
            town = owner[part].repeat_interleave(6)
            tp = _tick('refine_kuhn', tp)
            for k in range(3):
                tets, attr, town = _clip(tets, attr, town, k)
            tp = _tick('clip', tp)
            if len(tets):
                E = torch.stack([tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0], tets[:, 3] - tets[:, 0]], 1)
                J = torch.linalg.det(E).abs()
                keep = J > 0
                tets, E, J, town = tets[keep], E[keep], J[keep], town[keep]
                tp = _tick('jacobians', tp)
                # per-tetrahedron moments first (tensor-product form as one batched matmul), then one add per tet:
                # the same sum as the per-point accumulation, in a different order
                for lo_t in range(0, len(tets), 65536):
                    P = tets[lo_t:lo_t + 65536, None, 0, :] + torch.einsum('qk,tkd->tqd', tref, E[lo_t:lo_t + 65536])   # T x q x 3
                    P2 = P * P
                    pw = torch.stack([torch.ones_like(P), P, P2, P2 * P, P2 * P2], -1)         # T x q x 3 x 5
                    X = pw[:, :, 0, :] * (J[lo_t:lo_t + 65536, None] * tw[None])[..., None]      # T x q x 5 (weights folded)
                    YZ = (pw[:, :, 1, :, None] * pw[:, :, 2, None, :]).reshape(len(P), -1, 25)  # T x q x 25
                    mom = torch.bmm(X.transpose(1, 2), YZ).reshape(len(P), 125)
                    tp = _tick('tet_moments', tp)
                    M.index_add_(0, town[lo_t:lo_t + 65536], mom)
                    tp = _tick('tet_index_add', tp)
        out[b0:b0 + nb] = M
    return (out * (1 / (2 * n)) ** 3).cpu().numpy()


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True); ap.add_argument('--s', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--levels', type=int, default=0)
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
        M = cell_moments(cells, n, taus, normal, offset, s, levels=a.levels)
        torch.cuda.synchronize(); sec = time.perf_counter() - t
        rec = dict(case=a.case, s=s, elements=int(len(cells)), seconds=sec)
        if a.compare:
            ref = np.load(Path(a.compare) / f'POLYREF_S{s}.npz')['moments']
            rec['max_rel_vs_numpy'] = float((np.abs(M - ref).max(1) / np.abs(ref).max(1).clip(1e-300)).max())
        np.savez(out / f'POLYREF_TORCH_S{s}_L{a.levels}.npz', moments=M)
        print(json.dumps(rec), flush=True)
