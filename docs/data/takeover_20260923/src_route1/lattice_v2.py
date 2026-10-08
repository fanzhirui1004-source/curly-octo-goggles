"""Lattice acceptance harness for second-batch (CUT_COVER80) cells: compliance and design sensitivities of a small
assembly in which one cut cell uses an approximate boundary operator and every other cell is exact.

Coordinates. A packet's trace coordinates are the complete box Q2 nodal values (node-major xyz, first
3 * scalar_box_trace_dimension coordinates; TRACE.npz box_boundary_original maps them to background DOFs) followed by
the cut cell's exact residual functionals on its cut surface (free surface of the part: never glued, never loaded).
Box coordinates of different cells are glued by their absolute grid position (half-cell units) and component.

Configuration ('x' or 'y'): the cut cell at the origin and its FULL parent as neighbour at -x (or -y), glued on the
shared face; the neighbour's far face (global x = -1, or y = -1) is clamped. Loads: uniform nodal forces in x, y and
z on the free box face of the cut cell that lies in the plane y = 0 (or x = 0), and on the neighbour's face in the
same plane: a cantilever whose free end is the cut cell. Six load cases.

Reference: exact packet operators for every cell, assembled densely on the free coordinates and Cholesky-factored.
Approximate: the cut cell's operator replaced by the skeleton's boundary action S_hat (matrix-free), solved with CG
preconditioned by the exact lattice factor, so the preconditioned spectrum lies in [1, max Rayleigh ratio].
Outputs per variant: compliance per load case, every cell's strain energy e_m = q_m^T S_m q_m (the design
sensitivity of a cell stiffness multiplier is dc/drho_m = -e_m), the energy split, and relative errors.
"""
import json, time
from pathlib import Path
import numpy as np
import torch
from scipy import linalg

PACKETS = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets')


def packed_upper_dense(path, n):
    p = np.load(path, mmap_mode='r')
    if p.shape[0] != n * (n + 1) // 2:
        raise ValueError('PACKED_LENGTH')
    S = np.empty((n, n))
    off = 0
    for i in range(n):
        S[i, i:] = p[off:off + n - i]; off += n - i
    B = 1024
    for i0 in range(0, n, B):
        i1 = min(n, i0 + B)
        S[i1:, i0:i1] = S[i0:i1, i1:].T
        blk = S[i0:i1, i0:i1]
        S[i0:i1, i0:i1] = np.triu(blk) + np.triu(blk, 1).T
    return S


def box_positions(case, n=32):
    z = np.load(PACKETS / case / 'TRACE.npz')
    bb, bn = z['box_boundary_original'], z['background_nodes']
    grid = np.stack(np.unravel_index(bn[bb // 3], (2 * n + 1,) * 3), axis=1)
    return grid, bb % 3


class Lattice:
    def __init__(self, modules, clamp, loads, n=32):
        """modules: list of (name, case, offset, S dense (m x m) or None for the approximated cell);
        clamp: (axis, plane) in global cell units; loads: list of (label, axis, plane, direction, module filter)."""
        self.n = n
        keys, self.idx, self.pos = {}, [], []
        nxt = 0
        for name, case, offset, S in modules:
            grid, comp = box_positions(case, n)
            m = S.shape[0] if S is not None else None
            g = grid + 2 * n * np.asarray(offset)
            ids = []
            for k in range(len(comp)):
                key = (int(g[k, 0]), int(g[k, 1]), int(g[k, 2]), int(comp[k]))
                if key not in keys:
                    keys[key] = nxt; nxt += 1
                ids.append(keys[key])
            self.pos.append((g, comp))
            self.idx.append(ids)            # box coordinates only for now; private coordinates appended below
        self.box_count = nxt
        self.m = []
        for (name, case, offset, S), ids in zip(modules, self.idx):
            m = int(json.loads((PACKETS / case / 'SAMPLE.json').read_text())['full_trace_dimension'])
            private = list(range(nxt, nxt + m - len(ids))); nxt += len(private)
            ids.extend(private); self.m.append(m)
        self.idx = [np.asarray(i) for i in self.idx]
        self.N = nxt
        # clamp and loads act on box coordinates of the given global planes
        clamped = np.zeros(self.N, dtype=bool)
        for (g, comp), ids in zip(self.pos, self.idx):
            on = g[:, clamp[0]] == 2 * n * clamp[1]
            clamped[ids[:len(comp)][on]] = True
        self.free = np.flatnonzero(~clamped)
        self.fmap = np.full(self.N, -1); self.fmap[self.free] = np.arange(len(self.free))
        F = np.zeros((self.N, len(loads)))
        self.load_labels = []
        for j, (label, axis, plane, direction, which) in enumerate(loads):
            self.load_labels.append(label)
            for mi, ((g, comp), ids) in enumerate(zip(self.pos, self.idx)):
                if which is not None and mi not in which:
                    continue
                on = (g[:, axis] == 2 * n * plane) & (comp == direction)
                F[ids[:len(comp)][on], j] = 1.0
            F[:, j] /= max(F[:, j].sum(), 1.0)
        F[clamped] = 0
        self.F = F[self.free]
        self.modules = modules

    def assemble_free(self, ops):
        K = np.zeros((len(self.free), len(self.free)))
        for ids, S in zip(self.idx, ops):
            f = self.fmap[ids]; keep = f >= 0
            K[np.ix_(f[keep], f[keep])] += S[np.ix_(keep, keep)]
        return K

    def module_q(self, U, mi):
        """module coordinates of free-coordinate solutions U (free x loads)."""
        f = self.fmap[self.idx[mi]]
        q = np.zeros((len(f), U.shape[1])); keep = f >= 0
        q[keep] = U[f[keep]]
        return q


def pcg(apply_K, apply_M, F, x0, tol=1e-11, maxit=200):
    """block of independent PCG runs (one per column), torch tensors on one device."""
    x = x0.clone(); r = F - apply_K(x)
    z = apply_M(r); p = z.clone(); rz = (r * z).sum(0)
    nF = F.norm(dim=0)
    hist = []
    for it in range(1, maxit + 1):
        Kp = apply_K(p)
        alpha = rz / (p * Kp).sum(0)
        x += alpha * p; r -= alpha * Kp
        rel = float((r.norm(dim=0) / nF).max()); hist.append(rel)
        if rel < tol:
            break
        z = apply_M(r); rz_new = (r * z).sum(0)
        p = z + (rz_new / rz) * p; rz = rz_new
    return x, it, hist


def run(case, variants, out, config='x', log=print, threads=16, device='cuda:0'):
    """variants: list of (label, apply) where apply(q: torch tensor m x k on device) -> S_hat q for the cut cell.
    Dense lattice algebra on the GPU: the free-coordinate matrix is Cholesky-factored once; K x = L (L^T x)."""
    t0 = time.perf_counter()
    dev = torch.device(device); dt = torch.float64
    full = case.replace('_cover01_r1', '_full')
    m_full = int(json.loads((PACKETS / full / 'SAMPLE.json').read_text())['full_trace_dimension'])
    m_cut = int(json.loads((PACKETS / case / 'SAMPLE.json').read_text())['full_trace_dimension'])
    S_full = packed_upper_dense(PACKETS / full / 'S_UPPER.npy', m_full)
    S_cut = packed_upper_dense(PACKETS / case / 'S_UPPER.npy', m_cut)
    ax = 0 if config == 'x' else 1           # glue axis
    other = 1 - ax                           # the load plane: y = 0 for config x, x = 0 for config y
    offset = [0, 0, 0]; offset[ax] = -1
    modules = [('cut', case, (0, 0, 0), S_cut), ('full', full, tuple(offset), S_full)]
    loads = [(f'cut_face_{"xyz"[d]}', other, 0, d, [0]) for d in range(3)] + \
            [(f'neighbour_face_{"xyz"[d]}', other, 0, d, [1]) for d in range(3)]
    lat = Lattice(modules, (ax, -1), loads)
    rec = dict(case=case, full=full, config=config, N=int(lat.N), free=int(len(lat.free)), m=lat.m,
               glued_box_coordinates=int(sum(len(i) for i in lat.idx) - lat.N), loads=lat.load_labels,
               load_nonzeros=[int((lat.F[:, j] != 0).sum()) for j in range(lat.F.shape[1])])
    log(json.dumps(dict(event='LATTICE', **rec)))
    if min(rec['load_nonzeros']) == 0:
        raise ValueError('EMPTY_LOAD')
    ops = [S_cut, S_full]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(threads):
        t = time.perf_counter()
        K = lat.assemble_free(ops)
        rec['assemble_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    Lt = torch.from_numpy(K).to(dev); del K
    Lt = torch.linalg.cholesky(Lt)            # in place of the matrix: only the factor stays on the GPU
    torch.cuda.synchronize(); rec['factor_seconds'] = time.perf_counter() - t
    Kmul = lambda X: Lt @ (Lt.T @ X)
    Minv = lambda R: torch.cholesky_solve(R, Lt)
    F = torch.from_numpy(lat.F).to(dev)
    U = Minv(F)
    rec['exact_residual'] = float((Kmul(U) - F).norm() / F.norm())
    c = (F * U).sum(0)
    f_idx = [torch.as_tensor(lat.fmap[lat.idx[mi]], device=dev) for mi in range(2)]
    if bool((f_idx[0] < 0).any()):
        raise ValueError('CUT_CELL_CLAMPED')
    Sc = torch.from_numpy(S_cut).to(dev); Sf = torch.from_numpy(S_full).to(dev)
    def module_q(X, mi):
        f = f_idx[mi]; q = torch.zeros((len(f), X.shape[1]), dtype=dt, device=dev); keep = f >= 0
        q[keep] = X[f[keep]]
        return q
    q = [module_q(U, mi) for mi in range(2)]
    e = torch.stack([(q[0] * (Sc @ q[0])).sum(0), (q[1] * (Sf @ q[1])).sum(0)])
    rec['exact'] = dict(compliance=c.tolist(), energy=e.tolist(), cut_energy_share=(e[0] / e.sum(0)).tolist(),
                        energy_identity=float(((e.sum(0) - c).abs() / c).max()), residual=rec['exact_residual'])
    log(json.dumps(dict(event='EXACT', compliance=c.tolist(), cut_share=(e[0] / e.sum(0)).tolist(),
                        identity=rec['exact']['energy_identity'], residual=rec['exact_residual'],
                        factor_seconds=rec['factor_seconds'])))
    fc = f_idx[0]
    rows = []
    for label, apply in variants:
        t = time.perf_counter()
        def apply_K(X):
            Y = Kmul(X)
            Xc = X[fc]
            Y[fc] += apply(Xc) - Sc @ Xc
            return Y
        with torch.no_grad():
            Uh, it, hist = pcg(apply_K, Minv, F, U.clone())
            ch = (F * Uh).sum(0)
            qh = [module_q(Uh, mi) for mi in range(2)]
            eh = torch.stack([(qh[0] * apply(qh[0])).sum(0), (qh[1] * (Sf @ qh[1])).sum(0)])
            d1 = (q[0] * (apply(q[0]) - Sc @ q[0])).sum(0)
        row = dict(variant=label, iterations=it, residual_history=hist[-3:], seconds=time.perf_counter() - t,
                   compliance=ch.tolist(), compliance_rel_error=((ch - c) / c).tolist(),
                   compliance_rel_error_first_order=(-d1 / c).tolist(),
                   cut_along_exact_rel=(d1 / e[0]).tolist(),
                   sensitivity_rel_error=((eh - e) / e).tolist(), cut_energy_share=(eh[0] / eh.sum(0)).tolist())
        rows.append(row)
        log(json.dumps(dict(event='VARIANT', variant=label, iterations=it, seconds=round(row['seconds'], 1),
                            compliance_err=[round(x, 6) for x in row['compliance_rel_error']],
                            cut_sens_err=[round(x, 6) for x in row['sensitivity_rel_error'][0]],
                            full_sens_err=[round(x, 6) for x in row['sensitivity_rel_error'][1]])))
    rec['variants'] = rows
    rec['seconds'] = time.perf_counter() - t0
    (Path(out) / f'LATTICE_{config}.json').write_text(json.dumps(rec, indent=2))
    del Lt, Sc, Sf
    torch.cuda.empty_cache()
    return rec
