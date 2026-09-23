"""Step 0: the lattice acceptance harness (v3) on the port set box U cut band.

Configurations (box coordinates, cell at the origin = the cell under test, its neighbour glued by absolute node position):
  x: neighbour at (-1, 0, 0), its far face x = -1 clamped; face loads on the plane y = 0
  y: neighbour at (0, -1, 0), its far face y = -1 clamped; face loads on the plane x = 0
Loads (unit total force, uniform over the loaded nodes), per configuration:
  gate:    test-cell face x/y/z and neighbour face x/y/z on the load plane (6)
  report:  uniform traction x/y/z on the test cell's cut surface (3; consistent nodal forces of the Q2 trace, from
           quadrature points on the plane inside material; only if the test cell is cut)
Cut-band DOFs are private to their cell (free unknowns of the lattice: a free cut surface unless loaded).
Reference: dense lattice stiffness from the exact dense port operators (fp64), Cholesky.
Operator under test: preconditioned CG with the reference factor (the spectrum lies in [1, max ratio] for an upper-bound
operator), matvec through op.apply; sensitivities -u^T dK/dtau_c u with u = op.field(q) of each cell.
Gate metrics: compliance relative error per load; sensitivity: per cell and load, the relative 2-norm error of the
8-corner vector (also the componentwise max over components >= 5% of the largest).
"""
import json, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import ops as OP

dev, dt = TE.dev, TE.dt


def plane_traction_weights(cell, pts_per_elem=6):
    """Consistent nodal weights of a uniform unit traction on the cut plane inside material: sum_i w_i N_i."""
    from element_polyref import CUBE
    n = cell.n
    nrm = np.asarray(cell.normal, float); nrm = nrm / np.linalg.norm(nrm)
    off = cell.offset
    a = np.eye(3)[np.argmin(np.abs(nrm))]
    t1 = np.cross(nrm, a); t1 /= np.linalg.norm(t1); t2 = np.cross(nrm, t1)
    h = 1.0 / (n * pts_per_elem)
    x0 = nrm * off
    R = np.sqrt(3.0)
    s = np.arange(-R, R + h, h) + h / 2
    S1, S2 = np.meshgrid(s, s, indexing='ij')
    P = x0[None] + S1.reshape(-1, 1) * t1[None] + S2.reshape(-1, 1) * t2[None]
    P = P[(P >= 0).all(1) & (P <= 1).all(1)]
    f = np.cos(2 * np.pi * P).sum(1)
    cube = np.asarray(CUBE, float)
    w8 = np.where(cube[:, None, :].astype(bool), P[None], 1 - P[None]).prod(-1)
    tau = np.asarray(cell.taus, float) @ w8
    P = P[np.abs(f) <= tau]
    c = np.minimum(np.floor(P * n).astype(np.int64), n - 1)
    lookup = {tuple(x): k for k, x in enumerate(cell.cells.tolist())}
    keep = np.asarray([tuple(x) in lookup for x in c.tolist()])
    P, c = P[keep], c[keep]
    xi = P * n - c
    L = np.stack([2 * (xi - .5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - .5)], -1)       # m x 3(axis) x 3(o)
    w = np.zeros(len(cell.nodes))
    M = 2 * n + 1
    for o in np.ndindex(3, 3, 3):
        g = 2 * c + np.asarray(o)
        ids = (g[:, 0] * M + g[:, 1]) * M + g[:, 2]
        loc = np.searchsorted(cell.nodes, ids)
        np.add.at(w, loc, h * h * L[:, 0, o[0]] * L[:, 1, o[1]] * L[:, 2, o[2]])
    return w, float(h * h * len(P))


class Lattice:
    def __init__(self, cells, config, n=32, log=print):
        """cells: list of dict(label, cell (teacher.Cell, assembled), offset (3,), T (dense port operator, host/device))."""
        self.cells, self.config, self.n, self.log = cells, config, n, log
        keys, idx, pos = {}, [], []
        for cd in cells:
            C = cd['cell']
            g = np.stack(np.unravel_index(C.port_node_ids, (2 * n + 1,) * 3), 1) + 2 * n * np.asarray(cd['offset'])
            gg = np.repeat(g, 3, 0); comp = np.tile(np.arange(3), len(g))
            priv = np.repeat(C.port_is_cut & ~C.port_is_box, 3)          # cut-band DOFs are private to the cell
            k = np.empty(len(gg), dtype=np.int64)
            for r in range(len(gg)):
                key = (int(gg[r, 0]), int(gg[r, 1]), int(gg[r, 2]), int(comp[r])) if not priv[r] else (cd['label'], r)
                if key not in keys:
                    keys[key] = len(keys)
                k[r] = keys[key]
            idx.append(k); pos.append((gg, comp, priv))
        N = len(keys)
        ax = 'xy'.index(config)
        clamped = np.zeros(N, dtype=bool)
        for (gg, comp, priv), k in zip(pos, idx):
            clamped[k[(gg[:, ax] == -2 * n) & ~priv]] = True
        self.free = np.flatnonzero(~clamped)
        self.fmap = np.full(N, -1); self.fmap[self.free] = np.arange(len(self.free))
        self.idx, self.pos, self.N = idx, pos, N
        lp = 1 - ax                                                     # load plane: y = 0 for config x, x = 0 for y
        F, labels, gate = [], [], []
        for mi in range(len(cells)):
            gg, comp, priv = pos[mi]
            for d in range(3):
                on = (gg[:, lp] == 0) & (comp == d) & ~priv
                if not on.any():
                    raise ValueError('EMPTY_LOAD')
                f = np.zeros(N); f[idx[mi][on]] = 1.0; f /= f.sum()
                F.append(f); labels.append(f'{cells[mi]["label"]}_face{"xy"[lp]}0_{"xyz"[d]}'); gate.append(True)
        C0 = cells[0]['cell']
        if C0.is_cut.any():
            w, area = plane_traction_weights(C0)
            wn = w[np.isin(C0.nodes, C0.port_node_ids)]                  # port nodes, NODES order
            self.cut_load_outside = float(1 - wn.sum() / w.sum())
            for d in range(3):
                f = np.zeros(N)
                sel = np.arange(len(wn)) * 3 + d
                np.add.at(f, idx[0][sel], wn)
                f /= f.sum()
                F.append(f); labels.append(f'{cells[0]["label"]}_cut_traction_{"xyz"[d]}'); gate.append(False)
            self.cut_area = area
        F = np.stack(F, 1); F[clamped] = 0
        self.F = torch.as_tensor(F[self.free], dtype=dt, device=dev)
        self.labels, self.gate = labels, np.asarray(gate)
        self.gather_idx = [torch.as_tensor(self.fmap[k], device=dev) for k in idx]

    # ------------------------------------------------------------ lattice vectors <-> cell port vectors
    def gather(self, U, i):
        f = self.gather_idx[i]
        q = torch.zeros((len(f), U.shape[1]), dtype=dt, device=dev)
        keep = f >= 0
        q[keep] = U[f[keep]]
        return q

    def scatter_add(self, Y, y, i):
        f = self.gather_idx[i]; keep = f >= 0
        Y.index_add_(0, f[keep], y[keep])

    def matvec(self, ops, U):
        Y = torch.zeros_like(U)
        for i, op in enumerate(ops):
            self.scatter_add(Y, op.apply(self.gather(U, i)), i)
        return Y

    # ------------------------------------------------------------ reference
    def reference(self):
        t0 = time.perf_counter()
        nf = len(self.free)
        K = torch.zeros((nf, nf), dtype=dt, device=dev)
        for i, cd in enumerate(self.cells):
            f = self.gather_idx[i]; keep = torch.nonzero(f >= 0).squeeze(1); fc = f[keep]
            T = cd['T']
            for r0 in range(0, len(keep), 2048):
                rows = keep[r0:r0 + 2048]
                blk = T[rows.to(T.device)].to(dev, dt)[:, keep]
                K.index_put_((f[rows][:, None], fc[None, :]), blk, accumulate=True)
                del blk
        gc.collect(); torch.cuda.empty_cache()
        info = torch.empty((), dtype=torch.int32, device=dev)
        torch.linalg.cholesky_ex(K, out=(K, info))                        # in place: one dense copy only
        if int(info) != 0:
            raise ValueError(f'LATTICE_CHOLESKY_INFO:{int(info)}')
        self.L = K.tril_(); del K                                          # zero the stale upper triangle in place
        gc.collect(); torch.cuda.empty_cache()
        U = torch.cholesky_solve(self.F, self.L)
        self.ref = self._measure([OP.ExactOp(cd['cell'], cd['T']) for cd in self.cells], U)
        self.ref['seconds'] = time.perf_counter() - t0
        return self.ref

    def _measure(self, ops, U):
        Lh = None
        if getattr(self, 'L', None) is not None:                              # room for the field factorizations
            Lh = self.L.cpu(); self.L = None
            gc.collect(); torch.cuda.empty_cache()
        try:
            return self._measure_(ops, U)
        finally:
            if Lh is not None:
                self.L = Lh.to(dev); del Lh

    def _measure_(self, ops, U):
        c = (self.F * U).sum(0)
        out = dict(compliance=c.cpu().numpy(), energy=[], sens=[])
        for i, (cd, op) in enumerate(zip(self.cells, ops)):
            q = self.gather(U, i)
            out['energy'].append((q * op.apply(q)).sum(0).cpu().numpy())
            u = op.field(q)
            out['sens'].append(cd['cell'].sens(u).cpu().numpy())                 # 8 x loads
            if hasattr(cd['cell'], 'sol_I') and cd['cell'].sol_I is not None and not getattr(op, 'keep_factor', False):
                cd['cell']._free()
        out['energy'] = np.stack(out['energy']); out['sens'] = np.stack(out['sens'])
        out['U'] = U
        return out

    # ------------------------------------------------------------ operator under test
    def evaluate(self, ops, tol=1e-10, maxit=300):
        t0 = time.perf_counter()
        X = torch.zeros_like(self.F); R = self.F.clone()
        Z = torch.cholesky_solve(R, self.L); P = Z.clone()
        rz = (R * Z).sum(0); r0 = R.norm(dim=0)
        it = 0
        for it in range(1, maxit + 1):
            AP = self.matvec(ops, P)
            alpha = rz / (P * AP).sum(0)
            X += alpha * P; R -= alpha * AP
            if (R.norm(dim=0) / r0).max() < tol:
                break
            Z = torch.cholesky_solve(R, self.L)
            rz_new = (R * Z).sum(0)
            P = Z + (rz_new / rz) * P; rz = rz_new
        res = self._measure(ops, X)
        res['pcg_iterations'] = it
        res['pcg_residual'] = float((R.norm(dim=0) / r0).max())
        res['seconds'] = time.perf_counter() - t0
        return res

    def compare(self, res):
        ref = self.ref
        ce = np.abs(res['compliance'] / ref['compliance'] - 1)
        out = dict(loads=self.labels, gate=self.gate.tolist(), compliance_rel_err=ce.tolist(),
                   pcg_iterations=res.get('pcg_iterations'))
        s0, s1 = ref['sens'], res['sens']                                          # cells x 8 x loads
        vec = np.linalg.norm(s1 - s0, axis=1) / np.linalg.norm(s0, axis=1)       # cells x loads
        comp = np.zeros_like(vec)
        for ci in range(s0.shape[0]):
            for j in range(s0.shape[2]):
                a = np.abs(s0[ci, :, j]); m = a >= 0.05 * a.max()
                comp[ci, j] = np.max(np.abs(s1[ci, m, j] / s0[ci, m, j] - 1))
        out['sens_vec_rel_err'] = vec.tolist(); out['sens_comp_rel_err'] = comp.tolist()
        g = self.gate
        out['gate_compliance_max'] = float(ce[g].max())
        out['gate_sens_max'] = float(vec[:, g].max())
        out['gate_pass'] = bool(ce[g].max() <= 0.03 and vec[:, g].max() <= 0.03)
        if (~g).any():
            out['cut_compliance_max'] = float(ce[~g].max()); out['cut_sens_max'] = float(vec[:, ~g].max())
        out['energy_share_test_cell'] = (ref['energy'][0] / ref['compliance']).tolist()
        return out


def dense_T(cell, body_dir, host):
    import schur_encoder as SE
    import os
    case = cell.case
    pd = Path(body_dir) / (case + '_portview'); pd.mkdir(exist_ok=True)
    for fn in ('NODES.npy', 'CELL_INDICES.npy', 'dofs.npy', 'GP_FACES.npy'):
        if not (pd / fn).exists():
            os.symlink(Path(body_dir) / case / fn, pd / fn)
    np.save(pd / 'BOX_NODES.npy', cell.port_node_ids)
    enc = SE.SchurEncoder(case, body_dir, precision='fp64', log=lambda s_: None, sub=case + '_portview', out_host=host)
    T, _ = enc.update()
    T = T.T.contiguous() if host else T.T.contiguous()
    if host:
        T = T.cpu()
    enc.free(); enc.Tbuf = None; del enc
    gc.collect(); torch.cuda.empty_cache()
    return T


_CACHE = {}


def prepared(case, body_dir):
    """teacher.Cell (assembled, dM) and its dense port operator on the host, cached per case."""
    if case not in _CACHE:
        C = TE.Cell(case, body_dir, log=lambda s_: None)
        C.assemble(); C.dmoments()
        _CACHE[case] = (C, dense_T(C, body_dir, host=True))
    return _CACHE[case]


def build(test_case, neighbour_case, config, body_dir, log=print):
    cells = []
    for label, case, off in (('test', test_case, (0, 0, 0)), ('nbr', neighbour_case, (-1, 0, 0) if config == 'x' else (0, -1, 0))):
        C, T = prepared(case, body_dir)
        cells.append(dict(label=label, cell=C, offset=off, T=T))
    return Lattice(cells, config, log=log)
