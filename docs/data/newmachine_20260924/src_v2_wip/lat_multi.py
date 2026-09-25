"""Multi-cell lattice on the port DOFs (generalises lattice3's two-cell gluing to an arbitrary arrangement of cells).

Layout: dict {(i, j, k) integer cell position: CellGeom}. Every cell is a unit TPMS cell on the (2n+1)^3 node grid (node id =
ravel of the grid coordinates, as teacher.Cell / lattice3); a port node of the cell at position p has the ABSOLUTE grid
coordinates g + 2n p. Port DOFs are glued by absolute node key (gx, gy, gz, component): cells sharing a face share the
face's port nodes (the union of both cells' node sets: a node present in one cell only is that cell's unknown, a free
boundary for the other, as in lattice3). Cut-band DOFs (port_is_cut & ~port_is_box) are private to their cell. Numbering
= first appearance (cell order, then the cell's port DOF order), so the pair lattice reproduces lattice3's numbering.
Lattice unknowns = all glued DOFs minus the clamped ones (the non-private DOFs on one lattice face, default x = min).
Loads (columns of F on the free DOFs): 'consistent' = the traction-consistent nodal weights of a unit uniform traction
x / y / z on the load face (lattice3.face_traction_weights per cell, summed over the cells on the face, each normalised to
unit total force), 'uniform' = equal nodal forces on the non-private DOFs on the load face; plus n_random iid Gaussian
loads (unit 2-norm, seeded).
Operators: ops[i] is the operator of cell i (apply(q): port reaction, q = (3 * port nodes, k), node-major xyz). Cells
sharing one operator OBJECT are batched: their gathered vectors are concatenated as columns and applied once (in chunks
of max_cols columns). matmat_sparse applies the lattice operator to a matrix with few nonzero columns per cell (coarse
bases): each cell only applies the operator to the columns it touches.
Port stiffness K_PP (for Jacobi / the sparse fine level): each cell's assembled K restricted to its port DOFs (interior
DOFs clamped), summed over the lattice on the free DOFs (upper triplets).
No factorisation / GPU dependence here: CPU tests build synthetic CellGeoms (t_lat_precond.py); from_teacher() adapts a
teacher.Cell (assembled).
"""
import os
import numpy as np
import torch

dt = torch.float64
AXES = {'x': 0, 'y': 1, 'z': 2}


def default_device():
    return torch.device(os.environ.get('OPL_DEV', 'cuda:0' if torch.cuda.is_available() else 'cpu'))


class CellGeom:
    """What the lattice needs from one cell KIND (a case), shared by all cells of that kind.
    port_node_ids: sorted node ids on the (2n+1)^3 grid (port DOF r = 3 * port node + component);
    kpp_fn() -> upper triplets (r, c, v) of K_PP on the local port DOFs (r <= c), torch;
    face_w_fn(axis, value) -> consistent traction weights per port node on the local face x_axis = value (0 / 1)."""

    def __init__(self, case, n, port_node_ids, port_is_box, port_is_cut, kpp_fn=None, face_w_fn=None, normal=None,
                 offset=None, cell=None):
        self.case, self.n, self.period = case, int(n), 2 * int(n)
        self.port_node_ids = np.asarray(port_node_ids, dtype=np.int64)
        self.grid = np.stack(np.unravel_index(self.port_node_ids, (2 * self.n + 1,) * 3), 1).astype(np.int64)
        self.priv = np.asarray(port_is_cut, bool) & ~np.asarray(port_is_box, bool)
        self.nport = 3 * len(self.port_node_ids)
        self._kpp_fn, self._kpp = kpp_fn, None
        self.face_w_fn = face_w_fn
        self.normal, self.offset, self.cell = normal, offset, cell

    def kpp(self):
        if self._kpp is None:
            if self._kpp_fn is None:
                raise ValueError(f'NO_KPP:{self.case}')
            self._kpp = self._kpp_fn()
        return self._kpp


def from_teacher(C):
    """CellGeom of an assembled teacher.Cell (K_PP from its upper CSR; consistent face weights from lattice3)."""
    def kpp():
        dev = C.vals.device
        pm = torch.zeros(C.nb, dtype=torch.bool, device=dev); pm[C.P.to(dev)] = True
        pnew = torch.full((C.nb,), -1, dtype=torch.long, device=dev); pnew[C.P.to(dev)] = torch.arange(C.np_, device=dev)
        ru, cu = C.ru.to(dev).long(), C.cu.to(dev).long()
        sel = torch.nonzero(pm[ru] & pm[cu] & (C.vals != 0)).squeeze(1)                # (the stored pattern holds zeros)
        return pnew[ru[sel]], pnew[cu[sel]], C.vals[sel]                                # P sorted: upper stays upper

    onport = np.isin(C.nodes, C.port_node_ids)

    def face_w(axis, value):
        import lattice3 as LT
        wv = LT.face_traction_weights(C, axis, float(value))
        return wv[onport], float(1 - wv[onport].sum() / max(wv.sum(), 1e-300))

    return CellGeom(C.case, C.n, C.port_node_ids, C.port_is_box, C.port_is_cut, kpp_fn=kpp, face_w_fn=face_w,
                    normal=C.normal, offset=C.offset, cell=C)


def _plane(offsets, axis, side, period):
    o = np.asarray(offsets)[:, axis]
    return (int(o.min()) * period, int(o.min())) if side == 'min' else ((int(o.max()) + 1) * period, int(o.max()))


class MultiLattice:
    def __init__(self, layout, clamp=('x', 'min'), load=('x', 'max'), loads='consistent', n_random=3, seed=0,
                 device=None, max_cols=None, log=print):
        self.device = default_device() if device is None else torch.device(device)
        self.log, self.max_cols = log, max_cols
        self.positions = [tuple(int(v) for v in p) for p in layout]
        self.geoms = [layout[p] for p in layout]
        periods = {g.period for g in self.geoms}
        if len(periods) != 1:
            raise ValueError('ONE_GRID_PERIOD_EXPECTED')
        P = self.period = periods.pop()
        offs = np.asarray(self.positions, np.int64)
        # ------------------------------------------------ glue by absolute node key (vectorised; first-appearance order)
        g_all, c_all, pr_all, cell_all = [], [], [], []
        for i, (p, G) in enumerate(zip(self.positions, self.geoms)):
            g = G.grid + P * np.asarray(p)
            g_all.append(np.repeat(g, 3, 0)); c_all.append(np.tile(np.arange(3), len(g)))
            pr_all.append(np.repeat(G.priv, 3)); cell_all.append(np.full(3 * len(g), i))
        g_all, c_all, pr_all, cell_all = map(np.concatenate, (g_all, c_all, pr_all, cell_all))
        gs = g_all - g_all.min(0)
        M = gs.max(0) + 1
        key = ((gs[:, 0] * M[1] + gs[:, 1]) * M[2] + gs[:, 2]) * 3 + c_all
        big = int(M.prod()) * 3
        key = np.where(pr_all, big + np.arange(len(key)), key)                       # private: a key of its own
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        rank = np.empty(len(uniq), np.int64); rank[np.argsort(first, kind='stable')] = np.arange(len(uniq))
        glob = rank[inv]
        self.N = N = len(uniq)
        bounds = np.cumsum([0] + [G.nport for G in self.geoms])
        self.idx = [glob[bounds[i]:bounds[i + 1]] for i in range(len(self.geoms))]
        gpos = np.zeros((N, 3), np.int64); gpos[glob] = g_all
        comp = np.zeros(N, np.int64); comp[glob] = c_all
        priv = np.zeros(N, bool); priv[glob] = pr_all
        mult = np.bincount(glob, minlength=N)
        # ------------------------------------------------ clamp
        ca = AXES[clamp[0]]
        cplane, _ = _plane(offs, ca, clamp[1], P)
        clamped = (gpos[:, ca] == cplane) & ~priv
        self.clamp, self.load_face = clamp, load
        self.free = np.flatnonzero(~clamped)
        self.nfree = len(self.free)
        self.fmap = np.full(N, -1, np.int64); self.fmap[self.free] = np.arange(self.nfree)
        dev = self.device
        self.gather_idx = [torch.as_tensor(self.fmap[k], device=dev) for k in self.idx]
        self._keep = []
        for f in self.gather_idx:
            kp = torch.nonzero(f >= 0).squeeze(1)
            self._keep.append((kp, f[kp]))
        self.xyz = torch.as_tensor(gpos[self.free] / P, dtype=dt, device=dev)           # free DOF positions, cell units
        self.comp = torch.as_tensor(comp[self.free], device=dev)
        self.mult = torch.as_tensor(mult[self.free], dtype=dt, device=dev)               # cells sharing the DOF
        self.priv = priv[self.free]
        # ------------------------------------------------ loads
        la = AXES[load[0]]
        lplane, lcell = _plane(offs, la, load[1], P)
        F, labels, self.face_load_outside = [], [], []
        if loads == 'consistent':
            Fc = np.zeros((N, 3))
            for i, (p, G) in enumerate(zip(self.positions, self.geoms)):
                if p[la] != lcell:
                    continue
                w, out = G.face_w_fn(la, 0.0 if load[1] == 'min' else 1.0)
                self.face_load_outside.append(out)
                for d in range(3):
                    np.add.at(Fc[:, d], self.idx[i][d::3], w)
            Fc[clamped] = 0
        elif loads == 'uniform':
            Fc = np.zeros((N, 3))
            on = (gpos[:, la] == lplane) & ~priv & ~clamped
            for d in range(3):
                Fc[on & (comp == d), d] = 1.0
        else:
            raise ValueError(f'LOADS:{loads}')
        for d in range(3):
            s = Fc[:, d].sum()
            if not s > 0:
                raise ValueError('EMPTY_LOAD')
            F.append(Fc[self.free, d] / s); labels.append(f'{loads}_{load[0]}{load[1]}_{"xyz"[d]}')
        rng = np.random.default_rng(seed)
        for r in range(n_random):
            f = rng.standard_normal(self.nfree)
            F.append(f / np.linalg.norm(f)); labels.append(f'random{r}')
        self.F = torch.as_tensor(np.stack(F, 1), dtype=dt, device=dev)
        self.labels = labels

    # ------------------------------------------------------------------ lattice vectors <-> cell port vectors
    def gather(self, U, i):
        kp, fk = self._keep[i]
        q = torch.zeros((self.geoms[i].nport, U.shape[1]), dtype=U.dtype, device=U.device)
        q[kp] = U[fk]
        return q

    def scatter_add(self, Y, y, i):
        kp, fk = self._keep[i]
        Y.index_add_(0, fk, y[kp].to(Y.dtype))

    def groups(self, ops, batch=True):
        """[(op, [cell indices])]: cells sharing one operator object form one group (batch=False: one group per cell)."""
        if not batch:
            return [(op, [i]) for i, op in enumerate(ops)]
        out, where = [], {}
        for i, op in enumerate(ops):
            if id(op) not in where:
                where[id(op)] = len(out); out.append((op, []))
            out[where[id(op)]][1].append(i)
        for op, cells in out:
            if len({self.geoms[i].nport for i in cells}) != 1:
                raise ValueError('SHARED_OPERATOR_ON_DIFFERENT_PORT_SETS')
        return out

    def apply_blocks(self, ops, blocks, batch=True, max_cols=None):
        """blocks[i]: (nport_i, k_i) port matrix of cell i (k_i may differ, 0 = skip) -> [S_i blocks[i]], one apply per
        operator object (columns concatenated, chunks of max_cols)."""
        mc = max_cols or self.max_cols
        outs = [None] * len(blocks)
        for op, cells in self.groups(ops, batch):
            cells = [i for i in cells if blocks[i] is not None and blocks[i].shape[1] > 0]
            if not cells:
                continue
            X = torch.cat([blocks[i] for i in cells], 1)
            if mc is None or X.shape[1] <= mc:
                Y = op.apply(X)
            else:
                Y = torch.cat([op.apply(X[:, c0:c0 + mc]) for c0 in range(0, X.shape[1], mc)], 1)
            c0 = 0
            for i in cells:
                k = blocks[i].shape[1]
                outs[i] = Y[:, c0:c0 + k]; c0 += k
        return outs

    def matvec(self, ops, U, batch=True, max_cols=None):
        outs = self.apply_blocks(ops, [self.gather(U, i) for i in range(len(self.geoms))], batch, max_cols)
        Y = torch.zeros_like(U)
        for i, y in enumerate(outs):
            self.scatter_add(Y, y, i)
        return Y

    def matmat_sparse(self, ops, Z, batch=True, max_cols=None):
        """A Z for a (free x m) matrix whose columns touch few cells: cell i applies its operator only to the columns
        that are nonzero on its ports."""
        blocks, cols = [], []
        for i in range(len(self.geoms)):
            Zi = self.gather(Z, i)
            J = torch.nonzero(Zi.abs().sum(0) > 0).squeeze(1)
            blocks.append(Zi[:, J]); cols.append(J)
        outs = self.apply_blocks(ops, blocks, batch, max_cols)
        AZ = torch.zeros_like(Z)
        for i, (y, J) in enumerate(zip(outs, cols)):
            if y is None:
                continue
            kp, fk = self._keep[i]
            AZ.index_put_((fk[:, None], J[None, :]), y[kp].to(Z.dtype), accumulate=True)
        return AZ

    # ------------------------------------------------------------------ assembled port stiffness
    def assemble_kpp(self):
        """Upper triplets (rows, cols, vals) of the assembled K_PP on the free DOFs (coalesced, sorted by row then col)."""
        R, Cc, V = [], [], []
        for i, G in enumerate(self.geoms):
            r, c, v = G.kpp()
            f = self.gather_idx[i].to(r.device)
            fr, fc = f[r], f[c]
            ok = (fr >= 0) & (fc >= 0)
            fr, fc = fr[ok], fc[ok]
            R.append(torch.minimum(fr, fc)); Cc.append(torch.maximum(fr, fc)); V.append(v[ok].to(dt))
        r, c, v = torch.cat(R), torch.cat(Cc), torch.cat(V)
        key, inv = torch.unique(r * self.nfree + c, return_inverse=True)
        vals = torch.zeros(len(key), dtype=dt, device=v.device).index_add_(0, inv, v)
        rows, cols = key // self.nfree, key % self.nfree
        return rows.to(self.device), cols.to(self.device), vals.to(self.device)

    def kpp_dense(self):
        """Dense symmetric K_PP (tests only)."""
        r, c, v = self.assemble_kpp()
        K = torch.zeros((self.nfree, self.nfree), dtype=dt, device=self.device)
        K.index_put_((r, c), v, accumulate=True)
        off = r != c
        K.index_put_((c[off], r[off]), v[off], accumulate=True)
        return K

    def info(self):
        return dict(cells=len(self.geoms), distinct_cases=sorted({G.case for G in self.geoms}), glued_dofs=int(self.N),
                    free_dofs=int(self.nfree), private_dofs=int(self.priv.sum()), loads=int(self.F.shape[1]),
                    labels=self.labels, clamp=list(self.clamp), load_face=list(self.load_face),
                    positions=[list(p) for p in self.positions], cases=[G.case for G in self.geoms],
                    face_load_outside=self.face_load_outside)


# ---------------------------------------------------------------------- layouts
def block_layout(shape, geom, extra=None):
    """nx x ny x nz cells of one CellGeom at positions (0..nx-1, ...); extra: {pos: CellGeom} added / overriding."""
    lay = {(i, j, k): geom for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])}
    lay.update(extra or {})
    return lay


def cut_layer(shape, full, cut):
    """The FULL block plus one layer of cut cells on the block face the cut normal points to (the removed side of the cut
    plane faces out of the lattice). Returns (layout, cut axis, side)."""
    nrm = np.asarray(cut.normal if cut.normal is not None else [0, 0, 1], float)
    a = int(np.argmax(np.abs(nrm)))
    side = 'max' if nrm[a] > 0 else 'min'
    lay = block_layout(shape, full)
    rng = [range(s) for s in shape]
    rng[a] = [shape[a]] if side == 'max' else [-1]
    for i in rng[0]:
        for j in rng[1]:
            for k in rng[2]:
                lay[(i, j, k)] = cut
    return lay, 'xyz'[a], side
