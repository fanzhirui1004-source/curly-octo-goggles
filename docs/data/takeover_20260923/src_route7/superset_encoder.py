"""Route 7: design-loop encoder on a superset sparsity pattern (fast_superset.py), so designs whose active cells and
ghost faces change inside the band need no new cuDSS plan.

Setup (once per band and box set): the pattern of K over all superset cells and all possible ghost faces; element
scatter map; the ghost values of ALL superset faces; A/C/D split for the box set of the setup design; the plan.
Per design (its own exact active cells, faces and box nodes from fast_prep3):
  - containment check (cells and faces inside the superset, same box set; otherwise the caller rebuilds),
  - moments of the design's active cells only, scattered at their superset rows,
  - ghost contributions of the superset faces that are not faces of this design subtracted,
  - superset nodes outside every active cell are decoupled (unit diagonal): interior unknowns without any coupling
    do not change the condensed box operator,
  - scaling, in-place values, cuDSS refactorization.
The box operator equals a fresh encode of the design on its own topology (checked in main).
Usage: superset_encoder.py <case> <body_dir> <superset_dir> <out_dir> <design_eps> [<design_eps> ...]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import design_encoder as DE
import box_encode as BX

dev, dt = DE.dev, DE.dt
ROOT = DE.ROOT


class SupersetEncoder:
    def __init__(self, case, sup_dir, box_nodes, precision='fp32', threads=16, s=4, levels=1, log=print):
        import element_moments as EM
        import polyref_torch_fast as PT
        from fractions import Fraction
        self.PT, self.s, self.levels, self.precision, self.threads, self.log = PT, s, levels, precision, threads, log
        t0 = time.perf_counter()
        ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
        self.case, self.n = case, int(ctx['n'])
        Emod = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
        lam = Emod * nu / ((1 + nu) * (1 - 2 * nu)); mu = Emod / (2 * (1 + nu))
        self.taus0 = [float(Fraction(v)) for v in ctx['case']['tau_corners']]
        self.normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
        self.offset = None if self.normal is None else float(ctx['case']['offset'])
        self.gamma = float(json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['gp']['gamma'])
        d = Path(sup_dir)
        nodes = np.load(d / 'NODES.npy'); cells = np.load(d / 'CELL_INDICES.npy'); dofs = np.load(d / 'dofs.npy')
        faces = np.load(d / 'GP_FACES.npy')
        n = self.n
        self.nodes_sup = nodes
        self.cell_keys = (cells[:, 0].astype(np.int64) * n + cells[:, 1]) * n + cells[:, 2]
        self.face_keys = self.cell_keys[faces[:, 0]] * 3 + faces[:, 2]
        self.cells_sup, self.faces_sup = cells, faces
        arr = {'NODES.npy': nodes, 'dofs.npy': dofs, 'CELL_INDICES.npy': cells}
        xi = EM.local_coordinates(case, n, arr)
        if len(np.unique(xi.reshape(len(xi), -1), axis=0)) != 1:
            raise ValueError('ONE_NODE_ORDERING_EXPECTED')
        Tm = torch.tensor(EM.pattern_operators(xi[0].reshape(27, 3), lam, mu, n)[1], dtype=dt, device=dev)
        iu = torch.triu_indices(81, 81, device=dev)
        self.Tm_up = Tm.reshape(125, 81, 81)[:, iu[0], iu[1]].contiguous()
        nb = 3 * len(nodes); self.nb = nb
        dofs_t = torch.as_tensor(dofs, dtype=torch.long, device=dev)
        r, c = dofs_t[:, iu[0]], dofs_t[:, iu[1]]
        key_e = torch.minimum(r, c) * nb + torch.maximum(r, c)
        # ghost faces of the superset: per-face upper keys (templates per axis)
        tpl = np.load(Path(sup_dir).parent / f'GP_TEMPLATES_n{n}.npz')
        canon = torch.as_tensor(tpl['canonical'], dtype=dt, device=dev)
        self.offs = torch.as_tensor(tpl['offsets'].astype(np.int64), device=dev)
        tk = canon.transpose(1, 2) @ canon
        self.iu135 = torch.triu_indices(tk.shape[1], tk.shape[1], device=dev)
        self.tk_up = tk[:, self.iu135[0], self.iu135[1]].contiguous()                  # 3 x 9180
        self.nodes_t = torch.as_tensor(nodes.astype(np.int64), device=dev)
        self.cells_t = torch.as_tensor(cells.astype(np.int64), device=dev)
        self.faces_t = torch.as_tensor(faces.astype(np.int64), device=dev)
        keys_g = []
        for lo in range(0, len(faces), 1024):
            keys_g.append(self.face_entry_keys(torch.arange(lo, min(lo + 1024, len(faces)), device=dev)).reshape(-1))
        key_g = torch.cat(keys_g); del keys_g
        self.U = torch.unique(torch.cat([key_e.reshape(-1), key_g]))
        self.pos_e = torch.searchsorted(self.U, key_e.reshape(-1)).reshape(len(cells), -1)
        del key_e
        self.base = torch.zeros(len(self.U), dtype=dt, device=dev)
        for lo in range(0, len(faces), 1024):                                            # all superset faces
            fi = torch.arange(lo, min(lo + 1024, len(faces)), device=dev)
            self.base.index_add_(0, torch.searchsorted(self.U, self.face_entry_keys(fi).reshape(-1)),
                                 self.gamma * self.tk_up[self.faces_t[fi, 2]].reshape(-1))
        del key_g
        ru, cu = self.U // nb, self.U % nb
        onbox = np.isin(nodes, box_nodes)
        if int(onbox.sum()) != len(box_nodes):
            raise ValueError('BOX_NODE_OUTSIDE_SUPERSET')
        self.box_nodes = np.asarray(box_nodes)
        self.box_node_ids = nodes[onbox]
        box = torch.as_tensor(np.repeat(onbox, 3), device=dev)
        self.box_dof = box
        bi = torch.nonzero(box).squeeze(1); ii = torch.nonzero(~box).squeeze(1)
        new = torch.empty(nb, dtype=torch.long, device=dev)
        new[bi] = torch.arange(len(bi), device=dev); new[ii] = torch.arange(len(ii), device=dev)
        self.new = new
        self.ni, self.nbx = len(ii), len(bi)
        br, bc = box[ru], box[cu]
        selA = torch.nonzero(~br & ~bc).squeeze(1)
        self.idxA = selA; self.rowA, self.colA = new[ru[selA]], new[cu[selA]]
        self.diagA = torch.nonzero(self.rowA == self.colA).squeeze(1)
        if len(self.diagA) != self.ni:
            raise ValueError('A_DIAGONAL_INCOMPLETE')
        counts = torch.bincount(self.rowA, minlength=self.ni)
        self.crowA = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(counts, 0)]).int()
        self.colA32 = self.colA.int()
        selC1 = torch.nonzero(~br & bc).squeeze(1); selC2 = torch.nonzero(br & ~bc).squeeze(1)
        self.idxC = torch.cat([selC1, selC2])
        self.C_rows = torch.cat([new[ru[selC1]], new[cu[selC2]]]); self.C_cols = torch.cat([new[cu[selC1]], new[ru[selC2]]])
        selD = torch.nonzero(br & bc).squeeze(1)
        self.idxD = selD; self.D_r, self.D_c = new[ru[selD]], new[cu[selD]]
        self.D_off = self.D_r != self.D_c
        self.solver = None
        DE.sync(); self.setup_seconds = time.perf_counter() - t0
        log(json.dumps(dict(event='SUPERSET_SETUP', case=case, seconds=self.setup_seconds, nnz_upper=int(len(self.U)),
                            interior=self.ni, box=self.nbx, cells=int(len(cells)), faces=int(len(faces)))))

    def face_entry_keys(self, fi):
        """upper-triangle pattern keys of superset faces fi (F x 9180)."""
        n, nb = self.n, self.nb
        g = 2 * self.cells_t[self.faces_t[fi, 0]][:, None, :] + self.offs[self.faces_t[fi, 2]]
        M = 2 * n + 1
        local = torch.searchsorted(self.nodes_t, (g[..., 0] * M + g[..., 1]) * M + g[..., 2])
        dofs = (3 * local[:, :, None] + torch.arange(3, device=dev)).reshape(len(fi), -1)
        r, c = dofs[:, self.iu135[0]], dofs[:, self.iu135[1]]
        return torch.minimum(r, c) * nb + torch.maximum(r, c)

    def update(self, taus, cur_dir):
        """Design with thickness corners taus and its exact topology in cur_dir (fast_prep3). Returns stage times, or
        None if the design leaves the superset (caller rebuilds)."""
        from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions, DirectSolverMatrixType, DirectSolverMatrixViewType
        import os
        st = {}
        DE.sync(); t = time.perf_counter()
        n = self.n
        cells = np.load(Path(cur_dir) / 'CELL_INDICES.npy'); faces = np.load(Path(cur_dir) / 'GP_FACES.npy')
        box_nodes = np.load(Path(cur_dir) / 'BOX_NODES.npy'); nodes = np.load(Path(cur_dir) / 'NODES.npy')
        ck = (cells[:, 0].astype(np.int64) * n + cells[:, 1]) * n + cells[:, 2]
        act = np.searchsorted(self.cell_keys, ck)
        fk = ck[faces[:, 0]] * 3 + faces[:, 2]
        fpos = np.searchsorted(self.face_keys, fk)
        inside = (act < len(self.cell_keys)).all() and np.array_equal(self.cell_keys[np.minimum(act, len(self.cell_keys) - 1)], ck) \
            and (fpos < len(self.face_keys)).all() and np.array_equal(self.face_keys[np.minimum(fpos, len(self.face_keys) - 1)], fk) \
            and np.array_equal(box_nodes, self.box_nodes)
        if not inside:
            return None
        st['map'] = time.perf_counter() - t; t = time.perf_counter()
        M = torch.as_tensor(self.PT.cell_moments(cells, n, taus, self.normal, self.offset, self.s, levels=self.levels), dtype=dt, device=dev)
        DE.sync(); st['moments'] = time.perf_counter() - t; t = time.perf_counter()
        vals = self.base.clone()
        act_t = torch.as_tensor(act, device=dev)
        for lo in range(0, len(M), 4096):
            ke = M[lo:lo + 4096] @ self.Tm_up
            vals.index_add_(0, self.pos_e[act_t[lo:lo + 4096]].reshape(-1), ke.reshape(-1))
        del M
        off = np.setdiff1d(np.arange(len(self.face_keys)), fpos)                      # superset faces not in this design
        for lo in range(0, len(off), 1024):
            fi = torch.as_tensor(off[lo:lo + 1024], device=dev)
            vals.index_add_(0, torch.searchsorted(self.U, self.face_entry_keys(fi).reshape(-1)),
                            -self.gamma * self.tk_up[self.faces_t[fi, 2]].reshape(-1))
        a = vals[self.idxA]
        # superset nodes outside every active cell: decoupled interior unknowns with unit diagonal
        loose = np.flatnonzero(~np.isin(self.nodes_sup, nodes))
        if len(loose):
            ld = torch.as_tensor((3 * loose[:, None] + np.arange(3)).reshape(-1), device=dev)
            if bool(self.box_dof[ld].any()):
                raise ValueError('LOOSE_BOX_NODE')
            a[self.diagA[self.new[ld]]] = 1.0
        s = 1 / torch.sqrt(a[self.diagA])
        scaled = a * s[self.rowA] * s[self.colA]
        self.s_scale = s
        if self.precision == 'fp32':
            self.A_up = torch.sparse_csr_tensor(self.crowA.long(), self.colA, a, size=(self.ni, self.ni))
            self.A_upT = self.A_up.t().to_sparse_csr()
            self.A_diag = a[self.diagA]
        self.C = torch.sparse_coo_tensor(torch.stack([self.C_rows, self.C_cols]), vals[self.idxC], (self.ni, self.nbx)).coalesce()
        self.Ct = self.C.t().coalesce()
        dv = vals[self.idxD]
        self.D = torch.sparse_coo_tensor(torch.stack([torch.cat([self.D_r, self.D_c[self.D_off]]), torch.cat([self.D_c, self.D_r[self.D_off]])]),
                                         torch.cat([dv, dv[self.D_off]]), (self.nbx, self.nbx)).coalesce()
        del vals
        DE.sync(); st['assemble'] = time.perf_counter() - t; t = time.perf_counter()
        fdt = torch.float32 if self.precision == 'fp32' else dt
        if self.solver is None:
            self.U_csr = torch.sparse_csr_tensor(self.crowA, self.colA32, scaled.to(fdt), size=(self.ni, self.ni))
            lib = os.environ.get('CUDSS_MT')
            opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER,
                                       **(dict(multithreading_lib=lib) if lib else {}))
            self.w = getattr(self, "panel", 8)
            self.b = torch.zeros((self.w, self.ni), dtype=fdt, device=dev).T
            self.solver = DirectSolver(self.U_csr, self.b, options=opts)
            if lib:
                self.solver.plan_config.host_nthreads = self.threads
            torch.cuda.empty_cache()
            self.solver.plan()
            DE.sync(); st['plan'] = time.perf_counter() - t; t = time.perf_counter()
        else:
            self.U_csr.values().copy_(scaled.to(fdt))
        self.solver.factorize()
        DE.sync(); st['factorize'] = time.perf_counter() - t
        return st

    # queries: same as design_encoder.CellEncoder
    _solve_scaled = DE.CellEncoder._solve_scaled
    solve = DE.CellEncoder.solve
    apply = DE.CellEncoder.apply


def main(case, body_dir, sup_dir, out, eps_list):
    from fractions import Fraction
    Path(out).mkdir(parents=True, exist_ok=True)
    rec = dict(case=case, superset=str(sup_dir), designs=[])
    base_box = np.load(Path(body_dir) / case / 'BOX_NODES.npy')
    enc = SupersetEncoder(case, sup_dir, base_box, precision='fp32')
    rec['setup_seconds'] = enc.setup_seconds
    g = torch.Generator(device=dev).manual_seed(1)
    q = torch.randn((enc.nbx, 8), dtype=dt, device=dev, generator=g)
    outputs = {}
    for eps in eps_list:
        sub = case if eps == '0' else f'{case}_eps{eps.replace("/", "_")}'
        taus = [t * float(1 + Fraction(eps)) for t in enc.taus0]
        st = enc.update(taus, Path(body_dir) / sub)
        if st is not None:
            outputs[eps] = enc.apply(q).cpu()
        rec['designs'].append(dict(eps=eps, sub=sub, stages=st))
        print(json.dumps(dict(eps=eps, stages=None if st is None else {k: round(v, 3) for k, v in st.items()})), flush=True)
    del enc
    import gc; gc.collect(); torch.cuda.empty_cache()
    # reference: a fresh encoder on each design's own topology (no superset)
    for row in rec['designs']:
        if row['stages'] is None:
            continue
        ref = DE.CellEncoder(case, body_dir, precision='fp32', log=lambda s_: None, sub=row['sub'])
        ref.update([t * float(1 + Fraction(row['eps'])) for t in ref.taus0])
        y = ref.apply(q.to(dev)).cpu()
        row['vs_fresh_own_topology'] = float((outputs[row['eps']] - y).norm() / y.norm())
        print(json.dumps(dict(eps=row['eps'], vs_fresh_own_topology=row['vs_fresh_own_topology'])), flush=True)
        del ref; gc.collect(); torch.cuda.empty_cache()
    (Path(out) / f'SUPERSET_{case}.json').write_text(json.dumps(rec, indent=2, default=float))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
