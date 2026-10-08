"""Route 7: the dense box operator T = K_bb - K_bi K_ii^-1 K_ib straight from a partial cuDSS factorization.

cuDSS (>= 0.6) can stop the factorization before the rows and columns marked as Schur indices and return the dense
Schur complement of that block. With the box DOFs marked, this is exactly the condensed box operator, without one
triangular solve per box column (the panel route in budget_probe.py needs 23 616 of them for a FULL cell).

Setup once per topology (body-lite from fast_prep3): upper pattern of K = K_body + gamma K_ghost over ALL DOFs,
element scatter map, constant ghost values, Schur index mask of the box DOFs.
Per design: moments -> element upper entries -> scatter -> Jacobi scaling -> cuDSS factorization in Schur mode ->
dense Schur block -> unscaled T (T = S_B^-1 T~ S_B^-1; the interior scaling cancels).
Low-level cuDSS bindings (nvmath.bindings.cudss), since nvmath's DirectSolver does not expose the Schur mode.
Usage: schur_encoder.py <body_dir> <out_dir> <ref_cols> <case> [<case> ...]
"""
import json, sys, time, gc, os
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import design_encoder as DE

dev, dt = DE.dev, DE.dt
ROOT = DE.ROOT
CUDA_R_32F, CUDA_R_64F, CUDA_R_32I = 0, 1, 10


class SchurEncoder:
    def __init__(self, case, body_dir, precision='fp64', threads=16, s=4, levels=1, log=print, sub=None, out_host=False):
        import element_moments as EM
        import polyref_torch_fast as PT
        import box_encode as BX
        from fractions import Fraction
        self.PT, self.s, self.levels, self.precision, self.threads, self.log = PT, s, levels, precision, threads, log
        self.out_host = out_host                        # Schur block copied to pinned host memory (FULL in fp64)
        self.T = {}
        t0 = time.perf_counter()
        ctx = json.loads((ROOT / 'packets' / case / 'FRESH_CONTEXT.json').read_text())
        self.case, self.n = case, int(ctx['n'])
        Emod = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
        lam = Emod * nu / ((1 + nu) * (1 - 2 * nu)); mu = Emod / (2 * (1 + nu))
        self.taus0 = [float(Fraction(v)) for v in ctx['case']['tau_corners']]
        self.normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
        self.offset = None if self.normal is None else float(ctx['case']['offset'])
        self.gamma = float(json.loads((ROOT / 'packets' / case / 'SAMPLE.json').read_text())['gp']['gamma'])
        d = Path(body_dir) / (sub or case)
        nodes = np.load(d / 'NODES.npy'); cells = np.load(d / 'CELL_INDICES.npy'); dofs = np.load(d / 'dofs.npy')
        self.cells = cells
        arr = {'NODES.npy': nodes, 'dofs.npy': dofs, 'CELL_INDICES.npy': cells}
        xi = EM.local_coordinates(case, self.n, arr)
        keys = np.unique(xi.reshape(len(xi), -1), axis=0)
        if len(keys) != 1:
            raise ValueError('ONE_NODE_ORDERING_EXPECTED')
        Tm = torch.tensor(EM.pattern_operators(keys[0].reshape(27, 3), lam, mu, self.n)[1], dtype=dt, device=dev)
        iu = torch.triu_indices(81, 81, device=dev)
        self.Tm_up = Tm.reshape(125, 81, 81)[:, iu[0], iu[1]].contiguous()
        nb = 3 * len(nodes); self.nb = nb
        dofs_t = torch.as_tensor(dofs, dtype=torch.long, device=dev)
        r, c = dofs_t[:, iu[0]], dofs_t[:, iu[1]]
        key_e = torch.minimum(r, c) * nb + torch.maximum(r, c)
        G = BX.ghost_faces_gpu(body_dir, sub or case, self.n, nb)
        gi, gv = G.indices(), G.values()
        up = gi[0] <= gi[1]
        key_g, val_g = gi[0][up] * nb + gi[1][up], gv[up]
        del G, gi, gv
        U = torch.unique(torch.cat([key_e.reshape(-1), key_g]))                          # sorted = CSR row-major order
        self.pos_e = torch.searchsorted(U, key_e.reshape(-1))
        self.base = torch.zeros(len(U), dtype=dt, device=dev)
        self.base.index_add_(0, torch.searchsorted(U, key_g), self.gamma * val_g)
        del key_e, key_g, val_g
        ru, cu = (U // nb), (U % nb)
        del U
        self.nnz = len(ru)
        self.crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(ru, minlength=nb), 0)]).int()
        self.ru, self.cu = ru.int(), cu.int()
        del ru, cu
        self.diag = torch.nonzero(self.ru == self.cu).squeeze(1)
        if len(self.diag) != nb:
            raise ValueError('DIAGONAL_INCOMPLETE')
        onbox = np.isin(nodes, np.load(d / 'BOX_NODES.npy'))
        self.box_node_ids = nodes[onbox]
        box = np.repeat(onbox, 3)
        self.schur_idx = box.astype(np.int32)                                            # host mask for cuDSS
        self.box_dofs = torch.as_tensor(np.flatnonzero(box), device=dev)
        self.nbx = int(box.sum())
        self.handle = None
        DE.sync(); self.T['setup'] = time.perf_counter() - t0
        log(json.dumps(dict(event='SCHUR_SETUP', case=case, seconds=self.T['setup'], nnz_upper=self.nnz, dofs=nb, box=self.nbx)))

    def _cudss_init(self, fdt):
        from nvmath.bindings import cudss
        self.cudss = cudss
        vt = CUDA_R_32F if fdt == torch.float32 else CUDA_R_64F
        h = cudss.create()
        cudss.set_stream(h, torch.cuda.current_stream().cuda_stream)
        lib = os.environ.get('CUDSS_MT')
        cfg = cudss.config_create()
        if lib:
            cudss.set_threading_layer(h, lib)
            v = np.array([self.threads], dtype=np.int32)
            cudss.config_set(cfg, int(cudss.ConfigParam.HOST_NTHREADS), v.ctypes.data, v.nbytes)
        v = np.array([1], dtype=np.int32)
        cudss.config_set(cfg, int(cudss.ConfigParam.SCHUR_MODE), v.ctypes.data, v.nbytes)
        data = cudss.data_create(h)
        cudss.data_set(h, data, int(cudss.DataParam.USER_SCHUR_INDICES), self.schur_idx.ctypes.data, self.schur_idx.nbytes)
        self.vals_buf = torch.empty(self.nnz, dtype=fdt, device=dev)
        A = cudss.matrix_create_csr(self.nb, self.nb, self.nnz, self.crow.data_ptr(), 0, self.cu.data_ptr(), self.vals_buf.data_ptr(),
                                    CUDA_R_32I, CUDA_R_32I, vt, int(cudss.MatrixType.SPD) if hasattr(cudss, 'MatrixType') else 3,
                                    2, 0)                                                # view UPPER, base ZERO
        self.xb = torch.zeros((1, self.nb), dtype=fdt, device=dev).T
        self.bb = torch.zeros((1, self.nb), dtype=fdt, device=dev).T
        X = cudss.matrix_create_dn(self.nb, 1, self.nb, self.xb.data_ptr(), vt, 0)
        B = cudss.matrix_create_dn(self.nb, 1, self.nb, self.bb.data_ptr(), vt, 0)
        self.handle, self.cfg, self.data, self.A, self.X, self.B, self.vt, self.fdt = h, cfg, data, A, X, B, vt, fdt
        self.Tbuf = None

    def update(self, taus=None, keep_scaled=False):
        st = {}
        taus = self.taus0 if taus is None else taus
        fdt = torch.float32 if self.precision == 'fp32' else dt
        DE.sync(); t = time.perf_counter()
        M = torch.as_tensor(self.PT.cell_moments(self.cells, self.n, taus, self.normal, self.offset, self.s, levels=self.levels),
                            dtype=dt, device=dev)
        DE.sync(); st['moments'] = time.perf_counter() - t; t = time.perf_counter()
        vals = self.base.clone()
        for lo in range(0, len(M), 4096):
            ke = M[lo:lo + 4096] @ self.Tm_up
            vals.index_add_(0, self.pos_e[lo * 3321:(lo + len(ke)) * 3321], ke.reshape(-1))
        del M
        s = 1 / torch.sqrt(vals[self.diag])
        vals.mul_(s[self.ru.long()]).mul_(s[self.cu.long()])
        self.s_scale = s
        first = self.handle is None
        if first:
            torch.cuda.empty_cache()
            self._cudss_init(fdt)
        self.vals_buf.copy_(vals)
        del vals
        DE.sync(); st['assemble'] = time.perf_counter() - t; t = time.perf_counter()
        cudss = self.cudss
        if first:
            cudss.execute(self.handle, int(cudss.Phase.ANALYSIS), self.cfg, self.data, self.A, self.X, self.B)
            DE.sync(); st['analysis'] = time.perf_counter() - t; t = time.perf_counter()
        cudss.execute(self.handle, int(cudss.Phase.FACTORIZATION), self.cfg, self.data, self.A, self.X, self.B)
        DE.sync(); st['factorize_schur'] = time.perf_counter() - t; t = time.perf_counter()
        if self.Tbuf is None:
            shape = np.zeros(3, dtype=np.int64); w = np.zeros(1, dtype=np.uint64)
            cudss.data_get(self.handle, self.data, int(cudss.DataParam.SCHUR_SHAPE), shape.ctypes.data, shape.nbytes, w.ctypes.data)
            self.schur_shape = shape.tolist()
            if shape[0] != self.nbx or shape[1] != self.nbx:
                raise ValueError(f'SCHUR_SHAPE_MISMATCH:{shape.tolist()}:{self.nbx}')
            self.Tbuf = (torch.zeros((self.nbx, self.nbx), dtype=fdt, pin_memory=True) if self.out_host
                         else torch.zeros((self.nbx, self.nbx), dtype=fdt, device=dev))  # column-major = transpose view
            self.Smat = cudss.matrix_create_dn(self.nbx, self.nbx, self.nbx, self.Tbuf.data_ptr(), self.vt, 0)
        hm = np.array([self.Smat], dtype=np.int64); w = np.zeros(1, dtype=np.uint64)
        cudss.data_get(self.handle, self.data, int(cudss.DataParam.SCHUR_MATRIX), hm.ctypes.data, hm.nbytes, w.ctypes.data)
        DE.sync(); st['get_schur'] = time.perf_counter() - t; t = time.perf_counter()
        Tt = self.Tbuf.T if fdt == dt else self.Tbuf.T.to(dt)                     # column-major storage
        sb = self.s_scale[self.box_dofs]
        if self.out_host:
            sb = sb.cpu()
        # which triangles did cuDSS fill? (a symmetric view may return one triangle only)
        k = min(256, self.nbx // 2)
        self.fill = dict(upper_block_norm=float(Tt[:k, k:2 * k].norm()), lower_block_norm=float(Tt[k:2 * k, :k].norm()))
        if self.fill['lower_block_norm'] == 0.0 and self.fill['upper_block_norm'] > 0.0:
            symmetrize_(Tt, 'upper')
        elif self.fill['upper_block_norm'] == 0.0 and self.fill['lower_block_norm'] > 0.0:
            symmetrize_(Tt, 'lower')
        Tt.div_(sb[:, None]).div_(sb[None, :])
        T = Tt
        DE.sync(); st['unscale'] = time.perf_counter() - t
        return T, st

    def free(self):
        if self.handle is None:
            return
        c = self.cudss
        for m in (self.A, self.X, self.B) + ((self.Smat,) if self.Tbuf is not None else ()):
            c.matrix_destroy(m)
        c.data_destroy(self.handle, self.data); c.config_destroy(self.cfg); c.destroy(self.handle)
        self.handle = None


def symmetrize_(T, src, blk=2048):
    """fill the other strict triangle of T in place from src ('upper' or 'lower'), block rows at a time."""
    n = T.shape[0]
    for r0 in range(0, n, blk):
        r1 = min(r0 + blk, n)
        b = T[r0:r1, r0:r1]
        if src == 'upper':
            T[r0:r1, :r0] = T[:r0, r0:r1].T
            T[r0:r1, r0:r1] = torch.triu(b) + torch.triu(b, 1).T
        else:
            T[:r0, r0:r1] = T[r0:r1, :r0].T
            T[r0:r1, r0:r1] = torch.tril(b) + torch.tril(b, -1).T


def sym_rel(T, blk=4096):
    num = den = 0.0
    for r0 in range(0, T.shape[0], blk):
        r1 = min(r0 + blk, T.shape[0])
        num += float(((T[r0:r1] - T[:, r0:r1].T) ** 2).sum()); den += float((T[r0:r1] ** 2).sum())
    return (num / den) ** 0.5


def used_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 30


def reference_columns(case, body_dir, cols):
    """columns of T from the panel route (fp32 factor + 2 fp64 refinement steps, ~1e-14)."""
    enc = DE.CellEncoder(case, body_dir, precision='fp32', log=lambda s_: None)
    enc.panel = 256
    enc.update()
    Q = torch.zeros((enc.nbx, len(cols)), dtype=dt, device=dev)
    Q[torch.as_tensor(cols, device=dev), torch.arange(len(cols), device=dev)] = 1.0
    Y = torch.cat([enc.apply(Q[:, j:j + 256]) for j in range(0, len(cols), 256)], 1).cpu()
    ids = enc.box_node_ids.copy()
    del enc, Q
    gc.collect(); torch.cuda.empty_cache()
    return Y, ids


def main(body_dir, out, ref_cols, cases, out_host=False):
    Path(out).mkdir(parents=True, exist_ok=True)
    for case in cases:
        rec = dict(case=case)
        g = np.random.default_rng(0)
        Yref = None
        for precision in ('fp32', 'fp64'):
            gc.collect(); torch.cuda.empty_cache()
            u0 = used_gb()
            host = precision == 'fp64' and out_host
            enc = SchurEncoder(case, body_dir, precision=precision, log=lambda s_: None, out_host=host)
            if Yref is None:
                cols = np.sort(g.choice(enc.nbx, size=min(ref_cols, enc.nbx), replace=False))
                del enc; gc.collect(); torch.cuda.empty_cache()
                Yref, ids = reference_columns(case, body_dir, cols)
                enc = SchurEncoder(case, body_dir, precision=precision, log=lambda s_: None, out_host=host)
                if not np.array_equal(ids, enc.box_node_ids):
                    raise ValueError('BOX_ORDER')
            T, st0 = enc.update()
            row = dict(setup=enc.T['setup'], first=st0, fill=enc.fill, schur_shape=enc.schur_shape, out_host=host,
                       device_gb=used_gb() - u0)
            Tc = T[:, torch.as_tensor(cols, device=T.device)].cpu()
            row['vs_panel_ref'] = float((Tc - Yref).norm() / Yref.norm())
            row['sym_rel'] = sym_rel(T)
            del T
            T2, st1 = enc.update([t_ * (1 + 1e-4) for t_ in enc.taus0])               # design step (analysis reused)
            row['design_step'] = st1
            del T2
            rec[precision] = row
            print(json.dumps(dict(case=case, precision=precision, **row), default=float), flush=True)
            enc.free(); del enc
        (Path(out) / f'SCHUR_{case}.json').write_text(json.dumps(rec, indent=2, default=float))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:], out_host=os.environ.get('SCHUR_OUT_HOST') == '1')
