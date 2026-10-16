"""Route 7: what a 100-cell lattice iteration costs beyond the design step.

Per cell (body-lite from fast_prep3):
  - device memory of one encoder with its fp32 cuDSS factor (device-level, so cuDSS's own allocations count),
    cuDSS memory estimates and factor nonzeros,
  - dense box operator T = D - C^T A^-1 C, all box columns, in panels of width W (one solver built for that width),
    with the fp64 refinement the design step uses; accuracy of 0/1/2 refinement steps on the first panel,
  - plan and factorization cost of the cuDSS reordering algorithms (FULL only, optional).
Usage: budget_probe.py <body_dir> <out_dir> <panel> <reorder:0|1> <case> [<case> ...]
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import design_encoder as DE

dev, dt = DE.dev, DE.dt


def used_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 30


def mem_est(me):
    try:
        return {k: np.asarray(me[k]).tolist() for k in me.dtype.names}
    except Exception:
        return str(me)


def csr_rows_dense(Mcsr, r0, r1, ncols):
    """rows r0:r1 of a CSR matrix as a dense (r1 - r0) x ncols fp64 block."""
    crow = Mcsr.crow_indices()
    a, b = int(crow[r0]), int(crow[r1])
    col = Mcsr.col_indices()[a:b]
    val = Mcsr.values()[a:b]
    row = torch.repeat_interleave(torch.arange(r1 - r0, device=dev), (crow[r0 + 1:r1 + 1] - crow[r0:r1]))
    out = torch.zeros((r1 - r0, ncols), dtype=dt, device=dev)
    out[row, col] = val
    return out


def dense_T(enc, W, out_dtype=torch.float32, log=print):
    Ct = enc.Ct.to_sparse_csr()                                   # nbx x ni
    Dc = enc.D.to_sparse_csr()                                    # nbx x nbx, symmetric
    nbx = enc.nbx
    T = torch.empty((nbx, nbx), dtype=out_dtype, device=dev)
    acc = {}
    DE.sync(); t0 = time.perf_counter()
    for j0 in range(0, nbx, W):
        j1 = min(j0 + W, nbx)
        R = csr_rows_dense(Ct, j0, j1, enc.ni).T.contiguous()    # C[:, j0:j1]
        if j0 == 0:                                               # refinement study on the first panel
            ref = -enc.solve(R, refine=3)
            yr = (torch.sparse.mm(Ct, ref) + csr_rows_dense(Dc, j0, j1, nbx).T).norm()
            for k in (0, 1, 2):
                z = -enc.solve(R, refine=k)
                acc[f'refine{k}_rel_to_refine3'] = float(torch.sparse.mm(Ct, z - ref).norm() / yr)
                if k < 2:
                    del z
            del ref
            DE.sync(); t0 = time.perf_counter()                   # time the production path from the 2nd panel on
            first = j1
        else:
            z = -enc.solve(R)
        T[:, j0:j1] = (torch.sparse.mm(Ct, z) + csr_rows_dense(Dc, j0, j1, nbx).T).to(out_dtype)
        del R, z
    DE.sync(); sec = time.perf_counter() - t0
    per_col = sec / max(1, nbx - first)
    sym = float((T - T.T).norm() / T.norm())
    return T, dict(seconds_measured=sec, columns_measured=nbx - first, seconds_all_columns=per_col * nbx,
                   sym_rel=sym, **acc)


def reorder_study(enc, algs, log=print):
    from nvmath.sparse.advanced import (DirectSolver, DirectSolverOptions, DirectSolverMatrixType,
                                        DirectSolverMatrixViewType, DirectSolverReorderingAlg)
    import os
    rows = []
    lib = os.environ.get('CUDSS_MT')
    log(json.dumps(dict(reordering_algs=[x for x in dir(DirectSolverReorderingAlg) if not x.startswith('_')])))
    for name in algs:
        alg = getattr(DirectSolverReorderingAlg, name, None)
        if alg is None:
            rows.append(dict(alg=name, error='NOT_IN_ENUM')); continue
        opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER,
                                   **(dict(multithreading_lib=lib) if lib else {}))
        b = torch.zeros((8, enc.ni), dtype=torch.float32, device=dev).T
        gc.collect(); torch.cuda.empty_cache()
        u0 = used_gb()
        row = dict(alg=name)
        try:
            s = DirectSolver(enc.U, b, options=opts)
            if lib:
                s.plan_config.host_nthreads = enc.threads
            s.plan_config.reordering_algorithm = alg
            DE.sync(); t = time.perf_counter(); s.plan(); DE.sync(); row['plan'] = time.perf_counter() - t
            me = s.plan_info.memory_estimates
            row['memory_estimates'] = mem_est(me)
            t = time.perf_counter(); s.factorize(); DE.sync(); row['factorize'] = time.perf_counter() - t
            t = time.perf_counter(); s.factorize(); DE.sync(); row['refactorize'] = time.perf_counter() - t
            row['lu_nnz'] = int(s.factorization_info.lu_nnz)
            row['device_gb_added'] = used_gb() - u0
            if hasattr(s, 'free'):
                s.free()
            del s
        except Exception as e:                                     # an algorithm may not support SPD upper view
            row['error'] = repr(e)[:300]
        del b
        gc.collect(); torch.cuda.empty_cache()
        rows.append(row)
        log(json.dumps(row))
    return rows


def main(body_dir, out, W, reorder, cases):
    Path(out).mkdir(parents=True, exist_ok=True)
    for case in cases:
        gc.collect(); torch.cuda.empty_cache()
        rec = dict(case=case, panel=W, device_gb_start=used_gb())
        enc = DE.CellEncoder(case, body_dir, precision='fp32', log=lambda s_: None)
        enc.panel = W
        rec['device_gb_after_setup'] = used_gb()
        rec['first_update'] = enc.update()
        rec['device_gb_after_factor'] = used_gb()
        rec['torch_allocated_gb'] = torch.cuda.memory_allocated() / 2 ** 30
        rec['lu_nnz'] = int(enc.solver.factorization_info.lu_nnz)
        me = enc.solver.plan_info.memory_estimates
        rec['memory_estimates'] = mem_est(me)
        rec['interior'], rec['box'] = enc.ni, enc.nbx
        rec['design_step'] = enc.update([t * (1 + 1e-4) for t in enc.taus0])
        print(json.dumps({k: v for k, v in rec.items()}, default=float), flush=True)
        torch.cuda.reset_peak_memory_stats()
        T, info = dense_T(enc, W)
        rec['dense_T'] = info
        rec['dense_T_fp32_gb'] = T.numel() * 4 / 2 ** 30
        rec['device_gb_peak_during_T'] = used_gb()
        rec['torch_peak_gb_during_T'] = torch.cuda.max_memory_allocated() / 2 ** 30
        print(json.dumps(dict(case=case, dense_T=info, device_gb=rec['device_gb_peak_during_T'])), flush=True)
        del T
        if reorder:
            if hasattr(enc.solver, 'free'):
                enc.solver.free()
            enc.solver = None
            gc.collect(); torch.cuda.empty_cache()
            rec['reorder'] = reorder_study(enc, ['DEFAULT', 'ALG_1', 'ALG_2', 'ALG_3'])
        (Path(out) / f'BUDGET_{case}.json').write_text(json.dumps(rec, indent=2, default=float))
        del enc
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
