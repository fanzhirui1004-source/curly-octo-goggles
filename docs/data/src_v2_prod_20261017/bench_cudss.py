"""GPU sparse direct factorization (cuDSS via nvmath-python) of the scaled interior block A, then S q = D q + C^T z
for 1, 7, 64 columns. Same teacher compiled blocks as bench_pardiso.py; reports times and the relative residual
of A z = -C q and the relative difference of S q against a scipy reference solve on the 1-column query."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
import nvmath
from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions, DirectSolverMatrixType, DirectSolverMatrixViewType

MN = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/diagnostics/MECHANICS_NETWORK_20260922_01')
case = sys.argv[1]
comp = MN / 'assets' / case / 'compiled'
A, C, D = [sparse.load_npz(comp / (n + '.npz')).tocsr() for n in 'ACD']
s = 1 / np.sqrt(A.diagonal())
As = sparse.diags(s) @ A @ sparse.diags(s)
up = sparse.triu(As, format='csr'); up.sort_indices()
dev = torch.device('cuda')
def gpu_csr(M):
    return torch.sparse_csr_tensor(torch.from_numpy(M.indptr.astype(np.int32)).to(dev), torch.from_numpy(M.indices.astype(np.int32)).to(dev),
                                   torch.from_numpy(M.data).to(dev), size=M.shape)
a = gpu_csr(up)
Cg = gpu_csr(C.tocsr()); CTg = gpu_csr(C.T.tocsr()); Dg = gpu_csr(D.tocsr()); sg = torch.from_numpy(s).to(dev)
rec = dict(case=case, interior=A.shape[0], boundary=D.shape[0], nnz_upper=int(up.nnz))
rng = np.random.default_rng(0)
import os
from nvmath.sparse.advanced import DirectSolverReorderingAlg
mt = os.environ.get('CUDSS_MT'); nthreads = int(os.environ.get('CUDSS_THREADS', '0')); reorder = os.environ.get('CUDSS_REORDER', 'DEFAULT')
opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER,
                           **(dict(multithreading_lib=mt) if mt else {}))
rec.update(mt=bool(mt), threads=nthreads, reorder=reorder)
for cols in [int(c) for c in os.environ.get('CUDSS_COLS', '1 7 64').split()]:
    q = torch.from_numpy(rng.standard_normal((D.shape[0], cols))).to(dev)
    torch.cuda.synchronize(); t = time.perf_counter()
    b = -(sg[:, None] * (Cg @ q))
    b = b.T.contiguous().T  # column-major for cuDSS
    with DirectSolver(a, b, options=opts) as solver:
        if nthreads:
            solver.plan_config.host_nthreads = nthreads
        solver.plan_config.reordering_algorithm = getattr(DirectSolverReorderingAlg, reorder)
        free0 = torch.cuda.mem_get_info()[0]
        solver.plan(); torch.cuda.synchronize(); t1 = time.perf_counter()
        solver.factorize(); torch.cuda.synchronize(); t2 = time.perf_counter()
        used_gb = (free0 - torch.cuda.mem_get_info()[0]) / 2**30
        x = solver.solve(); torch.cuda.synchronize(); t3 = time.perf_counter()
        # second solve with the same factor (the query cost that matters)
        q2 = torch.from_numpy(rng.standard_normal((D.shape[0], cols))).to(dev)
        b2 = (-(sg[:, None] * (Cg @ q2))).T.contiguous().T
        torch.cuda.synchronize(); t4 = time.perf_counter()
        solver.reset_operands(b=b2); x2 = solver.solve()
        Sq = Dg @ q2 + CTg @ (sg[:, None] * x2)
        torch.cuda.synchronize(); t5 = time.perf_counter()
    z = s[:, None] * x2.cpu().numpy()
    res = np.linalg.norm(A @ z + C @ q2.cpu().numpy()) / np.linalg.norm(C @ q2.cpu().numpy())
    rec[f'cols_{cols}'] = dict(plan=t1 - t, factor=t2 - t1, first_solve=t3 - t2, query=t5 - t4, residual=float(res), device_gb_after_factor=used_gb)
    print(json.dumps(dict(cols=cols, **rec[f'cols_{cols}'])), flush=True)
rec['peak_gpu_gb'] = torch.cuda.max_memory_allocated() / 2**30
print(json.dumps(rec))
