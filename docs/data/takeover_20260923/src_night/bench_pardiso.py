"""Query-cost comparison: factor the interior block once (PARDISO, scaled SPD), then solve for q panels of
1, 7, 64 columns and apply S q = D q + C^T z. Teacher compiled blocks; timings only."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from stage_cutfem_solver.pardiso import PardisoSPD
MN = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/diagnostics/MECHANICS_NETWORK_20260922_01')
case, threads = sys.argv[1], int(sys.argv[2])
comp = MN / 'assets' / case / 'compiled'
A, C, D = [sparse.load_npz(comp / (n + '.npz')).tocsr() for n in 'ACD']
s = 1 / np.sqrt(A.diagonal()); up = sparse.triu(A, format='csr'); up.data *= np.repeat(s, np.diff(up.indptr)) * s[up.indices]
rec = dict(case=case, threads=threads, interior=A.shape[0], boundary=D.shape[0])
t = time.perf_counter()
with PardisoSPD(up, threads=threads) as f:
    rec['factor_seconds'] = time.perf_counter() - t
    rng = np.random.default_rng(0)
    for cols in (1, 7, 64):
        q = rng.standard_normal((D.shape[0], cols))
        t = time.perf_counter()
        z = s[:, None] * f.solve(s[:, None] * (-(C @ q)))
        Sq = D @ q + C.T @ z
        rec[f'query_{cols}'] = time.perf_counter() - t
print(json.dumps(rec))
