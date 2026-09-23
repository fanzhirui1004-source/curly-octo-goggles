import numpy as np, time
from scipy import sparse
for c in ['fresh_train_0003_full', 'fresh_train_0013_d0_v1']:
    P = sparse.load_npz(f'/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/packets/{c}/ORIGINAL_FROM_TRACE_FREE.npz').tocsr()
    r = np.diff(P.indptr)
    print(c, P.shape, P.nnz, 'row nnz hist', np.bincount(r)[:8].tolist(), 'max', r.max(), 'ones', float(np.mean(P.data == 1)))
    Pc = P.tocsc(); cc = np.diff(Pc.indptr); print(' col nnz hist', np.bincount(cc)[:8].tolist(), 'max', cc.max())
