"""build_K.py <sysdir> : rebuild the whole-lattice stiffness K_indptr/K_indices/K_data.npy from the per-cell systems
cell<i>.npz of lat_dump.py. K is the sum over cells of the local full matrices mapped by l2g, exactly as
lat_direct_cpu.global_matrix assembles it; the number of stored entries is checked against meta.json (nnz_full)."""
import sys, json, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
S = Path(sys.argv[1])
meta = json.loads((S / 'meta.json').read_text())
n = meta['global_dofs']
K = None
for i in range(len(meta['cells'])):
    C = np.load(S / f'cell{i}.npz')
    l2g = C['l2g']; nl = len(l2g)
    A = sp.csr_matrix((C['data'], C['indices'], C['indptr']), shape=(nl, nl)).tocoo()
    B = sp.csr_matrix((A.data, (l2g[A.row], l2g[A.col])), shape=(n, n))
    K = B if K is None else K + B
    del A, B, C; gc.collect()
    print('cell', i, 'nnz so far', K.nnz, flush=True)
K.sort_indices()
assert K.nnz == meta['nnz_full'], (K.nnz, meta['nnz_full'])
np.save(S / 'K_indptr.npy', K.indptr.astype(np.int64)); np.save(S / 'K_indices.npy', K.indices.astype(np.int32))
np.save(S / 'K_data.npy', K.data)
print('K_OK', n, K.nnz, flush=True)
