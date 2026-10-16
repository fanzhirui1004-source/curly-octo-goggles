import numpy as np, torch
from scipy import sparse
from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions, DirectSolverMatrixType, DirectSolverMatrixViewType
n = 2000
M = sparse.random(n, n, 0.002, random_state=0); M = M + M.T + sparse.eye(n) * 10
up = sparse.triu(M, format='csr'); up.sort_indices()
dev = torch.device('cuda')
a = torch.sparse_csr_tensor(torch.from_numpy(up.indptr.astype(np.int32)).to(dev), torch.from_numpy(up.indices.astype(np.int32)).to(dev), torch.from_numpy(up.data).to(dev), size=up.shape)
opts = DirectSolverOptions(sparse_system_type=DirectSolverMatrixType.SPD, sparse_system_view=DirectSolverMatrixViewType.UPPER)
b = torch.randn(n, 1, dtype=torch.float64, device=dev).T.contiguous().T
s = DirectSolver(a, b, options=opts); s.plan(); s.factorize(); x = s.solve()
print('1col', float(np.linalg.norm(M @ x.cpu().numpy() - b.cpu().numpy())))
for k in (7, 64, 1):
    b2 = torch.randn(n, k, dtype=torch.float64, device=dev).T.contiguous().T
    try:
        s.reset_operands(b=b2); x2 = s.solve()
        print(k, x2.shape, float(np.linalg.norm(M @ x2.cpu().numpy() - b2.cpu().numpy())))
    except Exception as e:
        print(k, 'ERR', type(e).__name__, str(e)[:200])
b1 = torch.randn(n, dtype=torch.float64, device=dev)
try:
    s.reset_operands(b=b1); x3 = s.solve(); print('1d', x3.shape)
except Exception as e:
    print('1d ERR', str(e)[:200])
s.free()
