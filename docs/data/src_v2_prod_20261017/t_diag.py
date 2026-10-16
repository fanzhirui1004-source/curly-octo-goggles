"""Dense exact port Schur complement T64 of a cell (make_T_gpu output): finiteness, symmetry, rigid-mode residual, and
positive semi-definiteness (Cholesky of T + shift I at shifts 1e-12, 1e-9 relative to max diag) and diagonal range."""
import sys, json
from pathlib import Path
import numpy as np, torch
body = Path(sys.argv[1])
for case in sys.argv[2:]:
    pd = body / (case + '_portview')
    T = torch.from_numpy(np.load(pd / 'T64.npy')).cuda()
    n2 = 65
    ids = np.load(pd / 'BOX_NODES.npy')
    xyz = torch.as_tensor(np.stack(np.unravel_index(ids, (n2,) * 3), 1) / (n2 - 1), dtype=T.dtype, device='cuda')
    m = len(ids)
    R = torch.zeros((3 * m, 6), dtype=T.dtype, device='cuda')
    for a in range(3):
        R[a::3, a] = 1
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    R[0::3, 3], R[1::3, 3] = -y, x; R[1::3, 4], R[2::3, 4] = -z, y; R[0::3, 5], R[2::3, 5] = z, -x
    d = torch.diagonal(T)
    r = dict(case=case, n=int(T.shape[0]), finite=bool(torch.isfinite(T).all()), sym=float((T - T.T).abs().max() / T.abs().max()),
             diag_min=float(d.min()), diag_max=float(d.max()), n_nonpos_diag=int((d <= 0).sum()),
             rigid=float((T @ R).norm() / (T.norm() * R.norm())))
    for sh in (1e-12, 1e-9):
        _, info = torch.linalg.cholesky_ex(T + sh * float(d.max()) * torch.eye(T.shape[0], dtype=T.dtype, device='cuda'))
        r[f'chol_ok_{sh:g}'] = int(info) == 0
    print(json.dumps(r), flush=True)
    del T, R; torch.cuda.empty_cache()
