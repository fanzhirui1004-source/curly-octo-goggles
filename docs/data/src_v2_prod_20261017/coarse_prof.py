"""Pieces of coarse_setup on a deployment cell (fastidx on): prolongation (scipy), colouring, probing products, near-mask
assembly, Jacobi scaling + Cholesky."""
import os, json, time, sys, math
import numpy as np
import models as MD  # noqa
import torch, teacher as TE, trainlib as TL, fastidx as FI
dev, dt = TE.dev, TE.dt
FI.ON = True
def T():
    torch.cuda.synchronize(); return time.perf_counter()
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
r = {}
t = T(); V, vc, slot = TL.coarse_prolong(C, 'Q1_17', with_meta=True); r['prolong_s'] = T() - t
t = T(); Vc = V.tocoo(); Vt = torch.sparse_coo_tensor(np.stack([Vc.row, Vc.col]), Vc.data, V.shape, dtype=dt, device=dev).coalesce(); VtT = Vt.t().coalesce(); r['to_torch_s'] = T() - t
t = T()
with C.hold64():
    x = torch.randn((C.nb, 128), dtype=dt, device=dev)
    for _ in range(9):
        y = C.K @ x
r['K128x9_s'] = T() - t
nc = V.shape[1]
A = torch.randn((nc, nc), dtype=dt, device=dev); A = A @ A.T / nc + torch.eye(nc, dtype=dt, device=dev)
t = T(); L, _ = TL._chol_jitter(A); r['chol_s'] = T() - t
t = T(); TL.coarse_setup(C, 'Q1_17'); r['coarse_setup_total_s'] = T() - t
r['nc'] = nc
print(json.dumps(dict(case=case, **r)), flush=True)
