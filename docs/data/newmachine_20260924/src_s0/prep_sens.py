"""Exact thickness sensitivities of every bank direction: s_c(q) = -u^T (dK/dtau_c) u, u = E q (teacher), 8 corners.
Saved as <split>_<class>_sens.npy (count x 8, float64) next to the banks.
Usage: prep_sens.py <body_dir> <data_dir> <case> [<case> ...]"""
import sys, json, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE

body, data = sys.argv[1], Path(sys.argv[2])
for case in sys.argv[3:]:
    t0 = time.perf_counter()
    C = TE.Cell(case, body, log=lambda s_: None)
    C.assemble(); C.factor(neumann=False); C.dmoments()
    d = data / case
    for split in ('train', 'val', 'test'):
        for cls in ('force', 'macro', 'grf'):
            Q = torch.as_tensor(np.load(d / f'{split}_{cls}.npy'), device='cuda').T.to(torch.float64)
            S = torch.cat([C.sens(C.extend(Q[:, j:j + 32])) for j in range(0, Q.shape[1], 32)], 1)
            np.save(d / f'{split}_{cls}_sens.npy', S.T.contiguous().cpu().numpy())
    print(json.dumps(dict(case=case, seconds=time.perf_counter() - t0)), flush=True)
    C._free(); del C; gc.collect(); torch.cuda.empty_cache()
