import os, sys, json
import numpy as np, torch
import models as MD  # noqa
import teacher as TE
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True)
M = C.moments(C.taus0)
bad = ~torch.isfinite(M).all(1)
r = dict(case=case, tet_triton=os.environ.get('OPL_TET_TRITON'), surface=C.surface, elements=int(len(M)), nan_elements=int(bad.sum()),
         vol_sum=float(M[~bad, 0].sum()), vol_min=float(M[~bad, 0].min()))
if bad.any():
    idx = torch.nonzero(bad).squeeze(1)[:5].cpu().numpy()
    r['bad_cells'] = C.cells[idx].tolist(); r['bad_M0'] = M[idx, 0].tolist()
np.save(f'/root/_ks/M_{case}_{os.environ.get("OPL_TET_TRITON")}.npy', M.cpu().numpy())
print(json.dumps(r), flush=True)
