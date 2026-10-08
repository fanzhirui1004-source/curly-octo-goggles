import json, sys
import models as MD  # noqa
import torch, teacher as TE
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True)
Tm = C.Tm.reshape(125, 81, 81)
nz = (Tm != 0)
colnz = nz.reshape(125, -1).any(0)
r = dict(total=int(Tm.numel()), nonzero=int(nz.sum()), frac=float(nz.float().mean()),
         entries_used=int(colnz.sum()), rank_Tf=int(torch.linalg.matrix_rank(Tm.reshape(125, -1)).item()),
         sym=float((Tm - Tm.transpose(1, 2)).abs().max()), per_m_nnz=[int(x) for x in nz.reshape(125, -1).sum(1)[:10]])
print(json.dumps(r))
