"""K_PP triplets: assembled route vs assemble_deploy, compared as sparse matrices over the union of patterns."""
import sys, json
import models as MD  # noqa
import torch, teacher as TE
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None); C.assemble(); C.lean(deploy=True, ke_moments=True, ghost_faces=True)
a = C._kpp_cache; n = C.np_; del C
D = TE.Cell(case, body, log=lambda s_: None, deploy=True); D.assemble_deploy(); b = D._kpp_cache
A = torch.sparse_coo_tensor(torch.stack([a[0], a[1]]), a[2], (n, n)).coalesce()
B = torch.sparse_coo_tensor(torch.stack([b[0], b[1]]), b[2], (n, n)).coalesce()
Dd = (A - B).coalesce().values().abs()
only_a = (A - B * 0).coalesce()
ka = set((a[0] * n + a[1]).tolist()); kb = set((b[0] * n + b[1]).tolist())
va = dict(zip((a[0] * n + a[1]).tolist(), a[2].tolist())); vb = dict(zip((b[0] * n + b[1]).tolist(), b[2].tolist()))
mx = float(a[2].abs().max())
print(json.dumps(dict(case=case, max_abs_diff_rel=float(Dd.max()) / mx, only_route=len(ka - kb), only_scratch=len(kb - ka),
    max_only_route_rel=max([abs(va[k]) for k in ka - kb] or [0]) / mx, max_only_scratch_rel=max([abs(vb[k]) for k in kb - ka] or [0]) / mx)))
