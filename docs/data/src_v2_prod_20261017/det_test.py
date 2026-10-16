"""OPL_TET_TRITON (explicit determinants) off / on: cell moments and reverse-mode summed sensitivity, times and differences."""
import os, json, time, sys
import models as MD  # noqa
import torch, teacher as TE, moments_ad as MA, polyref_torch_fast as PT, fastidx as FI
dev, dt = TE.dev, TE.dt
FI.ON = True
def T():
    torch.cuda.synchronize(); return time.perf_counter()
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
u = torch.randn((C.nb, 3), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
g = C.energy_density(u).sum(2, keepdim=True)
r, M, S = {}, {}, {}
for on in ('0', '1', '1', '0'):
    os.environ['OPL_TET_TRITON'] = on
    t = T(); M[on] = torch.as_tensor(PT.cell_moments(C.cells, C.n, C.taus, C.normal, C.offset, C.s, levels=C.levels)); r['moments_s_' + on] = T() - t
    t = T(); S[on] = MA.cell_sens(C, g, batch=2048, skip_full=True).cpu(); r['sens_s_' + on] = T() - t
r['moments_rel'] = float((M['1'] - M['0']).abs().max() / M['0'].abs().max())
r['sens_rel'] = float((S['1'] - S['0']).norm() / S['0'].norm())
print(json.dumps(dict(case=case, **r)), flush=True)
