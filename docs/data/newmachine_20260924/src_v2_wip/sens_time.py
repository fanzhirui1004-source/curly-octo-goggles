"""Sensitivity cost of one deployment cell: moments (polyref), reverse-mode summed-objective sensitivity at several row
batches (moments_ad.cell_sens), central differences (dmoments + sens); agreement ad vs fd."""
import os, json, time, sys
import models as MD  # noqa
import torch, teacher as TE, moments_ad as MA, fastidx as FI
dev, dt = TE.dev, TE.dt
FI.ON = True
def T():
    torch.cuda.synchronize(); return time.perf_counter()
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
u = torch.randn((C.nb, 6), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
r = {}
t = T(); C.moments(C.taus); r['moments_s'] = T() - t
t = T(); g = C.energy_density(u).sum(2, keepdim=True); r['energy_density_s'] = T() - t
S = {}
for b in (2048,):
    torch.cuda.reset_peak_memory_stats()
    t = T(); S[b] = MA.cell_sens(C, g, batch=b); r[f'ad_{b}_s'] = T() - t
    r[f'ad_{b}_peakGB'] = torch.cuda.max_memory_allocated() / 1e9
t = T(); s_sk = MA.cell_sens(C, g, batch=2048, skip_full=True); r['ad_skip_s'] = T() - t
cells, taus, nrm, off = MA.cell_rows(C); r['full_fraction'] = float(MA.full_rows(cells, C.n, taus, nrm, off, s=C.s).float().mean())
r['skip_vs_ad'] = float((s_sk - S[2048]).norm() / S[2048].norm())
MA.CHECKPOINT = False; torch.cuda.reset_peak_memory_stats()
for b in (256, 512):
    t = T(); s_nc = MA.cell_sens(C, g, batch=b, skip_full=True); r[f'ad_skip_nockpt_{b}_s'] = T() - t
    r[f'ad_skip_nockpt_{b}_peakGB'] = torch.cuda.max_memory_allocated() / 1e9
r['nockpt_vs_ad'] = float((s_nc - S[2048]).norm() / S[2048].norm()); MA.CHECKPOINT = True
t = T(); C.dmoments(); s_fd = C.sens(u).sum(1, keepdim=True); r['fd_s'] = T() - t
r['ad_vs_fd'] = float((S[2048] - s_fd).norm() / s_fd.norm())
r['ad_batch_spread'] = max(float((S[b] - S[2048]).norm() / S[2048].norm()) for b in S)
print(json.dumps(dict(case=case, E=int(len(C.M)), **r)), flush=True)
