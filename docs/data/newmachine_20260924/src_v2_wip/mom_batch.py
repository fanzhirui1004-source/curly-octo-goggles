"""Moments front end: polyref_torch_fast.cell_moments per cell (batch 2048 / whole cell) vs moments_ad.moments_rows
(no grad) per cell and over several cells at once; agreement with the per-cell reference."""
import os, json, time, sys
import numpy as np
import models as MD  # noqa
import torch, teacher as TE, polyref_torch_fast as PT, moments_ad as MA
dev, dt = TE.dev, TE.dt
def T():
    torch.cuda.synchronize(); return time.perf_counter()
Cs = [TE.Cell(c.split(':')[1], c.split(':')[0], log=lambda s_: None, deploy=True) for c in sys.argv[1].split(',')]
r = {}
ref = []
t = T()
for C in Cs:
    ref.append(torch.as_tensor(PT.cell_moments(C.cells, C.n, C.taus0, C.normal, C.offset, C.s, levels=C.levels), dtype=dt, device=dev))
r['pt_2048_s'] = T() - t
for b in (4096, 8192):
    t = T()
    out = [torch.as_tensor(PT.cell_moments(C.cells, C.n, C.taus0, C.normal, C.offset, C.s, levels=C.levels, batch=b), dtype=dt, device=dev) for C in Cs]
    r[f'pt_{b}_s'] = T() - t
    r[f'pt_{b}_rel'] = max(float((o - m).abs().max() / m.abs().max()) for o, m in zip(out, ref))
    print(json.dumps(r), flush=True); del out; torch.cuda.empty_cache()
rows = []
for C in Cs:
    C.taus = list(C.taus0)
    rows.append(MA.cell_rows(C))
with torch.no_grad():
    for b in (2048, 8192):
        t = T()
        out = [MA.moments_rows(c, Cs[0].n, tt, nn, oo, s=Cs[0].s, levels=Cs[0].levels, batch=b) for c, tt, nn, oo in rows]
        r[f'mr_percell_{b}_s'] = T() - t
        r[f'mr_percell_{b}_rel'] = max(float((o - m).abs().max() / m.abs().max()) for o, m in zip(out, ref))
    cat_c = np.concatenate([x[0] for x in rows]); cat_t = torch.cat([x[1] for x in rows]); cat_n = torch.cat([x[2] for x in rows]); cat_o = torch.cat([x[3] for x in rows])
    for b in (4096, 8192, 16384):
        torch.cuda.reset_peak_memory_stats()
        t = T()
        allm = MA.moments_rows(cat_c, Cs[0].n, cat_t, cat_n, cat_o, s=Cs[0].s, levels=Cs[0].levels, batch=b)
        r[f'mr_all_{b}_s'] = T() - t; r[f'mr_all_{b}_peakGB'] = torch.cuda.max_memory_allocated() / 1e9
        print(json.dumps(r), flush=True)
    parts = torch.split(allm, [len(x[0]) for x in rows])
    r['mr_all_rel'] = max(float((o - m).abs().max() / m.abs().max()) for o, m in zip(parts, ref))
r['cells'] = len(Cs); r['elements'] = int(sum(len(C.cells) for C in Cs))
print(json.dumps(r), flush=True)
