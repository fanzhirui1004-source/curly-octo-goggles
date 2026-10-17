"""Item 3: the reverse-mode sensitivity pass of one full plate cell (moments_ad.cell_sens, skip_full, the energy and the
volume column, as opt_design --sens ad): seconds per cell and device peak above the base for row batches 1024 ... all
rows, the result's largest relative difference to batch 2048; the split of one 2048-row batch into the forward pass
(with graph) and the two backward passes; and two cells' passes run concurrently from two threads on two CUDA streams
against one after the other (wall time).
usage: python t_sensb.py BODY_DIR:CASE"""
import os, sys, json, time, threading
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import numpy as np
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import moments_ad as MA

FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
g = torch.Generator(device='cuda'); g.manual_seed(0)
q = torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g)
with torch.no_grad():
    u = f.field(q).to(torch.float64)
    gd = C.energy_density(u)
    e0 = torch.zeros((gd.shape[0], gd.shape[1], 1), dtype=gd.dtype, device=gd.device); e0[:, 0, 0] = 1.0
    G = torch.cat([gd, e0], 2)
del f
r = dict(case=case, elements=int(len(C.cells)))


def timed(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    b = torch.cuda.memory_allocated(); t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, round(time.perf_counter() - t, 3), round((torch.cuda.max_memory_allocated() - b) / 2 ** 20, 1)


ref, _, _ = timed(lambda: MA.cell_sens(C, G, batch=2048, skip_full=True))
for bt in (1024, 2048, 4096, 8192, 1 << 20):
    MA.cell_sens(C, G, batch=bt, skip_full=True)                            # warm (compiles, caches)
    ts, pk = [], 0
    for _ in range(3):
        s, t_, p_ = timed(lambda: MA.cell_sens(C, G, batch=bt, skip_full=True))
        ts.append(t_); pk = max(pk, p_)
    r[f'batch_{bt}'] = dict(s=round(float(np.median(ts)), 3), peak_MiB=pk, rel_vs_2048=float((s - ref).abs().max() / ref.abs().max()))
    print(json.dumps({f'batch_{bt}': r[f'batch_{bt}']}), flush=True)
# split of one 2048-row batch: forward with graph, backward energy, backward volume
cells, taus, nrm, off = MA.cell_rows(C)
keep = torch.nonzero(~MA.full_rows(cells, C.n, taus, nrm, off, s=C.s)).squeeze(1)
kc = keep.cpu().numpy()[:2048]
kk = keep[:2048]
import fast_tet2 as FT2
tb = taus[kk].detach().clone().requires_grad_(True)
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.enable_grad():
    FT2.SKIP_FWD[0] = True
    Mb = MA.moments_rows(np.asarray(cells)[kc], C.n, tb, nrm[kk], off[kk], s=C.s, levels=C.levels, batch=len(kk))
    FT2.SKIP_FWD[0] = False
    torch.cuda.synchronize(); t1 = time.perf_counter()
    gk0, = torch.autograd.grad((Mb * G[kk][:, :, 0]).sum(), tb, retain_graph=True)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    FT2.E0_HINT[0] = True
    gk1, = torch.autograd.grad((Mb * G[kk][:, :, 1]).sum(), tb)
    FT2.E0_HINT[0] = False
    torch.cuda.synchronize(); t3 = time.perf_counter()
r['split_2048'] = dict(forward_s=round(t1 - t0, 3), backward_energy_s=round(t2 - t1, 3), backward_volume_s=round(t3 - t2, 3))
del Mb, tb
# two cells concurrently (the same cell twice) on two streams from two threads, against sequential
seq, t_seq, _ = timed(lambda: [MA.cell_sens(C, G, batch=2048, skip_full=True) for _ in range(2)])
outs = [None, None]
sts = [torch.cuda.Stream(), torch.cuda.Stream()]


def work(i):
    with torch.cuda.stream(sts[i]):
        outs[i] = MA.cell_sens(C, G, batch=2048, skip_full=True)
    sts[i].synchronize()


def par():
    th = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    return outs


par()
_, t_par, pk_par = timed(par)
r['two_cells'] = dict(sequential_s=t_seq, threads_streams_s=t_par, peak_MiB=pk_par,
                      rel=float(max((o - ref).abs().max() / ref.abs().max() for o in outs)))
print(json.dumps(r), flush=True)
