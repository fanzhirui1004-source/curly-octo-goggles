"""Sensitivity batch-size study (src_v3): the reverse-mode sensitivity pass (moments_ad.cell_sens, skip_full, energy and
volume columns, as opt_design --sens ad) on plate cells, row batches 512 ... 8192, with the tetrahedron VJP as before
(TetMoments2 in chunks of 65536 + index_add, path 'old') and row-indexed (OPL_V3_SENS_ROWS: fast_tet2.TetMomentsRows,
path 'rows'). Per cell and configuration:
  - wall seconds per pass: interleaved rounds (every configuration once per round, the order rotated each round),
    median / min / max; then rounds with the allocator cache emptied before every pass (as t_sensb / t_sensc timed);
  - device peak above the base (allocated) and the growth of the reserved memory;
  - from one profiled pass (torch.profiler): row batches, tetrahedra, kernel launches, host synchronisations and the
    time the host spent blocked in them, summed device time (kernels, copies, fills): wall minus device time is the time
    the GPU waited for the host;
  - forward / backward (energy) / backward (volume) split, summed over the batches;
  - result against path 'old' at batch 2048 (bitwise, largest relative difference).
Fit per path: wall = a + c * batches (c: the cost of one more row batch). Host load (load average, the container's CPU
throttling counters) before and after.
usage: python t_sensd.py BODY_DIR CASE [CASE ...] [--rounds 5] [--rounds-empty 3]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import numpy as np
import torch
from torch.autograd import DeviceType
from torch.profiler import profile, ProfilerActivity
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import moments_ad as MA, fast_tet2 as FT2

ap = argparse.ArgumentParser(); ap.add_argument('body'); ap.add_argument('cases', nargs='+')
ap.add_argument('--rounds', type=int, default=5); ap.add_argument('--rounds-empty', type=int, default=3)
ap.add_argument('--batches', default='512,1024,2048,4096,8192')
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
BATCHES = [int(x) for x in a.batches.split(',')]
CONFIGS = [(p, b) for p in ('old', 'rows') for b in BATCHES]
NT = [0]
_a2, _ar = FT2.TetMoments2.apply, FT2.TetMomentsRows.apply


def c2(tets, tref, tw):
    NT[0] += len(tets)
    return _a2(tets, tref, tw)


def cr(tets, own, n, tref, tw):
    NT[0] += len(tets)
    return _ar(tets, own, n, tref, tw)


FT2.TetMoments2.apply, FT2.TetMomentsRows.apply = c2, cr
LAUNCH = {'cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC', 'cuLaunchKernelEx'}
BLOCK = {'cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize', 'cudaMemcpy', 'cudaMemcpyAsync'}
SYNC = {'cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize'}


def host_load():
    d = dict(loadavg=[round(x, 1) for x in os.getloadavg()])
    try:
        for line in open('/sys/fs/cgroup/cpu.stat'):
            k, v = line.split()
            if k in ('nr_periods', 'nr_throttled', 'throttled_usec', 'usage_usec'):
                d[k] = int(v)
    except OSError:
        pass
    return d


def setup(case):
    C = TE.Cell(case, a.body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, a.body, BD.TMP / case)
    geo = TL.Geo(case, a.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo)
    g = torch.Generator(device='cuda'); g.manual_seed(0)
    q = torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g)
    with torch.no_grad():
        u = f.field(q).to(torch.float64)
        gd = C.energy_density(u)
        e0 = torch.zeros((gd.shape[0], gd.shape[1], 1), dtype=gd.dtype, device=gd.device); e0[:, 0, 0] = 1.0
        G = torch.cat([gd, e0], 2)
    del f, u, gd, e0, geo
    return C, G


def run(C, G, path, bt):
    MA.ROWS = path == 'rows'
    return MA.cell_sens(C, G, batch=bt, skip_full=True)


def wall(C, G, path, bt, empty=False):
    torch.cuda.synchronize()
    if empty:
        torch.cuda.empty_cache()
    t = time.perf_counter()
    run(C, G, path, bt)
    torch.cuda.synchronize()
    return time.perf_counter() - t


def memory(C, G, path, bt):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    b, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    NT[0] = 0
    s = run(C, G, path, bt)
    torch.cuda.synchronize()
    return s, dict(peak_MiB=round((torch.cuda.max_memory_allocated() - b) / 2 ** 20, 1),
                   reserved_growth_MiB=round((torch.cuda.max_memory_reserved() - r0) / 2 ** 20, 1), tets=NT[0])


def profiled(C, G, path, bt):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        run(C, G, path, bt)
        torch.cuda.synchronize()
        w = time.perf_counter() - t
    ev = prof.events()
    dev = [e for e in ev if e.device_type == DeviceType.CUDA]
    cpu = [e for e in ev if e.device_type == DeviceType.CPU]
    return dict(wall_profiled_s=round(w, 3), device_s=round(sum(e.time_range.elapsed_us() for e in dev) / 1e6, 3),
                device_ops=len(dev), launches=sum(e.name in LAUNCH for e in cpu), syncs=sum(e.name in SYNC for e in cpu),
                blocked_s=round(sum(e.time_range.elapsed_us() for e in cpu if e.name in BLOCK) / 1e6, 3),
                aten_ops=sum(e.name.startswith('aten::') for e in cpu))


def split(C, G, path, bt):
    """cell_sens(skip_full) / moments_vjp written out with a timer around the forward and each backward pass."""
    MA.ROWS = path == 'rows'
    cells, taus, nrm, off = MA.cell_rows(C)
    sf = getattr(C, 'surface', 'P')
    keep = torch.nonzero(~MA.full_rows(cells, C.n, taus, nrm, off, s=C.s, surface=sf)).squeeze(1)
    ck = np.asarray(cells)[keep.cpu().numpy()]
    tk, nk, ok, gk = taus[keep], nrm[keep], off[keep], G[keep]
    tf = te = tv = 0.0
    for b0 in range(0, len(ck), bt):
        b1 = min(len(ck), b0 + bt)
        tb = tk[b0:b1].detach().clone().requires_grad_(True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.enable_grad():
            FT2.SKIP_FWD[0] = True
            Mb = MA.moments_rows(ck[b0:b1], C.n, tb, nk[b0:b1], ok[b0:b1], s=C.s, levels=C.levels, surface=sf, batch=b1 - b0)
            FT2.SKIP_FWD[0] = False
            torch.cuda.synchronize(); t1 = time.perf_counter()
            torch.autograd.grad((Mb * gk[b0:b1, :, 0]).sum(), tb, retain_graph=True)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            FT2.E0_HINT[0] = True
            torch.autograd.grad((Mb * gk[b0:b1, :, 1]).sum(), tb)
            FT2.E0_HINT[0] = False
            torch.cuda.synchronize(); t3 = time.perf_counter()
        tf += t1 - t0; te += t2 - t1; tv += t3 - t2
        del Mb, tb
    return dict(forward_s=round(tf, 3), backward_energy_s=round(te, 3), backward_volume_s=round(tv, 3))


res = dict(host_start=host_load(), device=torch.cuda.get_device_name(), torch=torch.__version__)
for case in a.cases:
    C, G = setup(case)
    cells, taus, nrm, off = MA.cell_rows(C)
    nrows = int((~MA.full_rows(cells, C.n, taus, nrm, off, s=C.s, surface=getattr(C, 'surface', 'P'))).sum())
    r = dict(kind=C.kind, elements=int(len(C.cells)), rows_reverse=nrows, configs={})
    for p, b in CONFIGS:                                                   # warm (Triton compiles, caches)
        run(C, G, p, b)
    ref, _ = memory(C, G, 'old', 2048)
    for p, b in CONFIGS:
        s, m = memory(C, G, p, b)
        r['configs'][f'{p}_{b}'] = dict(path=p, batch=b, batches=-(-nrows // b), equal=bool(torch.equal(s, ref)),
                                        rel=float((s - ref).abs().max() / ref.abs().max()), **m)
    W = {k: [] for k in r['configs']}
    WE = {k: [] for k in r['configs']}
    loads = []
    for rd in range(a.rounds):
        o = CONFIGS[rd % len(CONFIGS):] + CONFIGS[:rd % len(CONFIGS)]
        for p, b in o:
            W[f'{p}_{b}'].append(wall(C, G, p, b))
        loads.append(host_load()['loadavg'][0])
    for rd in range(a.rounds_empty):
        o = CONFIGS[rd % len(CONFIGS):] + CONFIGS[:rd % len(CONFIGS)]
        for p, b in o:
            WE[f'{p}_{b}'].append(wall(C, G, p, b, empty=True))
    for p, b in CONFIGS:
        k = f'{p}_{b}'
        x = r['configs'][k]
        x.update(wall_s=dict(median=round(float(np.median(W[k])), 3), min=round(min(W[k]), 3), max=round(max(W[k]), 3)),
                 wall_empty_cache_s=dict(median=round(float(np.median(WE[k])), 3), min=round(min(WE[k]), 3),
                                         max=round(max(WE[k]), 3)) if WE[k] else None,
                 profile=profiled(C, G, p, b))
        pr = x['profile']
        pr.update(per_batch=dict(launches=round(pr['launches'] / x['batches']), syncs=round(pr['syncs'] / x['batches']),
                                 aten_ops=round(pr['aten_ops'] / x['batches'])),
                  gpu_idle_s=round(x['wall_s']['median'] - pr['device_s'], 3))
        print(json.dumps({case: {k: x}}), flush=True)
    for p, b in CONFIGS:
        if b in (1024, 2048, 4096):
            r['configs'][f'{p}_{b}']['split'] = split(C, G, p, b)
    fit = {}
    for p in ('old', 'rows'):
        xs = np.array([r['configs'][f'{p}_{b}']['batches'] for b in BATCHES], float)
        ys = np.array([r['configs'][f'{p}_{b}']['wall_s']['median'] for b in BATCHES], float)
        if len(set(xs)) > 1:
            c, a0 = np.polyfit(xs, ys, 1)
            fit[p] = dict(per_batch_s=round(float(c), 4), fixed_s=round(float(a0), 3))
    r['fit_wall_vs_batches'] = fit
    r['loadavg_rounds'] = loads
    res[case] = r
    print(json.dumps({case: {k: v for k, v in r.items() if k != 'configs'}}), flush=True)
    del C, G
    torch.cuda.empty_cache()
MA.ROWS = False
res['host_end'] = host_load()
print(json.dumps(res), flush=True)
