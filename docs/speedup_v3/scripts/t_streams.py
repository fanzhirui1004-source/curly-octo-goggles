"""Concurrency probe: S_hat of several cells at B = 1, one after another on one stream vs on separate CUDA streams
(issued from one thread), and the CPU time to issue one S_hat. Outputs per cell are compared with the sequential run.
usage: python t_streams.py BODY_DIR CASE1,CASE2[,...] [--rep 10] [--B 1]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

ap = argparse.ArgumentParser(); ap.add_argument('body'); ap.add_argument('cases')
ap.add_argument('--rep', type=int, default=10); ap.add_argument('--B', type=int, default=1)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
nets, qs = [], []
g = torch.Generator(device='cuda')
for i, case in enumerate(a.cases.split(',')):
    C = TE.Cell(case, a.body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, a.body, BD.TMP / case)
    geo = TL.Geo(case, a.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo)
    g.manual_seed(i)
    q = torch.randn((C.np_, a.B), dtype=torch.float64, device='cuda', generator=g)
    with torch.no_grad():
        f.s_hat(q)
    nets.append(f); qs.append(q)
n = len(nets)
torch.cuda.synchronize()
res = dict(cases=a.cases.split(','), B=a.B)
with torch.no_grad():
    # CPU issue time of one S_hat while the GPU is kept busy (sleep kernel ahead of it)
    ts = []
    for r in range(5):
        torch.cuda._sleep(int(2e9)); t0 = time.perf_counter(); nets[0].s_hat(qs[0]); ts.append(time.perf_counter() - t0)
        torch.cuda.synchronize()
    res['cpu_issue_ms'] = round(1e3 * min(ts), 2)
    # sequential
    ref = [f.s_hat(q) for f, q in zip(nets, qs)]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(a.rep):
        for f, q in zip(nets, qs):
            f.s_hat(q)
    torch.cuda.synchronize(); res['seq_ms_per_cell'] = round(1e3 * (time.perf_counter() - t0) / (a.rep * n), 2)
    # one stream per cell
    streams = [torch.cuda.Stream() for _ in range(n)]
    main = torch.cuda.current_stream()
    for s in streams:
        s.wait_stream(main)
    outs = [None] * n

    def round_():
        for i, (f, q, s) in enumerate(zip(nets, qs, streams)):
            with torch.cuda.stream(s):
                outs[i] = f.s_hat(q)
        for s in streams:
            main.wait_stream(s)
    round_(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.rep):
        round_()
    torch.cuda.synchronize(); res['par_ms_per_cell'] = round(1e3 * (time.perf_counter() - t0) / (a.rep * n), 2)
    res['max_rel_vs_seq'] = [float((o - r_).abs().max() / r_.abs().max()) for o, r_ in zip(outs, ref)]
    res['peak_GiB'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
print(json.dumps(res), flush=True)
