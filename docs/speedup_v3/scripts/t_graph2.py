"""Concurrency with CUDA graphs: S_hat (B = 1) of several resident cells captured as one graph each (own memory pool),
replayed one after another on one stream vs round robin on 2 / 3 streams; eager sequential for reference.
usage: python t_graph2.py BODY_DIR CASE1,CASE2,CASE3 [--rep 20]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

ap = argparse.ArgumentParser(); ap.add_argument('body'); ap.add_argument('cases'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
nets, qs, graphs, outs = [], [], [], []
g = torch.Generator(device='cuda')
for i, case in enumerate(a.cases.split(',')):
    C = TE.Cell(case, a.body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, a.body, BD.TMP / case)
    geo = TL.Geo(case, a.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo); C.store64()
    g.manual_seed(i)
    q = torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g)
    with torch.no_grad():
        f.s_hat(q)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f.s_hat(q); f.s_hat(q)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        G = torch.cuda.CUDAGraph()
        with torch.cuda.graph(G):
            y = f.s_hat(q)
    nets.append(f); qs.append(q); graphs.append(G); outs.append(y)
n = len(nets)
torch.cuda.synchronize()
res = dict(cases=a.cases.split(','), mem_GiB=round(torch.cuda.memory_reserved() / 2 ** 30, 2))


def timed(fn):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(a.rep):
        fn()
    torch.cuda.synchronize(); return round(1e3 * (time.perf_counter() - t) / (a.rep * n), 2)


with torch.no_grad():
    res['eager_seq_ms_per_cell'] = timed(lambda: [f.s_hat(q) for f, q in zip(nets, qs)])
    res['graph_seq_ms_per_cell'] = timed(lambda: [G.replay() for G in graphs])
    main = torch.cuda.current_stream()
    for k in (2, 3):
        if k > n:
            continue
        sts = [torch.cuda.Stream() for _ in range(k)]

        def rnd():
            ev = torch.cuda.Event(); ev.record(main)
            for i, G in enumerate(graphs):
                st = sts[i % k]; st.wait_event(ev)
                with torch.cuda.stream(st):
                    G.replay()
            for st in sts:
                main.wait_stream(st)
        res[f'graph_{k}streams_ms_per_cell'] = timed(rnd)
print(json.dumps(res), flush=True)
