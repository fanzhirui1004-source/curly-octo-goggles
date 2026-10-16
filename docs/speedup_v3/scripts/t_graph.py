"""CUDA graph probe: capture one resident cell's S_hat (B = 1) and replay it; output vs eager (atomics noise), GPU time
per S_hat eager vs replay, CPU time to issue (eager) vs to replay, extra memory of the graph pool. A capture failure
names the first operation that synchronises with the host (or is not capturable).
usage: python t_graph.py BODY_DIR:CASE"""
import os, sys, json, time, traceback
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
C.store64()
g = torch.Generator(device='cuda'); g.manual_seed(0)
q = torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g)
r = dict(case=case)
with torch.no_grad():
    y0 = f.s_hat(q); f.s_hat(q); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        f.s_hat(q)
    torch.cuda.synchronize(); r['eager_ms'] = round(1e3 * (time.perf_counter() - t) / 20, 2)
    torch.cuda._sleep(int(2e9)); t = time.perf_counter(); f.s_hat(q); r['eager_issue_ms'] = round(1e3 * (time.perf_counter() - t), 2)
    torch.cuda.synchronize()
    qs = q.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            f.s_hat(qs)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    m0 = torch.cuda.memory_reserved()
    G = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(G):
            ys = f.s_hat(qs)
        r['captured'] = True
    except Exception as ex:                                                               # noqa
        r['captured'] = False; r['error'] = ''.join(traceback.format_exception_only(type(ex), ex))[-600:]
        r['trace'] = traceback.format_exc()[-3000:]
    if r['captured']:
        torch.cuda.synchronize()
        r['graph_pool_MiB'] = round((torch.cuda.memory_reserved() - m0) / 2 ** 20, 1)
        G.replay(); torch.cuda.synchronize()
        r['rel_vs_eager'] = float((ys - y0).abs().max() / y0.abs().max())
        t = time.perf_counter()
        for _ in range(20):
            G.replay()
        torch.cuda.synchronize(); r['replay_ms'] = round(1e3 * (time.perf_counter() - t) / 20, 2)
        torch.cuda._sleep(int(2e9)); t = time.perf_counter(); G.replay(); r['replay_issue_ms'] = round(1e3 * (time.perf_counter() - t), 3)
        torch.cuda.synchronize()
        q2 = torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g)
        qs.copy_(q2); G.replay(); y2 = f.s_hat(q2); torch.cuda.synchronize()
        r['rel_new_input'] = float((ys - y2).abs().max() / y2.abs().max())
print(json.dumps(r), flush=True)
