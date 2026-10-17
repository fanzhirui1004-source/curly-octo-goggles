"""P2 under CUDA-graph replay (the solve's case): one full plate cell's S_hat captured as a graph with the previous and
with the new float32 element product (ke_symv.TUNE2 off / on), replay time per application (B = 1, 3: eager S_hat is
bound by the CPU's launches, so kernel time shows only in replays), the kernel-level profile of a replay, and all given
cells on two streams (one graph per cell, as the solve runs them).
usage: python t_symv3.py BODY_DIR:CASE[,CASE2,...] [--rep 30]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import ke_symv as KS
from torch.profiler import profile, ProfilerActivity

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=30)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, cases = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)


def cell(case):
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo)
    with torch.no_grad():
        f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
    if hasattr(C, 'store64'):
        C.store64(allow=True)
    return C, f


g = torch.Generator(device=dev); g.manual_seed(0)
cells = [cell(c) for c in cases.split(',')]


def rhs(C, B):
    q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
    return q - C.Q @ (C.Q.T @ q)


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


_CAPS = {}


class Graphed:
    def __init__(self, f, np_, B, pool=None, cap=0):
        self.qs = torch.zeros((np_, B), dtype=torch.float64, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                f.s_hat(self.qs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        if cap not in _CAPS:
            _CAPS[cap] = torch.cuda.Stream()
        with torch.no_grad(), torch.cuda.graph(self.g, pool=pool, stream=_CAPS[cap]):
            self.out = f.s_hat(self.qs)
        torch.cuda.synchronize()

    def __call__(self, q):
        self.qs.copy_(q)
        self.g.replay()
        return self.out


def kprof(fn, reps=5):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    rows = [e for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    tot = sum(e.self_device_time_total for e in rows) / 1e3 / reps
    top = sorted(rows, key=lambda e: -e.self_device_time_total)[:8]
    return dict(kernel_ms=round(tot, 3), top=[(e.key[:60], round(e.self_device_time_total / 1e3 / reps, 3), e.count // reps) for e in top])


res = dict(cases=cases)
with torch.no_grad():
    C, f = cells[0]
    for B in (1, 3):
        q = rhs(C, B)
        r = {}
        outs = {}
        for tune in (False, True):
            KS.TUNE2 = tune
            e0, te = timed(lambda: f.s_hat(q), a.rep)
            G = Graphed(f, C.np_, B)
            y, tg = timed(lambda: G(q), a.rep)
            outs[tune] = y.clone()
            k = 'new' if tune else 'old'
            r[k] = dict(ms_eager=round(te, 2), ms_graph=round(tg, 2), prof=kprof(lambda: G(q)))
            del G
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        r['rel_new_old'] = float(((outs[True] - outs[False]).norm(dim=0) / outs[False].norm(dim=0)).max())
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': r}), flush=True)
    if len(cells) > 1:
        ns = 2
        st = [torch.cuda.Stream() for _ in range(ns)]
        qs = [rhs(C_, 1) for C_, _ in cells]
        for tune in (False, True):
            KS.TUNE2 = tune
            pools = [None] * ns
            Gs = []
            for j, (C_, f_) in enumerate(cells):
                G = Graphed(f_, C_.np_, 1, pool=pools[j % ns], cap=100 + j % ns + (10 if tune else 0))
                pools[j % ns] = G.g.pool()
                Gs.append(G)

            def graph_all():
                cur = torch.cuda.current_stream()
                for s_ in st:
                    s_.wait_stream(cur)
                ys = []
                for j, G in enumerate(Gs):
                    with torch.cuda.stream(st[j % ns]):
                        ys.append(G(qs[j]))
                for s_ in st:
                    cur.wait_stream(s_)
                return ys
            _, tg = timed(graph_all, a.rep)
            _, t1 = timed(lambda: [G(q_) for G, q_ in zip(Gs, qs)], a.rep)
            res[f'streams2_{"new" if tune else "old"}'] = dict(cells=len(cells), ms_graph_all_2streams=round(tg, 2),
                                                              ms_graph_all_1stream=round(t1, 2))
            print(json.dumps({f'streams2_{"new" if tune else "old"}': res[f'streams2_{"new" if tune else "old"}']}), flush=True)
            del Gs, pools
            torch.cuda.synchronize(); torch.cuda.empty_cache()
print(json.dumps(res), flush=True)
