"""CUDA-graph prototype (src_v3, idea #1): the two-stream PCG matvec is bound by the CPU's kernel launches (~1400 per
S_hat, ~20 ms of launches for ~9 ms of GPU work at B = 1). Here one plate cell's FastNet.s_hat is captured as a CUDA
graph (static input and output, intermediates in the graph's private pool) and replayed:
  - capture: does it work at all (no host synchronisation inside S_hat), its time, the pool's device memory and the
    host memory of the graph (RSS);
  - replay against eager: relative difference (the scatters' atomics: compared with eager against itself) and the time
    per application, B = 1 and 3, slot weights stored and recomputed in every application;
  - all given cells on 1-4 streams (round robin, as lat_multi's matvec issues them): eager against one graph per cell
    (one shared memory pool per stream).
usage: python t_graph.py BODY_DIR:CASE[,CASE2,...] [--rep 20] [--streams 1,2,3,4]"""
import os, sys, json, time, argparse, traceback
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
ap.add_argument('--streams', default='1,2,3,4'); ap.add_argument('--hang-s', type=float, default=120)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, cases = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)


def rss_gib():
    return int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2 ** 30


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
    """s_hat of one FastNet as a CUDA graph for a fixed width B (static input qs, static output out). Graphs that may be
    replayed at the same time are captured on different capture streams (cap): cuBLAS keeps one workspace per (handle,
    stream), and a graph keeps the capture stream's workspace; two concurrent replays sharing it can deadlock (stream-K
    GEMMs synchronise through it)."""

    def __init__(self, f, np_, B, pool=None, cap=0):
        self.qs = torch.zeros((np_, B), dtype=torch.float64, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                f.s_hat(self.qs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize(); torch.cuda.empty_cache()                # (graph() empties the cache itself)
        r0, m0, t = rss_gib(), torch.cuda.memory_reserved(), time.perf_counter()
        self.g = torch.cuda.CUDAGraph()
        if cap not in _CAPS:
            _CAPS[cap] = torch.cuda.Stream()
        with torch.no_grad(), torch.cuda.graph(self.g, pool=pool, stream=_CAPS[cap]):
            self.out = f.s_hat(self.qs)
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - t
        self.rss_MiB = (rss_gib() - r0) * 1024
        self.pool_MiB = (torch.cuda.memory_reserved() - m0) / 2 ** 20

    def __call__(self, q):
        self.qs.copy_(q)
        self.g.replay()
        return self.out


import threading
_PH = ['start', time.time()]


def phase(name):
    _PH[0], _PH[1] = name, time.time()
    print(json.dumps(dict(phase=name)), flush=True)


def _watchdog():                                                     # a GPU deadlock cannot be interrupted otherwise
    while True:
        time.sleep(5)
        if time.time() - _PH[1] > a.hang_s:
            print(json.dumps(dict(hang=_PH[0], seconds=round(time.time() - _PH[1]))), flush=True)
            os._exit(3)


threading.Thread(target=_watchdog, daemon=True).start()
res = dict(cases=cases)
with torch.no_grad():
    C, f = cells[0]
    for mode in ('stored', 'recompute'):
        if mode == 'recompute':
            if f._rc is None and getattr(f, '_rc_src', None) is None:
                res['recompute'] = 'no recompute inputs on this cell'
                break
            f.recompute_slots()
        for B in (1, 3):
            q = rhs(C, B)
            s0, t0 = timed(lambda: f.s_hat(q), a.rep)
            s0 = s0.clone()
            s0b = f.s_hat(q).clone()
            r = dict(ms_eager=round(t0, 2), rel_eager_rerun=float(((s0b - s0).norm(dim=0) / s0.norm(dim=0)).max()))
            try:
                phase(f'{mode} B={B}')
                G = Graphed(f, C.np_, B)
                s1, t1 = timed(lambda: G(q), a.rep)
                r.update(ms_graph=round(t1, 2), rel_graph=float(((s1 - s0).norm(dim=0) / s0.norm(dim=0)).max()),
                         capture_s=round(G.capture_s, 3), pool_MiB=round(G.pool_MiB, 1), rss_MiB=round(G.rss_MiB, 1))
                q2 = rhs(C, B)                                               # a second right-hand side through the same graph
                e2 = f.s_hat(q2).clone()
                r['rel_graph_q2'] = float(((G(q2) - e2).norm(dim=0) / e2.norm(dim=0)).max())
                del G
            except Exception as e:                                       # noqa: BLE001 (report why capture fails)
                r['error'] = repr(e)[:400]
                r['trace'] = traceback.format_exc()[-2500:]
                torch.cuda.synchronize()
            res[f'{mode}_B{B}'] = r
            print(json.dumps({f'{mode}_B{B}': r}), flush=True)
    if len(cells) > 1:
        f._assign_slots(*f._slot_tensors()); f._rc = None                 # the first cell back to stored slot weights
        for ns in [int(x) for x in a.streams.split(',')]:
            st = [torch.cuda.Stream() for _ in range(ns)]
            qs = [rhs(C_, 1) for C_, _ in cells]

            def eager_all():
                cur = torch.cuda.current_stream()
                for s_ in st:
                    s_.wait_stream(cur)
                ys = []
                for j, (C_, f_) in enumerate(cells):
                    with torch.cuda.stream(st[j % ns]):
                        ys.append(f_.s_hat(qs[j]))
                for s_ in st:
                    cur.wait_stream(s_)
                return ys
            r = dict(cells=len(cells))
            try:
                pools = [None] * ns
                Gs = []
                for j, (C_, f_) in enumerate(cells):                     # one shared pool per stream
                    phase(f'capture ns={ns} cell {j}')
                    G = Graphed(f_, C_.np_, 1, pool=pools[j % ns], cap=100 + j % ns)
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
                phase(f'eager ns={ns}')
                ye, te = timed(eager_all, a.rep)
                ye = [y.clone() for y in ye]
                phase(f'graphs ns={ns}')
                yg, tg = timed(graph_all, a.rep)
                r.update(ms_eager_all=round(te, 2), ms_graph_all=round(tg, 2),
                         rel_graph=max(float(((g_ - e_).norm(dim=0) / e_.norm(dim=0)).max()) for g_, e_ in zip(yg, ye)),
                         pools_MiB=round(sum(G.pool_MiB for G in Gs), 1))
                del Gs, pools
            except Exception as e:                                       # noqa: BLE001
                r.update(error=repr(e)[:400], trace=traceback.format_exc()[-2000:])
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            res[f'streams_{ns}'] = r
            print(json.dumps({f'streams_{ns}': r}), flush=True)
        res['ms_eager_single'] = [round(timed(lambda: f_.s_hat(rhs(C_, 1)), a.rep)[1], 2) for C_, f_ in cells]
print(json.dumps(res), flush=True)
