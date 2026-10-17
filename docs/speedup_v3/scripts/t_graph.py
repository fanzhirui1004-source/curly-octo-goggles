"""CUDA-graph prototype (src_v3, idea #1): the two-stream PCG matvec is bound by the CPU's kernel launches (~1400 per
S_hat, ~20 ms of launches for ~9 ms of GPU work at B = 1). Here one plate cell's FastNet.s_hat is captured as a CUDA
graph (static input and output, intermediates in the graph's private pool) and replayed:
  - capture: does it work at all (no host synchronisation inside S_hat), its time, the pool's device memory and the
    host memory of the graph (RSS);
  - replay against eager: relative difference (the scatters' atomics: compared with eager against itself) and the time
    per application, B = 1 and 3, slot weights stored and recomputed in every application;
  - two cells on two streams: eager (as lat_multi's two-stream matvec issues them) against two graphs.
usage: python t_graph.py BODY_DIR:CASE[,CASE2] [--rep 20]"""
import os, sys, json, time, argparse, traceback
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
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


class Graphed:
    """s_hat of one FastNet as a CUDA graph for a fixed width B (static input qs, static output out)."""

    def __init__(self, f, np_, B, pool=None):
        self.qs = torch.zeros((np_, B), dtype=torch.float64, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                f.s_hat(self.qs)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        r0, m0, t = rss_gib(), torch.cuda.memory_reserved(), time.perf_counter()
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g, pool=pool):
            self.out = f.s_hat(self.qs)
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter() - t
        self.rss_MiB = (rss_gib() - r0) * 1024
        self.pool_MiB = (torch.cuda.memory_reserved() - m0) / 2 ** 20

    def __call__(self, q):
        self.qs.copy_(q)
        self.g.replay()
        return self.out


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
    if len(cells) > 1 and all('error' not in res.get(f'stored_B{B}', {}) for B in (1,)):
        (C1, f1), (C2, f2) = cells[0], cells[1]                       # (the first cell recomputes its slot weights now)
        st = [torch.cuda.Stream(), torch.cuda.Stream()]
        q1, q2 = rhs(C1, 1), rhs(C2, 1)

        def eager2():
            cur = torch.cuda.current_stream()
            for s_ in st:
                s_.wait_stream(cur)
            with torch.cuda.stream(st[0]):
                y1 = f1.s_hat(q1)
            with torch.cuda.stream(st[1]):
                y2 = f2.s_hat(q2)
            for s_ in st:
                cur.wait_stream(s_)
            return y1, y2
        try:
            G1, G2 = Graphed(f1, C1.np_, 1), Graphed(f2, C2.np_, 1)

            def graph2():
                cur = torch.cuda.current_stream()
                for s_ in st:
                    s_.wait_stream(cur)
                with torch.cuda.stream(st[0]):
                    y1 = G1(q1)
                with torch.cuda.stream(st[1]):
                    y2 = G2(q2)
                for s_ in st:
                    cur.wait_stream(s_)
                return y1, y2
            _, te = timed(eager2, a.rep)
            _, tg = timed(graph2, a.rep)
            _, t_1 = timed(lambda: f1.s_hat(q1), a.rep)
            _, t_2 = timed(lambda: f2.s_hat(q2), a.rep)
            res['two_streams'] = dict(ms_eager_pair=round(te, 2), ms_graph_pair=round(tg, 2),
                                      ms_eager_single=[round(t_1, 2), round(t_2, 2)])
        except Exception as e:                                           # noqa: BLE001
            res['two_streams'] = dict(error=repr(e)[:400], trace=traceback.format_exc()[-2000:])
print(json.dumps(res), flush=True)
