"""The float32 element product's kernel choice measured where it runs: inside S_hat of a full plate cell captured as a CUDA
graph (the solve's case). For each candidate (ke_symv.CFG[BP] = (kernel, BI, warps, stages)): the graph replay time of
S_hat and the element-product kernels' device time per call (torch.profiler over the replays), B = 1 (plate PCG),
3 (hlat222 PCG), 16 (coarse A Z); then the best B = 1 choice against the current one with two cells on two streams.
(t_symv4's cold-L2 micro-benchmark did not predict the in-situ times.)
usage: python t_symv5.py BODY_DIR:CASE[,CASE2] [--rep 20]"""
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
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
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


cells = [cell(c) for c in cases.split(',')]
g = torch.Generator(device=dev); g.manual_seed(0)


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep                                  # ms


_CAPS = {}


class Graphed:
    def __init__(self, f, np_, B, pool=None, cap=0):
        self.qs = torch.zeros((np_, B), dtype=torch.float64, device=dev)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                f.s_hat(self.qs)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        if cap not in _CAPS:
            _CAPS[cap] = torch.cuda.Stream()
        with torch.no_grad(), torch.cuda.graph(self.g, pool=pool, stream=_CAPS[cap]):
            self.out = f.s_hat(self.qs)
        torch.cuda.synchronize()

    def __call__(self, q):
        self.qs.copy_(q); self.g.replay(); return self.out


def symv_ms(fn, reps=5):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    rows = [e for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA and e.key.startswith('_symv')]
    ms = sum(e.self_device_time_total for e in rows) / 1e3 / reps
    n = sum(e.count for e in rows) // reps
    return round(ms, 3), n


CANDS = {1: [None, ('old', 8, 2, 0), ('up', 32, 1, 0), ('up', 32, 2, 0), ('up', 8, 2, 0), ('upp', 8, 1, 2), ('upp', 16, 1, 2),
             ('old', 16, 1, 0)],
         3: [None, ('old', 16, 1, 0), ('old', 8, 2, 0), ('upp', 8, 1, 2), ('up', 16, 2, 0)],
         16: [None, ('old', 4, 2, 0), ('old', 16, 1, 0), ('old', 8, 1, 0), ('old', 16, 2, 0)]}
res = dict(cases=cases)
C, f = cells[0]
with torch.no_grad():
    for B, cands in CANDS.items():
        BP = 1 if B == 1 else (4 if B == 3 else 16)
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        q = q - C.Q @ (C.Q.T @ q)
        rows, ref = [], None
        for cfg in cands:
            KS.CFG = {} if cfg is None else {BP: cfg}
            G = Graphed(f, C.np_, B)
            ms = ev(lambda: G(q), a.rep)
            sm, n = symv_ms(lambda: G(q))
            y = G(q).clone()
            if ref is None:
                ref = y
            rows.append(dict(cfg=cfg or 'default', ms_graph=round(ms, 2), symv_ms=sm, symv_calls=n,
                             rel=float(((y - ref).norm(dim=0) / ref.norm(dim=0)).max())))
            print(json.dumps(dict(B=B, **rows[-1])), flush=True)
            del G; torch.cuda.synchronize(); torch.cuda.empty_cache()
        KS.CFG = {}
        res[f'B{B}'] = rows
    if len(cells) > 1:
        best1 = min(res['B1'], key=lambda r: r['ms_graph'])['cfg']
        st = [torch.cuda.Stream() for _ in range(2)]
        qs = [torch.randn((C_.np_, 1), dtype=torch.float64, device=dev, generator=g) for C_, _ in cells]
        for tag, cfg in (('default', None), ('best', None if best1 == 'default' else tuple(best1))):
            KS.CFG = {} if cfg is None else {1: cfg}
            pools = [None, None]; Gs = []
            for j, (C_, f_) in enumerate(cells):
                G = Graphed(f_, C_.np_, 1, pool=pools[j % 2], cap=100 + j % 2 + (10 if tag == 'best' else 0))
                pools[j % 2] = G.g.pool(); Gs.append(G)

            def run_all():
                cur = torch.cuda.current_stream()
                for s_ in st:
                    s_.wait_stream(cur)
                for j, G in enumerate(Gs):
                    with torch.cuda.stream(st[j % 2]):
                        G(qs[j])
                for s_ in st:
                    cur.wait_stream(s_)
            res[f'streams2_{tag}'] = dict(cfg=str(cfg), ms=round(ev(run_all, a.rep), 2))
            print(json.dumps({f'streams2_{tag}': res[f'streams2_{tag}']}), flush=True)
            del Gs, pools; torch.cuda.synchronize(); torch.cuda.empty_cache()
        KS.CFG = {}
print(json.dumps(res), flush=True)
