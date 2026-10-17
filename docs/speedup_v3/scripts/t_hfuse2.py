"""P8 check: the hyperedge layer in one kernel (hyperfused.layer, OPL_V3_HFUSE) against fused_hyper's gather -> bmm ->
scatter on a full plate cell's real layers (element, face and fringe hyperedges), B = 1, 3, 16: time per layer
application (cold L2: a 256 MB buffer rewritten before each call, its time subtracted) for tile choices (EB, FC, warps),
relative difference; then S_hat with OPL_V3_HFUSE off / on as CUDA-graph replays (the solve's case) and eagerly, B = 1, 3,
16: time and difference against the off path's rerun difference.
usage: python t_hfuse2.py BODY_DIR:CASE [--rep 20]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import fused_hyper as FH, hyperfused as HF

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
with torch.no_grad():
    f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
if hasattr(C, 'store64'):
    C.store64(allow=True)
g = torch.Generator(device=dev); g.manual_seed(0)
buf = torch.zeros(64 * 2 ** 20, dtype=torch.float32, device=dev)


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep                                  # ms


def cold(fn, rep):                                                  # us
    return 1000 * (ev(lambda: (buf.add_(1.0), fn()), rep) - ev(lambda: buf.add_(1.0), rep))


def rel(x, y):
    return float((x - y).abs().max() / y.abs().max())


res = dict(case=case, layers=[])
layers = [('elem', f.elem[0])]
for nm in ('face', 'fringe'):
    L_ = getattr(f, nm, None)
    if L_:
        layers.append((nm, L_[0]))
with torch.no_grad(), f.slots():
    for nm, hl in layers:
        row = dict(layer=nm, Eh=int(hl.Eh), S=int(hl.hn.shape[1]), H=int(hl.H), N=int(hl.N))
        for B in (1, 3, 16):
            X = torch.randn((hl.N, B, hl.W.shape[1]), dtype=torch.float32, device=dev, generator=g)
            y0 = hl.fwd_res(X).clone()                                       # current path (gather, bmm, scatter)
            y0b = hl.fwd_res(X).clone()
            r = dict(us_cur=round(cold(lambda: hl.fwd_res(X), a.rep), 1), rel_rerun=rel(y0b, y0))
            best = None
            for eb in (2, 4, 8):
                for fc in (8, 16):
                    if eb * hl.H * fc * hl.W.shape[1] > 8192:                     # register tile limit (larger tiles
                        continue                                                 # take minutes to compile)
                    for w in (4, 8):
                        try:
                            run = lambda: HF.layer(X.reshape(hl.N, -1), hl.hn, hl.a, hl.b, hl.deg, hl.W,
                                                   X.reshape(hl.N, -1).clone(), False, True, eb=eb, fc=fc, warps=w)
                            yy = run()
                            us = cold(run, a.rep)
                            if best is None or us < best[0]:
                                best = (round(us, 1), eb, fc, w, rel(yy.view_as(y0), y0))
                        except Exception as ex:                          # noqa: BLE001 (register limits)
                            pass
            r['best_fused'] = best
            best4 = None                                                         # the 4-head dot form (layer4)
            for eb in (16, 32):
                for w in (2, 4, 8):
                    try:
                        run = lambda: HF.layer4(X.reshape(hl.N, -1), hl.hn, hl.a, hl.b, hl.deg, hl.W,
                                                X.reshape(hl.N, -1).clone(), False, True, eb=eb, warps=w)
                        yy = run()
                        us = cold(run, a.rep)
                        if best4 is None or us < best4[0]:
                            best4 = (round(us, 1), eb, w, rel(yy.view_as(y0), y0))
                    except Exception as ex:                              # noqa: BLE001
                        r.setdefault('errors4', []).append(repr(ex)[:100])
            r['best_fused4'] = best4
            row[f'B{B}'] = r
        res['layers'].append(row)
        print(json.dumps(row), flush=True)

_CAPS = {}


class Graphed:
    def __init__(self, f, np_, B):
        self.qs = torch.zeros((np_, B), dtype=torch.float64, device=dev)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                f.s_hat(self.qs)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g):
            self.out = f.s_hat(self.qs)
        torch.cuda.synchronize()

    def __call__(self, q):
        self.qs.copy_(q); self.g.replay(); return self.out


with torch.no_grad():
    for B in (1, 3, 16):
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        q = q - C.Q @ (C.Q.T @ q)
        out = {}
        r = {}
        for on in (False, True):
            FN.HFUSE = on
            te = ev(lambda: f.s_hat(q), a.rep)
            G = Graphed(f, C.np_, B)
            tg = ev(lambda: G(q), a.rep)
            out[on] = G(q).clone(); out[(on, 'b')] = G(q).clone()
            r['on' if on else 'off'] = dict(ms_eager=round(te, 2), ms_graph=round(tg, 2))
            del G; torch.cuda.synchronize(); torch.cuda.empty_cache()
        FN.HFUSE = False
        r['rel_on_off'] = float(((out[True] - out[False]).norm(dim=0) / out[False].norm(dim=0)).max())
        r['rel_off_rerun'] = float(((out[(False, 'b')] - out[False]).norm(dim=0) / out[False].norm(dim=0)).max())
        res[f's_hat_B{B}'] = r
        print(json.dumps({f's_hat_B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
