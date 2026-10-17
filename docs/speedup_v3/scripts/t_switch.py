"""S_hat of a full plate cell with a module switch off / on, as the solve runs it (CUDA-graph replay) and eagerly,
B = 1, 3, 16: time, and the relative difference on / off against the off path's rerun difference (atomics).
Switches (comma list, each MODULE.ATTR, set to True for 'on'): e.g. fastnet.HFUSE, gffused.ON.
Also, with --gf, the ghost-face product alone (teacher.GhostFaces.matvec, float32) with gffused off / on and tiles.
usage: python t_switch.py BODY_DIR:CASE --sw fastnet.HFUSE,gffused.ON [--rep 20] [--gf]"""
import os, sys, json, time, argparse, importlib
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--sw', required=True)
ap.add_argument('--rep', type=int, default=20); ap.add_argument('--gf', action='store_true')
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
SW = []
for s_ in a.sw.split(','):                                             # MODULE.ATTR (on: True) or MODULE.ATTR=VALUE
    s_, _, val = s_.partition('=')
    mod, attr = s_.rsplit('.', 1)
    m_ = importlib.import_module(mod)
    SW.append((m_, attr, getattr(m_, attr), int(val) if val else True))


def set_sw(on):
    for m, at, off_v, on_v in SW:
        setattr(m, at, on_v if on else off_v)


h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
set_sw(False)
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


res = dict(case=case, switches=a.sw)
with torch.no_grad():
    if a.gf:
        import gffused as GFF
        gf = C._Kc.gf
        nb = C.nb
        for B in (1, 3, 16):
            x = torch.randn((nb, B), dtype=torch.float32, device=dev, generator=g).contiguous()
            GFF.ON = False
            y0 = gf.matvec(x, torch.zeros_like(x)).clone(); y0b = gf.matvec(x, torch.zeros_like(x)).clone()
            r = dict(us_cur=round(cold(lambda: gf.matvec(x, torch.zeros_like(x)), a.rep), 1),
                     rel_rerun=float((y0b - y0).abs().max() / y0.abs().max()), faces=int(gf.faces))
            best = None
            for fb in (16, 32, 64):
                for kc in (16, 32, 64):
                    for w in (2, 4, 8):
                        try:
                            yy = GFF.matvec(gf, x, torch.zeros_like(x), fb=fb, kc=kc, warps=w).clone()
                            us = cold(lambda: GFF.matvec(gf, x, torch.zeros_like(x), fb=fb, kc=kc, warps=w), a.rep)
                            if best is None or us < best[0]:
                                best = (round(us, 1), fb, kc, w, float((yy - y0).abs().max() / y0.abs().max()))
                        except Exception as ex:                          # noqa: BLE001
                            r.setdefault('errors', []).append(repr(ex)[:100])
            r['best_fused'] = best
            try:                                                                 # P9b two-phase (no atomics)
                y2 = GFF.matvec2(gf, x, torch.zeros_like(x)).clone()
                r['us_twophase'] = round(cold(lambda: GFF.matvec2(gf, x, torch.zeros_like(x)), a.rep), 1)
                r['rel_twophase'] = float((y2 - y0).abs().max() / y0.abs().max())
                r['twophase_maxdeg'] = int(gf._gf_inc['maxd'])
            except Exception as ex:                                              # noqa: BLE001
                r['twophase_error'] = repr(ex)[:200]
            res[f'gf_B{B}'] = r
            print(json.dumps({f'gf_B{B}': r}), flush=True)
    for B in (1, 3, 16):
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        q = q - C.Q @ (C.Q.T @ q)
        out = {}
        r = {}
        for on in (False, True):
            set_sw(on)
            te = ev(lambda: f.s_hat(q), a.rep)
            G = Graphed(f, C.np_, B)
            tg = ev(lambda: G(q), a.rep)
            out[on] = G(q).clone(); out[(on, 'b')] = G(q).clone()
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CUDA]) as p_:
                for _ in range(5):
                    G(q)
                torch.cuda.synchronize()
            rows_ = [e_ for e_ in p_.key_averages() if e_.device_type == torch.autograd.DeviceType.CUDA]
            top_ = [(e_.key[:50], round(e_.self_device_time_total / 5e3, 3), e_.count // 5)
                    for e_ in sorted(rows_, key=lambda e_: -e_.self_device_time_total)[:14]]
            r['on' if on else 'off'] = dict(ms_eager=round(te, 2), ms_graph=round(tg, 2),
                                            kernel_ms=round(sum(e_.self_device_time_total for e_ in rows_) / 5e3, 2), top=top_)
            del G; torch.cuda.synchronize(); torch.cuda.empty_cache()
        set_sw(False)
        r['rel_on_off'] = float(((out[True] - out[False]).norm(dim=0) / out[False].norm(dim=0)).max())
        r['rel_off_rerun'] = float(((out[(False, 'b')] - out[False]).norm(dim=0) / out[False].norm(dim=0)).max())
        res[f's_hat_B{B}'] = r
        print(json.dumps({f's_hat_B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
