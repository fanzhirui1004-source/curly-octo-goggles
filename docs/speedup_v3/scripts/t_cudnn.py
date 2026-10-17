"""cuDNN algorithm choice for the network's 3-D convolutions: S_hat (B = 1, 3) with torch.backends.cudnn.benchmark off
(heuristic choice, the current default) vs on (timed choice per shape), TF32 stays off (OPL_CONV_FP32=1); output diff.
usage: python t_cudnn.py BODY_DIR:CASE"""
import os, sys, json, time
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
f = FN.FastNet(h.add(geo), geo); C.store64()
g = torch.Generator(device='cuda'); g.manual_seed(0)
r = dict(case=case, allow_tf32=bool(torch.backends.cudnn.allow_tf32))


def timed(fn, rep=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, round(1e3 * (time.perf_counter() - t) / rep, 2)


with torch.no_grad():
    for B in (1, 3):
        q = torch.randn((C.np_, B), dtype=torch.float64, device='cuda', generator=g)
        torch.backends.cudnn.benchmark = False
        y0, t0 = timed(lambda: f.s_hat(q))
        q32 = q.to(torch.float32); cc = f.RPpinv @ q32
        e0, te0 = timed(lambda: f.ext(q32 - f.RP @ cc))
        torch.backends.cudnn.benchmark = True
        y1, t1 = timed(lambda: f.s_hat(q))
        e1, te1 = timed(lambda: f.ext(q32 - f.RP @ cc))
        torch.backends.cudnn.benchmark = False
        y2, t2 = timed(lambda: f.s_hat(q))
        r[f'B{B}'] = dict(shat_ms=(t0, t1, t2), ext_ms=(te0, te1), rel_bench=float((y1 - y0).abs().max() / y0.abs().max()),
                          rel_rerun=float((y2 - y0).abs().max() / y0.abs().max()),
                          ext_rel=float((e1 - e0).abs().max() / e0.abs().max()))
print(json.dumps(r), flush=True)
