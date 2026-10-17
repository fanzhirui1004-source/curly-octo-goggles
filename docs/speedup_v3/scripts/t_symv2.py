"""Check of P2 (ke_symv.symv_ with OPL_V3_SYMV2: the single-read kernel for one / two columns, retuned tiles otherwise)
on S_hat of a full plate cell: time and difference against the previous kernel configuration, compared with the
previous configuration against itself (the scatters' atomics), B = 1, 3, 16.
usage: python t_symv2.py BODY_DIR:CASE [--rep 10]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import ke_symv as KS

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=10)
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


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


res = dict(case=case)
with torch.no_grad():
    for B in (1, 3, 16):
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        q = q - C.Q @ (C.Q.T @ q)
        KS.TUNE2 = False
        s0, t0 = timed(lambda: f.s_hat(q), a.rep)
        s0 = s0.clone(); s0b = f.s_hat(q).clone()
        KS.TUNE2 = True
        s1, t1 = timed(lambda: f.s_hat(q), a.rep)
        r = dict(ms_old=round(t0, 2), ms_new=round(t1, 2), rel_new=float(((s1 - s0).norm(dim=0) / s0.norm(dim=0)).max()),
                 rel_old_rerun=float(((s0b - s0).norm(dim=0) / s0.norm(dim=0)).max()))
        res[f's_hat_B{B}'] = r
        print(json.dumps({f's_hat_B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
