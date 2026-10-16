"""Stored float64 partial-element matrices (teacher.Cell.store64, OPL_V3_KE64=store): the stored chunks must equal a fresh
formation bitwise; S_hat time with the stored matrices vs formed per product; S_hat output difference (atomics noise).
usage: python t_ke64.py BODY_DIR:CASE [--B 1,3,16]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI


def timed(fn, rep=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1e3 * (time.perf_counter() - t) / rep


ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--B', default='1,3,16')
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
g = torch.Generator(device='cuda'); g.manual_seed(0)
with torch.no_grad():
    f.s_hat(torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g))
    C.store64()
    stored = [t.clone() for t in C._k64_part]; dp = C._dp64_part.clone()
    C._k64_part = C._dp64_part = None
    C._form64_part()
    r = dict(case=case, chunks=len(stored), MB=round(sum(t.numel() * t.element_size() for t in stored) / 2 ** 20, 1),
             bitwise=all(torch.equal(x, y) for x, y in zip(stored, C._k64_part)) and torch.equal(dp, C._dp64_part))
    out = {}
    for B in [int(x) for x in a.B.split(',')]:
        q = torch.randn((C.np_, B), dtype=torch.float64, device='cuda', generator=g)
        C._k64_part = C._dp64_part = None
        y0, t0 = timed(lambda: f.s_hat(q))
        C._form64_part()
        y1, t1 = timed(lambda: f.s_hat(q))
        C._k64_part = C._dp64_part = None
        y2, t2 = timed(lambda: f.s_hat(q))
        out[B] = dict(ms_form=round(t0, 2), ms_store=round(t1, 2), ms_form2=round(t2, 2),
                      rel_store=float((y1 - y0).abs().max() / y0.abs().max()), rel_rerun=float((y2 - y0).abs().max() / y0.abs().max()))
    r['B'] = out
print(json.dumps(r), flush=True)
