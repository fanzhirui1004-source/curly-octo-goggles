"""The U-Net's 3-D convolutions of one full plate cell (FastNet._conv: input = the dense grid as a channels-last view,
F = 32 channels, 3x3x3, padding 1), per level and B = 1, 3, 16: time of the current call against
  - contiguous NCDHW input (heuristic algorithm),
  - torch.backends.cudnn.benchmark (timed algorithm choice), channels-last view and contiguous,
  - explicit channels-last-3d input and weight,
with the relative difference to the current call (rounding: another algorithm / layout sums in another order) and
whether each variant is deterministic (rerun bitwise equal). TF32 stays off (OPL_CONV_FP32=1).
usage: python t_conv2.py BODY_DIR:CASE [--rep 20]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import torch.nn.functional as Fn
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

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
g = torch.Generator(device=dev); g.manual_seed(0)


def timed(fn, rep):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep


def rel(x, y):
    return float((x.float() - y.float()).abs().max() / y.abs().max())


res = dict(case=case, allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
           levels=[])
with torch.no_grad():
    for l in range(f.levels):
        t = f.c.trans[l]
        if 'dense_idx' not in t or 'e_src' in t:
            continue
        m_ = t['m']
        W = f.convs[l][0]
        F_ = W.shape[1]
        row = dict(l=l, m=m_, F=F_, active=int(len(t['dense_idx'])), voxels=m_ ** 3)
        for B in (1, 3, 16):
            dense = torch.zeros((m_ ** 3, B, F_), device=dev)
            dense[t['dense_idx']] = torch.randn((len(t['dense_idx']), B, F_), device=dev, generator=g)
            xv = dense.permute(1, 2, 0).reshape(B, F_, m_, m_, m_)                 # current: channels-last view
            xc = xv.contiguous()
            xl = xv.contiguous(memory_format=torch.channels_last_3d)
            Wl = W.contiguous(memory_format=torch.channels_last_3d)
            torch.backends.cudnn.benchmark = False
            y0 = Fn.conv3d(xv, W, padding=1); y0b = Fn.conv3d(xv, W, padding=1)
            r = dict(us_current=round(timed(lambda: Fn.conv3d(xv, W, padding=1), a.rep), 1), det_current=bool(torch.equal(y0, y0b)))
            y = Fn.conv3d(xc, W, padding=1)
            r.update(us_ncdhw=round(timed(lambda: Fn.conv3d(xc, W, padding=1), a.rep), 1), rel_ncdhw=rel(y, y0),
                     det_ncdhw=bool(torch.equal(y, Fn.conv3d(xc, W, padding=1))))
            y = Fn.conv3d(xl, Wl, padding=1)
            r.update(us_cl=round(timed(lambda: Fn.conv3d(xl, Wl, padding=1), a.rep), 1), rel_cl=rel(y, y0))
            torch.backends.cudnn.benchmark = True
            y = Fn.conv3d(xv, W, padding=1)
            r.update(us_bench_current=round(timed(lambda: Fn.conv3d(xv, W, padding=1), a.rep), 1), rel_bench_current=rel(y, y0),
                     det_bench_current=bool(torch.equal(y, Fn.conv3d(xv, W, padding=1))))
            y = Fn.conv3d(xc, W, padding=1)
            r.update(us_bench_ncdhw=round(timed(lambda: Fn.conv3d(xc, W, padding=1), a.rep), 1), rel_bench_ncdhw=rel(y, y0),
                     det_bench_ncdhw=bool(torch.equal(y, Fn.conv3d(xc, W, padding=1))))
            torch.backends.cudnn.benchmark = False
            r['us_contig_copy'] = round(timed(lambda: xv.contiguous(), a.rep), 1)
            import conv3 as C3                                                   # src_v3 P10: Triton implicit GEMM
            dn = dense.contiguous()
            for bm in (32, 64, 128):
                for w in (2, 4, 8):
                    try:
                        y = C3.conv3d(dn, W, m_, bm=bm, warps=w)
                        us = timed(lambda: C3.conv3d(dn, W, m_, bm=bm, warps=w), a.rep)
                        rr = rel(y.permute(1, 2, 0).reshape(B, F_, m_, m_, m_), y0)
                        if 'us_triton' not in r or us < r['us_triton']:
                            r.update(us_triton=round(us, 1), triton_cfg=(bm, w), rel_triton=rr)
                    except Exception as ex:                              # noqa: BLE001
                        r.setdefault('triton_errors', []).append(repr(ex)[:80])
            row[f'B{B}'] = r
        res['levels'].append(row)
        print(json.dumps(row), flush=True)
print(json.dumps(res), flush=True)
