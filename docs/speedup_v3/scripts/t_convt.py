"""Check of S27 (OPL_V3_CONVT=1, fastnet._convt): every transposed U-Net convolution of a full plate cell computed as the
forward convolution with the flipped, channel-swapped weight, against cuDNN's conv_transpose3d (which is itself not
reproducible run to run: the original is compared with its own rerun), B = 1, 3, 16; the convolution's time each way;
then S_hat (B = 1, 3, 16) with S27 off / on: time and the difference against the original's rerun difference.
usage: python t_convt.py BODY_DIR:CASE [--rep 10]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import torch.nn.functional as Fn
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

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


def rel(x, y):
    return float((x - y).abs().max() / y.abs().max())


res = dict(case=case, conv=[])
with torch.no_grad():
    for l in range(f.levels):
        t = f.c.trans[l]
        if 'dense_idx' not in t or 'e_src' in t:
            continue
        m_ = t['m']
        for B in (1, 3, 16):
            x = torch.randn((B, f.W_in.shape[1], m_, m_, m_), dtype=torch.float32, device=dev, generator=g)
            for k in range(2 * f.cpl):
                W = f.convs[l][k]
                y0, t0 = timed(lambda: Fn.conv_transpose3d(x, W, padding=1), a.rep)
                y0b = Fn.conv_transpose3d(x, W, padding=1)
                W2 = FN._shared_flipT(f.convs[l])[k]
                y1, t1 = timed(lambda: Fn.conv3d(x, W2, padding=1), a.rep)
                y1b = Fn.conv3d(x, W2, padding=1)
                res['conv'].append(dict(l=l, m=m_, B=B, k=k, ms_convT=round(t0, 3), ms_conv=round(t1, 3),
                                        rel_new=rel(y1, y0), rel_orig_rerun=rel(y0b, y0), new_rerun_equal=bool(torch.equal(y1, y1b))))
    cv = res['conv']
    res['conv_summary'] = dict(n=len(cv), max_rel_new=max(r['rel_new'] for r in cv), max_rel_orig_rerun=max(r['rel_orig_rerun'] for r in cv),
                               new_deterministic=all(r['new_rerun_equal'] for r in cv),
                               ms_convT=round(sum(r['ms_convT'] for r in cv), 2), ms_conv=round(sum(r['ms_conv'] for r in cv), 2))
    print(json.dumps(res['conv_summary']), flush=True)
    for B in (1, 3, 16):
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev, generator=g)
        q = q - C.Q @ (C.Q.T @ q)
        FN.CONVT = False
        s0, t0 = timed(lambda: f.s_hat(q), a.rep)
        s0 = s0.clone(); s0b = f.s_hat(q).clone()
        FN.CONVT = True
        s1, t1 = timed(lambda: f.s_hat(q), a.rep)
        FN.CONVT = False
        r = dict(ms_orig=round(t0, 2), ms_convt=round(t1, 2),
                 rel_new=float(((s1 - s0).norm(dim=0) / s0.norm(dim=0)).max()),
                 rel_orig_rerun=float(((s0b - s0).norm(dim=0) / s0.norm(dim=0)).max()))
        res[f's_hat_B{B}'] = r
        print(json.dumps({f's_hat_B{B}': r}), flush=True)
res.pop('conv')
print(json.dumps(res), flush=True)
