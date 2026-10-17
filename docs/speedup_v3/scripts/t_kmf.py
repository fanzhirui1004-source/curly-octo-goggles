"""P13 (OPL_V3_KMF): the matrix-free float32 element product (kmf.py: 125 Gauss points with fitted weights) against
the formed-matrix product (ke_symv on the packed float32 matrices of _Kfp32.hold) on a full plate cell, B = 1, 2, 4:
  - accuracy: both against the float64 template product sum_e P_e^T (M_e T) P_e x (float64 moments and templates);
  - device time per product alone (warm, back to back; and cold: a 256 MB buffer rewritten before each call), for
    a sweep of tiles (EB elements, PB points, warps); the formed-matrix product with its formation GEMM separately.
usage: python t_kmf.py BODY_DIR:CASE [--rep 30]"""
import os, sys, json, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import ke_symv as KS, kmf as KMF

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=30)
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
g = torch.Generator(device=dev); g.manual_seed(0)
buf = torch.zeros(64 * 2 ** 20, dtype=torch.float32, device=dev)
f32, f64 = torch.float32, torch.float64


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep                              # us


def cold(fn, rep):
    return ev(lambda: (buf.add_(1.0), fn()), rep) - ev(lambda: buf.add_(1.0), rep)


Kc = C._Kc
E = len(C.M)
res = dict(case=case, elements=E, nb=int(C.nb))
with torch.no_grad():
    nd, WT, DN, lam, mu = Kc.mf_state()
    res['state_MB'] = dict(nodes=round(nd.numel() * 4 / 2 ** 20, 2), weights=round(WT.numel() * 4 / 2 ** 20, 2),
                           packed=round(E * 3321 * 4 / 2 ** 20, 1))
    W = WT[:125].t().double()
    res['weights'] = dict(neg_frac=float((W < 0).double().mean()),
                          abs_over_sum_median=float((W.abs().sum(1) / W.sum(1).abs().clamp_min(1e-300)).median()),
                          abs_over_sum_max=float((W.abs().sum(1) / W.sum(1).abs().clamp_min(1e-300)).max()))
    Kc._form_p()
    P = Kc._kp_tmp
    dofs = C.dofs.contiguous()
    form = lambda: [torch.matmul(Kc.m32(lo, lo + C._lean_chunk), Kc.Tup32, out=P[lo:lo + C._lean_chunk])
                    for lo in range(0, E, C._lean_chunk)]
    res['form_us'] = round(ev(form, 10), 1)
    Tf = C.Tm.reshape(125, 81 * 81)
    X4 = torch.randn((C.nb, 4), dtype=f32, device=dev, generator=g).contiguous()
    Y64 = torch.zeros((C.nb, 4), dtype=f64, device=dev)
    for lo in range(0, E, 2048):                                          # float64 reference, all four columns
        de = C.dofs[lo:lo + 2048]
        Ke = (C.M[lo:lo + 2048] @ Tf).reshape(-1, 81, 81)
        Y64.index_add_(0, de.reshape(-1), torch.bmm(Ke, X4.double()[de]).reshape(-1, 4))
        del Ke
    for B in (1, 2, 4):
        x = X4[:, :B].contiguous()
        y64 = Y64[:, :B]
        nrm = y64.abs().max()
        yP = torch.zeros_like(x); KS.symv_(yP, P, dofs, x)
        yM = torch.zeros_like(x); KMF.matvec_(yM, nd, WT, DN, x, lam, mu)
        yM2 = torch.zeros_like(x); KMF.matvec_(yM2, nd, WT, DN, x, lam, mu)
        r = dict(rel_packed_f64=float((yP.double() - y64).abs().max() / nrm), rel_mf_f64=float((yM.double() - y64).abs().max() / nrm),
                 rel_mf_rerun=float((yM2 - yM).abs().max() / yM.abs().max()),
                 l2_packed_f64=float((yP.double() - y64).norm() / y64.norm()), l2_mf_f64=float((yM.double() - y64).norm() / y64.norm()))
        y = torch.zeros_like(x)
        r['packed_us'] = dict(warm=round(ev(lambda: KS.symv_(y, P, dofs, x), a.rep), 1), cold=round(cold(lambda: KS.symv_(y, P, dofs, x), a.rep), 1))
        r['mf_default_us'] = dict(warm=round(ev(lambda: KMF.matvec_(y, nd, WT, DN, x, lam, mu), a.rep), 1),
                                  cold=round(cold(lambda: KMF.matvec_(y, nd, WT, DN, x, lam, mu), a.rep), 1))
        sweep = []
        for eb in (16, 32, 64):
            for pb in (16, 32):
                for w in (2, 4, 8):
                    try:
                        yy = torch.zeros_like(x); KMF.matvec_(yy, nd, WT, DN, x, lam, mu, eb=eb, pb=pb, warps=w)
                        err = float((yy.double() - y64).abs().max() / nrm)
                        us = cold(lambda: KMF.matvec_(y, nd, WT, DN, x, lam, mu, eb=eb, pb=pb, warps=w), a.rep)
                        sweep.append((round(us, 1), eb, pb, w, err))
                    except Exception as ex:                                  # noqa: BLE001
                        sweep.append((None, eb, pb, w, repr(ex)[:120]))
                print(json.dumps({f'B{B}_sweep_eb{eb}': [s_ for s_ in sweep if s_[1] == eb]}), flush=True)
        ok = sorted([s_ for s_ in sweep if s_[0] is not None])
        r['mf_best'] = ok[:4]
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
