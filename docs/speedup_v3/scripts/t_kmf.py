"""P13 (OPL_V3_KMF): the matrix-free float32 element product (kmf.py: 125 Gauss points with fitted weights; CUDA C++
sum-factorised kernel kmf_kernel.cu, and the Triton dot form) against the formed-matrix product (ke_symv on the packed
float32 matrices of _Kfp32.hold) on a full plate cell, B = 1, 2, 4:
  - accuracy: all against the float64 template product sum_e P_e^T (M_e T) P_e x (float64 moments and templates);
  - device time per product alone (warm, back to back; and cold: a 256 MB buffer rewritten before each call): the CUDA
    kernel with 2 / 4 / 8 elements per block, the Triton form with its default tiles; the formation GEMM separately.
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
    KMF.CUDA = False
    stT = KMF.state(C)
    KMF.CUDA = True
    stC = KMF.state(C)
    res['cuda'] = dict(kind=stC['kind'], lib_error=KMF._LIB_ERR)
    res['state_MB'] = dict(nodes=round(stC['nd'].numel() * 4 / 2 ** 20, 2), weights=round(stC['w'].numel() * 4 / 2 ** 20, 2)
                           if stC['kind'] == 'cuda' else None, packed=round(E * 3321 * 4 / 2 ** 20, 1))
    W = stT['WT'][:125].t().double()
    res['weights'] = dict(neg_frac=float((W < 0).double().mean()),
                          abs_over_sum_median=float((W.abs().sum(1) / W.sum(1).abs().clamp_min(1e-300)).median()),
                          abs_over_sum_max=float((W.abs().sum(1) / W.sum(1).abs().clamp_min(1e-300)).max()))
    print(json.dumps(res), flush=True)
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
        yT = torch.zeros_like(x); KMF.apply_(yT, stT, x)
        r = dict(rel_packed_f64=float((yP.double() - y64).abs().max() / nrm), rel_triton_f64=float((yT.double() - y64).abs().max() / nrm),
                 l2_packed_f64=float((yP.double() - y64).norm() / y64.norm()), l2_triton_f64=float((yT.double() - y64).norm() / y64.norm()))
        y = torch.zeros_like(x)
        r['packed_us'] = dict(warm=round(ev(lambda: KS.symv_(y, P, dofs, x), a.rep), 1), cold=round(cold(lambda: KS.symv_(y, P, dofs, x), a.rep), 1))
        r['triton_us'] = dict(warm=round(ev(lambda: KMF.apply_(y, stT, x), a.rep), 1), cold=round(cold(lambda: KMF.apply_(y, stT, x), a.rep), 1))
        if stC['kind'] == 'cuda':
            for epb in (2, 4, 8):
                yC = torch.zeros_like(x); KMF.apply_(yC, stC, x, epb=epb)
                yC2 = torch.zeros_like(x); KMF.apply_(yC2, stC, x, epb=epb)
                r[f'cuda{epb}'] = dict(rel_f64=float((yC.double() - y64).abs().max() / nrm), l2_f64=float((yC.double() - y64).norm() / y64.norm()),
                                       rel_rerun=float((yC2 - yC).abs().max() / yC.abs().max()),
                                       warm_us=round(ev(lambda: KMF.apply_(y, stC, x, epb=epb), a.rep), 1),
                                       cold_us=round(cold(lambda: KMF.apply_(y, stC, x, epb=epb), a.rep), 1))
        if stC['kind'] == 'cuda' and Kc.gf is not None:                   # ghost faces: kernel vs current float32 path
            gst = KMF.gf_state(C, stC['nd'])
            x64 = x.double()
            g64 = C.GF64.matvec(x64, torch.zeros_like(x64)) * C.gamma
            gn = g64.abs().max()
            gcur = Kc.gf.matvec(x, torch.zeros_like(x)).mul_(Kc.gamma)
            r['gf'] = dict(faces=gst['nf'], rel_cur_f64=float((gcur.double() - g64).abs().max() / gn),
                           cur_us=dict(warm=round(ev(lambda: Kc.gf.matvec(x, torch.zeros_like(x)).mul_(Kc.gamma), a.rep), 1),
                                       cold=round(cold(lambda: Kc.gf.matvec(x, torch.zeros_like(x)).mul_(Kc.gamma), a.rep), 1)))
            sms = torch.cuda.get_device_properties(0).multi_processor_count
            for fpb in (2, 4, 8):
                for blk in (2 * sms, 4 * sms, 8 * sms, 0):
                    gk = KMF.gf_apply_(torch.zeros_like(x), gst, x, fpb=fpb, blocks=blk)
                    r['gf'][f'fpb{fpb}_blk{blk}'] = dict(rel_f64=float((gk.double() - g64).abs().max() / gn),
                                                         warm=round(ev(lambda: KMF.gf_apply_(y, gst, x, fpb=fpb, blocks=blk), a.rep), 1),
                                                         cold=round(cold(lambda: KMF.gf_apply_(y, gst, x, fpb=fpb, blocks=blk), a.rep), 1))
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
