"""P14 (OPL_V3_KMF64): the float64 K product of a deployed plate cell without element matrices (kmf.state64: partial
elements matrix-free at the 125 Gauss points with float64 weights, full elements by fullke's shared-matrix GEMM, ghost
faces through float64 45 x 45 Gram matrices) against the current product (stored float64 packed matrices of the
partial elements, ke_symv; ghost faces by template GEMMs), B = 1, 3, 16: relative difference, device time per product
(warm, back to back; cold: a 256 MB buffer rewritten before each call), and the bytes each keeps per cell.
usage: python t_kmf64.py BODY_DIR:CASE [--rep 20]"""
import os, sys, json, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import torch
import teacher as TE, fastidx as FI, fullke as FK, kmf as KMF

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FI.ON = True
body, case = a.spec.split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
KMF.ON64 = False
stored = C.store64(allow=True)
g = torch.Generator(device=dev); g.manual_seed(0)
buf = torch.zeros(64 * 2 ** 20, dtype=torch.float32, device=dev)


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


nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts)
fk = FK.setup(C)['f64']
res = dict(case=case, elements=len(C.M), partial=int(fk['n_part']), full=int(fk['n_full']), stored64=stored,
           stored_MB=round(nbytes(C._k64_part or []) / 2 ** 20, 1) if stored else 0.0,
           gf64_iTc_MB=round(C.GF64.iTc.numel() * 4 / 2 ** 20, 2) if C.GF64.iTc is not None else None)
with torch.no_grad():
    KMF.ON64 = True
    st = C.mf64_state()
    res['cuda'] = dict(kind=KMF.kind(), lib_error=KMF._LIB_ERR)
    res['mf64_MB'] = round(nbytes([st['nd'], st['wl'], st['wm'], st['gf']['fe'], st['gf']['fn'], st['gf']['G'], st['gf']['nd']]) / 2 ** 20, 2)
    print(json.dumps(res), flush=True)
    for B in (1, 3, 16):
        x = torch.randn((C.nb, B), dtype=torch.float64, device=dev, generator=g)
        KMF.ON64 = False
        y0 = C.Kx(x).clone()
        t0 = dict(warm=round(ev(lambda: C.Kx(x), a.rep), 1), cold=round(cold(lambda: C.Kx(x), a.rep), 1))
        KMF.ON64 = True
        y1 = C.Kx(x).clone(); y1b = C.Kx(x).clone()
        t1 = dict(warm=round(ev(lambda: C.Kx(x), a.rep), 1), cold=round(cold(lambda: C.Kx(x), a.rep), 1))
        nrm = y0.abs().max()
        parts = {}
        xc = x.contiguous()
        parts['gf64_kernel'] = round(cold(lambda: KMF.gf_apply64_(torch.zeros_like(xc), st['gf'], xc), a.rep), 1)
        parts['elem64_kernel'] = round(cold(lambda: KMF.apply64_(torch.zeros_like(xc), st, xc), a.rep), 1)
        parts['gf64_current'] = round(cold(lambda: C.GF64.matvec(xc, torch.zeros_like(xc)).mul_(C.gamma), a.rep), 1)
        r = dict(rel_max=float((y1 - y0).abs().max() / nrm), rel_l2=float((y1 - y0).norm() / y0.norm()),
                 rel_rerun=float((y1b - y1).abs().max() / nrm), current_us=t0, mf64_us=t1, parts_cold_us=parts)
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
