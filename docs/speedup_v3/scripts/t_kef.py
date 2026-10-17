"""P13b (OPL_V3_KMF_EF): the float32 correction product's elements and ghost faces in one CUDA kernel (kmf.apply_ef_,
kef_kernel: the element of a face handles it, its own 27 nodes' face rows join the element result before the one atomic
scatter) against the two kernels of P13 (kmf.gf_apply_, then kmf.apply_) on a full plate cell, B = 1, 2, 4, 16:
  - accuracy against the float64 product (float64 moments and templates; float64 ghost faces);
  - device time per product (warm, back to back; cold: a 256 MB buffer rewritten before each call), the fused kernel
    for grid caps of 2 / 4 / 8 / 16 blocks per SM and none, and columns per block by B (0) or one (1).
usage: python t_kef.py BODY_DIR:CASE [--rep 30]"""
import os, sys, json, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import torch
import teacher as TE, fastidx as FI, kmf as KMF

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=30)
a = ap.parse_args()
FI.ON = True
body, case = a.spec.split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
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


E = len(C.M)
sms = torch.cuda.get_device_properties(0).multi_processor_count
with torch.no_grad():
    st = C._Kc.mf_state()
    res = dict(case=case, elements=E, nb=int(C.nb), sms=sms, kind=st['kind'], lib_error=KMF._LIB_ERR, points=KMF.PTS,
               faces=st['gf']['nf'] if 'gf' in st else None)
    print(json.dumps(res), flush=True)
    efn = KMF.efn_table(st['gf'], E)
    res['elements_with_faces'] = int((efn >= 0).any(1).sum())
    Tf = C.Tm.reshape(125, 81 * 81)
    X = torch.randn((C.nb, 16), dtype=f32, device=dev, generator=g).contiguous()  # B columns: the first B of these
    X64 = X.double()
    Y64 = C.GF64.matvec(X64, torch.zeros_like(X64)).mul_(C.gamma)       # float64 reference, all 16 columns
    for lo in range(0, E, 1024):
        de = C.dofs[lo:lo + 1024]
        Ke = (C.M[lo:lo + 1024] @ Tf).reshape(-1, 81, 81)
        Y64.index_add_(0, de.reshape(-1), torch.bmm(Ke, X64[de]).reshape(-1, 16))
        del Ke
    del X64
    for B in (1, 2, 4, 16):
        x = X[:, :B].contiguous()
        y64 = Y64[:, :B]
        nrm = y64.abs().max()
        acc = lambda y_: dict(rel_f64=float((y_.double() - y64).abs().max() / nrm), l2_f64=float((y_.double() - y64).norm() / y64.norm()))
        two = lambda y_: KMF.apply_(KMF.gf_apply_(y_, st['gf'], x), st, x)
        y = torch.zeros_like(x)
        r = dict(two=acc(two(torch.zeros_like(x))))
        r['two']['warm'] = round(ev(lambda: two(y), a.rep), 1)
        r['two']['cold'] = round(cold(lambda: two(y), a.rep), 1)
        r['two']['gf_cold'] = round(cold(lambda: KMF.gf_apply_(y, st['gf'], x), a.rep), 1)
        r['two']['el_cold'] = round(cold(lambda: KMF.apply_(y, st, x), a.rep), 1)
        for nbk in ((0, 1) if B > 1 else (0,)):
            for bps in (2, 4, 8, 16, 0):
                blk = bps * sms
                y1 = KMF.apply_ef_(torch.zeros_like(x), st, x, nb=nbk, blocks=blk)
                y2 = KMF.apply_ef_(torch.zeros_like(x), st, x, nb=nbk, blocks=blk)
                d = acc(y1)
                d['rel_rerun'] = float((y2 - y1).abs().max() / y1.abs().max())
                d['warm'] = round(ev(lambda: KMF.apply_ef_(y, st, x, nb=nbk, blocks=blk), a.rep), 1)
                d['cold'] = round(cold(lambda: KMF.apply_ef_(y, st, x, nb=nbk, blocks=blk), a.rep), 1)
                r[f'ef_nb{nbk}_bps{bps}'] = d
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
