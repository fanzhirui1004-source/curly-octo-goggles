"""P15 (OPL_V3_KMF_GK): the ghost-face product in the Kronecker form (every face's Gram matrix gamma A (x) m (x) m in the
face's (s, u, v) node order; kmf.gk_apply_, gk_kernel: 15 threads per face, the tangential pass in registers) against
the current kernels (gf_kernel for one column, gf2_kernel from two), float32 and float64, B = 1, 2, 4, 16, on a full
plate cell: accuracy against the float64 product of the templates (GhostFaces), device time per product (warm, back to
back; cold: a 256 MB buffer rewritten before each call) for grid caps of 4 / 8 / 16 / 32 blocks per SM and none.
usage: python t_gk.py BODY_DIR:CASE [--rep 30]"""
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


sms = torch.cuda.get_device_properties(0).multi_processor_count
with torch.no_grad():
    nd = KMF.nodes_tensor(C)
    g32, g64 = KMF.gf_state(C, nd, torch.float32), KMF.gf_state(C, nd, torch.float64)
    res = dict(case=case, faces=g32['nf'], sms=sms, lib_error=None, factored=g32['gk'] is not None)
    try:
        KMF._lib()
    except Exception as ex:                                             # noqa: BLE001
        res['lib_error'] = repr(ex)[:1500]
    if g32['gk'] is not None:
        t = g64['gk'].cpu()
        res['A'] = [round(float(v), 8) for v in t[:25]]
        res['m'] = [float(v) for v in t[25:]]
        res['state_MB'] = dict(fsuv=round(g32['fsuv'].numel() * 4 / 2 ** 20, 2), fnode=round(g32['fnode'].numel() * 4 / 2 ** 20, 2))
    print(json.dumps(res), flush=True)
    X = torch.randn((C.nb, 16), dtype=torch.float64, device=dev, generator=g)
    Y64 = C.GF64.matvec(X, torch.zeros_like(X)).mul_(C.gamma)           # float64 reference (template GEMMs), 16 columns
    for B in (1, 2, 4, 16):
        r = {}
        for dt, gst in ((torch.float32, g32), (torch.float64, g64)):
            x = X[:, :B].to(dt).contiguous()
            y64 = Y64[:, :B]
            nrm = y64.abs().max()
            app = KMF.gf_apply_ if dt == torch.float32 else KMF.gf_apply64_
            y = torch.zeros_like(x)
            tag = 'f32' if dt == torch.float32 else 'f64'
            yc = app(torch.zeros_like(x), gst, x, gk=False)
            r[f'{tag}_cur'] = dict(rel_f64=float((yc.double() - y64).abs().max() / nrm),
                                   warm=round(ev(lambda: app(y, gst, x, gk=False), a.rep), 1),
                                   cold=round(cold(lambda: app(y, gst, x, gk=False), a.rep), 1))
            if gst['gk'] is None:
                continue
            for bps in (4, 8, 16, 32, 0):
                for nbk in ((0, 1) if B > 1 else (0,)):
                    blk = bps * sms
                    y1 = KMF.gk_apply_(torch.zeros_like(x), gst, x, blocks=blk, nb=nbk)
                    y2 = KMF.gk_apply_(torch.zeros_like(x), gst, x, blocks=blk, nb=nbk)
                    r[f'{tag}_gk_bps{bps}_nb{nbk}'] = dict(
                        rel_f64=float((y1.double() - y64).abs().max() / nrm), l2_f64=float((y1.double() - y64).norm() / y64.norm()),
                        rel_rerun=float((y2 - y1).abs().max() / y1.abs().max()),
                        warm=round(ev(lambda: KMF.gk_apply_(y, gst, x, blocks=blk, nb=nbk), a.rep), 1),
                        cold=round(cold(lambda: KMF.gk_apply_(y, gst, x, blocks=blk, nb=nbk), a.rep), 1))
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
