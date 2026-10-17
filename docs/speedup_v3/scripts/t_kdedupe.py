"""P2b check: in the float32 element product (packed per-element K_e, formed as teacher._Kfp32.hold forms them: chunked
GEMM of the float32 moments with T_up), how many full elements have a K_e bitwise equal to one representative row, is
the GEMM deterministic (two formations bitwise equal), and the time of ke_symv.symv_ with the element-to-row map
(shared elements read the representative's row) against without, B = 1, 3, 16, with the difference of the results
compared with the product against itself (atomics).
usage: python t_kdedupe.py BODY_DIR:CASE[,CASE2,...] [--rep 50]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
os.environ.setdefault('OPL_KE_SYMV', '1')
import torch
import teacher as TE
import ke_symv as KS
import moments_ad as MA

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=50)
a = ap.parse_args()
body, cases = a.spec.split(':')
dev = 'cuda'
f32 = torch.float32


def timed(fn, rep):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep


for case in cases.split(','):
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    Tup = C.Tm_up.reshape(125, -1).to(f32)
    ch = C._lean_chunk
    E = len(C.M)

    def form():
        P = torch.empty((E, Tup.shape[1]), dtype=f32, device=dev)
        for lo in range(0, E, ch):
            torch.matmul(C.M[lo:lo + ch].to(f32), Tup, out=P[lo:lo + ch])
        return P
    P = form()
    res = dict(case=case, E=E, chunk=ch, deterministic=bool(torch.equal(P, form())))
    cells, taus, nrm, off = MA.cell_rows(C)
    full = torch.nonzero(MA.full_rows(cells, C.n, taus, nrm, off, s=C.s, surface=getattr(C, 'surface', 'P'))).squeeze(1)
    res['n_full'] = int(len(full))
    if len(full):
        R32 = C.M[full].to(f32)
        uq, inv, cnt = torch.unique(R32, dim=0, return_inverse=True, return_counts=True)
        k = int(torch.argmax(cnt))
        rep = int(full[torch.nonzero(inv == k)[0, 0]])
        eq = (P[full] == P[rep][None]).all(1)
        shared = full[eq]
        res.update(m32_patterns=int(len(uq)), m32_mode=int(cnt[k]), shared=int(len(shared)),
                   shared_frac=round(len(shared) / E, 4))
        PU = torch.unique(P[full], dim=0)
        res['full_P_patterns'] = int(len(PU))
    else:
        rep, shared = 0, full
    kmap = torch.arange(E, dtype=torch.int32, device=dev)
    kmap[shared] = rep
    dofs = C.dofs.contiguous()
    nb = int(dofs.max()) + 1
    g = torch.Generator(device=dev); g.manual_seed(0)
    for B in (1, 3, 16):
        x = torch.randn((nb, B), dtype=f32, device=dev, generator=g).contiguous()

        def run(km):
            y = torch.zeros((nb, B), dtype=f32, device=dev)
            KS.symv_(y, P, dofs, x, kmap=km)
            return y
        y0, y0b, y1 = run(None), run(None), run(kmap)
        sc = float(y0.abs().max())
        res[f'B{B}'] = dict(us_nomap=round(timed(lambda: run(None), a.rep), 1), us_map=round(timed(lambda: run(kmap), a.rep), 1),
                            diff_map=float((y1 - y0).abs().max()) / sc, diff_rerun=float((y0b - y0).abs().max()) / sc)
    print(json.dumps(res), flush=True)
