"""The float32 element product under the solve's conditions (P2, second round). In S_hat the 151 MB of packed element
matrices do not stay in L2 between two products (other kernels run in between; t_symv3: the single-read kernel gave
nothing there although it was 28% faster back to back), so every product streams P from DRAM. Here a 256 MB buffer is
rewritten before each product (its own time subtracted) and every kernel variant is timed cold:
  ke_symv._symv (BI, warps), ke_symv2._symv_up (BI, warps), ke_symv2._symv_up_p (software-pipelined row blocks:
  BI, warps, stages), each also with the element-to-row map of P2b (full elements with a bitwise-equal matrix read one
  row); B = 1, 2, 3 (and the warm time of the best, for reference). Differences against the current kernel (rounding of
  the summation order; the map: same values).
usage: python t_symv4.py BODY_DIR:CASE[,CASE2] [--rep 40]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import torch
import triton
import teacher as TE
import ke_symv as KS, ke_symv2 as K2
import moments_ad as MA

ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=40)
a = ap.parse_args()
body, cases = a.spec.split(':')
dev = 'cuda'
f32 = torch.float32
buf = torch.zeros(64 * 2 ** 20, dtype=f32, device=dev)                 # 256 MB: more than the L2


def flush():
    buf.add_(1.0)


def ev_time(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep


def cold(fn, rep):
    return ev_time(lambda: (flush(), fn()), rep) - ev_time(flush, rep)


for case in cases.split(','):
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    Tup = C.Tm_up.reshape(125, -1).to(f32)
    ch, E = C._lean_chunk, len(C.M)
    P = torch.empty((E, Tup.shape[1]), dtype=f32, device=dev)
    for lo in range(0, E, ch):
        torch.matmul(C.M[lo:lo + ch].to(f32), Tup, out=P[lo:lo + ch])
    cells, taus, nrm, off = MA.cell_rows(C)
    full = torch.nonzero(MA.full_rows(cells, C.n, taus, nrm, off, s=C.s, surface=getattr(C, 'surface', 'P'))).squeeze(1)
    kmap = None
    if len(full) > 1:
        _, inv, cnt = torch.unique(C.M[full].to(f32), dim=0, return_inverse=True, return_counts=True)
        rep_ = int(full[torch.nonzero(inv == int(torch.argmax(cnt)))[0, 0]])
        sh = full[(P[full] == P[rep_][None]).all(1)]
        kmap = torch.arange(E, dtype=torch.int32, device=dev); kmap[sh] = rep_
    dofs = C.dofs.contiguous()
    nb = int(dofs.max()) + 1
    g = torch.Generator(device=dev); g.manual_seed(0)
    res = dict(case=case, E=E, shared=int((kmap != torch.arange(E, device=dev)).sum()) if kmap is not None else 0,
               P_MB=round(P.numel() * 4 / 2 ** 20, 1))
    for B in (1, 2, 3):
        x = torch.randn((nb, B), dtype=f32, device=dev, generator=g).contiguous()
        y = torch.zeros((nb, B), dtype=f32, device=dev)
        BP = triton.next_power_of_2(B)
        KS.TUNE2 = False
        y0 = torch.zeros_like(y); KS.symv_(y0, P, dofs, x)
        sc = float(y0.abs().max())
        rows = {}

        def run(name, fn):
            try:
                yy = torch.zeros_like(y); fn(yy)
                rows[name] = dict(us=round(cold(lambda: fn(y), a.rep), 1), d=round(float((yy - y0).abs().max()) / sc, 9))
            except Exception as ex:                                      # noqa: BLE001 (resource limits of a config)
                rows[name] = dict(error=repr(ex)[:120])
        for km in (None, kmap):
            tag = 'm' if km is not None else ''
            mp = dict(kmap_ptr=km if km is not None else dofs, MAP=km is not None)
            for bi in (4, 8, 16):
                for w in (1, 2, 4):
                    run(f'old{tag}_bi{bi}_w{w}', lambda yy, bi=bi, w=w, mp=mp: KS._symv[(E,)](
                        P, dofs, x, yy, B, B, BP=BP, BI=bi, NBLK=(81 + bi - 1) // bi, num_warps=w, **mp))
            if B <= 2:
                for bi in (8, 16, 32):
                    for w in (1, 2, 4, 8):
                        run(f'up{tag}_bi{bi}_w{w}', lambda yy, bi=bi, w=w, mp=mp: K2._symv_up[(E,)](
                            P, dofs, x, yy, B, B, BP=BP, BI=bi, NBLK=(81 + bi - 1) // bi, ONEHOT=True, num_warps=w, **mp))
                        for st in (2, 3, 4):
                            run(f'upp{tag}_bi{bi}_w{w}_s{st}', lambda yy, bi=bi, w=w, st=st, km=km: K2.symv_p(
                                yy, P, dofs, x, BI=bi, warps=w, stages=st, kmap=km))
        ok = {k: v for k, v in rows.items() if 'us' in v}
        best = min(ok.items(), key=lambda kv: kv[1]['us'])
        best_nomap = min(((k, v) for k, v in ok.items() if 'm_' not in k and not k.startswith(('oldm', 'upm', 'uppm'))),
                         key=lambda kv: kv[1]['us'])
        cur = 'old_bi8_w2'
        res[f'B{B}'] = dict(current=ok.get(cur), best=best, best_nomap=best_nomap, rows=rows)
        print(json.dumps({f'B{B}': dict(current=ok.get(cur), best=best, best_nomap=best_nomap)}), flush=True)
    print(json.dumps(res), flush=True)
