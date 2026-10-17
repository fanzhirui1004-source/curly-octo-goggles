"""Check of the block-structured coarse inverse (src_v3 S15, coarse_blk, OPL_COARSE_BLK) against the packed inverse (S2)
on deployment cells: each cell is built twice (TL.COARSE_BLK off / on); compared: the coarse solve (and both against the
float64 solve), S_hat on seeded directions; recorded: the stored bytes of the coarse inverse, the block layout, the
coarse setup time, the warm-up time, the time of a coarse solve and of S_hat (B = 1, 3), the allocated-memory peak of
the set-up.
usage: python t_cblk.py BODY_DIR:CASE[,CASE2,...] [--rep 20]"""
import os, sys, json, time, argparse, gc
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import coarse_blk as CB

dev, dt = TE.dev, TE.dt


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, cases = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
BD.TMP.mkdir(parents=True, exist_ok=True)
for case in cases.split(','):
    outs, rec = {}, dict(case=case)
    for mode in (False, True):
        TL.COARSE_BLK = mode
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
        BD.netdata(C, body, BD.TMP / case)
        geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        if h.model is not None:
            h.model.caches.pop(case, None)
        f = FN.FastNet(h.add(geo), geo)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.no_grad():
            f.s_hat(torch.zeros((C.np_, 1), dtype=dt, device=dev))     # warm-up: coarse setup + inverse
        torch.cuda.synchronize(); t2 = time.perf_counter()
        r = dict(build_s=round(t1 - t0, 3), warmup_s=round(t2 - t1, 3), coarse_setup_s=round(C._c_seconds, 3),
                 peak_alloc_GiB=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), inv_err=C._c_inv_err,
                 blk=getattr(C, '_c_blk_info', None), nc=C._c_nc)
        if getattr(C, '_cBlk', None) is not None:
            r['coarse_MiB'] = round(CB.nbytes(C._cBlk) / 2 ** 20, 2)
        else:
            r['coarse_MiB'] = round(C._cAp32.numel() * 4 / 2 ** 20, 2) if C._cAp32 is not None else None
        g = torch.Generator(device=dev); g.manual_seed(1)
        x = torch.randn((C.ni, 3), dtype=dt, device=dev, generator=g)
        y, r['coarse_solve_ms_B3'] = timed(lambda: TL._coarse_solve(C, x), a.rep)
        _, r['coarse_solve_ms_B1'] = timed(lambda: TL._coarse_solve(C, x[:, :1]), a.rep)
        q = torch.randn((C.np_, 3), dtype=dt, device=dev, generator=g)
        q = q - C.Q @ (C.Q.T @ q)
        with torch.no_grad():
            s3, r['s_hat_ms_B3'] = timed(lambda: f.s_hat(q), a.rep)
            _, r['s_hat_ms_B1'] = timed(lambda: f.s_hat(q[:, :1]), a.rep)
        outs[mode] = (y, s3)
        rec['blk' if mode else 'packed'] = r
        print(json.dumps({case: {('blk' if mode else 'packed'): r}}), flush=True)
        del C, geo, f, x, q
    (y0, s0), (y1, s1) = outs[False], outs[True]
    rec['coarse_solve_rel_diff'] = float((y1 - y0).norm() / y0.norm())
    rec['s_hat_rel_diff_max_col'] = float(((s1 - s0).norm(dim=0) / s0.norm(dim=0)).max())
    print(json.dumps(dict(case=case, coarse_solve_rel_diff=rec['coarse_solve_rel_diff'],
                          s_hat_rel_diff_max_col=rec['s_hat_rel_diff_max_col'])), flush=True)
    del outs, y0, y1, s0, s1
