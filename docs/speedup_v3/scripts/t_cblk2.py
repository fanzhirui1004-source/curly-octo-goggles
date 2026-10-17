"""Two-stream S_hat throughput with the block coarse inverse (S15) against the packed inverse (S2), as the PCG matvec
issues it (lat_multi.apply_blocks: B = 1, cells round robin on two CUDA streams): time per cell for packed / block
(cuSPARSE CSR couplings, coarse_blk.solve) / block with gather + index_add couplings (no cuSPARSE), plus the CPU time
to issue one S_hat (no synchronisation, short queue) and one coarse solve.
usage: python t_cblk2.py BODY_DIR:CASE1,CASE2 [--rep 40]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'
import models as MD  # noqa
import torch
import torch.nn.functional as Fn
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import coarse_blk as CB
import symtile as ST

dev, dt = TE.dev, TE.dt
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=40)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, cases = a.spec.split(':')
cases = cases.split(',')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
BD.TMP.mkdir(parents=True, exist_ok=True)


def build(case, blk):
    TL.COARSE_BLK = blk
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    if h.model is not None:
        h.model.caches.pop(case, None)
    f = FN.FastNet(h.add(geo), geo)
    with torch.no_grad():
        f.s_hat(torch.zeros((C.np_, 1), dtype=dt, device=dev))
    if hasattr(C, 'store64'):
        C.store64(allow=True)
    g = torch.Generator(device=dev); g.manual_seed(3)
    q = torch.randn((C.np_, 1), dtype=dt, device=dev, generator=g)
    return C, f, q


def solve_ia(blk, z):
    """coarse_blk.solve with the couplings as gather + index_add (COO from the CSR arrays) instead of cuSPARSE."""
    nc, ns, Np = blk['dims']
    if '_coo' not in blk:
        G, Gt = blk['G'], blk['Gt']
        rg = torch.repeat_interleave(torch.arange(Np, device=dev), G.crow_indices().diff())
        rt = torch.repeat_interleave(torch.arange(ns, device=dev), Gt.crow_indices().diff())
        blk['_coo'] = (rg, G.col_indices(), G.values(), rt, Gt.col_indices(), Gt.values())
    rg, cg, vg, rt, ct, vt = blk['_coo']
    zp = Fn.pad(z, (0, 0, 0, 1))[blk['perm']]
    y = ST.matmul_tiles(blk['D'], blk['TI'], blk['TJ'], Np, zp)
    t = z[blk['sep']].index_add(0, rt, vt[:, None] * y[ct], alpha=-1)
    xs = ST.matmul(blk['S'], ns, t)
    xo = ST.matmul_tiles(blk['D'], blk['TI'], blk['TJ'], Np, zp.index_add(0, rg, vg[:, None] * xs[cg], alpha=-1))
    return torch.cat([xo, xs])[blk['inv']]


def pair_time(cells, rep):
    """ms per cell: cells alternate on two streams (as apply_blocks with OPL_V3_STREAMS=2)."""
    main = torch.cuda.current_stream()
    sts = [torch.cuda.Stream(), torch.cuda.Stream()]

    def one():
        ev = torch.cuda.Event(); ev.record(main)
        for i, (C, f, q) in enumerate(cells):
            s = sts[i % 2]
            s.wait_event(ev)
            with torch.cuda.stream(s):
                f.s_hat(q)
        for s in sts:
            main.wait_stream(s)
    with torch.no_grad():
        one(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(rep):
            one()
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / rep / len(cells)


def seq_time(cells, rep):
    with torch.no_grad():
        for C, f, q in cells:
            f.s_hat(q)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(rep):
            for C, f, q in cells:
                f.s_hat(q)
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / rep / len(cells)


def cpu_issue(fn, rep=20):
    """CPU ms to issue fn() when the GPU queue is short (synchronised before each call)."""
    tot = 0.0
    for _ in range(rep):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); tot += time.perf_counter() - t
    torch.cuda.synchronize()
    return 1000 * tot / rep


res = {}
packed = [build(c, False) for c in cases] * 2                         # 4 applies per round (2 per stream)
blk = [build(c, True) for c in cases] * 2
assert all(getattr(C, '_cBlk', None) is not None for C, _, _ in blk[:len(cases)])
for name, cells in (('packed', packed), ('blk', blk)):
    res[name] = dict(seq_ms=seq_time(cells, a.rep), pair_ms=pair_time(cells, a.rep))
    C, f, q = cells[0]
    with torch.no_grad():
        res[name]['cpu_shat_ms'] = cpu_issue(lambda: f.s_hat(q))
        z = torch.randn((C._c_nc, 1), dtype=torch.float32, device=dev)
        if name == 'blk':
            res[name]['cpu_coarse_ms'] = cpu_issue(lambda: CB.solve(C._cBlk, z), 50)
            res['blk_ia'] = dict(cpu_coarse_ms=cpu_issue(lambda: solve_ia(C._cBlk, z), 50))
            d = float((solve_ia(C._cBlk, z) - CB.solve(C._cBlk, z)).norm() / CB.solve(C._cBlk, z).norm())
            res['blk_ia']['rel_diff_vs_blk'] = d
        else:
            res[name]['cpu_coarse_ms'] = cpu_issue(lambda: ST.matmul(C._cAp32, C._c_nc, z), 50)
    print(json.dumps({name: res[name]}), flush=True)
orig = CB.solve
CB.solve = solve_ia                                                    # couplings without cuSPARSE
res['blk_ia'].update(seq_ms=seq_time(blk, a.rep), pair_ms=pair_time(blk, a.rep))
C, f, q = blk[0]
with torch.no_grad():
    res['blk_ia']['cpu_shat_ms'] = cpu_issue(lambda: f.s_hat(q))
CB.solve = orig
print(json.dumps({'blk_ia': res['blk_ia']}), flush=True)
print(json.dumps(res), flush=True)
