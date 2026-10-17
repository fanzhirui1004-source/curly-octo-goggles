"""Item 2c (src_v3 OPL_V3_COARSE_ZT=0, OPL_V3_COARSE_TT=0): the coarse space's transposed copies. On a plate lattice's
SparseCoarse basis Z (q1r): Z^T R by the stored Z^T CSR against torch.sparse.mm(Z.t(), R) (no second copy), and T^T z by
the stored contiguous T^T against the transposed view of T (random T of the coarse space's shape), B = 1 and 3: largest
relative difference, rerun difference, device time per call (CUDA events), device peak above the base, bytes of Z^T.
usage: python t_coarseT.py BODY_DIR LAYOUT.json [--rep 50]"""
import os, sys, json, argparse, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import torch
import teacher as TE, lat_multi as LM, lat_precond as PR

ap = argparse.ArgumentParser(); ap.add_argument('body'); ap.add_argument('layout'); ap.add_argument('--rep', type=int, default=50)
a = ap.parse_args()
dev = torch.device('cuda')
L = json.load(open(a.layout))
lay = {}
for c in L['cells']:
    case = c['case'] + '_o000'
    C = TE.Cell(case, a.body, log=lambda s_: None, deploy=True)
    C.assemble_deploy(C.taus0)
    C._kpp_cache = None
    lay[tuple(c['position'])] = LM.from_teacher(C)
lat = LM.MultiLattice(lay, clamp=('x', 'min'), load=('x', 'max'), loads='consistent', n_random=0, device=dev, max_cols=16,
                      log=lambda s_: None)


def ev(fn, rep):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep                              # us


def peak(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); b = torch.cuda.memory_allocated()
    fn(); torch.cuda.synchronize()
    return round((torch.cuda.max_memory_allocated() - b) / 2 ** 20, 2)


with torch.no_grad():
    Z, ZT, verts, keep = PR.coarse_z(lat, 'q1r')
    n, nc = Z.shape
    res = dict(cells=len(L['cells']), n=n, nc=nc, Z_nnz=int(Z.values().numel()),
               ZT_MiB=round(sum(t.numel() * t.element_size() for t in (ZT.crow_indices(), ZT.col_indices(), ZT.values())) / 2 ** 20, 2))
    g = torch.Generator(device=dev); g.manual_seed(0)
    T = torch.randn((nc, nc - 10), dtype=torch.float64, device=dev, generator=g)
    TTc = T.T.contiguous()
    for B in (1, 3):
        R = torch.randn((n, B), dtype=torch.float64, device=dev, generator=g)
        ya = torch.sparse.mm(ZT, R); yb = torch.sparse.mm(Z.t(), R); yb2 = torch.sparse.mm(Z.t(), R)
        z = torch.randn((nc, B), dtype=torch.float64, device=dev, generator=g)
        ta, tb = TTc @ z, T.t() @ z
        res[f'B{B}'] = dict(
            zt=dict(rel=float((yb - ya).abs().max() / ya.abs().max()), rerun_equal=bool(torch.equal(yb, yb2)),
                    us_stored=round(ev(lambda: torch.sparse.mm(ZT, R), a.rep), 1),
                    us_transposed=round(ev(lambda: torch.sparse.mm(Z.t(), R), a.rep), 1),
                    peak_MiB_stored=peak(lambda: torch.sparse.mm(ZT, R)), peak_MiB_transposed=peak(lambda: torch.sparse.mm(Z.t(), R))),
            tt=dict(equal=bool(torch.equal(ta, tb)), rel=float((tb - ta).abs().max() / ta.abs().max()),
                    us_stored=round(ev(lambda: TTc @ z, a.rep), 1), us_view=round(ev(lambda: T.t() @ z, a.rep), 1)))
        print(json.dumps({f'B{B}': res[f'B{B}']}), flush=True)
print(json.dumps(res), flush=True)
