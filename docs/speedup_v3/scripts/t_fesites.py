"""Where a deployment cell's front end spends its float64 device time: dense linear-algebra calls (torch.linalg.* and
torch.cholesky_*, torch.matmul / mm / bmm on float64) wrapped to record the calling src_v3 line, shapes and synchronised
time, during one full plate cell's setup as opt_design runs it (constructor, assemble_deploy, network files, geometry,
FastNet, warm-up application, slim_deploy) -- second cell timed, after a first cell warms the kernels up.
usage: python t_fesites.py BODY_DIR:CASE1,CASE2"""
import os, sys, json, time, traceback, collections
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

dev = 'cuda'
FN.FUSED = True; FI.ON = True
body, cases = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
REC = collections.defaultdict(lambda: [0, 0.0])
ON = [False]


def site():
    out = []
    for fr in reversed(traceback.extract_stack()[:-2]):
        if 'src_v3' in fr.filename and 'fesites' not in fr.filename:
            out.append(f'{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}')
            if len(out) == 2:
                break
    return ' < '.join(out) or '?'


def wrap(name, fn, f64only=False):
    def w(*args, **kw):
        if not ON[0]:
            return fn(*args, **kw)
        ts = [a for a in args if torch.is_tensor(a)]
        if f64only and not (ts and ts[0].dtype == torch.float64):
            return fn(*args, **kw)
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn(*args, **kw)
        torch.cuda.synchronize(); dt_ = time.perf_counter() - t
        key = (name, site(), str(ts[0].dtype)[6:] if ts else '', ' x '.join(str(tuple(x.shape)) for x in ts[:2]))
        REC[key][0] += 1; REC[key][1] += dt_
        return r
    return w


for nm in ('cholesky', 'cholesky_ex', 'solve', 'inv', 'inv_ex', 'eigh', 'eigvalsh', 'lu_factor', 'lu', 'qr', 'svd',
           'solve_triangular', 'det', 'slogdet', 'lstsq'):
    if hasattr(torch.linalg, nm):
        setattr(torch.linalg, nm, wrap('linalg.' + nm, getattr(torch.linalg, nm)))
for nm in ('cholesky_solve', 'cholesky_inverse', 'triangular_solve'):
    setattr(torch, nm, wrap(nm, getattr(torch, nm)))
for nm in ('mm', 'bmm', 'matmul'):
    setattr(torch, nm, wrap(nm, getattr(torch, nm), f64only=True))
_mt = torch.Tensor.__matmul__
torch.Tensor.__matmul__ = wrap('@', _mt, f64only=True)


def front(case, rec):
    ON[0] = rec
    t = time.perf_counter()
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    model = h.add(geo)
    f = FN.FastNet(model, geo)
    with torch.no_grad():
        f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
    torch.cuda.synchronize()
    ON[0] = False
    return time.perf_counter() - t


cs = cases.split(',')
front(cs[0], False)
REC.clear()
wall = front(cs[1], True)
rows = sorted(REC.items(), key=lambda kv: -kv[1][1])
res = dict(case=cs[1], wall_s=round(wall, 3), recorded_s=round(sum(v[1] for v in REC.values()), 3),
           sites=[dict(op=k[0], site=k[1], dtype=k[2], shapes=k[3], n=v[0], ms=round(1e3 * v[1], 2)) for k, v in rows[:40]])
print(json.dumps(dict(case=res['case'], wall_s=res['wall_s'], recorded_s=res['recorded_s'])), flush=True)
for r in res['sites']:
    print('  %8.2f ms %4d  %-22s %-8s %-70s %s' % (r['ms'], r['n'], r['op'], r['dtype'], r['site'][:70], r['shapes']), flush=True)
print(json.dumps(res), flush=True)
