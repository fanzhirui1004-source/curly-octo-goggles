"""Where S_hat's matrix products come from: torch.mm / bmm / matmul / addmm / Tensor.__matmul__ wrapped to record the
calling src_v3 line, the operand shapes and dtype during one S_hat of a full plate cell (B = 1 and 16); prints the
aggregated call sites (count, dtype, shapes). No timing (the kernel profile has the times by kernel name).
usage: python t_gemmsites.py BODY_DIR:CASE"""
import os, sys, json, traceback, collections
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

dev = 'cuda'
FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
with torch.no_grad():
    f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
if hasattr(C, 'store64'):
    C.store64(allow=True)

REC = collections.Counter()
ON = [False]


def site():
    for fr in reversed(traceback.extract_stack()[:-2]):
        if 'src_v3' in fr.filename and 'gemmsites' not in fr.filename:
            return f'{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}'
    return '?'


def wrap(name, fn):
    def w(*args, **kw):
        if ON[0]:
            ts = [a for a in args if torch.is_tensor(a)][:2]
            REC[(name, site(), str(ts[0].dtype) if ts else '', ' x '.join(str(tuple(t.shape)) for t in ts))] += 1
        return fn(*args, **kw)
    return w


torch.mm = wrap('mm', torch.mm); torch.bmm = wrap('bmm', torch.bmm); torch.matmul = wrap('matmul', torch.matmul)
torch.addmm = wrap('addmm', torch.addmm)
_mt = torch.Tensor.__matmul__
torch.Tensor.__matmul__ = wrap('@', _mt)
res = {}
with torch.no_grad():
    for B in (1, 16):
        q = torch.randn((C.np_, B), dtype=torch.float64, device=dev)
        REC.clear(); ON[0] = True
        f.s_hat(q)
        ON[0] = False
        rows = sorted(REC.items(), key=lambda kv: (kv[0][2] != 'torch.float64', -kv[1]))
        res[f'B{B}'] = [dict(op=k[0], site=k[1], dtype=k[2], shapes=k[3], n=v) for k, v in rows]
        print(f'== B={B}', flush=True)
        for r in res[f'B{B}']:
            print('  %4d  %-6s %-14s %-60s %s' % (r['n'], r['op'], r['dtype'][6:], r['site'][:60], r['shapes']), flush=True)
print(json.dumps(res), flush=True)
