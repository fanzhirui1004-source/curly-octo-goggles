"""P11 check (OPL_V3_CLAMPFREE): in FastNet's forward, the input of every clamped hyperedge layer already carries the
port values X0 on the port rows, so where(port, X0, X) == X. Checked bitwise at every layer call of the forward network
map for B = 1, 3, 16 on a full plate cell (random port data), then the forward map itself with the switch off / on
(no atomics differ in the comparison if the gather / scatter order is the same: reported as max abs difference).
usage: python t_clampfree.py BODY_DIR:CASE"""
import os, sys, json
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
CHK = dict(calls=0, port_rows_equal=True)
_orig = FN._Hyper.fwd_clamped


def fwd_clamped(self, X, X0, pmask, pm8, own=False):
    CHK['calls'] += 1
    w = torch.where(pmask, X0, X)
    if not torch.equal(w, X):
        CHK['port_rows_equal'] = False
    return _orig(self, X, X0, pmask, pm8, own=own)


FN._Hyper.fwd_clamped = fwd_clamped
g = torch.Generator(device=dev); g.manual_seed(0)
res = dict(case=case)
with torch.no_grad():
    for B in (1, 3, 16):
        qd = torch.randn((C.np_, B), dtype=torch.float32, device=dev, generator=g)
        FN.CLAMPFREE = False
        y0 = f.ext(qd).clone(); y0b = f.ext(qd).clone()
        FN.CLAMPFREE = True
        y1 = f.ext(qd).clone()
        FN.CLAMPFREE = False
        res[f'B{B}'] = dict(calls=CHK['calls'], port_rows_equal=CHK['port_rows_equal'],
                            max_abs_on_off=float((y1 - y0).abs().max()), max_abs_rerun=float((y0b - y0).abs().max()),
                            scale=float(y0.abs().max()))
        print(json.dumps({f'B{B}': res[f'B{B}']}), flush=True)
print(json.dumps(res), flush=True)
