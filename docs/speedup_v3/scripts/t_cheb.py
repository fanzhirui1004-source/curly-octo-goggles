"""chebfused check: trainlib.smooth_tail / _cheb / smooth_tail_T with OPL_V3_CHEB on vs off.

mock: a synthetic deployment cell (dense SPD float32 K product through a _Kfp32-like object, deterministic GEMM) -> the
      fused iterates must be bitwise equal to the PyTorch recurrence (CUDA).
cell: the first cell of a layout (real _Kc; deterministic algorithms, OPL_KE_SYMV=0 so every product is reproducible):
      bitwise again, then the same with the default (atomic) products, and the timing of both paths.
usage: python t_cheb.py mock | python t_cheb.py cell BODY_DIR:CASE"""
import os, sys, json, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
mode = sys.argv[1]
if mode == 'cell' and os.environ.get('DET', '1') == '1':
    os.environ['OPL_KE_SYMV'] = '0'; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import torch
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def same(a, b):
    return bool(torch.equal(a, b)), float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


if mode == 'mock':
    import types
    sys.modules.setdefault('models', types.ModuleType('models'))
    import trainlib as TL
    import chebfused as CF
    torch.manual_seed(0)
    nb, B = 3000, 5
    A = torch.randn(nb, nb, dtype=torch.float64) / nb ** 0.5
    K = (A @ A.t() + 0.05 * torch.eye(nb, dtype=torch.float64)).to(dev)
    K32 = K.to(torch.float32)

    class Kc:
        def mm32(self, x32):
            return K32 @ x32

        def __matmul__(self, x):
            return self.mm32(x.to(torch.float32)).to(x.dtype)

    C = types.SimpleNamespace()
    C._Kc = Kc(); C.K = K; C._deploy = True; C.nb = nb
    perm = torch.randperm(nb)
    C.I = perm[:2400].sort().values.to(dev); C.ni = 2400; C.dK = torch.diagonal(K).contiguous()
    C.P = perm[2400:].sort().values.to(dev)
    lmin, lmax = TL.tail_bounds(C, 30.0)
    u = torch.randn(nb, B, dtype=torch.float32, device=dev)
    w = torch.randn(2400, B, dtype=torch.float64, device=dev)
    zb = torch.randn(2400, B, dtype=torch.float64, device=dev)
    y = torch.randn(nb, B, dtype=torch.float64, device=dev)
    res = {}
    with torch.no_grad():
        for on in (False, True):
            TL.CHEB = on
            res[on] = dict(tail32=TL.smooth_tail(C, u, 8), tail64=TL.smooth_tail(C, u.double(), 8),
                           cheb=TL._cheb(C, w, zb, 8, 30.0), cheb1=TL._cheb(C, w[:, :1].contiguous(), zb[:, :1].contiguous(), 8, 30.0),
                           tailT=TL.smooth_tail_T(C, y, 8))
    print(json.dumps({k: same(res[True][k], res[False][k]) for k in res[False]}), flush=True)
    sys.exit(0)

import models as MD  # noqa
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
FN.FUSED = True; FI.ON = True
body, case = sys.argv[2].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
if h.model is not None:
    h.model.caches.pop(case, None)
f = FN.FastNet(h.add(geo), geo)
m = f.model
det = os.environ.get('DET', '1') == '1'
g = torch.Generator(device='cuda'); g.manual_seed(0)
out = {}
with torch.no_grad():
    f.s_hat(torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g))     # warm-up, coarse setup
    for B in (1, 3, 16):
        g.manual_seed(B)
        u = torch.randn((C.nb, B), dtype=torch.float32, device='cuda', generator=g)
        y = torch.randn((C.nb, B), dtype=torch.float64, device='cuda', generator=g)
        q = torch.randn((C.np_, B), dtype=torch.float64, device='cuda', generator=g)
        for on in (False, True, False):
            TL.CHEB = on
            torch.use_deterministic_algorithms(det, warn_only=True)          # only around the two tails
            with TL._held(C):
                r = dict(tail=TL.smooth_tail(C, u, m.smooth_k, m.smooth_alpha), tailT=TL.smooth_tail_T(C, y, m.smooth_k, m.smooth_alpha))
            torch.use_deterministic_algorithms(False)
            r['field'] = f.field(q)
            r['shat'] = f.s_hat(q)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(5):
                f.s_hat(q)
            torch.cuda.synchronize(); r['t'] = (time.time() - t0) / 5
            out.setdefault(B, []).append(r)
    rep = {}
    for B, (a, b, a2) in out.items():
        rep[B] = {k: dict(fused=same(b[k], a[k]), rerun=same(a2[k], a[k])) for k in a if k != 't'}
        rep[B]['t_ms'] = (round(1e3 * a['t'], 2), round(1e3 * b['t'], 2), round(1e3 * a2['t'], 2))
print(json.dumps(dict(case=case, det=det, rep=rep)), flush=True)
