"""Kc (float32 correction stiffness) time vs columns; tail transpose fused vs unfused; S_hat total both ways."""
import os, json, time, sys
import models as MD  # noqa
import torch, teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
dev, dt, f32 = TE.dev, TE.dt, torch.float32
def timed(fn, rep=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t) / rep
FI.ON = os.environ.get('FI', '1') == '1'; FN.FUSED = True; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_TAILT_FUSED'] = '1'
body, case = sys.argv[1].split(':')
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo); m = f.model
r = {}
for B in (6, 12):
    x = torch.randn((C.nb, B), dtype=dt, device=dev)
    r[f'Kc_{B}'] = timed(lambda: C._Kc @ x)
    r[f'K64_{B}'] = timed(lambda: C.K @ x)
    x32 = x.to(f32)
    r[f'ghost32_{B}'] = timed(lambda: C._Kc.gf.matvec(x32, torch.zeros_like(x32)))
print(json.dumps(r), flush=True)
q = torch.randn((C.np_, 6), dtype=dt, device=dev)
y = torch.randn((C.nb, 6), dtype=f32, device=dev)
f.s_hat(q)
out = {}
for fz in ('1', '0'):
    os.environ['OPL_TAILT_FUSED'] = fz
    out['tail_T_' + fz] = timed(lambda: TL.smooth_tail_T(C, y, m.smooth_k, m.smooth_alpha))
    out['shat_' + fz] = timed(lambda: f.s_hat(q))
print(json.dumps(out), flush=True)
