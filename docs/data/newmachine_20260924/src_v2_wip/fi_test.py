"""OPL_FASTIDX check: S_hat q, K x, Kc x with row gathers by torch.take vs indexing (a deployment cell); run-to-run
noise of the baseline (atomic adds) for scale; times at B = 6 and 12."""
import os, json, time, sys
import models as MD  # noqa
import torch, teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
dev, dt, f32 = TE.dev, TE.dt, torch.float32
def timed(fn, rep=5):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep
FN.FUSED = True; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_TAILT_FUSED'] = '0'
body, case = sys.argv[1].split(':')
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
for B in (6, 12):
    q = torch.randn((C.np_, B), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(B))
    x = torch.randn((C.nb, B), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(B + 1))
    r = {}
    for on in (False, True):
        FI.ON = on
        s, r[f'shat_ms_{on}'] = timed(lambda: f.s_hat(q))
        k, r[f'K64_ms_{on}'] = timed(lambda: C.K @ x)
        kc, r[f'Kc_ms_{on}'] = timed(lambda: C._Kc @ x)
        r[on] = (s, k, kc)
    FI.ON = False; s2 = f.s_hat(q)
    (s0, k0, c0), (s1, k1, c1) = r.pop(False), r.pop(True)
    r.update(B=B, shat_rel=float((s1 - s0).norm() / s0.norm()), shat_noise=float((s2 - s0).norm() / s0.norm()),
             K64_rel=float((k1 - k0).norm() / k0.norm()), Kc_rel=float((c1 - c0).norm() / c0.norm()))
    print(json.dumps(r), flush=True)
