import os, json, time, sys
import models as MD  # noqa
import torch, teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
dev, dt = TE.dev, TE.dt
def timed(fn, rep=3):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t) / rep
FI.ON = True; FN.FUSED = True; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_TAILT_FUSED'] = '1'
body, case = sys.argv[1].split(':')
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
r = {}
for B in (1, 3, 6, 16, 24, 48):
    q = torch.randn((C.np_, B), dtype=dt, device=dev)
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    r[f'ms_{B}'] = timed(lambda: f.s_hat(q)); r[f'peakGB_{B}'] = (torch.cuda.max_memory_allocated() - base) / 1e9
print(json.dumps(dict(case=case, **r)))
