import sys, json, os
os.environ['OPL_KE_SYMV'] = '1'
import models as MD  # noqa
import torch, teacher as TE
body, case = sys.argv[1].split(':')
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
Kc = C._Kc
print('elements', len(C.dofs), 'nb', C.nb)
for sym in (False, True):
    Kc.symv = sym
    x = torch.randn((C.nb, 3), dtype=torch.float32, device='cuda')
    with Kc.hold():
        for _ in range(3): Kc @ x
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(10): Kc @ x
            torch.cuda.synchronize()
    rows = sorted(((e.key, e.device_time_total / 10000, e.count // 10) for e in p.key_averages()), key=lambda z: -z[1])
    print('symv' if sym else 'dense', [(k[:60], round(t, 4), n) for k, t, n in rows[:8]])
