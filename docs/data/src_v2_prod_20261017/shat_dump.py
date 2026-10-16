"""S_hat q of a deploy cell under the current OPL_* switches (same q), saved for shat_cmp; plus float64 K u."""
import os, sys, torch
import models as MD                                                    # noqa: F401
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD
FN.FUSED = True; os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
body, case = sys.argv[1].split(':')
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
g = torch.Generator(device='cuda').manual_seed(0)
q = torch.randn((C.np_, 6), dtype=TE.dt, device='cuda', generator=g)
u = torch.randn((C.nb, 3), dtype=TE.dt, device='cuda', generator=g)
torch.save(dict(shat=f.s_hat(q).cpu(), Ku=(C.K @ u).cpu()), sys.argv[2])
