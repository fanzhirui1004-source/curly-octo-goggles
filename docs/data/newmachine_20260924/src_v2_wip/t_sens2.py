"""sens2 (reassociated) vs sens on real fields; timing. Usage: t_sens2.py <body> <case> [<case> ...]"""
import sys, json, time
import torch
import teacher as TE
for case in sys.argv[2:]:
    C = TE.Cell(case, sys.argv[1], log=lambda s_: None); C.assemble(); C.dmoments()
    g = torch.Generator(device=TE.dev).manual_seed(0)
    u = torch.randn((C.nb, 128), dtype=TE.dt, device=TE.dev, generator=g)
    TE.sync(); t = time.perf_counter(); s1 = torch.cat([C.sens(u[:, j:j + 32]) for j in range(0, 128, 32)], 1); TE.sync(); t1 = time.perf_counter() - t
    t = time.perf_counter(); s2 = C.sens2(u); TE.sync(); t2 = time.perf_counter() - t
    print(json.dumps(dict(case=case, rel=float(((s2 - s1).norm(dim=0) / s1.norm(dim=0)).max()), sens_s=t1, sens2_s=t2)), flush=True)
    C._free(); del C; torch.cuda.empty_cache()
