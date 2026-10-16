"""Smoke test of cpu_sens on one cell (host): exact field u = E q for random port values q (interior factor by
diag_sens.cpu_factor), sensitivity by cpu_sens.cell_sens (light record) vs Cell.dmoments + Cell.sens (central differences,
the reference route of lat_scale.py --sens fd). Usage: smoke_sens.py <case> [body]"""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import sys, time, json
import numpy as np
import torch
import trainlib as TL
import teacher as TE
import box_encode as BX
import cpu_sens as CS
BX.dev = TL.dev
case = sys.argv[1]; body = sys.argv[2] if len(sys.argv) > 2 else '/root/autodl-tmp/OPL/S4/body'
t = time.perf_counter(); C = TE.Cell(case, body, log=lambda s_: None); C.assemble(); t_cell = time.perf_counter() - t
DS.cpu_factor(C)
q = torch.as_tensor(np.random.default_rng(0).standard_normal((C.np_, 3)), dtype=torch.float64)
u = C.extend(q)
Lc = CS.light(C)
t = time.perf_counter(); s_ad, tt = CS.cell_sens(Lc, u.numpy()); t_ad = time.perf_counter() - t
t = time.perf_counter(); C.dmoments(); s_fd = C.sens(u).sum(1).numpy(); t_fd = time.perf_counter() - t
print(json.dumps(dict(case=case, dofs=int(C.nb), threads=torch.get_num_threads(), cell_s=t_cell, ad_s=t_ad, ad_parts=tt, fd_s=t_fd,
                      rel_diff=float(np.linalg.norm(s_ad - s_fd) / np.linalg.norm(s_fd)), s_ad=s_ad.tolist(), s_fd=s_fd.tolist())))
