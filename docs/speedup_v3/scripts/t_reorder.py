"""K_PP factor with cuDSS reordering algorithms (OPL_V3_CUDSS_REORDER; one algorithm per process): on a plate lattice's
K_PP (as assemble_kpp_csr / SparseSPD build it), the host memory peak of the plan (ru_maxrss before / after, MiB),
seconds of plan + factorisation, cuDSS's memory estimates (factor on the device), seconds per solve (B = 1), and the
solution of fixed right-hand sides saved to OUT.npy (compared across algorithms by the caller).
usage: OPL_V3_CUDSS_REORDER=ALG python t_reorder.py BODY_DIR LAYOUT.json OUT.npy"""
import os, sys, json, time, resource
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import numpy as np
import torch
import teacher as TE, lat_multi as LM, lat_precond as PR, kppstore as KS

body, layout, out = sys.argv[1:4]
dev = torch.device('cuda')
L = json.load(open(layout))
lay, host = {}, {}
for c in L['cells']:
    case = c['case'] + '_o000'
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True)
    C.assemble_deploy(C.taus0)
    r, cc, v = C._kpp_cache
    host[case] = KS.put(case, r, cc, v)
    C._kpp_cache = None
    g = LM.from_teacher(C)
    g._kpp_fn = (lambda cs: (lambda: KS.to_device(host[cs], dev)))(case)
    lay[tuple(c['position'])] = g
lat = LM.MultiLattice(lay, clamp=('x', 'min'), load=('x', 'max'), loads='consistent', n_random=0, device=dev, max_cols=16,
                      log=lambda s_: None)
crow, cols, vals = lat.assemble_kpp_csr()
n = lat.nfree
torch.cuda.synchronize(); torch.cuda.empty_cache()
rss0 = [int(l.split()[1]) / 1024 for l in open('/proc/self/status') if l.startswith('VmRSS')][0]      # current, MiB
t = time.perf_counter()
sol = PR.SparseSPD(['csr', crow, cols, vals], None, None, n, 'auto')
torch.cuda.synchronize()
t_fac = time.perf_counter() - t
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
g = torch.Generator(device=dev); g.manual_seed(0)
R = torch.randn((n, 3), dtype=torch.float64, device=dev, generator=g)
y = sol(R); torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(10):
    sol(R[:, :1])
torch.cuda.synchronize()
t_sol = (time.perf_counter() - t) / 10
np.save(out, y.cpu().numpy())
res = dict(alg=os.environ.get('OPL_V3_CUDSS_REORDER', 'DEFAULT'), n=n, nnz=int(len(cols)), plan_factor_s=round(t_fac, 2),
           solve_ms=round(1000 * t_sol, 2), rss_before_MiB=round(rss0), maxrss_after_MiB=round(rss1), plan_host_peak_MiB=round(rss1 - rss0),
           mem_est={k: v for k, v in (sol.mem_est or {}).items() if isinstance(v, (int, float, bool))})
KS.cleanup()
print(json.dumps(res), flush=True)
