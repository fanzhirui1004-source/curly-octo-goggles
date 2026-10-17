"""Check of teacher.SPDSolver._release_matrix (src_v3 S21, OPL_V3_KPP_FREEA): after the factorisation the CSR operand's
storages are released; the solutions of the same right-hand sides before and after must be bitwise equal (the solve
phase reads only the factors). Synthetic SPD matrix: 3-D 7-point Laplacian plus a diagonal shift, Jacobi scaled, upper
CSR, as lat_precond.SparseSPD hands it over. Also: device memory released, and the float32 factor's relative difference.
usage: python t_kppfree.py [--m 80]"""
import os, sys, json, argparse, time
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
import torch
import teacher as TE

ap = argparse.ArgumentParser(); ap.add_argument('--m', type=int, default=80)
a = ap.parse_args()
dev = 'cuda'
m = a.m
n = m ** 3
i = torch.arange(n, device=dev)
x, y, z = i // (m * m), (i // m) % m, i % m
rows, cols, vals = [i], [i], [torch.full((n,), 6.1, dtype=torch.float64, device=dev)]
for d, ok in ((1, z < m - 1), (m, y < m - 1), (m * m, x < m - 1)):
    r = i[ok]
    rows.append(r); cols.append(r + d); vals.append(torch.full((len(r),), -1.0, dtype=torch.float64, device=dev))
r, c, v = torch.cat(rows), torch.cat(cols), torch.cat(vals)
o = torch.argsort(r * n + c); r, c, v = r[o], c[o], v[o]
s = 1 / torch.sqrt(torch.full((n,), 6.1, dtype=torch.float64, device=dev))
v = v * s[r] * s[c]
crow = torch.zeros(n + 1, dtype=torch.long, device=dev); crow[1:] = torch.cumsum(torch.bincount(r, minlength=n), 0)
crow, c = crow.int(), c.int()
g = torch.Generator(device=dev); g.manual_seed(0)
R = torch.randn((n, 3), dtype=torch.float64, device=dev, generator=g)
res = dict(n=n, nnz_upper=int(len(v)))
TE.FREE_A = False
sol = TE.SPDSolver(crow.clone(), c.clone(), v.clone(), n)
y0 = sol.solve(R); y0b = sol.solve(R[:, :1])
torch.cuda.synchronize(); a0 = torch.cuda.memory_allocated()
sol._release_matrix()
torch.cuda.synchronize(); a1 = torch.cuda.memory_allocated()
y1 = sol.solve(R); y1b = sol.solve(R[:, :1])
res.update(released_MiB=round((a0 - a1) / 2 ** 20, 1), equal_B3=bool(torch.equal(y0, y1)), equal_B1=bool(torch.equal(y0b, y1b)),
           max_abs_diff=float((y0 - y1).abs().max()))
sol.free(); del sol
sol32 = TE.SPDSolver(crow.clone(), c.clone(), v.clone(), n, fdt=torch.float32)
y32 = sol32.solve(R)
res['fp32_rel_diff'] = float((y32 - y0).norm() / y0.norm())
res['fp32_mem_est'] = sol32.mem_est.get('permanent_device_memory') if isinstance(sol32.mem_est, dict) else None
print(json.dumps(res), flush=True)
