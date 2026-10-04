"""Whole-lattice iterative solution of the Table 5 route (a) system with PETSc (CPU, MPI): AMG-PCG or BDDC-PCG.
Reads the export of lat_dump.py. Conjugate gradients on the unscaled system, converged in the UNPRECONDITIONED relative
residual ||K u - f|| / ||f|| (the measure of the direct solutions), for every load to --rtol; along the way the iterate
is captured where the relative residual first falls below 1e-3, 1e-4, ..., and its compliance f^T u and energy-norm
difference from the final iterate are reported.
  --method gamg   smoothed-aggregation AMG (PCGAMG), block size 3, rigid-body near-nullspace from the DOF positions
  --method hypre  BoomerAMG (nodal coarsening, block size 3, coordinates for the interpolation variant)
  --method bddc   BDDC (PCBDDC on a MATIS matrix, one subdomain per cell, i.e. ranks = cells), vertices, edges, faces,
                  MUMPS for the local Dirichlet / Neumann problems and the coarse problem
Timing: matrix build (from the export; not part of a solver's cost), PCSetUp (setup), and per load the CG solve.
Usage: mpiexec -n <ranks> python petsc_solve.py <sysdir> <out.json> --method gamg|hypre|bddc [--rtol 1e-9] [--loads 6]
       [--maxit 5000] [extra PETSc options after --]"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
argv = sys.argv[1:]
extra = []
if '--' in argv:
    k = argv.index('--'); extra = argv[k + 1:]; argv = argv[:k]
import petsc4py
petsc4py.init([sys.argv[0]] + extra)
from petsc4py import PETSc
comm = PETSc.COMM_WORLD; rank, size = comm.getRank(), comm.getSize()
ap = argparse.ArgumentParser()
ap.add_argument('sysdir'); ap.add_argument('out'); ap.add_argument('--method', required=True)
ap.add_argument('--rtol', type=float, default=1e-9); ap.add_argument('--loads', type=int, default=6)
ap.add_argument('--maxit', type=int, default=5000)
a = ap.parse_args(argv)
S = Path(a.sysdir)
meta = json.loads((S / 'meta.json').read_text())
b_all = np.load(S / 'b.npy', mmap_mode='r'); n = b_all.shape[0]
xyz = np.load(S / 'xyz.npy', mmap_mode='r'); comp = np.load(S / 'comp.npy', mmap_mode='r')
opt = PETSc.Options()
log = lambda *s: (print(*s, flush=True) if rank == 0 else None)
rec = dict(method=a.method, ranks=size, n=n, layout=meta['layout'], rtol=a.rtol, options=extra, petsc=PETSc.Sys.getVersion())

blocked = bool((n % 3 == 0) and np.array_equal(np.asarray(comp[:3 * 1000]), np.tile([0, 1, 2], 1000)))
if blocked:                                                          # verify node triplets everywhere (cheap, chunked)
    for s0 in range(0, n, 3 * 2 ** 20):
        c = np.asarray(comp[s0:s0 + 3 * 2 ** 20]); x = np.asarray(xyz[s0:s0 + 3 * 2 ** 20])
        if not (np.array_equal(c, np.tile([0, 1, 2], len(c) // 3)) and np.abs(x[0::3] - x[1::3]).max() == 0
                and np.abs(x[0::3] - x[2::3]).max() == 0):
            blocked = False; break
rec['blocked_xyz_triplets'] = blocked

t = time.perf_counter()
if a.method in ('gamg', 'hypre'):
    bs = 3 if blocked else 1
    nn = n // bs
    lo_n = rank * nn // size; hi_n = (rank + 1) * nn // size
    r0, r1 = lo_n * bs, hi_n * bs
    ip = np.load(S / 'K_indptr.npy', mmap_mode='r'); ix = np.load(S / 'K_indices.npy', mmap_mode='r')
    dv = np.load(S / 'K_data.npy', mmap_mode='r')
    p0, p1 = int(ip[r0]), int(ip[r1])
    lip = (np.asarray(ip[r0:r1 + 1]) - p0).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ(size=((r1 - r0, n), (r1 - r0, n)), bsize=bs,
                              csr=(lip, np.asarray(ix[p0:p1]).astype(PETSc.IntType), np.asarray(dv[p0:p1])), comm=comm)
    A.setOption(PETSc.Mat.Option.SYMMETRIC, True); A.setOption(PETSc.Mat.Option.SPD, True)
    A.assemble()
    if bs == 3:
        cv = PETSc.Vec().createMPI(((hi_n - lo_n) * 3, nn * 3), bsize=3, comm=comm)
        cv.setArray(np.asarray(xyz[r0:r1:3]).reshape(-1).copy())
        A.setNearNullSpace(PETSc.NullSpace().createRigidBody(cv))
    rng = (r0, r1)
elif a.method == 'bddc':
    nc = len(meta['cells'])
    assert size == nc, f'bddc: one rank per cell ({nc}), got {size}'
    C = np.load(S / f'cell{rank}.npz')
    l2g = C['l2g'].astype(PETSc.IntType)
    lg = PETSc.LGMap().create(l2g, comm=comm)
    A = PETSc.Mat().createIS((n, n), lgmapr=lg, lgmapc=lg, comm=comm)
    nl = len(l2g)
    Al = PETSc.Mat().createAIJ(size=(nl, nl), csr=(C['indptr'].astype(PETSc.IntType), C['indices'].astype(PETSc.IntType),
                                                     C['data']), comm=PETSc.COMM_SELF)
    Al.setOption(PETSc.Mat.Option.SYMMETRIC, True); Al.assemble()
    A.setISLocalMat(Al); A.setOption(PETSc.Mat.Option.SYMMETRIC, True); A.setOption(PETSc.Mat.Option.SPD, True)
    A.assemble()
    rng = A.getOwnershipRange()
else:
    raise SystemExit('method')
comm.tompi4py().Barrier()
rec['matrix_build_s'] = time.perf_counter() - t
log('BUILT', json.dumps(dict(method=a.method, n=n, ranks=size, s=rec['matrix_build_s'], blocked=blocked)))

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType('cg'); ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
pc = ksp.getPC()
if a.method == 'gamg':
    pc.setType('gamg')
elif a.method == 'hypre':
    pc.setType('hypre'); pc.setHYPREType('boomeramg')
    if blocked:
        pc.setCoordinates(np.asarray(xyz[rng[0]:rng[1]:3]).copy())
else:
    pc.setType('bddc')
    for k, v in [('pc_bddc_use_faces', 'true'), ('pc_bddc_dirichlet_pc_factor_mat_solver_type', 'mumps'),
                 ('pc_bddc_neumann_pc_factor_mat_solver_type', 'mumps'), ('pc_bddc_coarse_pc_factor_mat_solver_type', 'mumps')]:
        if not opt.hasName(k):
            opt.setValue(k, v)
ksp.setTolerances(rtol=a.rtol, atol=0.0, max_it=a.maxit)
ksp.setFromOptions()
x, bv = A.createVecs()
comm.tompi4py().Barrier()
t = time.perf_counter(); ksp.setUp(); pc.setUp(); comm.tompi4py().Barrier()
rec['setup_s'] = time.perf_counter() - t
log('SETUP', rec['setup_s'])

levels = [10.0 ** -k for k in range(3, 13)]
loads = []
for j in range(min(a.loads, b_all.shape[1])):
    bv.setArray(np.asarray(b_all[rng[0]:rng[1], j]).copy())
    bn = bv.norm()
    x.set(0.0)
    caps, hist = {}, []
    t0 = [None]

    def mon(k_, it, rn):
        el = time.perf_counter() - t0[0]
        rel = rn / bn
        hist.append((it, el, rel))
        for L in levels:
            if rel <= L and L not in caps:
                caps[L] = dict(it=it, s=el, u=k_.buildSolution().copy())
    ksp.setMonitor(mon)
    comm.tompi4py().Barrier()
    t0[0] = time.perf_counter()
    ksp.solve(bv, x)
    comm.tompi4py().Barrier()
    ts = time.perf_counter() - t0[0]
    ksp.monitorCancel()
    r = bv.duplicate(); A.mult(x, r); r.axpy(-1.0, bv)
    Kx = bv.duplicate(); A.mult(x, Kx)
    ex = x.dot(Kx)
    C_fin = bv.dot(x)
    pts = []
    for L in sorted(caps, reverse=True):
        u = caps[L]['u']; d = u.copy(); d.axpy(-1.0, x); Kd = d.duplicate(); A.mult(d, Kd)
        pts.append(dict(level=L, it=caps[L]['it'], s=caps[L]['s'], compliance=bv.dot(u),
                        rel_compliance_diff=abs(bv.dot(u) / C_fin - 1), rel_energy_diff=float(np.sqrt(max(d.dot(Kd), 0) / ex))))
    rd = dict(load=j, its=ksp.getIterationNumber(), reason=int(ksp.getConvergedReason()), solve_s=ts,
              rel_residual=r.norm() / bn, compliance=C_fin, points=pts, hist=hist[::max(1, len(hist) // 200)])
    loads.append(rd)
    log('LOAD', json.dumps({k: v for k, v in rd.items() if k not in ('hist', 'points')}))
    if rank == 0:
        rec['loads'] = loads
        Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
if rank == 0:
    rec['loads'] = loads
    rec['total_setup_plus_solves_s'] = rec['setup_s'] + sum(l['solve_s'] for l in loads)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
log('DONE', rec.get('total_setup_plus_solves_s'))
