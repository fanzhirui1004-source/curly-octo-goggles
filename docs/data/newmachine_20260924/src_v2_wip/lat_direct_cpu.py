"""P1 cost baseline, type 1: the WHOLE lattice solved directly on the host CPU, without substructuring -- every cell's
full cut-cell stiffness (retained and interior DOFs) assembled into one global matrix and factorised once by MKL PARDISO,
then solved for the lattice's load cases. This is the full-scale finite element analysis that a practitioner would run
instead of condensing cells.
  cells       teacher.Cell on the host: setup + assemble (cut-cell integration, stiffness), per cell
  numbering   retained DOFs glued exactly as lat_multi.MultiLattice (same clamp, same free set, same loads F); interior DOFs
              of each cell appended after the free retained DOFs (cell by cell, in C.I order)
  assembly    upper triplets of every cell mapped to global indices (clamped retained DOFs dropped), duplicates summed,
              symmetric Jacobi scaling D^-1/2 K D^-1/2 (as bench_cpu.py's interior factor)
  analysis    PARDISO phase 11 first: predicted peak / permanent / factor memory (iparm 15-17, kB); the numerical
              factorisation is skipped when the prediction exceeds --mem-gb
  factor      phase 22 (numerical factorisation), solve phase 33 for all load cases at once
  mtypes      11 = real nonsymmetric LU (pypardiso default, the setting of bench_cpu.py / Table 5) and 2 = real symmetric
              positive definite Cholesky on the upper triangle; each run separately with its own analysis
Outputs per mtype: seconds per phase, PARDISO memory, process peak RSS, compliance F_j^T u_j per load, relative residual
||K u - f|| / ||f|| per load (unscaled system).
Usage: lat_direct_cpu.py <out.json> <layout.json> [--body /root/autodl-tmp/OPL/S4/body] [--mtypes 11,2] [--mem-gb 70]
Environment as bench_cpu.py: OPL_DEV=cpu, CUDA_VISIBLE_DEVICES=, MKL_NUM_THREADS=OMP_NUM_THREADS=16, PYPARDISO_MKL_RT."""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import os, sys, json, time, argparse, gc, resource
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import teacher as TE
import lat_multi as LM


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20            # ru_maxrss is in kB on Linux


def global_matrix(Cs, lat):
    """Upper-triangle global stiffness (CSR, unscaled) over [free retained DOFs | interior DOFs of cell 0 | cell 1 | ...]."""
    nfree = lat.nfree
    offs, n = [], nfree
    for C in Cs:
        offs.append(n); n += int(C.ni)
    R, Cc, V = [], [], []
    for i, C in enumerate(Cs):
        g = np.full(int(C.nb), -1, np.int64)
        P, I = C.P.cpu().numpy(), C.I.cpu().numpy()
        g[P] = lat.fmap[lat.idx[i]]                                          # local port order = C.P order (from_teacher)
        g[I] = offs[i] + np.arange(len(I))
        ru, cu, v = C.ru.cpu().numpy().astype(np.int64), C.cu.cpu().numpy().astype(np.int64), C.vals.cpu().numpy()
        gr, gc = g[ru], g[cu]
        ok = (gr >= 0) & (gc >= 0) & (v != 0)
        gr, gc, v = gr[ok], gc[ok], v[ok]
        R.append(np.minimum(gr, gc)); Cc.append(np.maximum(gr, gc)); V.append(v.astype(np.float64))
    K = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(n, n)).tocsr()
    K.sum_duplicates()
    return K, n, offs


def pardiso_run(Ku, F, mtype, mem_gb, log):
    import pypardiso
    n = Ku.shape[0]
    d = Ku.diagonal()
    s = 1 / np.sqrt(np.where(d > 0, d, 1.0))
    S = sp.diags(s)
    Us = (S @ Ku @ S).tocsr()
    A = Us if mtype == 2 else (Us + sp.triu(Us, 1).T).tocsr()
    A.sort_indices()
    ps = pypardiso.PyPardisoSolver(mtype=mtype)
    r = dict(mtype=mtype, nnz_stored=int(A.nnz))
    b = np.zeros((n, F.shape[1])); b[:F.shape[0]] = F
    bs = np.asfortranarray(b * s[:, None])                                   # PARDISO reads right-hand sides column-major
    ps._check_A(A)                                                          # CSR flag (iparm 12), sorted indices, no empty rows
    t = time.perf_counter()
    ps.set_phase(11); ps._call_pardiso(A, bs)
    r['analysis_s'] = time.perf_counter() - t
    ip = ps.get_iparms()
    r['predicted_kB'] = dict(peak_analysis=int(ip.get(15, 0)), permanent=int(ip.get(16, 0)), factor_solve=int(ip.get(17, 0)))
    pred_gb = (r['predicted_kB']['permanent'] + r['predicted_kB']['factor_solve']) / 2 ** 20
    r['predicted_GB'] = pred_gb
    log(dict(event='ANALYSIS', **r))
    if pred_gb > mem_gb:
        r['skipped'] = f'predicted {pred_gb:.1f} GB > {mem_gb} GB'
        ps.free_memory(everything=True)
        return r
    t = time.perf_counter()
    ps.set_phase(22); ps._call_pardiso(A, bs)
    r['factor_s'] = time.perf_counter() - t
    ip = ps.get_iparms()
    r['after_factor_kB'] = dict(peak_analysis=int(ip.get(15, 0)), permanent=int(ip.get(16, 0)), factor_solve=int(ip.get(17, 0)))
    t = time.perf_counter()
    ps.set_phase(33); x = ps._call_pardiso(A, bs)
    r['solve_s'] = time.perf_counter() - t
    u = np.asarray(x).reshape(n, -1) * s[:, None]
    Kfull = (Ku + sp.triu(Ku, 1).T).tocsr()
    res = Kfull @ u - b
    r['rel_residual'] = [float(np.linalg.norm(res[:, j]) / np.linalg.norm(b[:, j])) for j in range(b.shape[1])]
    r['compliance'] = [float(F[:, j] @ u[:F.shape[0], j]) for j in range(F.shape[1])]
    r['peak_rss_GB'] = rss_gb()
    ps.free_memory(everything=True)
    return r


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layout')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
    ap.add_argument('--mtypes', default='11,2'); ap.add_argument('--mem-gb', type=float, default=70.0)
    ap.add_argument('--n-random', type=int, default=3)
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    L = json.loads(Path(a.layout).read_text())
    rec = dict(layout=L['name'], threads=dict(MKL=os.environ.get('MKL_NUM_THREADS'), OMP=os.environ.get('OMP_NUM_THREADS'),
               torch=torch.get_num_threads()), cpu=os.cpu_count(), loadavg_start=os.getloadavg(), cells={})
    Cs, lay = [], {}
    t0 = time.perf_counter()
    for c in L['cells']:
        t = time.perf_counter(); C = TE.Cell(c['case'], a.body, log=lambda s_: None); ts = time.perf_counter() - t
        t = time.perf_counter(); C.assemble(); ta = time.perf_counter() - t
        rec['cells'][c['case']] = dict(setup_s=ts, assembly_s=ta, dofs=int(C.nb), retained=int(C.np_), interior=int(C.ni))
        Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
    rec['cells_s'] = time.perf_counter() - t0
    t = time.perf_counter()
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random,
                          device='cpu', log=lambda s_: None)
    rec['lattice_s'] = time.perf_counter() - t
    assert [G.case for G in lat.geoms] == [C.case for C in Cs]
    F = lat.F.cpu().numpy().astype(np.float64)
    t = time.perf_counter()
    Ku, n, offs = global_matrix(Cs, lat)
    rec['global_assembly_s'] = time.perf_counter() - t
    rec.update(global_dofs=int(n), free_retained=int(lat.nfree), interior=int(n - lat.nfree), nnz_upper=int(Ku.nnz),
               loads=lat.labels, rss_after_assembly_GB=rss_gb())
    log(dict(event='ASSEMBLED', **{k: v for k, v in rec.items() if k != 'cells'}))
    for C in Cs:
        C._free()
    Cs = None; gc.collect()
    rec['runs'] = []
    for mt in (int(m) for m in a.mtypes.split(',')):
        r = pardiso_run(Ku, F, mt, a.mem_gb, log)
        rec['runs'].append(r)
        log(dict(event='RUN', **r))
        Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    log(dict(event='DONE'))


if __name__ == '__main__':
    main(sys.argv[1:])
