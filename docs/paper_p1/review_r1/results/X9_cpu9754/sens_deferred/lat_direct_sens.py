"""Table 5 route (a) with design sensitivities: the whole-lattice direct solution of lat_direct_cpu2.py (unchanged: same
cells, lattice, clamp, six loads, global matrix, Jacobi scaling, MKL PARDISO Cholesky through pardiso_direct with the
tuned iparm, phases 11 / 22 / 33 timed), followed by the sensitivity stage of the learned route on the host:
  sens   per cell, the exact full field of the three face loads is read from the global solution (retained DOFs through
         the lattice numbering, interior DOFs from the cell's block, clamped DOFs zero), then cpu_sens.cell_sens:
         energy densities summed over the face loads, one reverse pass through the element moments -> s_c (8 values).
         Timed per cell (energy density, reverse pass) and in total; it runs after PARDISO has released its factor.
The cells keep only what the sensitivity needs (cpu_sens.light) when the route releases them after the global assembly.
NEW script; lat_direct_cpu2.py is unchanged and its record keys are kept (plus 'sens_*', 'sens').
Usage: lat_direct_sens.py <out.json> <layout.json> [--body /root/autodl-tmp/OPL/S4/body] [--iparm-file f]
       [--margin-gib 4] [--n-random 3] [--ad-batch 256] [--no-sens]
Environment as lat_direct_cpu2.py (OPL_DEV=cpu, PYPARDISO_MKL_RT, MKL_NUM_THREADS = OMP_NUM_THREADS)."""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import os, sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import trainlib as TL
import teacher as TE
import box_encode as BX
import lat_multi as LM
import lat_direct_cpu as LD
import pardiso_direct as PD
import cpu_sens as CS

BX.dev = TL.dev


def pardiso_run(A, s, b, F, iparm, margin, log):
    """lat_direct_cpu2.pardiso_run for mtype 2, also returning the unscaled solution u (n, k)."""
    n = A.shape[0]
    r = dict(mtype=2, nnz_stored=int(A.nnz), iparm_requested={str(k): v for k, v in iparm.items()},
             loadavg_start=PD.loadavg(), cpu_stat_start=PD.cpu_stat())
    P = PD.Pardiso(A, 2, iparm)
    ok = PD.reset_peak()
    r['analysis_s'] = P.phase(11)
    r['analysis_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    r['predicted_kB'] = P.mem_kB(); r['predicted_GiB'] = P.mem_GiB()
    av = PD.avail_gib()
    r['avail_before_factor_GiB'] = av
    r['cgroup_before_factor'] = PD.cgroup()
    log(dict(event='ANALYSIS', **{k: v for k, v in r.items() if k not in ('cpu_stat_start', 'cgroup_before_factor')}))
    need = r['predicted_GiB']['total'] + margin
    if need > av:
        r['skipped'] = f"predicted {r['predicted_GiB']['total']:.2f} GiB + margin {margin} GiB > available {av:.2f} GiB"
        P.release()
        return r, None
    ok = PD.reset_peak()
    r['factor_s'] = P.phase(22)
    r['factor_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    r['pardiso'] = P.info()
    bs = np.asfortranarray(b * s[:, None])
    ok = PD.reset_peak()
    t = time.perf_counter(); x = P.solve(bs); r['solve_s'] = time.perf_counter() - t
    r['solve_peak_rss_GiB'] = PD.peak_rss_gib() if ok else None
    P.release()
    Ax = PD.sym_upper_matvec(A, x)
    res = (Ax - bs) / s[:, None]
    u = x * s[:, None]
    r['rel_residual'] = [float(np.linalg.norm(res[:, j]) / np.linalg.norm(b[:, j])) for j in range(b.shape[1])]
    r['compliance'] = [float(F[:, j] @ u[:F.shape[0], j]) for j in range(F.shape[1])]
    r['loadavg_end'] = PD.loadavg(); cs = PD.cpu_stat()
    r['throttled_s'] = (cs.get('throttled_usec', 0) - r['cpu_stat_start'].get('throttled_usec', 0)) / 1e6
    r['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
    return r, u


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layout')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
    ap.add_argument('--iparm-file', default=None)
    ap.add_argument('--margin-gib', type=float, default=4.0); ap.add_argument('--n-random', type=int, default=3)
    ap.add_argument('--ad-batch', type=int, default=256); ap.add_argument('--no-sens', action='store_true')
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    iparm = PD.tuned_iparm()
    if a.iparm_file and Path(a.iparm_file).exists():
        iparm = {int(k): int(v) for k, v in json.loads(Path(a.iparm_file).read_text())['iparm'].items()}
    L = json.loads(Path(a.layout).read_text())
    rec = dict(layout=L['name'], machine=__import__('platform').node(), env=PD.env_record(), loadavg_start=PD.loadavg(),
               route='(a) whole-lattice direct solve (lat_direct_cpu2) + sensitivities (cpu_sens)', cells={})
    save = lambda: Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    Cs, lay, lights = [], {}, []
    PD.reset_peak()
    t0 = time.perf_counter()
    for c in L['cells']:
        t = time.perf_counter(); C = TE.Cell(c['case'], a.body, log=lambda s_: None); ts = time.perf_counter() - t
        t = time.perf_counter(); C.assemble(); ta = time.perf_counter() - t
        rec['cells'][c['case']] = dict(setup_s=ts, assembly_s=ta, dofs=int(C.nb), retained=int(C.np_), interior=int(C.ni))
        Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
        lights.append(CS.light(C))
    rec['cells_s'] = time.perf_counter() - t0
    t = time.perf_counter()
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random,
                          device='cpu', log=lambda s_: None)
    rec['lattice_s'] = time.perf_counter() - t
    assert [G.case for G in lat.geoms] == [C.case for C in Cs]
    F = lat.F.cpu().numpy().astype(np.float64)
    t = time.perf_counter()
    Ku, n, offs = LD.global_matrix(Cs, lat)
    rec['global_assembly_s'] = time.perf_counter() - t
    gmaps = []                                                              # local DOF -> global row (-1: clamped)
    for i, C in enumerate(Cs):                                              # (as LD.global_matrix; not timed)
        g = np.full(int(C.nb), -1, np.int64)
        g[C.P.cpu().numpy()] = lat.fmap[lat.idx[i]]
        g[C.I.cpu().numpy()] = offs[i] + np.arange(int(C.ni))
        gmaps.append(g)
    labels = list(lat.labels)
    rec.update(global_dofs=int(n), free_retained=int(lat.nfree), interior=int(n - lat.nfree), nnz_upper=int(Ku.nnz),
               loads=labels, front_end_peak_rss_GiB=PD.peak_rss_gib())
    log(dict(event='ASSEMBLED', **{k: v for k, v in rec.items() if k not in ('cells', 'env')}))
    for C in Cs:                                                            # release the cells: only K, F (and the light
        C._free()                                                           # sensitivity records) remain
    for G in lat.geoms:
        G.cell, G._kpp, G._kpp_fn, G.face_w_fn = None, None, None, None
    Cs = lay = lat = None; gc.collect()
    b = np.zeros((n, F.shape[1])); b[:F.shape[0]] = F
    d = Ku.diagonal()
    s = 1 / np.sqrt(np.where(d > 0, d, 1.0))
    Sd = sp.diags(s)
    t = time.perf_counter()
    Us = (Sd @ Ku @ Sd).tocsr(); Us.sort_indices()
    rec['scaling_s'] = time.perf_counter() - t
    del Ku; gc.collect()
    rec['rss_before_pardiso_GiB'] = PD.rss_gib()
    r, u = pardiso_run(Us, s, b, F, iparm, a.margin_gib, log)
    rec['runs'] = [r]
    log(dict(event='RUN', **{k: v for k, v in r.items() if k not in ('cpu_stat_start',)}))
    save()
    if u is not None and not a.no_sens:
        fcols = [j for j, l in enumerate(labels) if not l.startswith('random')]
        rec['sens_loads'] = [labels[j] for j in fcols]
        uf = np.ascontiguousarray(u[:, fcols]); del u; gc.collect()
        PD.reset_peak()
        sens, per = [], {}
        t = time.perf_counter()
        for Lc, g in zip(lights, gmaps):
            uc = np.zeros((int(Lc.nb), len(fcols)))
            ok = g >= 0
            uc[ok] = uf[g[ok]]
            tc = time.perf_counter()
            sc, tt = CS.cell_sens(Lc, uc, batch=a.ad_batch)
            per[Lc.case] = dict(seconds=time.perf_counter() - tc, **tt, s=sc.tolist())
            sens.append(sc)
        rec['sens_s'] = time.perf_counter() - t
        rec['sens_peak_rss_GiB'] = PD.peak_rss_gib()
        rec['sens'] = per
        rec['sens_settings'] = dict(objective='sum of the face-load compliances', method='moments_ad.cell_sens (reverse mode)',
                                    skip_full=True, ad_batch=a.ad_batch, torch_threads=torch.get_num_threads())
        log(dict(event='SENS', seconds=rec['sens_s'], torch_threads=torch.get_num_threads()))
    r0 = rec['runs'][0]
    if 'factor_s' in r0:
        st = dict(cells=rec['cells_s'], lattice=rec['lattice_s'], global_assembly=rec['global_assembly_s'],
                  scaling=rec['scaling_s'], analysis=r0['analysis_s'], factor=r0['factor_s'], solve=r0['solve_s'],
                  sens=rec.get('sens_s', 0.0))
        rec['stages_s'] = st
        rec['total_s'] = sum(st.values())
        rec['total_without_sens_s'] = rec['total_s'] - st['sens']
    rec['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
    save()
    log(dict(event='DONE', total_s=rec.get('total_s'), process_peak_rss_GiB=rec['process_peak_rss_GiB']))


if __name__ == '__main__':
    main(sys.argv[1:])
