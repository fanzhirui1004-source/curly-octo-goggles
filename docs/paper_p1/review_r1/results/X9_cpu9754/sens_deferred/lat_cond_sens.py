"""Table 5 route (b) with design sensitivities: the conventional exact condensation of lat_cond_cpu.py (unchanged: per
cell setup + assembly, dense S_i by the PARDISO Schur option on the Jacobi-scaled cell matrix, exact block Cholesky of
the condensed system in the substructuring order, six loads), followed by
  recovery  per cell, the interior displacements of the three face loads from the condensed solution,
              u_I = -K_II^-1 K_IP u_P ,
            with the cell's upper stiffness kept from step 1 (CSR, unscaled): K_II is Jacobi scaled and factorised by MKL
            PARDISO Cholesky (tuned iparm, phases 11 / 22) and solved (phase 33) for the three loads, then released.
            The Schur handles of step 1 are not kept for this: for the eight-cell lattices their PARDISO memory (6-15 GiB
            per cell) together with the held S_i would exceed the machine, so the interior factor is recomputed.
  sens      cpu_sens.cell_sens on the full cell field (as route (a) and the learned route).
Checks per cell: relative interior residual ||K_II u_I + K_IP u_P|| / ||K_IP u_P||; for the lattice, sum_c u_c^T K_c u_c
against sum_k f_k^T U_k over the face loads.
NEW script; lat_cond_cpu.py is unchanged and its record keys are kept (plus 'recovery_*', 'sens_*', 'sens').
Usage: lat_cond_sens.py <out.json> <layout.json> [--body ...] [--iparm-file f] [--margin-gib 4] [--n-random 3]
       [--ref f1.json,...] [--ad-batch 256] [--no-sens]
Environment as lat_cond_cpu.py (OPL_DEV=cpu, PYPARDISO_MKL_RT, MKL_NUM_THREADS = OMP_NUM_THREADS = 16)."""
import diag_sens as DS                                                   # noqa: F401  first: CPU environment
import sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import trainlib as TL
import teacher as TE
import box_encode as BX
import lat_multi as LM
import pardiso_direct as PD
import lat_cond_cpu as LC
import cpu_sens as CS

BX.dev = TL.dev


def upper_csr(C):
    """The cell's assembled upper stiffness as a scipy CSR (unscaled; arrays copied out of the cell)."""
    return sp.csr_matrix((C.vals.cpu().numpy().copy(), C.cu.cpu().numpy().astype(np.int32),
                          C.crow.cpu().numpy().astype(np.int64)), shape=(int(C.nb), int(C.nb)))


def recover(Lc, Kup, uP, iparm, margin):
    """u (nb, k) full field from the retained displacements uP (np_, k) by the interior solve; record of the step."""
    r = {}
    t = time.perf_counter()
    P_, I_ = Lc.P, Lc.I
    KI = Kup[I_]
    KII = KI[:, I_].tocsr()                                                  # upper triangle of K_II (I_ ascending)
    KIP = (KI[:, P_] + Kup[P_][:, I_].T).tocsr()                             # K_IP from both triangles
    del KI
    rhs = -(KIP @ uP)
    d = KII.diagonal(); sA = 1 / np.sqrt(d)
    Sd = sp.diags(sA)
    A = (Sd @ KII @ Sd).tocsr(); A.sort_indices()
    r['blocks_s'] = time.perf_counter() - t
    H = PD.Pardiso(A, 2, iparm)
    r['analysis_s'] = H.phase(11)
    r['predicted_GiB'] = H.mem_GiB()
    av = PD.avail_gib()
    if r['predicted_GiB']['total'] + margin > av:
        H.release()
        r['skipped'] = f"interior factor: predicted {r['predicted_GiB']['total']:.2f} GiB + margin > available {av:.2f} GiB"
        return None, r
    r['factor_s'] = H.phase(22)
    t = time.perf_counter(); y = H.solve(np.asfortranarray(rhs * sA[:, None])); r['solve_s'] = time.perf_counter() - t
    r['pardiso'] = H.info()
    H.release()
    uI = y * sA[:, None]
    res = PD.sym_upper_matvec(KII, uI) - rhs
    r['interior_rel_residual'] = float(np.linalg.norm(res) / np.linalg.norm(rhs))
    u = np.zeros((int(Lc.nb), uP.shape[1])); u[P_] = uP; u[I_] = uI
    r['energy'] = CS.energy(Lc, Kup, u)
    r['seconds'] = r['blocks_s'] + r['analysis_s'] + r['factor_s'] + r['solve_s']
    return u, r


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layout')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body'); ap.add_argument('--iparm-file', default=None)
    ap.add_argument('--margin-gib', type=float, default=4.0); ap.add_argument('--n-random', type=int, default=3)
    ap.add_argument('--ref', default='')
    ap.add_argument('--ad-batch', type=int, default=256); ap.add_argument('--no-sens', action='store_true')
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    iparm = PD.tuned_iparm()
    if a.iparm_file and Path(a.iparm_file).exists():
        iparm = {int(k): int(v) for k, v in json.loads(Path(a.iparm_file).read_text())['iparm'].items()}
    L = json.loads(Path(a.layout).read_text())
    rec = dict(layout=L['name'], machine=__import__('platform').node(),
               route='(b) conventional exact condensation (lat_cond_cpu, block solver) + interior recovery + sensitivities',
               env=PD.env_record(), iparm={str(k): v for k, v in iparm.items()}, loadavg_start=PD.loadavg(),
               cpu_stat_start=PD.cpu_stat(), cells={})
    save = lambda: Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    t_all = time.perf_counter()
    S_of, lay, lights, kups = {}, {}, {}, {}
    # 1. cells (as lat_cond_cpu.main; the light sensitivity record and the upper CSR are kept)
    for c in L['cells']:
        r = {}
        PD.reset_peak()
        t = time.perf_counter(); C = TE.Cell(c['case'], a.body, log=lambda s_: None); r['setup_s'] = time.perf_counter() - t
        t = time.perf_counter(); C.assemble(); r['assembly_s'] = time.perf_counter() - t
        r.update(dofs=int(C.nb), retained=int(C.np_), interior=int(C.ni), front_end_peak_rss_GiB=PD.peak_rss_gib())
        G = LM.from_teacher(C)
        fw = {(ax, v): G.face_w_fn(ax, v) for ax in range(3) for v in (0.0, 1.0)}
        G.face_w_fn = lambda ax, v, fw=fw: fw[(ax, float(v))]
        G._kpp_fn = None; G.cell = None
        lay[tuple(c['position'])] = G
        S, sr = LC.cell_schur(C, iparm, a.margin_gib, log)
        r['schur'] = sr
        if not a.no_sens:
            t = time.perf_counter()
            lights[c['case']] = CS.light(C); kups[c['case']] = upper_csr(C)
            r['keep_s'] = time.perf_counter() - t
            r['kept_csr_GiB'] = (kups[c['case']].data.nbytes + kups[c['case']].indices.nbytes + kups[c['case']].indptr.nbytes) / PD.GIB
        rec['cells'][c['case']] = r
        C._free(); C = None; gc.collect()
        log(dict(event='CELL', case=c['case'], **{k: v for k, v in r.items() if k != 'schur'},
                 schur_s=sr.get('seconds'), skipped=sr.get('skipped')))
        if S is None:
            rec['skipped'] = f"cell {c['case']}: {sr['skipped']}"
            save(); return
        S_of[c['case']] = S
        rec['held_S_GiB'] = sum(8 * float(s_.shape[0]) ** 2 for s_ in S_of.values()) / PD.GIB
        save()
    rec['cells_s'] = time.perf_counter() - t_all
    rec['dense_S_total_GiB'] = rec['held_S_GiB']
    rec['kept_csr_total_GiB'] = sum(x.get('kept_csr_GiB', 0.0) for x in rec['cells'].values())
    rec['rss_after_cells_GiB'] = PD.rss_gib()
    # 2. lattice
    t = time.perf_counter()
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random,
                          device='cpu', log=lambda s_: None)
    rec['lattice_s'] = time.perf_counter() - t
    F = lat.F.cpu().numpy().astype(np.float64); nfree = int(lat.nfree)
    gl = [lat.fmap[lat.idx[i]] for i in range(len(lat.geoms))]
    cases = [G.case for G in lat.geoms]
    labels = list(lat.labels)
    rec.update(free_retained=nfree, loads=labels, solver='block')
    rec['solver_description'] = ('exact block Cholesky of the block-sparse condensed matrix in the substructuring order '
                                 '(cell-private retained DOFs eliminated per cell, then the dense shared-DOF interface), '
                                 'dense MKL kernels (dpotrf, dtrsm, dsyrk, dgemm)')
    S_list = [S_of.pop(c) for c in cases]
    t = time.perf_counter()
    U = LC.block_solve(gl, S_list, F, rec, a.margin_gib)
    if U is None:
        save(); return
    rec['condensed_s'] = time.perf_counter() - t
    rec['compliance'] = [float(F[:, j] @ U[:, j]) for j in range(F.shape[1])]
    rf, mt, rc = LC.reference([x for x in a.ref.split(',') if x], F.shape[1])
    if rc is not None:
        rec['reference'] = dict(file=rf, mtype=mt, compliance=rc,
                                rel_diff=[abs(x / y - 1) for x, y in zip(rec['compliance'], rc)])
    rec['front_end_s'] = sum(c['setup_s'] + c['assembly_s'] for c in rec['cells'].values())
    rec['schur_s'] = sum(c['schur']['seconds'] for c in rec['cells'].values())
    rec['total_without_sens_s'] = time.perf_counter() - t_all - sum(c.get('keep_s', 0.0) for c in rec['cells'].values())
    save()
    # 3. interior recovery and sensitivities
    if not a.no_sens:
        fcols = [j for j, l in enumerate(labels) if not l.startswith('random')]
        rec['sens_loads'] = [labels[j] for j in fcols]
        Uf = np.ascontiguousarray(U[:, fcols])
        PD.reset_peak()
        rec_cells, per, energy = {}, {}, 0.0
        t_rec = t_sens = 0.0
        for i, cs in enumerate(cases):
            Lc, g = lights.pop(cs), gl[i]
            uP = np.zeros((int(Lc.np_), len(fcols))); ok = g >= 0; uP[ok] = Uf[g[ok]]
            t = time.perf_counter()
            u, rr = recover(Lc, kups.pop(cs), uP, iparm, a.margin_gib)
            t_rec += time.perf_counter() - t
            rec_cells[cs] = rr
            if u is None:
                rec['skipped'] = f'recovery {cs}: {rr["skipped"]}'
                save(); return
            energy += rr['energy']
            t = time.perf_counter()
            sc, tt = CS.cell_sens(Lc, u, batch=a.ad_batch)
            dt_ = time.perf_counter() - t; t_sens += dt_
            per[cs] = dict(seconds=dt_, **tt, s=sc.tolist())
            del u; gc.collect()
            log(dict(event='SENS_CELL', case=cs, recovery_s=rr['seconds'], sens_s=dt_, res=rr['interior_rel_residual']))
        comp_f = float(sum(F[:, j] @ U[:, j] for j in fcols))
        rec['recovery'] = rec_cells
        rec['recovery_s'] = t_rec
        rec['sens_s'] = t_sens
        rec['sens'] = per
        rec['energy_check'] = dict(sum_cell_energy=energy, sum_face_compliance=comp_f, rel_diff=abs(energy / comp_f - 1))
        rec['recovery_sens_peak_rss_GiB'] = PD.peak_rss_gib()
        rec['sens_settings'] = dict(objective='sum of the face-load compliances', method='moments_ad.cell_sens (reverse mode)',
                                    skip_full=True, ad_batch=a.ad_batch, torch_threads=torch.get_num_threads(),
                                    recovery='interior Cholesky of K_II per cell (PARDISO, tuned iparm), recomputed')
    rec['total_s'] = time.perf_counter() - t_all - sum(c.get('keep_s', 0.0) for c in rec['cells'].values())
    rec['keep_s'] = sum(c.get('keep_s', 0.0) for c in rec['cells'].values())
    rec['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
    rec['loadavg_end'] = PD.loadavg(); cs_ = PD.cpu_stat()
    rec['throttled_s'] = (cs_.get('throttled_usec', 0) - rec['cpu_stat_start'].get('throttled_usec', 0)) / 1e6
    save()
    log(dict(event='DONE', **{k: rec.get(k) for k in ('front_end_s', 'schur_s', 'condensed_s', 'recovery_s', 'sens_s', 'total_s',
                                                     'compliance', 'reference', 'energy_check', 'process_peak_rss_GiB')}))


if __name__ == '__main__':
    main(sys.argv[1:])
