"""Revision (lattice-level cost table, route (b)): CONVENTIONAL EXACT CONDENSATION of a whole lattice on the host CPU. NEW
script; nothing else changes. Same cells, numbering, clamp and loads as lat_direct_cpu(2).py / lat_multi.MultiLattice.
Per repetition (one process):
  1 per cell, one at a time: host front end (teacher.Cell setup + assembly, as lat_direct_cpu), lattice geometry of the
    cell (lat_multi.from_teacher; face weights cached so the cell can be released), then the dense Schur complement onto
    the cell's retained DOFs by MKL PARDISO Cholesky (mtype 2) with the Schur-complement option (iparm(36) = 1, perm marks
    the retained DOFs) on the symmetrically Jacobi-scaled full cell matrix, unscaled afterwards; the cell is released and
    its dense S_i (np_i x np_i, float64) is HELD until the condensed assembly (their total is reported).
  2 MultiLattice (clamp y = min, consistent loads on y = max + 3 random loads): free retained DOFs, loads F.
  3 condensed assembly: K_hat = sum_i A_i^T S_i A_i over the free retained DOFs, as an upper-triangle CSR in the
    substructuring order (every cell's private DOFs cell by cell, then the DOFs shared by several cells); built in two
    passes (exact structure, then values cell by cell, each S_i released right after its contribution).
  4 condensed solve: SPARSE Cholesky of this block-sparse matrix by MKL PARDISO (mtype 2, the tuned iparm) with the
    substructuring order supplied as the fill-reducing ordering (iparm(5) = 1, identity perm on the reordered matrix):
    cell-private blocks are eliminated first (dense supernodes = exact static condensation of each cell's private
    retained DOFs), then the shared interface. METIS is not run on the ~1e9-nonzero condensed graph. The 32-bit PARDISO
    interface is used when nnz < 2**31, the 64-bit one (pardiso_64, int64 arrays) otherwise. All six loads in one solve.
Memory guards (as E1): before each Schur and before the condensed factorisation, the predicted PARDISO memory (+ the dense
Schur output, + margin) must fit in memory.high - container anonymous memory; before the condensed assembly, the exact CSR
size (+ margin) must fit. Otherwise the run records 'skipped' with the predictions and stops.
Outputs: per-phase seconds (front end, Schur, lattice, structure, values, analysis, factor, solve), PARDISO memory (kB and
kB / 2**20 GiB) per cell and global, total dense S (GiB), CSR size (GiB), peak RSS per phase and process lifetime (GiB),
compliance per load, relative residual of the condensed system, relative compliance difference to the whole-lattice
direct solution (--ref JSON files, first that has 'compliance').
Usage: lat_cond_cpu.py <out.json> <layout.json> [--body /root/autodl-tmp/OPL/S4/body] [--iparm-file f] [--margin-gib 4]
       [--n-random 3] [--ref f1.json,f2.json] [--ilp64]
       lat_cond_cpu.py --selftest      (tiny check of the ordering / 32- and 64-bit paths)
Environment as lat_direct_cpu2.py (OPL_DEV=cpu, PYPARDISO_MKL_RT, MKL_NUM_THREADS = OMP_NUM_THREADS = 16)."""
import os, sys, json, time, argparse, gc, ctypes
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pardiso_direct as PD


class Pardiso64(PD.Pardiso):
    """pardiso_64 (ILP64 interface): int64 pt / iparm / ia / ja / perm and integer arguments."""

    def __init__(self, A, mtype, iparm, perm=None):
        self.mtype, self.n, self.msglvl = int(mtype), int(A.shape[0]), 0
        self.pt = np.zeros(64, np.int64)
        self.iparm = np.zeros(64, np.int64)
        self.defaults = False
        self.iparm[0] = 1
        for k, v in iparm.items():
            self.iparm[int(k) - 1] = int(v)
        self.a = np.ascontiguousarray(A.data, dtype=np.float64)
        off = 0 if self.iparm[34] == 1 else 1
        self.ia = np.ascontiguousarray(A.indptr, dtype=np.int64) + off if off else np.ascontiguousarray(A.indptr, dtype=np.int64)
        self.ja = np.ascontiguousarray(A.indices, dtype=np.int64) + off if off else np.ascontiguousarray(A.indices, dtype=np.int64)
        self.perm = np.zeros(max(self.n, 1), np.int64) if perm is None else np.ascontiguousarray(perm, dtype=np.int64)
        self._one = np.zeros(1)
        self.released = False
        L = PD.lib()
        self.fn = L.pardiso_64
        self.fn.argtypes = [ctypes.c_void_p] * 16
        self.fn.restype = None

    def _call(self, phase, b, x, nrhs):
        err = ctypes.c_int64(0)
        i64 = lambda v: ctypes.byref(ctypes.c_int64(int(v)))
        self.fn(self.pt.ctypes.data, i64(1), i64(1), i64(self.mtype), i64(phase), i64(self.n), self.a.ctypes.data,
                self.ia.ctypes.data, self.ja.ctypes.data, self.perm.ctypes.data, i64(nrhs), self.iparm.ctypes.data,
                i64(0), b.ctypes.data, x.ctypes.data, ctypes.byref(err))
        if err.value != 0:
            raise PD.PardisoError(f'PARDISO_64 phase {phase} error {err.value}')


def handle(A, iparm, ilp64, perm=None):
    return Pardiso64(A, 2, iparm, perm=perm) if ilp64 else PD.Pardiso(A, 2, iparm, perm=perm)


# ------------------------------------------------------------------ cell stage
def cell_schur(C, iparm, margin, log):
    """Dense S of the retained DOFs (C.P order) by the Schur option; returns (S, record) or (None, record) if skipped."""
    t = time.perf_counter()
    ru, cu, v = C.ru.long().numpy(), C.cu.long().numpy(), C.vals.numpy()
    s = 1 / np.sqrt(v[C.diag.numpy()])
    U = sp.csr_matrix((v * s[ru] * s[cu], (ru, cu)), shape=(C.nb, C.nb)); U.sum_duplicates(); U.sort_indices()
    del ru, cu
    P_ = C.P.numpy(); ns = len(P_)
    perm = np.zeros(C.nb, np.int32); perm[P_] = 1
    r = dict(schur_rows=ns, build_s=time.perf_counter() - t)
    ip = dict(iparm); ip[36] = 1; ip[5] = 0
    H = PD.Pardiso(U, 2, ip, perm=perm)
    PD.reset_peak()
    r['analysis_s'] = H.phase(11)
    r['predicted_GiB'] = H.mem_GiB()
    need = r['predicted_GiB']['total'] + 8 * ns * ns / PD.GIB + margin
    r['avail_GiB'] = av = PD.avail_gib()
    if need > av:
        H.release()
        r['skipped'] = f'Schur: predicted {need - margin:.2f} GiB + margin {margin} > available {av:.2f} GiB'
        return None, r
    x = np.zeros(max(C.nb, ns * ns))
    r['factor_schur_s'] = H.phase(22, x=x)
    r['seconds'] = r['analysis_s'] + r['factor_schur_s']
    r['pardiso'] = H.info(); r['peak_rss_GiB'] = PD.peak_rss_gib()
    H.release()
    del U, H
    S = x[:ns * ns].reshape(ns, ns)
    sP = s[P_]
    S /= sP[:, None]; S /= sP[None, :]
    r['symmetry_rel'] = float(np.abs(S[:, :64] - S[:64, :].T).max() / np.abs(S[:, :64]).max())
    r['dense_S_GiB'] = 8 * ns * ns / PD.GIB
    return S, r


# ------------------------------------------------------------------ condensed assembly
def structure(gl, nfree):
    """gl[i]: free global index per local retained DOF of cell i (-1 = clamped). Returns the substructuring order
    (new index per free DOF), per-cell sorted new indices + local positions, per-row column 'source', and indptr."""
    mult = np.zeros(nfree, np.int64)
    for g in gl:
        mult[g[g >= 0]] += 1
    owner = np.full(nfree, -1, np.int64)
    for i, g in enumerate(gl):
        gg = g[g >= 0]
        owner[gg[mult[gg] == 1]] = i
    new = np.full(nfree, -1, np.int64); k = 0
    for i, g in enumerate(gl):                                                     # private DOFs, cell by cell
        gg = g[g >= 0]; pv = np.unique(gg[mult[gg] == 1])
        new[pv] = np.arange(k, k + len(pv)); k += len(pv)
    sh = np.flatnonzero(mult > 1)
    new[sh] = np.arange(k, k + len(sh)); n_priv = k
    assert (new >= 0).all()
    cells = []
    for g in gl:
        loc = np.flatnonzero(g >= 0)
        nn = new[g[loc]]
        o = np.argsort(nn, kind='stable')
        cells.append(dict(loc=loc[o], N=nn[o]))                                   # N sorted new indices, loc local rows
    # shared rows: column set = union of the N of the cells containing the row, grouped by cell signature
    memb = [[] for _ in range(len(sh))]
    for i, c in enumerate(cells):
        sN = c['N'][c['N'] >= n_priv]
        for r_ in sN:
            memb[r_ - n_priv].append(i)
    sig = {}
    for j, m in enumerate(memb):
        sig.setdefault(tuple(m), []).append(n_priv + j)
    U = {t: np.unique(np.concatenate([cells[i]['N'] for i in t])) for t in sig}
    lens = np.zeros(nfree, np.int64)
    for i, c in enumerate(cells):
        N = c['N']; npv = int((N < n_priv).sum())
        lens[N[:npv]] = len(N) - np.arange(npv)                                     # private row a: columns N[a:]
    rowsig = {}
    for t, rows in sig.items():
        rows = np.asarray(rows); u = U[t]
        lens[rows] = len(u) - np.searchsorted(u, rows)
        for r_ in rows:
            rowsig[int(r_)] = t
    indptr = np.zeros(nfree + 1, np.int64); np.cumsum(lens, out=indptr[1:])
    return dict(new=new, n_priv=n_priv, n_shared=len(sh), cells=cells, U=U, rowsig=rowsig, indptr=indptr,
                signatures={','.join(map(str, t)): len(v) for t, v in sig.items()})


def fill_indices(st, idx_dtype):
    ip = st['indptr']; nnz = int(ip[-1])
    ind = np.empty(nnz, idx_dtype)
    for c in st['cells']:
        N = c['N']; npv = int((N < st['n_priv']).sum())
        for a in range(npv):
            r_ = N[a]; ind[ip[r_]:ip[r_ + 1]] = N[a:]
    for r_, t in st['rowsig'].items():
        u = st['U'][t]; ind[ip[r_]:ip[r_ + 1]] = u[np.searchsorted(u, r_):]
    return ind


def add_cell(st, i, S, data, ind):
    c = st['cells'][i]; N, loc = c['N'], c['loc']; ip = st['indptr']; n_priv = st['n_priv']
    for a in range(len(N)):
        r_ = N[a]
        vals = S[loc[a], loc[a:]]
        if r_ < n_priv:                                                            # private row: columns are exactly N[a:]
            data[ip[r_]:ip[r_] + len(vals)] += vals
        else:
            row = ind[ip[r_]:ip[r_ + 1]]
            data[ip[r_] + np.searchsorted(row, N[a:])] += vals


# ------------------------------------------------------------------ main
def reference(files, nl):
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
            for r in d.get('runs', []):
                if 'compliance' in r and len(r['compliance']) == nl:
                    return f, r['mtype'], r['compliance']
        except Exception:                                                           # noqa: BLE001
            continue
    return None, None, None


def main(argv):
    import diag_sens as DS                                               # noqa: F401  CPU environment (first torch user)
    import torch
    import trainlib as TL
    import teacher as TE
    import box_encode as BX
    import lat_multi as LM
    BX.dev = TL.dev
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layout')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body'); ap.add_argument('--iparm-file', default=None)
    ap.add_argument('--margin-gib', type=float, default=4.0); ap.add_argument('--n-random', type=int, default=3)
    ap.add_argument('--ref', default=''); ap.add_argument('--ilp64', action='store_true')
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    iparm = PD.tuned_iparm()
    if a.iparm_file and Path(a.iparm_file).exists():
        iparm = {int(k): int(v) for k, v in json.loads(Path(a.iparm_file).read_text())['iparm'].items()}
    L = json.loads(Path(a.layout).read_text())
    rec = dict(layout=L['name'], route='conventional exact condensation (dense S_i by PARDISO Schur option; sparse Cholesky '
               'of the assembled condensed matrix, substructuring order)', env=PD.env_record(),
               iparm={str(k): v for k, v in iparm.items()}, loadavg_start=PD.loadavg(), cpu_stat_start=PD.cpu_stat(), cells={})
    save = lambda: Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    t_all = time.perf_counter()
    S_of, lay = {}, {}
    # 1. cells
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
        S, sr = cell_schur(C, iparm, a.margin_gib, log)
        r['schur'] = sr
        rec['cells'][c['case']] = r
        C._free(); C = G_ = None; gc.collect()
        log(dict(event='CELL', case=c['case'], **{k: v for k, v in r.items() if k != 'schur'},
                 schur_s=sr.get('seconds'), skipped=sr.get('skipped')))
        if S is None:
            rec['skipped'] = f"cell {c['case']}: {sr['skipped']}"
            rec['dense_S_total_GiB_predicted'] = sum(8 * float(x['retained']) ** 2 for x in rec['cells'].values()) / PD.GIB
            save(); return
        S_of[c['case']] = S
        rec['held_S_GiB'] = sum(8 * float(s_.shape[0]) ** 2 for s_ in S_of.values()) / PD.GIB
        save()
    rec['cells_s'] = time.perf_counter() - t_all
    rec['dense_S_total_GiB'] = rec['held_S_GiB']
    rec['rss_after_cells_GiB'] = PD.rss_gib()
    # 2. lattice
    t = time.perf_counter()
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random,
                          device='cpu', log=lambda s_: None)
    rec['lattice_s'] = time.perf_counter() - t
    F = lat.F.cpu().numpy().astype(np.float64); nfree = int(lat.nfree)
    gl = [lat.fmap[lat.idx[i]] for i in range(len(lat.geoms))]
    cases = [G.case for G in lat.geoms]
    rec.update(free_retained=nfree, loads=lat.labels)
    # 3. condensed assembly
    t = time.perf_counter(); st = structure(gl, nfree); rec['structure_s'] = time.perf_counter() - t
    nnz = int(st['indptr'][-1])
    ilp64 = a.ilp64 or nnz >= 2 ** 31 - 1
    isz = 8 if ilp64 else 4
    rec.update(nnz_upper=nnz, n_private=int(st['n_priv']), n_shared=int(st['n_shared']), signatures=st['signatures'],
               interface='pardiso_64 (int64)' if ilp64 else 'pardiso (int32)',
               csr_GiB=(nnz * (8 + isz) + (nfree + 1) * 8) / PD.GIB)
    av = PD.avail_gib(); rec['avail_before_assembly_GiB'] = av
    if rec['csr_GiB'] + a.margin_gib > av:
        rec['skipped'] = f"condensed CSR {rec['csr_GiB']:.2f} GiB + margin > available {av:.2f} GiB"
        save(); return
    PD.reset_peak()
    t = time.perf_counter(); ind = fill_indices(st, np.int64 if ilp64 else np.int32); rec['indices_s'] = time.perf_counter() - t
    data = np.zeros(nnz)
    t = time.perf_counter()
    for i, cs in enumerate(cases):
        add_cell(st, i, S_of.pop(cs), data, ind); gc.collect()
    rec['values_s'] = time.perf_counter() - t
    rec['assembly_peak_rss_GiB'] = PD.peak_rss_gib()
    ip = st['indptr'] if ilp64 else st['indptr'].astype(np.int32)
    A = sp.csr_matrix((data, ind, ip), shape=(nfree, nfree))
    newF = np.zeros_like(F); newF[st['new']] = F
    save()
    # 4. condensed sparse Cholesky, substructuring order (identity perm on the reordered matrix)
    ipc = dict(iparm); ipc[5] = 1; ipc[36] = 0
    perm = np.arange(nfree)
    H = handle(A, ipc, ilp64, perm=perm)
    PD.reset_peak()
    rec['analysis_s'] = H.phase(11)
    rec['predicted_GiB'] = H.mem_GiB()
    av = PD.avail_gib(); rec['avail_before_factor_GiB'] = av
    if rec['predicted_GiB']['total'] + a.margin_gib > av:
        rec['skipped'] = f"condensed factor: predicted {rec['predicted_GiB']['total']:.2f} GiB + margin > available {av:.2f} GiB"
        H.release(); save(); return
    rec['factor_s'] = H.phase(22)
    b = np.asfortranarray(newF)
    t = time.perf_counter(); X = H.solve(b); rec['solve_s'] = time.perf_counter() - t
    rec['pardiso'] = H.info(); rec['factor_solve_peak_rss_GiB'] = PD.peak_rss_gib()
    H.release()
    res = PD.sym_upper_matvec(A, X) - b
    rec['rel_residual'] = [float(np.linalg.norm(res[:, j]) / np.linalg.norm(b[:, j])) for j in range(b.shape[1])]
    rec['compliance'] = [float(newF[:, j] @ X[:, j]) for j in range(b.shape[1])]
    rf, mt, rc = reference([x for x in a.ref.split(',') if x], b.shape[1])
    if rc is not None:
        rec['reference'] = dict(file=rf, mtype=mt, compliance=rc,
                                rel_diff=[abs(x / y - 1) for x, y in zip(rec['compliance'], rc)])
    rec['condensed_s'] = rec['structure_s'] + rec['indices_s'] + rec['values_s'] + rec['analysis_s'] + rec['factor_s'] + rec['solve_s']
    rec['front_end_s'] = sum(c['setup_s'] + c['assembly_s'] for c in rec['cells'].values())
    rec['schur_s'] = sum(c['schur']['seconds'] for c in rec['cells'].values())
    rec['total_s'] = time.perf_counter() - t_all
    rec['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
    rec['loadavg_end'] = PD.loadavg(); cs_ = PD.cpu_stat()
    rec['throttled_s'] = (cs_.get('throttled_usec', 0) - rec['cpu_stat_start'].get('throttled_usec', 0)) / 1e6
    save()
    log(dict(event='DONE', **{k: rec.get(k) for k in ('front_end_s', 'schur_s', 'condensed_s', 'total_s', 'compliance',
                                                     'rel_residual', 'reference', 'process_peak_rss_GiB')}))


def selftest():
    """Random SPD 'cells' glued on a chain: condensed assembly + both interfaces against a dense solve."""
    rng = np.random.default_rng(0)
    nfree = 60; gl = []
    for lo, hi, ncl in ((0, 30, 0), (20, 50, 0), (40, 60, 10)):
        g = np.concatenate([np.arange(lo, hi), -np.ones(ncl, np.int64)]).astype(np.int64)
        rng.shuffle(g); gl.append(g)
    Ss = []
    for g in gl:
        m = len(g); M = rng.standard_normal((m, m)); Ss.append(M @ M.T + m * np.eye(m))
    K = np.zeros((nfree, nfree))
    for g, S in zip(gl, Ss):
        v = np.flatnonzero(g >= 0); K[np.ix_(g[v], g[v])] += S[np.ix_(v, v)]
    used = np.flatnonzero(np.diag(K) > 0); assert len(used) == nfree, 'selftest coverage'
    st = structure(gl, nfree)
    out = {}
    for ilp64 in (False, True):
        ind = fill_indices(st, np.int64 if ilp64 else np.int32); data = np.zeros(int(st['indptr'][-1]))
        for i, S in enumerate(Ss):
            add_cell(st, i, S, data, ind)
        A = sp.csr_matrix((data, ind, st['indptr'] if ilp64 else st['indptr'].astype(np.int32)), shape=(nfree, nfree))
        Kn = np.zeros_like(K); Kn[np.ix_(st['new'], st['new'])] = K
        e_asm = float(np.abs(np.triu(Kn) - A.toarray()).max() / np.abs(K).max())
        f = rng.standard_normal((nfree, 2)); fn = np.zeros_like(f); fn[st['new']] = f
        ip = PD.tuned_iparm(); ip[5] = 1
        H = handle(A, ip, ilp64, perm=np.arange(nfree)); H.phase(11); H.phase(22); X = H.solve(np.asfortranarray(fn)); H.release()
        e_sol = float(np.abs(X[st['new']] - np.linalg.solve(K, f)).max() / np.abs(np.linalg.solve(K, f)).max())
        out['ilp64' if ilp64 else 'lp64'] = dict(assembly_err=e_asm, solve_err=e_sol)
    print(json.dumps(out))


if __name__ == '__main__':
    if sys.argv[1:] == ['--selftest']:
        selftest()
    else:
        main(sys.argv[1:])
