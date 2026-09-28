"""Direct MKL PARDISO calls through ctypes for the revision host baselines (bench_cpu2.py, lat_direct_cpu2.py, ref_valid2.py).

Why not pypardiso.solve(): on EVERY call it (i) checks that A is the factorised matrix -- a SHA-1 hash of indices, indptr
and data when A.nnz > 5e7, a full array comparison against a stored copy otherwise -- and (ii) builds 1-based int32 copies
of indptr and indices. Pardiso below keeps one handle per matrix, passes the CSR arrays of A once with zero-based indexing
(iparm(35) = 1), so a solve (phase 33) is a single library call with no per-call work on A. With iparm=None it reproduces
pypardiso's configuration (iparm(1) = 0, all MKL defaults, one-based arrays built once).

iparm numbering below is 1-based as in the MKL documentation (iparm(k) = iparm[k-1]).
Memory reported by PARDISO: iparm(15) peak of the analysis phase, iparm(16) permanent memory from analysis on, iparm(17)
memory of numerical factorisation and solution; all in kB (1 kB = 1024 bytes), converted with kB / 2**20 = GiB.
Helpers: cgroup limits / usage of the container, process peak RSS with reset (VmHWM via /proc/self/clear_refs), MKL version.
"""
import os, ctypes, ctypes.util, time, json
import numpy as np
import scipy.sparse as sp

GIB = 2 ** 30
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        p = os.environ.get('PYPARDISO_MKL_RT') or ctypes.util.find_library('mkl_rt')
        _LIB = ctypes.CDLL(p)
        f = _LIB.pardiso
        f.argtypes = [ctypes.c_void_p] * 16
        f.restype = None
    return _LIB


def mkl_version():
    try:
        L = lib()
        buf = ctypes.create_string_buffer(256)
        L.mkl_get_version_string.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.mkl_get_version_string(buf, 256)
        return buf.value.decode(errors='replace').strip()
    except Exception as e:                                                          # noqa: BLE001
        return f'unavailable: {e!r}'[:200]


def mkl_max_threads():
    try:
        L = lib(); L.MKL_Get_Max_Threads.restype = ctypes.c_int
        return int(L.MKL_Get_Max_Threads())
    except Exception:                                                               # noqa: BLE001
        return None


# ------------------------------------------------------------------ container / process memory
def _read(p):
    try:
        return open(p).read().strip()
    except OSError:
        return None


def cgroup():
    """Container limits and usage (cgroup v2): memory.max / memory.high (bytes and GiB), anonymous memory in use, CPU quota."""
    st = {}
    for line in (_read('/sys/fs/cgroup/memory.stat') or '').splitlines():
        k, _, v = line.partition(' ')
        if k in ('anon', 'file', 'shmem'):
            st[k] = int(v)
    cpu = (_read('/sys/fs/cgroup/cpu.max') or 'max 100000').split()
    mx, hi = _read('/sys/fs/cgroup/memory.max'), _read('/sys/fs/cgroup/memory.high')
    toint = lambda s: None if s in (None, 'max') else int(s)
    r = dict(memory_max_bytes=toint(mx), memory_high_bytes=toint(hi), anon_bytes=st.get('anon'), file_bytes=st.get('file'),
             cpu_quota_cores=None if cpu[0] == 'max' else int(cpu[0]) / int(cpu[1]), nproc=len(os.sched_getaffinity(0)),
             os_cpu_count=os.cpu_count())
    for k in ('memory_max', 'memory_high', 'anon', 'file'):
        b = r.get(k + '_bytes')
        r[k + '_GiB'] = None if b is None else b / GIB
    return r


def cpu_stat():
    d = {}
    for line in (_read('/sys/fs/cgroup/cpu.stat') or '').splitlines():
        k, _, v = line.partition(' ')
        d[k] = int(v)
    return d


def avail_gib():
    """GiB the container can still allocate before memory.high (or memory.max) counting anonymous memory only (page cache
    is reclaimable)."""
    c = cgroup()
    lim = c['memory_high_bytes'] or c['memory_max_bytes']
    if lim is None or c['anon_bytes'] is None:
        return float('inf')
    return (lim - c['anon_bytes']) / GIB


_MAXHWM = [0.0]


def reset_peak():
    """Reset the process high-water mark (VmHWM; the value before the reset is kept for lifetime_peak_gib); returns False
    where the kernel does not allow it."""
    _MAXHWM[0] = max(_MAXHWM[0], status_gib('VmHWM'))
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


def status_gib(key):
    for line in open('/proc/self/status'):
        if line.startswith(key + ':'):
            return int(line.split()[1]) / 2 ** 20                                   # kB -> GiB
    return float('nan')


def peak_rss_gib():
    return status_gib('VmHWM')


def lifetime_peak_gib():
    return max(_MAXHWM[0], status_gib('VmHWM'))


def rss_gib():
    return status_gib('VmRSS')


def loadavg():
    return list(os.getloadavg())


# ------------------------------------------------------------------ settings
def tuned_iparm(ordering=3, two_level=1, par_solve=1, schur=0, check=0):
    """Explicit settings (1-based keys) for real SPD Cholesky (mtype 2), see the module docstring and the chain report."""
    return {1: 1,            # user-supplied values (no defaults filled in)
            2: ordering,     # fill-in reducing ordering: 2 = METIS nested dissection, 3 = parallel (OpenMP) METIS
            4: 0,            # no iterative (CGS) solve
            5: 0,            # no user permutation (perm is the Schur mask when iparm(36) != 0)
            6: 0,            # solution returned in x, b unchanged
            8: 0,            # no iterative refinement beyond the automatic default (none without perturbed pivots)
            10: 8,           # pivot perturbation 1e-8 (symmetric default; not used by SPD Cholesky)
            11: 0,           # no scaling (SPD; the matrices are already symmetrically Jacobi scaled)
            13: 0,           # no weighted matching (SPD)
            18: -1,          # report number of nonzeros in the factor (output in iparm(18))
            19: -1,          # report factorisation Mflop (output in iparm(19))
            24: two_level,   # parallel factorisation control: 0 = classic, 1 = two-level (better scaling at many threads)
            25: par_solve,   # parallel forward/backward solve: 0 = sequential, 1 = parallel, 2 = parallel (alternative)
            27: check,       # matrix checker (smoke tests only)
            28: 0,           # double precision
            34: 0,           # conditional numerical reproducibility off
            35: 1,           # zero-based ia / ja (no index copies)
            36: schur,       # Schur complement: 0 off, 1 = dense Schur complement returned in x (perm marks its rows)
            56: 0, 60: 0}    # no pivot callback; in-core


class PardisoError(RuntimeError):
    pass


class Pardiso:
    """One PARDISO handle for a CSR matrix A (float64, sorted indices; mtype 2: upper triangle incl. the diagonal, 11: full).
    iparm: dict {1-based index: value} (sets iparm(1) = 1 implicitly) or None for MKL defaults (pypardiso's setting)."""

    def __init__(self, A, mtype, iparm=None, msglvl=0, perm=None):
        if not sp.isspmatrix_csr(A) and not (sp.issparse(A) and A.format == 'csr'):
            raise TypeError('CSR expected')
        if not A.has_sorted_indices:
            A.sort_indices()
        self.mtype, self.n, self.msglvl = int(mtype), int(A.shape[0]), int(msglvl)
        self.pt = np.zeros(64, np.int64)
        self.iparm = np.zeros(64, np.int32)
        self.defaults = iparm is None
        if iparm is not None:
            self.iparm[0] = 1
            for k, v in iparm.items():
                self.iparm[int(k) - 1] = int(v)
        self.a = np.ascontiguousarray(A.data, dtype=np.float64)
        if self.defaults or self.iparm[34] == 0:                                        # one-based: copies built ONCE
            self.ia = A.indptr.astype(np.int32) + 1
            self.ja = A.indices.astype(np.int32) + 1
        else:
            self.ia = np.ascontiguousarray(A.indptr, dtype=np.int32)                    # zero-based, no copy if int32
            self.ja = np.ascontiguousarray(A.indices, dtype=np.int32)
        self.perm = np.zeros(max(self.n, 1), np.int32) if perm is None else np.ascontiguousarray(perm, dtype=np.int32)
        self._one = np.zeros(1)
        self.released = False

    def _call(self, phase, b, x, nrhs):
        err = ctypes.c_int32(0)
        i32 = lambda v: ctypes.byref(ctypes.c_int32(int(v)))
        lib().pardiso(self.pt.ctypes.data, i32(1), i32(1), i32(self.mtype), i32(phase), i32(self.n),
                      self.a.ctypes.data, self.ia.ctypes.data, self.ja.ctypes.data, self.perm.ctypes.data, i32(nrhs),
                      self.iparm.ctypes.data, i32(self.msglvl), b.ctypes.data, x.ctypes.data, ctypes.byref(err))
        if err.value != 0:
            raise PardisoError(f'PARDISO phase {phase} error {err.value}')

    def phase(self, ph, x=None):
        """Phases without right-hand sides (11 analysis, 22 factorisation, 12 both); x receives the Schur complement."""
        t = time.perf_counter()
        xx = self._one if x is None else x
        self._call(ph, self._one, xx, 1)
        return time.perf_counter() - t

    def solve(self, b, x=None):
        """Phase 33 for b (n,) or (n, k); b is used column-major (a Fortran copy is made only if needed)."""
        b = np.asarray(b, dtype=np.float64)
        if b.ndim == 2 and b.shape[1] > 1 and not b.flags.f_contiguous:
            b = np.asfortranarray(b)
        nrhs = 1 if b.ndim == 1 else b.shape[1]
        if x is None:
            x = np.empty_like(b, order='F')
        self._call(33, b, x, nrhs)
        return x

    def mem_kB(self):
        return dict(peak_analysis=int(self.iparm[14]), permanent=int(self.iparm[15]), factor_solve=int(self.iparm[16]))

    def mem_GiB(self):
        m = self.mem_kB()
        return dict(peak_analysis=m['peak_analysis'] / 2 ** 20, permanent=m['permanent'] / 2 ** 20,
                    factor_solve=m['factor_solve'] / 2 ** 20, total=(m['permanent'] + m['factor_solve']) / 2 ** 20)

    def info(self):
        ip = self.iparm
        return dict(iparm_1based={str(k + 1): int(v) for k, v in enumerate(ip) if v != 0}, nnz_factor=int(ip[17]),
                    mflop_factor=int(ip[18]), perturbed_pivots=int(ip[13]), peak_mem_kB=self.mem_kB(),
                    peak_mem_GiB=self.mem_GiB())

    def release(self):
        if not self.released:
            try:
                self._call(-1, self._one, self._one, 1)
            finally:
                self.released = True

    def __del__(self):
        try:
            self.release()
        except Exception:                                                           # noqa: BLE001
            pass


def sym_upper_matvec(U, x):
    """Symmetric product from an upper-triangle CSR (incl. diagonal)."""
    return U @ x + U.T @ x - U.diagonal()[:, None] * x if x.ndim == 2 else U @ x + U.T @ x - U.diagonal() * x


def env_record():
    import platform
    r = dict(mkl=mkl_version(), mkl_max_threads=mkl_max_threads(), MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'),
             OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'), mkl_rt=os.environ.get('PYPARDISO_MKL_RT'),
             python=platform.python_version(), numpy=np.__version__, cgroup=cgroup(), loadavg=loadavg(),
             host=platform.node())
    try:
        import scipy; r['scipy'] = scipy.__version__
        import torch; r['torch'] = torch.__version__; r['torch_threads'] = torch.get_num_threads()
    except Exception:                                                               # noqa: BLE001
        pass
    return r


if __name__ == '__main__':                                                          # tiny self-test: 3D Laplacian
    import sys
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    L1 = sp.diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1])
    I1 = sp.identity(m)
    K = (sp.kron(sp.kron(L1, I1), I1) + sp.kron(sp.kron(I1, L1), I1) + sp.kron(sp.kron(I1, I1), L1)).tocsr()
    U = sp.triu(K).tocsr(); U.sort_indices()
    b = np.random.default_rng(0).standard_normal((K.shape[0], 3))
    out = dict(env=env_record(), n=K.shape[0])
    for name, (A, mt, ip) in dict(chol=(U, 2, tuned_iparm(check=1)), lu_default=(K, 11, None)).items():
        P = Pardiso(A, mt, ip)
        ta = P.phase(11); tf = P.phase(22); x = P.solve(b)
        res = np.linalg.norm(K @ x - b) / np.linalg.norm(b)
        out[name] = dict(analysis_s=ta, factor_s=tf, rel_res=float(res), **P.info())
        P.release()
    # Schur complement of the last 50 unknowns vs dense reference
    ns = 50; perm = np.zeros(K.shape[0], np.int32); perm[-ns:] = 1
    P = Pardiso(U, 2, tuned_iparm(schur=1), perm=perm)
    xs = np.zeros(max(K.shape[0], ns * ns))
    P.phase(12, x=xs)
    S = xs[:ns * ns].reshape(ns, ns)
    Kd = K.toarray(); ii = np.arange(K.shape[0] - ns)
    Sref = Kd[-ns:, -ns:] - Kd[-ns:, ii] @ np.linalg.solve(Kd[np.ix_(ii, ii)], Kd[ii, -ns:])
    out['schur'] = dict(rel_err=float(np.linalg.norm(S - Sref) / np.linalg.norm(Sref)),
                        rel_err_symmetrised_lower=float(np.linalg.norm(np.tril(S) + np.tril(S, -1).T - Sref) / np.linalg.norm(Sref)),
                        rel_err_symmetrised_upper=float(np.linalg.norm(np.triu(S) + np.triu(S, 1).T - Sref) / np.linalg.norm(Sref)))
    P.release()
    print(json.dumps(out, indent=1, default=str))
