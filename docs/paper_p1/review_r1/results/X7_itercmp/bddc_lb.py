"""Lower bound of the cost of BDDC with exact local solvers on a lattice export of lat_dump.py (one subdomain per cell).
BDDC factorises, for every cell, its Dirichlet problem (the block of the cell's DOFs not shared with other cells) and its
Neumann problem (the whole local matrix, made nonsingular by the primal constraints); both factorisations must be redone
whenever the cell's geometry changes, i.e. in every design iteration. Here each is factorised once, one cell after the
other, with the configuration of the whole-lattice direct solution of Table 5 (pardiso_direct.Pardiso: symmetric Jacobi
scaling, real SPD Cholesky mtype 2, tuned iparm; analysis (phase 11) and numerical factorisation (phase 22) timed).
The Neumann matrix of a floating cell is singular; it is factorised with a diagonal shift of 1e-10 (scaled matrix, unit
diagonal), which leaves the factorisation work unchanged. The constrained Neumann solve, the coarse problem and the PCG
iterations of BDDC are not included, so the sum is a lower bound.
Usage: PYPARDISO_MKL_RT=.../libmkl_rt.so.2 MKL_NUM_THREADS=16 python bddc_lb.py <sysdir> <out.json>"""
import sys, json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pardiso_direct as PD

S = Path(sys.argv[1]); out = Path(sys.argv[2])
meta = json.loads((S / 'meta.json').read_text())
nfree = meta['free_retained']
rec = dict(layout=meta['layout'], env=PD.env_record(), iparm=PD.tuned_iparm(), cells=[])


def factor(A, shift=0.0):
    """Jacobi-scaled SPD Cholesky of A (full CSR) with the Table 5 configuration; returns seconds and factor nnz."""
    d = A.diagonal(); s = 1 / np.sqrt(d)
    U = sp.triu(sp.diags(s) @ A @ sp.diags(s) + (sp.diags(np.full(A.shape[0], shift)) if shift else 0), format='csr')
    U.sort_indices()
    P = PD.Pardiso(U, 2, PD.tuned_iparm())
    ta = P.phase(11); tf = P.phase(22)
    info = P.info(); P.release()
    return ta, tf, info


tot = 0.0
for i in range(len(meta['cells'])):
    C = np.load(S / f'cell{i}.npz')
    l2g = C['l2g']; nl = len(l2g)
    A = sp.csr_matrix((C['data'], C['indices'], C['indptr']), shape=(nl, nl))
    inner = np.nonzero(l2g >= nfree)[0]                   # cell DOFs not shared with any other cell
    aD, fD, iD = factor(A[inner][:, inner].tocsr())
    aN, fN, iN = factor(A, shift=1e-10)
    r = dict(cell=i, dofs=nl, dirichlet_dofs=len(inner), dirichlet_analysis_s=aD, dirichlet_factor_s=fD,
             neumann_analysis_s=aN, neumann_factor_s=fN, dirichlet_info=iD, neumann_info=iN)
    rec['cells'].append(r); tot += aD + fD + aN + fN
    print(json.dumps(dict(cell=i, dofs=nl, dirichlet=len(inner), D=aD + fD, N=aN + fN)), flush=True)
rec['sum_s'] = tot
rec['process_peak_rss_GiB'] = PD.lifetime_peak_gib()
out.write_text(json.dumps(rec, indent=1, default=float))
print('SUM', tot, 'peak_rss_GiB', rec['process_peak_rss_GiB'], flush=True)
