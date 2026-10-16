"""Section 6.11 homogenisation comparison, step 1: effective elasticity tensor C^H(tau) of the uncut Schwarz-P sheet cell
with UNIFORM corner thickness, by periodic homogenisation on the same discrete CutFEM model (n = 32, Q2, ghost penalty)
that defines the reference and the design problem.  New script.

For each tau: a FULL packet with eight equal corners (template: a FULL cell packet of the plate layout), its body by
fast_prep4 in the frozen CPU environment, the assembled stiffness K (teacher.Cell, upper CSR incl. ghost penalty) moved to
the host; nodes on opposite faces of the unit box are identified (the geometry is periodic for uniform tau: asserted), the
fine displacement is u = E x + P w with periodic fluctuation w (one node fixed against translation), and for the six
unit macro strains (Voigt order xx, yy, zz, yz, xz, xy, engineering shear) the fluctuations solve P^T K P w = -P^T K u_E by
MKL PARDISO (host).  C^H_ij = u_i^T K u_j over the unit cell volume 1; the material volume fraction rho(tau) is the sum of
the zeroth element moments.  The cubic symmetry of the Schwarz-P cell is reported (C11 ~ C22 ~ C33, C12 ~ C13 ~ C23,
C44 ~ C55 ~ C66, off-diagonal shear coupling ~ 0).
Output: <out>/homog_cells.json: per tau: rho, C^H (6 x 6), symmetry deviations, DOFs, seconds.
Usage: homog_cell.py <out_dir> --template <packet dir of a FULL cell> [--taus 0.17,...,0.70] [--workers 8]
"""
import json, time, os, sys, argparse, subprocess, shutil
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--template', required=True)
ap.add_argument('--taus', default='0.17,0.22,0.27,0.32,0.37,0.42,0.47,0.52,0.57,0.62,0.67,0.70')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--frozen', default='/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/run_frozen_python.sh')
ap.add_argument('--frozen-cwd', default='/root/autodl-tmp/CUTFEM_DEPENDENCIES_20260924/root/autodl-tmp/CLAUDE_TAKEOVER_20260923/COVER_G/src')
ap.add_argument('--templates', default='/root/autodl-tmp/OPL/S4/body/GP_TEMPLATES_n32.npz')
A = ap.parse_args()
OUT = Path(A.out); PK, BODY = OUT / 'packets', OUT / 'body'
for d in (PK, BODY):
    d.mkdir(parents=True, exist_ok=True)
os.environ['OPL_PACKETS_EXTRA'] = ':'.join([str(PK)] + [p for p in os.environ.get('OPL_PACKETS_EXTRA', '').split(':') if p])

import torch                                                            # noqa: E402
import scipy.sparse as sp                                               # noqa: E402
import models as MD                                                     # noqa: E402,F401
import teacher as TE                                                    # noqa: E402

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def packets(taus):
    tpl = Path(A.template)
    cases = []
    for t in taus:
        case = f'homogP_t{int(round(t * 1e4)):05d}'
        d = PK / case
        if not (d / 'FRESH_CONTEXT.json').exists():
            cx = json.loads((tpl / 'FRESH_CONTEXT.json').read_text())
            cs = cx['case']
            if cs.get('kind') != 'FULL':
                raise ValueError('TEMPLATE_MUST_BE_FULL')
            cs['tau_corners'] = [format(t, '.12f')] * 8
            cs['case_id'] = case
            cx.setdefault('provenance', {})['homog_cell'] = dict(tau=t, template=str(tpl))
            d.mkdir(parents=True, exist_ok=True)
            (d / 'FRESH_CONTEXT.json').write_text(json.dumps(cx, indent=1))
            shutil.copy(tpl / 'SAMPLE.json', d / 'SAMPLE.json')
        cases.append(case)
    return cases


def bodies(cases):
    if not (BODY / Path(A.templates).name).exists():
        shutil.copy(A.templates, BODY / Path(A.templates).name)
    todo = [c for c in cases if not (BODY / c / 'PREP.json').exists()]
    if todo:
        src = Path(__file__).resolve().parent / 'fast_prep4.py'
        one = OUT / 'body_one.sh'
        one.write_text('#!/bin/bash\n[ -f $1/$2/PREP.json ] && exit 0\n'
                       f'cd {A.frozen_cwd} && OMP_NUM_THREADS=1 timeout 1800 {A.frozen} {src} 1 $1 $2 > $1/$2.log 2>&1 || touch $1/$2.failed\n')
        one.chmod(0o755)
        (OUT / 'todo.txt').write_text('\n'.join(todo) + '\n')
        subprocess.run(['bash', '-c', f'xargs -a {OUT}/todo.txt -P {A.workers} -I{{}} {one} {BODY} {{}}'], check=False)
    bad = [c for c in cases if not (BODY / c / 'PREP.json').exists()]
    if bad:
        raise RuntimeError(f'BODY_FAIL {bad}')


def homogenise(case):
    t0 = time.perf_counter()
    C = TE.Cell(case, str(BODY), log=lambda s_: None); C.assemble()
    rho = float(C.M[:, 0].sum())
    nb, n = C.nb, C.n
    ru, cu, v = C.ru.long().cpu().numpy(), C.cu.long().cpu().numpy(), C.vals.cpu().numpy()
    U = sp.csr_matrix((v, (ru, cu)), shape=(nb, nb))
    K = (U + sp.triu(U, 1).T).tocsr()
    nodes = np.asarray(C.nodes)
    g = np.stack(np.unravel_index(nodes, (2 * n + 1,) * 3), 1)             # grid coordinates 0..2n
    x = g / (2 * n)                                                         # unit-cell coordinates
    # periodic identification: representative grid coordinate g mod 2n
    rep = g % (2 * n)
    key = (rep[:, 0] * (2 * n + 1) + rep[:, 1]) * (2 * n + 1) + rep[:, 2]
    ukeys, inv = np.unique(key, return_inverse=True)
    # every image must exist: a node on a face needs its partner (periodic geometry for uniform tau)
    face = (g == 0) | (g == 2 * n)
    cnt = np.bincount(inv)
    expected = 2 ** face.sum(1)
    if not np.all(cnt[inv] == expected):
        raise ValueError(f'NON_PERIODIC_NODE_SET {case}: {int((cnt[inv] != expected).sum())} nodes')
    nr = len(ukeys)
    rows = np.arange(nb); cols = 3 * inv.repeat(3) + np.tile(np.arange(3), len(nodes))
    Pm = sp.csr_matrix((np.ones(nb), (rows, cols)), shape=(nb, 3 * nr))
    Kr = (Pm.T @ K @ Pm).tocsr()
    fix = np.arange(3)                                                      # reduced node 0 fixed (translation)
    keep = np.setdiff1d(np.arange(3 * nr), fix)
    Kk = Kr[keep][:, keep].tocsr()
    UE = np.zeros((nb, 6))
    for j, (a, b) in enumerate(VOIGT):
        E = np.zeros((3, 3))
        if a == b:
            E[a, a] = 1.0
        else:
            E[a, b] = E[b, a] = 0.5                                         # engineering shear strain 1
        UE[:, j] = (x @ E.T).reshape(-1)
    rhs = -(Pm.T @ (K @ UE))[keep]
    import pypardiso
    ps = pypardiso.PyPardisoSolver(mtype=11)
    W = np.zeros((3 * nr, 6))
    W[keep] = ps.solve(Kk, np.ascontiguousarray(rhs))
    ps.free_memory(everything=True)
    Ufull = UE + Pm @ W
    CH = Ufull.T @ (K @ Ufull)
    CH = 0.5 * (CH + CH.T)
    res = np.abs((Pm.T @ (K @ Ufull))[keep]).max() / np.abs(Pm.T @ (K @ UE)).max()
    d = np.diag(CH)
    sym = dict(C11_spread=float(np.ptp(d[:3]) / d[:3].mean()),
               C12_spread=float(np.ptp([CH[0, 1], CH[0, 2], CH[1, 2]]) / np.mean([CH[0, 1], CH[0, 2], CH[1, 2]])),
               C44_spread=float(np.ptp(d[3:]) / d[3:].mean()),
               coupling_max=float(max(np.abs(CH[:3, 3:]).max(), np.abs(CH[3, 4]), np.abs(CH[3, 5]), np.abs(CH[4, 5])) / d[:3].mean()))
    out = dict(case=case, tau=float(C.taus0[0]), rho=rho, dofs=int(nb), reduced_nodes=int(nr), CH=CH.tolist(), symmetry=sym,
               equilibrium_residual=float(res), seconds=time.perf_counter() - t0)
    C._free(); del C, K, U, Kr, Kk
    torch.cuda.empty_cache()
    return out


def main():
    taus = [float(t) for t in A.taus.split(',') if t]
    cases = packets(taus)
    t = time.perf_counter(); bodies(cases); tb = time.perf_counter() - t
    rows = []
    for c in cases:
        r = homogenise(c)
        rows.append(r)
        print(json.dumps(dict(event='CELL', **{k: v for k, v in r.items() if k != 'CH'},
                              C11=r['CH'][0][0], C12=r['CH'][0][1], C44=r['CH'][3][3])), flush=True)
        (OUT / 'homog_cells.json').write_text(json.dumps(dict(template=A.template, bodies_s=tb, cells=rows), indent=1))


if __name__ == '__main__':
    main()
