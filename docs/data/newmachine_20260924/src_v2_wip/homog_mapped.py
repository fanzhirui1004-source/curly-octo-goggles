"""Direction A: periodic homogenisation of AFFINELY mapped uncut Schwarz-P cells vs cheap mappings of the reference tensor
(new script; runs on the GPU server; P1 bodies are only read).

For x = A X (constant A) an affinely mapped periodic lattice is periodic again, so its exact homogenised tensor follows from
a periodic cell problem on the mapped cell: K~ from mapped_cell.MappedCell (same discrete CutFEM model as P1), nodes on
opposite faces of the reference box identified (periodicity in reference grid = periodicity of the mapped lattice), macro
strain in PHYSICAL coordinates u_E = E x (x = mapped node positions), fluctuation periodic, one node fixed against
translation. For the six unit strains (Voigt xx, yy, zz, yz, xz, xy, engineering shear): C^H_ab = u_a^T K~ u_b / det A.
Compared with
  rot   C^H_ref rotated by the polar factor R of A (stretch ignored),
  push  full push-forward  C_ijkl = (1/det A) A_iI A_jJ A_kK A_lL C^H_ref,IJKL (the reference tensor carried by the map),
by the generalised eigenvalues of (C_approx, C_exact) on the 6 Voigt strains: lambda_max - 1 and 1 - lambda_min are the
largest over- and under-estimates of the strain energy (stiffness), i.e. the range of directional errors.
Checks: identity map reproduces P1's C^H (homog_cells.json), rot30 equals the rotated reference tensor.
Usage: homog_mapped.py <out.json> <body_dir> <cases (comma)> <maps.json> --maps id,rot30,... [--p1 homog_cells.json]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

import mapped_cell as MC
import teacher as TE

dev, dt = TE.dev, TE.dt
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt_to_tensor(Cv):
    C = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    C[p, q, r, s] = Cv[a][b]
    return C


def tensor_to_voigt(C):
    return np.array([[C[i, j, k, l] for (k, l) in VOIGT] for (i, j) in VOIGT])


def push(Cv, A):
    C = voigt_to_tensor(Cv)
    return tensor_to_voigt(np.einsum('iI,jJ,kK,lL,IJKL->ijkl', A, A, A, A, C) / np.linalg.det(A))


def rotate(Cv, R):
    C = voigt_to_tensor(Cv)
    return tensor_to_voigt(np.einsum('iI,jJ,kK,lL,IJKL->ijkl', R, R, R, R, C))


def compare(Ca, Ce):
    """Generalised eigenvalues of (Ca, Ce) in Voigt form with engineering shear (energy = e^T C e)."""
    L = np.linalg.cholesky(0.5 * (Ce + Ce.T))
    Li = np.linalg.inv(L)
    ev = np.linalg.eigvalsh(Li @ (0.5 * (Ca + Ca.T)) @ Li.T)
    return dict(over=float(ev.max() - 1), under=float(1 - ev.min()))


def periodic_CH(C):
    """Exact homogenised tensor (Voigt, engineering shear) of the mapped cell C (assembled MappedCell)."""
    n = C.n; M = 2 * n + 1
    g = np.stack(np.unravel_index(C.nodes, (M,) * 3), 1)
    rep = np.ravel_multi_index(tuple((g % (2 * n)).T), (M,) * 3)            # representative node: coordinates mod 2n
    ureps, ridx = np.unique(rep, return_inverse=True)                       # reduced node index per node
    present = np.isin(ureps, C.nodes)
    if not present.all():
        raise ValueError('PERIODIC: representative nodes missing (geometry not periodic?)')
    nr = len(ureps)
    rd = torch.as_tensor(3 * ridx[:, None] + np.arange(3)[None], device=dev).reshape(-1)  # reduced dof per cell dof
    ru, cu, v = C.ru.long(), C.cu.long(), C.vals.to(dt)
    a, b = rd[ru], rd[cu]
    # full symmetric reduced matrix (both triangles), then keep upper
    rr = torch.cat([a, b[ru != cu]]); cc = torch.cat([b, a[ru != cu]]); vv = torch.cat([v, v[ru != cu]])
    keep = rr <= cc
    rr, cc, vv = rr[keep], cc[keep], vv[keep]
    fixed = torch.arange(3, device=dev)                                     # reduced node 0 fixed (translation)
    ok = (rr >= 3) & (cc >= 3)
    rr, cc, vv = rr[ok] - 3, cc[ok] - 3, vv[ok]
    nd = 3 * nr - 3
    key = rr * nd + cc
    uk, inv = torch.unique(key, return_inverse=True)
    vals = torch.zeros(len(uk), dtype=dt, device=dev).index_add_(0, inv, vv)
    r2, c2 = uk // nd, uk % nd
    d = vals[r2 == c2]
    s = torch.zeros(nd, dtype=dt, device=dev); s[r2[r2 == c2]] = 1 / torch.sqrt(d)
    crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(r2, minlength=nd), 0)])
    sol = TE.SPDSolver(crow.int(), c2.int(), (vals * s[r2] * s[c2]).contiguous(), nd)
    x = torch.as_tensor(C.xyz, dtype=dt, device=dev); x = x - x.mean(0)
    UE = torch.zeros((C.nb, 6), dtype=dt, device=dev)
    for col, (i, j) in enumerate(VOIGT):                                    # u_i = E_ij x_j, engineering shear
        if i == j:
            UE[i::3, col] = x[:, i]
        else:
            UE[i::3, col] = 0.5 * x[:, j]; UE[j::3, col] = 0.5 * x[:, i]
    KU = C.K @ UE                                                           # (nb, 6)
    Rr = torch.zeros((3 * nr, 6), dtype=dt, device=dev).index_add_(0, rd, -KU)   # P^T (-K u_E)
    w = torch.zeros((3 * nr, 6), dtype=dt, device=dev)
    w[3:] = s[:, None] * sol.solve((s[:, None] * Rr[3:]).contiguous())
    U = UE + w[rd]
    CH = (U.T @ (C.K @ U)).cpu().numpy()
    sol.free()
    return 0.5 * (CH + CH.T), dict(reduced_dofs=nd, nodes=len(C.nodes))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('body'); ap.add_argument('cases'); ap.add_argument('mapsjson')
    ap.add_argument('--maps', default='id,rot30,strx2,strz0.5,strg2,shear0.3')
    ap.add_argument('--p1', default='')
    a = ap.parse_args(argv)
    specs = {}
    for f in a.mapsjson.split(','):
        specs.update({m['name']: m['spec'] for m in json.loads(Path(f).read_text())})
    p1 = {c['case']: c for c in json.loads(Path(a.p1).read_text())['cells']} if a.p1 else {}
    out = []
    for case in a.cases.split(','):
        ref = None
        for mname in a.maps.split(','):
            t0 = time.perf_counter()
            spec = specs[mname]
            A = np.eye(3) if spec['kind'] == 'identity' else np.asarray(spec['A'], float)
            C = MC.MappedCell(case, a.body, spec, log=lambda s_: None); C.assemble()
            CH, info = periodic_CH(C)
            CH = CH / np.linalg.det(A)
            rec = dict(case=case, map=mname, A=A.tolist(), CH=CH.tolist(), seconds=time.perf_counter() - t0, **info)
            if mname == 'id':
                ref = CH
                if case in p1:
                    Cp = np.asarray(p1[case]['CH'])
                    rec['vs_P1_rel'] = float(np.abs(CH - Cp).max() / np.abs(Cp).max())
            if ref is not None and mname != 'id':
                U_, S_, Vh = np.linalg.svd(A); R = U_ @ Vh
                rec['kappa'] = float(S_[0] / S_[-1])
                rec['rot'] = compare(rotate(ref, R), CH)
                rec['push'] = compare(push(ref, A), CH)
            out.append(rec)
            print(json.dumps({k: rec[k] for k in rec if k not in ('CH', 'A')}), flush=True)
            C._free(); del C; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
