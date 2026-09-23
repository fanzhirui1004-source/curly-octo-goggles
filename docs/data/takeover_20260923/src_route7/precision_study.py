"""Route 7: is the single-precision Schur-mode box operator good enough?

For each cell: T64 (fp64 partial factorization, ~1e-14) and T32 (fp32 partial factorization, ~1e-6).
  - rigid-mode residual |T R| / |T| (R: the six rigid modes on the box nodes, orthonormal),
  - T32p = P T32 P + (rigid part of T64 = 0): the exact rigid null space restored by projection,
  - whitened spectrum mu = eig(T64^-1/2 T32p T64^-1/2) on the complement of the rigid modes.
    If (1 - d) T64 <= T32p <= (1 + d) T64 for every cell, the assembled lattice obeys the same bounds for ANY lattice
    size and ANY load, so compliance errors stay below d / (1 - d); d = max |mu - 1| is lattice-size independent.
  - the same spectrum without the projection (cut cells), and the spectrum of T64 itself (soft end).
Lattice check (config x of lattice_v2, box coordinates): cut cell at the origin, its FULL parent at -x, far face
clamped, six face loads; compliance and cell energies with (T64, T64) vs (T32p, T32p) vs (T32, T32).
Usage: precision_study.py <body_dir> <out_dir> <cut_case> <full_case> [<extra_cut_case> ...]
"""
import json, sys, time, gc, os
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schur_encoder as SE

dev, dt = SE.dev, SE.dt


def rigid_basis(node_ids, n):
    g = np.stack(np.unravel_index(node_ids, (2 * n + 1,) * 3), 1) / (2 * n)
    c = g - g.mean(0)
    R = np.zeros((len(g), 3, 6))
    for a in range(3):
        R[:, a, a] = 1.0
    for a in range(3):                                   # rotation about axis a: e_a x (x - c)
        e = np.zeros(3); e[a] = 1.0
        R[:, :, 3 + a] = np.cross(e[None, :], c)
    Q, _ = np.linalg.qr(R.reshape(-1, 6))
    return torch.as_tensor(Q, dtype=dt, device=dev)


def project_(X, R):
    """X <- P X P, P = I - R R^T (in place)."""
    X -= R @ (R.T @ X)
    X -= (X @ R) @ R.T
    return X


def schur_T(case, body_dir, precision, host):
    """T on the host: fp64 for the fp64 factorization, fp32 storage for the fp32 one."""
    enc = SE.SchurEncoder(case, body_dir, precision=precision, log=lambda s_: None, out_host=host)
    T, st = enc.update()
    ids, n = enc.box_node_ids.copy(), enc.n
    T = T.T.contiguous() if host else (T.to(torch.float32) if precision == 'fp32' else T).T.contiguous().cpu()   # symmetric
    enc.free(); enc.Tbuf = None; del enc
    gc.collect(); torch.cuda.empty_cache()
    return T, ids, n, st


def spectra(T64, T32, R, log, unprojected=True):
    """whitened spectra of T32 (projected, and optionally raw) against T64 on the rigid complement.
    T64, T32: host tensors; device copies are made one at a time (a FULL cell is 4.5 GB per copy)."""
    out = {}
    A = T64.to(dev, dt); B = T32.to(dev, dt)
    nrm = float(A.norm())
    out['rel_frobenius'] = float((B - A).norm() / nrm)
    out['rigid_residual_64'] = float((A @ R).norm() / nrm)
    out['rigid_residual_32'] = float((B @ R).norm() / nrm)
    del B; gc.collect(); torch.cuda.empty_cache()
    project_(A, R)
    ev = torch.linalg.eigvalsh(A).sort().values
    alpha = float(ev[-1])
    out['T64_eigs_low'] = ev[:12].tolist(); out['T64_eig_max'] = alpha
    out['T64_soft_ratio'] = float(ev[6] / ev[-1])                        # 7th = softest deformation mode
    del ev; gc.collect(); torch.cuda.empty_cache()
    A += alpha * (R @ R.T)
    L = torch.linalg.cholesky(A); del A
    gc.collect(); torch.cuda.empty_cache()
    variants = [('projected', True)] + ([('raw', False)] if unprojected else [])
    for name, proj in variants:
        B = T32.to(dev, dt)
        if proj:
            project_(B, R)
        B += alpha * (R @ R.T)
        X = torch.linalg.solve_triangular(L, B, upper=False); del B
        Xt = X.T.contiguous(); del X
        B = torch.linalg.solve_triangular(L, Xt, upper=False); del Xt          # L^-1 B L^-T
        C = B + B.T; del B
        C.mul_(0.5)
        mu = torch.linalg.eigvalsh(C); del C
        dev_ = (mu - 1).abs()
        out[f'mu_{name}'] = {'min': float(mu.min()), 'max': float(mu.max()), 'dev_max': float(dev_.max()),
                             'above_1e-4': int((dev_ > 1e-4).sum()), 'above_1e-3': int((dev_ > 1e-3).sum()),
                             'above_1e-2': int((dev_ > 1e-2).sum()), 'count': int(len(mu))}
        del dev_
        log(json.dumps({name: out[f'mu_{name}']}))
        del mu; gc.collect(); torch.cuda.empty_cache()
    del L; gc.collect(); torch.cuda.empty_cache()
    return out


def lattice_check(cells, n=32, log=print):
    """cells: list of (label, offset, node_ids, {variant: T (device fp64)}). Config x: clamp x = -1, loads on y = 0."""
    keys, idx, pos = {}, [], []
    for label, off, ids, _ in cells:
        g = np.stack(np.unravel_index(ids, (2 * n + 1,) * 3), 1) + 2 * n * np.asarray(off)
        gg = np.repeat(g, 3, 0); comp = np.tile(np.arange(3), len(g))
        k = []
        for r in range(len(gg)):
            key = (int(gg[r, 0]), int(gg[r, 1]), int(gg[r, 2]), int(comp[r]))
            if key not in keys:
                keys[key] = len(keys)
            k.append(keys[key])
        idx.append(np.asarray(k)); pos.append((gg, comp))
    N = len(keys)
    clamped = np.zeros(N, dtype=bool)
    for (gg, comp), k in zip(pos, idx):
        clamped[k[gg[:, 0] == -2 * n]] = True
    free = np.flatnonzero(~clamped); fmap = np.full(N, -1); fmap[free] = np.arange(len(free))
    F = np.zeros((N, 6)); labels = []
    for j, (mi, d) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]):
        gg, comp = pos[mi]
        on = (gg[:, 1] == 0) & (comp == d)
        if not on.any():
            raise ValueError('EMPTY_LOAD')
        F[idx[mi][on], j] = 1.0; F[:, j] /= F[:, j].sum(); labels.append(f'{cells[mi][0]}_face_{"xyz"[d]}')
    F[clamped] = 0
    Ft = torch.as_tensor(F[free], dtype=dt, device=dev)
    variants = list(cells[0][3].keys())
    res = dict(N=int(N), free=int(len(free)), loads=labels)
    for v in variants:
        K = torch.zeros((len(free), len(free)), dtype=dt, device=dev)
        for (label, off, ids, Ts), k in zip(cells, idx):
            f = torch.as_tensor(fmap[k], device=dev); keep = torch.nonzero(f >= 0).squeeze(1)
            fc = f[keep]
            for r0 in range(0, len(keep), 2048):
                rows = keep[r0:r0 + 2048]
                blk = Ts[v][rows.cpu()].to(dev, dt)[:, keep]
                K.index_put_((f[rows][:, None], fc[None, :]), blk, accumulate=True)
                del blk
        L = torch.linalg.cholesky(K); del K
        U = torch.cholesky_solve(Ft, L); del L
        c = (Ft * U).sum(0)
        e = []
        for (label, off, ids, Ts), k in zip(cells, idx):
            f = torch.as_tensor(fmap[k], device=dev); keep = f >= 0
            q = torch.zeros((len(k), 6), dtype=dt, device=dev); q[keep] = U[f[keep]]
            Tq = torch.cat([Ts[v][r0:r0 + 2048].to(dev, dt) @ q for r0 in range(0, len(k), 2048)])
            e.append((q * Tq).sum(0)); del Tq
        e = torch.stack(e)
        res[v] = dict(compliance=c.tolist(), energy=e.tolist(), identity=float(((e.sum(0) - c).abs() / c).max()))
        del U; gc.collect(); torch.cuda.empty_cache()
    ref = res[variants[0]]
    for v in variants[1:]:
        c0, c1 = np.asarray(ref['compliance']), np.asarray(res[v]['compliance'])
        e0, e1 = np.asarray(ref['energy']), np.asarray(res[v]['energy'])
        res[v]['compliance_rel_err_max'] = float(np.abs(c1 / c0 - 1).max())
        res[v]['energy_rel_err_max'] = float(np.abs(e1 / e0 - 1).max())
        log(json.dumps(dict(variant=v, compliance_rel_err_max=res[v]['compliance_rel_err_max'],
                            energy_rel_err_max=res[v]['energy_rel_err_max'])))
    return res


def main(body_dir, out, cut, full, extra):
    Path(out).mkdir(parents=True, exist_ok=True)
    rec = dict(cut=cut, full=full, extra=extra, cells={})
    lat = {}
    for case in [cut] + extra + [full]:
        is_full = case == full
        t0 = time.perf_counter()
        T64, ids, n, st64 = schur_T(case, body_dir, 'fp64', host=is_full)
        T32, ids2, _, st32 = schur_T(case, body_dir, 'fp32', host=False)
        if not np.array_equal(ids, ids2):
            raise ValueError('BOX_ORDER')
        R = rigid_basis(ids, n)
        row = dict(box=int(T64.shape[0]), schur64=st64, schur32=st32)
        row.update(spectra(T64, T32, R, log=print, unprojected=True))
        row['seconds'] = time.perf_counter() - t0
        rec['cells'][case] = row
        print(json.dumps(dict(case=case, **{k: v for k, v in row.items() if k not in ('schur64', 'schur32')})), flush=True)
        if case in (cut, full):
            T32p = project_(T32.to(dev, dt), R).to(torch.float32).cpu()
            lat[case] = (ids, {'T64': T64, 'T32p': T32p, 'T32': T32})
        del R
        gc.collect(); torch.cuda.empty_cache()
    cells = [('cut', (0, 0, 0), lat[cut][0], lat[cut][1]), ('full', (-1, 0, 0), lat[full][0], lat[full][1])]
    rec['lattice_x'] = lattice_check(cells, log=print)
    (Path(out) / 'PRECISION.json').write_text(json.dumps(rec, indent=2, default=float))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
