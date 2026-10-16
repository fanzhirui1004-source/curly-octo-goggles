"""P2 item B, step 1: how compressible is a cell's dense condensed operator S (ports x ports)?
Input: <body>/<case>_portview/T64.npy (exact dense S, from make_T_gpu/make_T_cpu) and BOX_NODES.npy (port node ids, the
teacher's port order: dof k <-> node k // 3, component k % 3). Ports are reordered by recursive coordinate bisection of the
node positions; two formats are tested on the reordered matrix, each at several truncation tolerances tol (singular values
below tol * ||S||_2 are dropped in every compressed block):
  BLR    flat blocks of --block dofs; diagonal blocks dense, off-diagonal blocks truncated SVD (upper triangle stored)
  HODLR  binary tree down to --leaf dofs; sibling off-diagonal blocks truncated SVD, leaves dense
Reported per (format, tol): storage (float64 words of the upper/symmetric representation) against the dense upper triangle,
relative matvec error on random vectors, and for 9 equilibrated linear load fields f (component e_i times x_j - mean):
compliance error |f^T q_c - f^T q| / f^T q and solution error ||q_c - q||_S / ||q||_S, with q = S^+ f, q_c = S_c^+ f (rigid
modes removed by a rigid-basis shift); whether S_c + shift is positive definite.
Usage: bcomp.py <out.json> <body>:<case>[,<body>:<case>...] [--tols 1e-4,1e-6,1e-8] [--block 1024] [--leaf 512] [--n 32]"""
import argparse, json, time
from pathlib import Path
import numpy as np
import scipy.linalg as sl


def bisect_order(X, idx, leaf, out):
    if len(idx) <= leaf:
        out.extend(idx.tolist()); return
    ext = X[idx].max(0) - X[idx].min(0)
    ax = int(np.argmax(ext))
    o = idx[np.argsort(X[idx, ax], kind='stable')]
    h = len(o) // 2
    bisect_order(X, o[:h], leaf, out); bisect_order(X, o[h:], leaf, out)


def rigid_basis(X):
    c = X.mean(0); d = X - c
    R = np.zeros((3 * len(X), 6))
    for i in range(3):
        R[i::3, i] = 1.0
    R[1::3, 3], R[2::3, 3] = -d[:, 2], d[:, 1]                   # rotation about x
    R[0::3, 4], R[2::3, 4] = d[:, 2], -d[:, 0]                   # about y
    R[0::3, 5], R[1::3, 5] = -d[:, 1], d[:, 0]                   # about z
    Q, _ = np.linalg.qr(R)
    return Q


def compress(S, blocks, dense, tmin, norm2):
    """Truncated SVD of every off-diagonal (upper) block, kept at the loosest rank needed (tol = tmin)."""
    fac = []
    for (r0, r1, c0, c1) in blocks:
        U, s, Vt = sl.svd(S[r0:r1, c0:c1], full_matrices=False, lapack_driver='gesdd')
        k = int((s > tmin * norm2).sum())
        fac.append((U[:, :k].copy(), s[:k].copy(), Vt[:k].copy()))
    return fac


def build(S, blocks, dense, fac, tol, norm2):
    Sc = np.zeros_like(S); words = 0
    for a, z in dense:
        Sc[a:z, a:z] = S[a:z, a:z]; words += (z - a) * (z - a + 1) // 2
    for (r0, r1, c0, c1), (U, s, Vt) in zip(blocks, fac):
        k = int((s > tol * norm2).sum())
        blk = (U[:, :k] * s[:k]) @ Vt[:k]
        Sc[r0:r1, c0:c1] = blk; Sc[c0:c1, r0:r1] = blk.T
        words += k * ((r1 - r0) + (c1 - c0))
    return Sc, words


def layout_blr(n, b):
    cuts = list(range(0, n, b)) + [n]
    dense = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    blocks = [(dense[i][0], dense[i][1], dense[j][0], dense[j][1]) for i in range(len(dense)) for j in range(i + 1, len(dense))]
    return blocks, dense


def layout_hodlr(n, leaf):
    blocks, dense = [], []

    def rec(a, z):
        if z - a <= leaf:
            dense.append((a, z)); return
        m = (a + z) // 2
        blocks.append((a, m, m, z)); rec(a, m); rec(m, z)
    rec(0, n)
    return blocks, dense


def solve_all(S, F, R, shift):
    A = S + shift * (R @ R.T)
    try:
        L = sl.cho_factor(A, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return None
    return sl.cho_solve(L, F, check_finite=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--tols', default='1e-4,1e-6,1e-8'); ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--leaf', type=int, default=512); ap.add_argument('--n', type=int, default=32)
    ap.add_argument('--scale', default='none', help="'jacobi': compress D^-1/2 S D^-1/2 (D = diag S), tolerances relative to its norm")
    ap.add_argument('--formats', default='BLR,HODLR')
    a = ap.parse_args()
    tols = [float(t) for t in a.tols.split(',')]
    out = {}
    for spec in a.cases.split(','):
        body, case = spec.split(':')
        t0 = time.perf_counter()
        pd = Path(body) / (case + '_portview')
        nodes = np.load(pd / 'BOX_NODES.npy')
        X = np.stack(np.unravel_index(nodes, (2 * a.n + 1,) * 3), 1).astype(float) / (2 * a.n)
        S = np.load(pd / 'T64.npy')
        n = S.shape[0]; assert n == 3 * len(nodes)
        order = []; bisect_order(X, np.arange(len(nodes)), 64, order)
        order = np.asarray(order)
        perm = (3 * order[:, None] + np.arange(3)[None]).reshape(-1)
        S = np.ascontiguousarray(S[np.ix_(perm, perm)]); Xp = X[order]
        R = rigid_basis(Xp)
        x = np.random.default_rng(0).standard_normal(n)
        for _ in range(30):
            x = S @ x; x /= np.linalg.norm(x)
        norm2 = float(x @ (S @ x))
        # loads: component i times (x_j - mean), equilibrated (rigid part removed)
        F = np.zeros((n, 9)); c = Xp - Xp.mean(0)
        for i in range(3):
            for j in range(3):
                F[i::3, 3 * i + j] = c[:, j]
        F -= R @ (R.T @ F)
        shift = norm2
        Q = solve_all(S, F, R, shift)
        Q -= R @ (R.T @ Q)
        comp = (F * Q).sum(0); eQ = np.sqrt((Q * (S @ Q)).sum(0))
        Xr = np.random.default_rng(1).standard_normal((n, 8)); SXr = S @ Xr
        rec = dict(ports=n, norm2=norm2, dense_upper_words=n * (n + 1) // 2, setup_s=time.perf_counter() - t0, formats={})
        if a.scale == 'jacobi':
            dsq = np.sqrt(np.diag(S).copy())
            Sw = S / dsq[:, None] / dsq[None, :]
            y = np.random.default_rng(0).standard_normal(n)
            for _ in range(30):
                y = Sw @ y; y /= np.linalg.norm(y)
            nw = float(y @ (Sw @ y))
        else:
            dsq, Sw, nw = None, S, norm2
        rec['scale'] = a.scale
        fmts = {'BLR': (layout_blr, a.block), 'HODLR': (layout_hodlr, a.leaf)}
        for name in a.formats.split(','):
            lay, arg = fmts[name]
            t = time.perf_counter()
            blocks, dense = lay(n, arg)
            fac = compress(Sw, blocks, dense, min(tols), nw)
            rec[f'{name}_svd_s'] = time.perf_counter() - t
            for tol in tols:
                Sc, words = build(Sw, blocks, dense, fac, tol, nw)
                if dsq is not None:
                    Sc *= dsq[:, None]; Sc *= dsq[None, :]
                Qc = solve_all(Sc, F, R, shift)
                row = dict(words=int(words), ratio=rec['dense_upper_words'] / words,
                           MB=8 * words / 1e6, matvec_rel=float(np.linalg.norm(Sc @ Xr - SXr) / np.linalg.norm(SXr)))
                if Qc is None:
                    row['spd'] = False
                else:
                    Qc -= R @ (R.T @ Qc); D = Qc - Q
                    row.update(spd=True, compliance_rel_max=float((np.abs((F * Qc).sum(0) - comp) / comp).max()),
                               solution_rel_max=float((np.sqrt((D * (S @ D)).sum(0)) / eQ).max()))
                rec['formats'][f'{name}:{tol:g}'] = row
                print(json.dumps(dict(case=case, fmt=name, tol=tol, **row)), flush=True)
                del Sc
            rec[f'{name}_s'] = time.perf_counter() - t
            del fac
        out[case] = rec
        Path(a.out).write_text(json.dumps(out, indent=1))
        del S


if __name__ == '__main__':
    main()
