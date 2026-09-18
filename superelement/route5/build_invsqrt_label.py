"""Build packed-upper A^{-1/2} labels (equivariant, PSD-by-construction, soft-weighted) in the exact layout of the Cholesky R_UPPER labels.

A = R^T R with R the frozen reference Cholesky factor.  M = A^{1/2} is the unique
symmetric positive definite square root.  M is stored as the packed upper triangle,
byte-for-byte the same layout as R_UPPER.npy, so factors.read_blocks reads it
unchanged: for a symmetric M the mask lr>=lc keeps exactly the six independent
entries of a diagonal 3x3 block and the full block off the diagonal.

No repair, no clipping, no rank truncation.  float64 throughout.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp/CUTFEM_M4_MULTI_20260917/dev')
from stage_cutfem_m4.factors import read_blocks, sample_pairs


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def unpack(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError('FP64_UPPER_REFERENCE_REQUIRED')
    out = np.zeros((d, d), dtype=np.float64)
    off = 0
    for row in range(d):
        out[row, row:] = p[off:off + d - row]
        off += d - row
    return out


def pack(dense, path):
    d = len(dense)
    out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=(d * (d + 1) // 2,))
    off = 0
    for row in range(d):
        out[off:off + d - row] = dense[row, row:]
        off += d - row
    out.flush()
    del out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--d', type=int, required=True)
    ap.add_argument('--seat', type=int, required=True)
    ap.add_argument('--patch-size', type=int, default=32)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    ref = args.reference
    d = args.d
    report = dict(seat=args.seat, d=d, reference=str(ref),
                  scope='symmetric positive definite square root of G = A^-1, so A_hat^-1 = M M; '
                        'stored in the frozen packed-upper layout')

    t0 = time.perf_counter()
    Rh = unpack(ref / 'R_UPPER.npy', d)
    R = torch.from_numpy(Rh).to(args.device)
    del Rh
    report['unpack_seconds'] = time.perf_counter() - t0

    if bool((R.diagonal() <= 0).any()) or bool((torch.tril(R, -1) != 0).any()):
        raise ValueError('REFERENCE_IS_NOT_UPPER_WITH_POSITIVE_DIAGONAL')

    t0 = time.perf_counter()
    A = R.T @ R
    A = (A + A.T) * .5
    G = torch.cholesky_inverse(R, upper=True)
    G = (G + G.T) * .5
    lam, V = torch.linalg.eigh(G)
    report['eigh_seconds'] = time.perf_counter() - t0
    report['lambda_min'] = float(lam.min())
    report['lambda_max'] = float(lam.max())
    report['condition'] = float(lam.max() / lam.min())
    if float(lam.min()) <= 0:
        raise ValueError('NONPOSITIVE_EIGENVALUE_NO_REPAIR')

    M = (V * lam.sqrt()) @ V.T
    M = (M + M.T) * .5
    del V

    # --- self-tests -------------------------------------------------------
    report['sym_residual'] = float((M - M.T).norm() / M.norm())
    report['square_residual'] = float((M @ M - G).norm() / G.norm())
    report['MMA_minus_I'] = float((M @ (M @ A) - torch.eye(d, dtype=torch.float64, device=args.device)).norm() / d ** .5)
    del G
    report['diag_min'] = float(M.diagonal().min())
    report['diag_max'] = float(M.diagonal().max())
    if float(M.diagonal().min()) <= 0:
        raise ValueError('NONPOSITIVE_SQRT_DIAGONAL')
    logp = M.diagonal().log()
    report['log_pivot_min'] = float(logp.min())
    report['log_pivot_max'] = float(logp.max())
    report['log_pivot_mean'] = float(logp.mean())
    report['log_pivot_std'] = float(logp.std())
    report['calibrate_log_bounds_ok'] = bool(logp.min() > -20 and logp.max() < 5)

    # whitened spectrum of the exact label, through the same path evaluation uses
    C = M @ R.T   # X^-1 = (M R^T)^T (M R^T) for A_hat^-1 = M^2; nu = 1/mu
    W = C.T @ C
    W = (W + W.T) * .5
    mu = torch.linalg.eigvalsh(W)
    report['label_eps_op'] = float((1.0 / mu - 1).abs().max())
    report['label_g'] = float(max(1.0 / mu.min(), mu.max()))
    del C, W, mu, A

    Mh = M.cpu().numpy()
    del M
    torch.cuda.empty_cache()

    out_path = ref / 'MINV_UPPER.npy'
    pack(Mh, out_path)

    # --- round trip through the UNCHANGED reader --------------------------
    rng = np.random.default_rng(20260917)
    packed = np.load(out_path, mmap_mode='r', allow_pickle=False)
    r, c, b = sample_pairs(d // 3, args.patch_size, 4096, rng)
    got = read_blocks(packed, r, c, d)
    want = np.zeros_like(got)
    a3 = np.arange(3)
    lr = 3 * r[:, None, None] + a3[None, :, None]
    lc = 3 * c[:, None, None] + a3[None, None, :]
    valid = lr >= lc
    want = np.where(valid, Mh[lr, lc], 0.)          # L = M^T = M
    report['read_blocks_roundtrip'] = float(np.abs(got - want).max())
    report['read_blocks_scale'] = float(np.abs(want).max())
    report['blocks_checked'] = int(len(r))
    if report['read_blocks_roundtrip'] != 0.0:
        raise ValueError('PACKED_ROUNDTRIP_NOT_EXACT')

    report['m_sha256'] = sha256(out_path)
    report['r_sha256'] = sha256(ref / 'R_UPPER.npy')
    report['bytes'] = out_path.stat().st_size
    (ref / 'INVSQRT_RESULT.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
