"""Route 5: the Cholesky factor of A^-1, in the same packed-upper layout.

Why this target.  Tonight's measurements say the binding error is on the soft end:
both trained arms have 1/mu_min far above mu_max, compliance f^T K^-1 f lives there,
and the entrywise loss is magnitude weighted, so it spends its accuracy on the stiff
end where nothing depends on it.  G = A^-1 has its mass on exactly the directions that
are soft in A, so the SAME magnitude-weighted loss lands where the physics is, with no
reweighting at all.

Why it is route 5 and not a fourth head.  With A_hat^-1 = L^T L, applying A_hat to a
vector is two triangular solves, so an assembled lattice can be solved iteratively and
the dense Schur operator is never formed anywhere -- which was the original reason S
could not enter topology optimisation.

Evaluation is cheaper too.  R_*^T A_hat^-1 R_* = (L R_*)^T (L R_*), so the whitened
spectrum comes from one matmul and its eigenvalues are 1/mu.  g = max(nu_max, 1/nu_min)
is unchanged, being symmetric under inversion.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v3')
from stage_cutfem_m4.factors import read_blocks, sample_pairs


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
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
    a = ap.parse_args()
    ref, d = a.reference, a.d
    report = dict(seat=a.seat, d=d, reference=str(ref),
                  scope='upper Cholesky factor L of G = A^-1, so that A_hat^-1 = L^T L',
                  convention='G = L^T L, matching A = R_*^T R_*')

    t = time.perf_counter()
    Rstar = torch.from_numpy(unpack(ref / 'R_UPPER.npy', d)).to(a.device)
    report['unpack_seconds'] = time.perf_counter() - t
    if bool((Rstar.diagonal() <= 0).any()) or bool((torch.tril(Rstar, -1) != 0).any()):
        raise ValueError('REFERENCE_IS_NOT_UPPER_WITH_POSITIVE_DIAGONAL')

    t = time.perf_counter()
    # G = A^-1 straight from the reference factor, no explicit A and no general inverse.
    G = torch.cholesky_inverse(Rstar, upper=True)
    G = (G + G.T) * .5
    report['inverse_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    L = torch.linalg.cholesky(G, upper=True)
    report['cholesky_seconds'] = time.perf_counter() - t

    # --- self-tests --------------------------------------------------------
    A = Rstar.T @ Rstar
    resid = L.T @ (L @ A) - torch.eye(d, dtype=torch.float64, device=a.device)
    report['L_T_L_times_A_minus_I'] = float(resid.norm() / d ** .5)
    del resid
    report['G_reconstruction_relative'] = float((L.T @ L - G).norm() / G.norm())
    pivot = L.diagonal()
    report['pivot_min'] = float(pivot.min())
    report['pivot_max'] = float(pivot.max())
    logp = pivot.log()
    report['log_pivot_min'] = float(logp.min())
    report['log_pivot_max'] = float(logp.max())
    report['log_pivot_mean'] = float(logp.mean())
    report['log_pivot_std'] = float(logp.std())
    report['legacy_log_pivot_bounds_ok'] = bool(logp.min() > -20 and logp.max() < 5)

    # The whitened spectrum of the EXACT label, through the cheap route.
    # X = R_*^-T A_hat R_*^-1 with A_hat = (L^T L)^-1 satisfies
    #     X^-1 = R_* L^T L R_*^T = (L R_*^T)^T (L R_*^T),
    # so Q = L R_*^T gives nu = 1/mu.  Note R_*^T, not R_*: the transposed pair
    # R_*^T A_hat^-1 R_* is a DIFFERENT matrix and its eigenvalues are not 1/mu.
    C = L @ Rstar.T
    W = C.T @ C
    W = (W + W.T) * .5
    nu = torch.linalg.eigvalsh(W)
    del C, W
    mu_min, mu_max = 1.0 / float(nu.max()), 1.0 / float(nu.min())
    report['label_nu_min'] = float(nu.min())
    report['label_nu_max'] = float(nu.max())
    report['label_eps_op'] = max(abs(mu_min - 1), abs(mu_max - 1))
    report['label_g'] = max(mu_max, 1.0 / mu_min)
    report['condition_of_A'] = float((Rstar.diagonal().max() / Rstar.diagonal().min()) ** 2)
    del nu, G, A, Rstar
    torch.cuda.empty_cache()

    host = L.cpu().numpy()
    del L
    torch.cuda.empty_cache()
    out_path = ref / 'G_UPPER.npy'
    pack(host, out_path)

    # --- round trip through the UNCHANGED reader ---------------------------
    rng = np.random.default_rng(20260918)
    packed = np.load(out_path, mmap_mode='r', allow_pickle=False)
    r, c, b = sample_pairs(d // 3, a.patch_size, 4096, rng)
    got = read_blocks(packed, r, c, d)
    a3 = np.arange(3)
    lr = 3 * r[:, None, None] + a3[None, :, None]
    lc = 3 * c[:, None, None] + a3[None, None, :]
    want = np.where(lr >= lc, host[lc, lr], 0.)      # the reader returns blocks of L^T
    report['read_blocks_roundtrip'] = float(np.abs(got - want).max())
    report['blocks_checked'] = int(len(r))
    if report['read_blocks_roundtrip'] != 0.0:
        raise ValueError('PACKED_ROUNDTRIP_NOT_EXACT')

    report['g_sha256'] = sha256(out_path)
    report['r_sha256'] = sha256(ref / 'R_UPPER.npy')
    report['bytes'] = out_path.stat().st_size
    (ref / 'INVERSE_RESULT.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
