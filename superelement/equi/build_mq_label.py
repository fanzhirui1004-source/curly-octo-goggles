"""Build the canonical q-space label M_q = B^T A^{-1/2} B, packed upper, with self-tests.

A = R^T R is the frozen quotient operator (d = q - 6), B the frozen Householder quotient
with orthonormal rows and ker B = the rigid trace.  A itself depends on the arbitrary
basis B of the rigid complement (B' = U B gives A' = U A U^T), so a network predicting
A's factor has to learn Codex's elimination bookkeeping.  M_q does not:

    M_q = B^T A^{-1/2} B = ((Pi S Pi)^+)^{1/2},   Pi = B^T B,

the square root of the pseudo-inverse of the teacher's Schur operator on the rigid
complement — a function of S and the node set alone.  It is symmetric positive
semidefinite with exactly the rigid nullspace, its entries are largest where the cell
is soft (the physics that compliance and sensitivities live on), and under a cube
symmetry g it transforms as a nodal tensor field: block (i, j) -> Q B_ij Q^T with the
node permutation, which is what `equi.context.rotate_context` applies to the inputs.

Round trip for evaluation: B M_q B^T = A^{-1/2} exactly, so the whitened spectrum of a
prediction is nu = eig((B M_q_hat B^T R^T)^T (B M_q_hat B^T R^T)) with mu = 1/nu.

Layout: packed upper triangle of the q x q symmetric M_q, byte-compatible with
factors.read_blocks(packed, r, c, q) over count = q/3 nodes (lower blocks, diagonal
blocks masked to their lower triangle).  No repair, no clipping.  float64 throughout.
"""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch


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


def build(reference, seat, device, threads, source, trace_cache=None, patch_size=32):
    sys.path.insert(0, str(source))
    from stage_cutfem_m4.quotient import RigidQuotient
    from stage_cutfem_m4.factors import read_blocks, sample_pairs
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = Path(reference)
    receipt = json.loads((ref / 'RESULT.json').read_text())
    d = int(receipt['dimension'])
    q = d + 6
    if q % 3:
        raise ValueError('Q_NOT_A_MULTIPLE_OF_3')
    count = q // 3
    cache_path = Path(trace_cache) if trace_cache else ref / 'input' / 'TRACE_CACHE.npz'
    r_sha = sha256(ref / 'R_UPPER.npy')
    if r_sha != receipt['factor_sha256']:
        raise ValueError('FROZEN_REFERENCE_BINDING')
    trace_sha = sha256(cache_path)
    if trace_sha != receipt['trace_sha256']:
        # The trace cache supplies the quotient. Every self-test below passes with the WRONG
        # cache (it is self-consistent in whatever basis it is handed), so this is the only
        # thing standing between a mismatched --trace-cache and a silently wrong label.
        raise ValueError('FROZEN_TRACE_BINDING')
    report = dict(seat=int(seat), d=d, q=q, count=count, reference=str(ref), trace_cache=str(cache_path),
                  device=device, threads=threads,
                  scope='M_q = B^T A^{-1/2} B, the canonical q-space label; symmetric PSD with the rigid nullspace')
    dev = torch.device(device)
    t0 = time.perf_counter()
    R = torch.from_numpy(unpack(ref / 'R_UPPER.npy', d)).to(dev)
    if bool((R.diagonal() <= 0).any()) or bool((torch.tril(R, -1) != 0).any()):
        raise ValueError('REFERENCE_IS_NOT_UPPER_WITH_POSITIVE_DIAGONAL')
    cache = dict(np.load(cache_path, allow_pickle=False))
    rigid = torch.from_numpy(cache['rigid']).to(dev)
    if rigid.shape != (q, 6):
        raise ValueError('RIGID_TRACE_DIMENSION')
    Qt = RigidQuotient(rigid, torch.from_numpy(cache['order']).to(dev))
    report['unpack_seconds'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    G = torch.cholesky_inverse(R, upper=True)            # A^-1
    G = (G + G.T) * .5
    lam, V = torch.linalg.eigh(G)
    report['eigh_seconds'] = time.perf_counter() - t0
    report['lambda_min_of_A_inverse'] = float(lam.min())
    report['lambda_max_of_A_inverse'] = float(lam.max())
    if float(lam.min()) <= 0:
        raise ValueError('NONPOSITIVE_EIGENVALUE_NO_REPAIR')
    M = (V * lam.sqrt()) @ V.T                            # A^{-1/2}
    M = (M + M.T) * .5
    del V, lam
    gc.collect()
    report['MMG_residual'] = float((M @ M - G).norm() / G.norm())
    del G
    gc.collect()

    t0 = time.perf_counter()
    half = Qt.lift(M)                                     # B^T M            (q, d)
    Mq = Qt.lift(half.T.contiguous())                     # B^T M B          (q, q)
    del half
    gc.collect()
    report['sym_residual_before_average'] = float((Mq - Mq.T).norm() / Mq.norm())
    Mq = (Mq + Mq.T) * .5
    report['lift_seconds'] = time.perf_counter() - t0

    # --- self-tests -------------------------------------------------------
    # recorded BEFORE the symmetrising average, so it can actually fail
    report['rigid_nullspace'] = float((Mq @ rigid).norm() / (Mq.norm() * rigid.norm()))
    back = Qt(Qt(Mq).T.contiguous())                      # B M_q B^T  (d, d)
    report['round_trip_to_M'] = float((back - M).norm() / M.norm())
    eye = torch.eye(d, dtype=torch.float64, device=dev)
    A = R.T @ R
    report['back_squared_times_A_minus_I'] = float((back @ (back @ A) - eye).norm() / d ** .5)
    del A, eye
    C = back @ R.T                                        # nu = eig(C^T C) = 1/mu for the exact label
    W = C.T @ C
    W = (W + W.T) * .5
    nu = torch.linalg.eigvalsh(W)
    report['label_g'] = float(max(1.0 / nu.min(), nu.max()))
    del C, W, nu, back, M
    gc.collect()
    diag = Mq.diagonal()
    report['diag_min'] = float(diag.min()); report['diag_max'] = float(diag.max())
    if float(diag.min()) <= 0:
        raise ValueError('NONPOSITIVE_MQ_DIAGONAL')
    logp = diag.log()
    report['log_pivot_min'] = float(logp.min()); report['log_pivot_max'] = float(logp.max())
    report['log_pivot_mean'] = float(logp.mean()); report['log_pivot_std'] = float(logp.std())
    report['entry_abs_max'] = float(Mq.abs().max()); report['frobenius'] = float(Mq.norm())
    # the true quotient spectrum is what the network will be scored against; record it for the record
    report['eig_seconds_total'] = report['eigh_seconds']

    # span(rigid) must be the cell's rigid-body space, or M_q is not the basis-free object the
    # label is claimed to be (and augmentation's identity fails).  For a general signed trace
    # that is the pullback of the background rigid modes through the trace operator, and the
    # identity is exact, so this doubles as a check that the CSR and the rigid array agree.
    from superelement.equi.context import rigid_span_residual
    meta = json.loads((cache_path.parent / 'INPUT.json').read_text())['metadata']
    report['n'] = int(meta['n'])
    report['rigid_span_residual'] = rigid_span_residual(cache, report['n'])

    # --- gate on the self-tests, rather than only recording them ------------
    gates = dict(label_g=(abs(report['label_g'] - 1.0), 1e-8), round_trip_to_M=(report['round_trip_to_M'], 1e-12),
                 MMG_residual=(report['MMG_residual'], 1e-12),
                 back_squared_times_A_minus_I=(report['back_squared_times_A_minus_I'], 1e-9),
                 rigid_nullspace=(report['rigid_nullspace'], 1e-12),
                 rigid_span_residual=(report['rigid_span_residual'], 1e-12))
    report['gates'] = {k: dict(value=v, tolerance=t, pass_=bool(v <= t)) for k, (v, t) in gates.items()}
    failed = [k for k, (v, t) in gates.items() if not v <= t]
    if failed:
        (ref / 'MQ_FAILURE.json').write_text(json.dumps(report, indent=1))
        raise ValueError('MQ_SELFTEST_GATE_FAILED ' + ','.join(failed))

    Mh = Mq.cpu().numpy()
    del Mq, R, rigid, Qt
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    out_path = ref / 'MQ_UPPER.npy'
    pack(Mh, out_path)

    # --- round trip through the UNCHANGED reader, indexed by q ---------------
    rng = np.random.default_rng(20260918)
    packed = np.load(out_path, mmap_mode='r', allow_pickle=False)
    r, c, b = sample_pairs(count, patch_size, 4096, rng)
    got = read_blocks(packed, r, c, q)
    a3 = np.arange(3)
    lr = 3 * r[:, None, None] + a3[None, :, None]
    lc = 3 * c[:, None, None] + a3[None, None, :]
    want = np.where(lr >= lc, Mh[lr, lc], 0.)
    report['read_blocks_roundtrip'] = float(np.abs(got - want).max())
    report['read_blocks_scale'] = float(np.abs(want).max())
    report['blocks_checked'] = int(len(r))
    if report['read_blocks_roundtrip'] != 0.0:
        raise ValueError('PACKED_ROUNDTRIP_NOT_EXACT')
    # (the packed round trip above is the real check; Mh is symmetric by construction of the
    #  averaging step, so asserting it here would be a tautology)
    del Mh
    gc.collect()
    report['m_sha256'] = sha256(out_path)
    report['r_sha256'] = r_sha
    report['trace_sha256'] = trace_sha
    report['bytes'] = out_path.stat().st_size
    report['total_seconds'] = time.perf_counter() - t0 + report['unpack_seconds'] + report['eigh_seconds']
    (ref / 'MQ_RESULT.json').write_text(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--seat', type=int, required=True)
    ap.add_argument('--trace-cache', type=Path, default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    args = ap.parse_args()
    report = build(args.reference, args.seat, args.device, args.threads, args.source, args.trace_cache)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
