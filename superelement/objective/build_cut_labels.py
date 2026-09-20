"""Build M_q labels for cut packets straight from the packet, in our own rigid quotient.

The original reference stage - packet S_UPPER -> A = B S B^T -> R_UPPER - is not on this box, and
rebuilding it does not reproduce the stored R_UPPER: our Householder quotient is a different basis
of the rigid complement, giving an operator with an identical Frobenius norm and a 0.374 relative
difference.  That is harmless for the object that matters, and the identity is the one the label was
designed around:

    A' = U A U^T with B' = U B  =>  B'^T A'^(-1/2) B' = B^T U^T U A^(-1/2) U^T U B = B^T A^(-1/2) B

so M_q = ((Pi S Pi)^+)^(1/2) with Pi = B^T B is a function of S and the node set alone.  Verified,
not assumed: on seat 100064 (q = 13248) this route reproduces the stored MQ_UPPER to 8.79e-13
relative, 2.32e-09 absolute, 2.40e-12 on the diagonal.  So labels built here can be mixed with the
139 that already exist.

It therefore skips the reference stage entirely: packet S_UPPER -> A -> eigh -> M_q, one dense
eigendecomposition rather than a Cholesky, an inverse, and a second eigendecomposition.  A
self-consistent R_UPPER is written beside it because evaluation reads one as the whitener, and the
whitened spectrum mu = eig(R^-T A_hat R^-1) is invariant to which basis it is.

    python -m superelement.objective.build_cut_labels --packets LIST.json --output DIR [--device cuda:0]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def unpack(path, n, device):
    """Packed upper triangle -> dense symmetric float64.

    Some packets store S_UPPER as float32 and some as float64 (SAMPLE.json records which, and the
    ingest carries a `source_float64_matrix_migration` field), so the source dtype is promoted here
    and returned, to be recorded in the label's report: a label built from a float32 operator is not
    the same object as one built from float64, and nothing downstream can tell them apart.
    """
    v = np.load(path, mmap_mode='r')
    if v.size != n * (n + 1) // 2:
        raise ValueError(f'PACKED_SIZE {v.size} for n={n} at {path}')
    source_dtype = str(v.dtype)
    M = torch.zeros((n, n), dtype=torch.float64, device=device)
    iu = torch.triu_indices(n, n, device=device)
    M[iu[0], iu[1]] = torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=torch.float64)
    M = M + torch.triu(M, 1).T
    return M, source_dtype


def pack(M, path):
    n = M.shape[0]
    iu = torch.triu_indices(n, n, device=M.device)
    np.save(path, M[iu[0], iu[1]].contiguous().cpu().numpy())
    return path


def build_one(job, out_root, device, quotient_module, tests=True):
    """One packet -> MQ_UPPER.npy, R_UPPER.npy, MQ_RESULT.json in out_root/<name>."""
    from stage_cutfem_m4.quotient import RigidQuotient
    dev = torch.device(device)
    name = job['name']
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'MQ_RESULT.json').exists():
        return dict(name=name, status='ALREADY_BUILT')
    packet = Path(job['packet'])
    cache_path = Path(job['trace_cache'])
    q = int(job['q'])
    report = dict(name=name, packet=str(packet), trace_cache=str(cache_path), q=q, d=q - 6,
                  device=device, route='packet S_UPPER -> our rigid quotient -> A -> eigh -> M_q',
                  basis_note='our Householder quotient is not the original pipeline\'s; M_q is '
                             'invariant to that choice, verified on seat 100064 to 8.79e-13')
    tick = time.perf_counter()
    cache = dict(np.load(cache_path, allow_pickle=False))
    if cache['rigid'].shape != (q, 6):
        raise ValueError(f'RIGID_SHAPE {cache["rigid"].shape} for q={q}')
    Qt = RigidQuotient(torch.from_numpy(cache['rigid']).to(dev),
                       torch.from_numpy(cache['order']).to(dev))
    S, source_dtype = unpack(packet / 'S_UPPER.npy', q, dev)
    report['source_operator_dtype'] = source_dtype
    report['load_seconds'] = time.perf_counter() - tick
    report['trace_operator_frobenius'] = float(S.norm())

    tick = time.perf_counter()
    A = Qt(Qt(S).T.contiguous())
    del S
    gc.collect()
    report['congruence_relative_asymmetry'] = float((A - A.T).norm() / A.norm())
    A = (A + A.T) * .5
    lam, V = torch.linalg.eigh(A)
    report['eigh_seconds'] = time.perf_counter() - tick
    report['lambda_min_of_A'] = float(lam.min())
    report['lambda_max_of_A'] = float(lam.max())
    report['condition_of_A'] = float(lam.max() / lam.min())
    if float(lam.min()) <= 0:
        raise ValueError('NONPOSITIVE_EIGENVALUE_NO_REPAIR')

    tick = time.perf_counter()
    Mhalf = (V * lam.rsqrt()) @ V.T                       # A^{-1/2}
    Mhalf = (Mhalf + Mhalf.T) * .5
    half = Qt.lift(Mhalf)
    Mq = Qt.lift(half.T.contiguous())                     # B^T A^{-1/2} B
    del half
    report['sym_residual_before_average'] = float((Mq - Mq.T).norm() / Mq.norm())
    Mq = (Mq + Mq.T) * .5
    report['lift_seconds'] = time.perf_counter() - tick

    if tests:
        rigid = torch.from_numpy(cache['rigid']).to(dev)
        report['rigid_nullspace'] = float((Mq @ rigid).norm() / (Mq.norm() * rigid.norm()))
        back = Qt(Qt(Mq).T.contiguous())
        report['round_trip_to_A_inverse_half'] = float((back - Mhalf).norm() / Mhalf.norm())
        eye = torch.eye(q - 6, dtype=torch.float64, device=dev)
        report['back_squared_times_A_minus_I'] = float((back @ (back @ A) - eye).norm() / (q - 6) ** .5)
        del back, eye, rigid
    del Mhalf, V, lam
    gc.collect()

    R, info = torch.linalg.cholesky_ex(A, upper=True)
    if int(info) != 0:
        raise ValueError('REFERENCE_CHOLESKY_FAILED')
    report['minimum_diagonal'] = float(R.diagonal().min())
    del A
    gc.collect()
    pack(R, out / 'R_UPPER.npy')
    del R
    gc.collect()
    pack(Mq, out / 'MQ_UPPER.npy')
    report['label_frobenius'] = float(Mq.norm())
    del Mq
    gc.collect()
    if dev.type == 'cuda':
        report['peak_gib'] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.empty_cache()
    report['mq_sha256'] = sha256(out / 'MQ_UPPER.npy')
    report['r_sha256'] = sha256(out / 'R_UPPER.npy')
    report['trace_sha256'] = sha256(cache_path)
    report['s_upper_sha256'] = sha256(packet / 'S_UPPER.npy')
    report['mq_bytes'] = (out / 'MQ_UPPER.npy').stat().st_size
    report['total_seconds'] = (report['load_seconds'] + report['eigh_seconds']
                               + report['lift_seconds'])
    (out / 'MQ_RESULT.json').write_text(json.dumps(report, indent=1) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packets', type=Path, required=True,
                    help='JSON list of {name, packet, trace_cache, q}')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--max-q', type=int, default=0,
                    help='skip packets above this q (the dense eigh needs about 3 q^2 doubles)')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--no-tests', action='store_true')
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(a.source))
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    jobs = json.loads(a.packets.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    done, failed, skipped = 0, 0, 0
    for job in jobs:
        if a.max_q and int(job['q']) > a.max_q:
            skipped += 1
            continue
        try:
            r = build_one(job, a.output, a.device, a.source, tests=not a.no_tests)
            done += 1
            print(json.dumps({k: r.get(k) for k in
                              ('name', 'q', 'status', 'total_seconds', 'eigh_seconds', 'peak_gib',
                               'condition_of_A', 'rigid_nullspace', 'back_squared_times_A_minus_I',
                               'mq_bytes')}), flush=True)
        except Exception as exc:
            failed += 1
            (a.output / f'{job["name"]}_FAILED.json').write_text(
                json.dumps(dict(name=job['name'], error=str(exc)[:400]), indent=1) + '\n')
            print(json.dumps(dict(name=job['name'], q=job.get('q'), error=str(exc)[:200])), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps(dict(phase='batch_complete', built=done, failed=failed, skipped=skipped)), flush=True)


if __name__ == '__main__':
    main()
