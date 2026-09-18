"""Six-platen response of a cell from a q-space M_q (label or prediction): the tau-test observable.

HEAD_AB_AND_TAU_20260917.md measured the teacher's d(trace H^-1)/d eps = -2.278 (relative)
on seat 0328 across the five operators eps in {-1e-3, -1e-4, 0, 1e-4, 1e-3}.  The
Phase-3 test asks the network for the same derivative.  This module turns any packed
M_q into that observable through the frozen fixture (bottom face clamped, top face a
rigid platen with six unit motions), so the same function scores the label (which must
reproduce the teacher's response exactly) and a prediction.

    python -m superelement.equi.platen --mq REFERENCE_0253/MQ_UPPER.npy --trace-cache .../TRACE_CACHE.npz --q 12828

Prints trace(H^-1), the six compliances and the free-equilibrium residual.  With
--compare RESPONSE.json (an EVAL_*/RESPONSE.json from run.py) the six compliances are
checked against that file's reference response.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def unpack(path, d, device):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path}')
    out = np.zeros((d, d), dtype=np.float64); off = 0
    for row in range(d):
        out[row, row:] = p[off:off + d - row]; off += d - row
    return torch.from_numpy(out).to(device)


def require_box_only(cache):
    """run.py:six_response refuses the fixture unless these three hold; so must this."""
    if not (np.all(np.diff(cache['indptr']) == 1) and np.all(cache['coefficients'] == 1)
            and np.all(np.asarray(cache['kind']) == 0)):
        raise ValueError('GENERAL_SIGNED_TRACE_FIXTURE_NOT_SUPPORTED')


def operator_from_mq(Mq_packed, cache, q, device):
    """A_hat = (B M_q B^T)^-2 in the frozen quotient, and the quotient itself."""
    from stage_cutfem_m4.quotient import RigidQuotient
    M = unpack(Mq_packed, q, device); M = M + torch.triu(M, 1).T
    rigid = torch.from_numpy(cache['rigid']).to(device)
    Qt = RigidQuotient(rigid, torch.from_numpy(cache['order']).to(device))
    Mhat = Qt(Qt(M).T.contiguous()); del M; gc.collect()
    Minv = torch.linalg.inv(Mhat); del Mhat
    A = Minv @ Minv; del Minv; gc.collect()
    return (A + A.T) * .5, Qt


@torch.no_grad()
def platen_response(A, Qt, cache, device):
    """Lift A to the full trace (S = B^T A B), clamp the bottom, drive the top platen."""
    from stage_cutfem_m4.response import end_registry, force_response
    q = Qt.dimension
    half = Qt.lift(A); S = Qt.lift(half.T.contiguous()).T.contiguous(); del half; gc.collect()
    prescribed, values, registry = end_registry(cache['support_centroid'])
    free = np.setdiff1d(np.arange(q), prescribed)
    i = torch.as_tensor(free, device=device); b = torch.as_tensor(prescribed, device=device)
    u = torch.zeros((q, 6), dtype=torch.float64, device=device); u[b] = torch.from_numpy(values).to(device)
    rhs = -(S @ u)[i]; block = S[i[:, None], i[None, :]]
    skew = float((block - block.T).norm() / block.norm())
    if skew > 1e-10:
        raise ValueError('RESPONSE_BLOCK_ASYMMETRY')
    scale = block.diagonal().rsqrt(); scaled = scale[:, None] * block * scale[None, :]
    root = torch.linalg.cholesky((scaled + scaled.T) * .5)
    u[i] = scale[:, None] * torch.cholesky_solve(scale[:, None] * rhs, root)
    force = S @ u
    residual = (scale[:, None] * force[i]).norm(dim=0) / (scale[:, None] * rhs).norm(dim=0)
    H = (torch.from_numpy(values).to(device).T @ force[b]).cpu().numpy()
    result, _ = force_response(H)
    C = np.asarray(result['load_basis_compliance'])
    return dict(trace_H_inverse=float(np.trace(C)), compliance=result['compliance'], H=H.tolist(),
                free_equilibrium_relative=residual.cpu().tolist(), registry=registry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mq', type=Path, required=True); ap.add_argument('--trace-cache', type=Path, required=True)
    ap.add_argument('--q', type=int, required=True); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--compare', type=Path, default=None, help='an EVAL_*/RESPONSE.json whose reference response to match')
    ap.add_argument('--output', type=Path, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(a.source))
    torch.set_num_threads(8)
    cache = dict(np.load(a.trace_cache, allow_pickle=False))
    require_box_only(cache)
    A, Qt = operator_from_mq(a.mq, cache, a.q, a.device)
    out = platen_response(A, Qt, cache, a.device)
    out['provenance'] = dict(mq=str(a.mq), trace_cache=str(a.trace_cache), q=int(a.q),
                             mq_sha256=sha256(a.mq), trace_sha256=sha256(a.trace_cache),
                             tau_corners=np.asarray(cache['tau_corners'], dtype=float).tolist())
    if a.compare:
        ref = json.loads(a.compare.read_text())['reference']
        want = np.asarray(ref['compliance']); got = np.asarray(out['compliance'])
        out['compare'] = dict(file=str(a.compare), max_relative_compliance_difference=float(np.abs(got / want - 1).max()),
                              reference_trace_H_inverse=float(np.trace(np.asarray(ref['load_basis_compliance']))))
    print(json.dumps({k: v for k, v in out.items() if k not in ('H', 'registry')}, indent=1))
    if a.output:
        a.output.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
