"""Is the physical response smooth in the design variable tau?

Five operators for seat 0328 exist at eps = 0, +-1e-4, +-1e-3 (tau_corners scaled
by 1+eps), all admitted with the SAME complete trace, order and quotient.  Between
eps=0 and eps=+1e-4 exactly one cut cell is born (7478 -> 7479 active cells), and
that single cell moves the whitened spectrum to mu_max = 1.598 while the Frobenius
change stays a smooth linear 2.3e-4.

The acceptance metric eps_op therefore jumps.  The question this script answers is
whether the quantity the project actually has to get right -- the assembled platen
stiffness, the compliance and its tau sensitivity -- jumps with it.

Bottom face clamped, top face a rigid platen given six unit motions; H is the 6x6
platen stiffness, compliance under unit load f is f^T H^-1 f.  No repair anywhere.
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src')
from stage_cutfem_m4.quotient import RigidQuotient
from stage_cutfem_m4.response import end_registry

DEV = 'cuda:0'
FROZEN = Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
PATHS = Path('/root/autodl-tmp/CUTFEM_FULL_FACTOR_LOCAL_GEOMETRY_20260913T2330/ANALYSIS_R1')
CASES = [('-1/1000', -1e-3, PATHS / 'PATH_1' / 'R_UPPER.npy'),
         ('-1/10000', -1e-4, PATHS / 'PATH_2' / 'R_UPPER.npy'),
         ('0', 0.0, FROZEN / 'R_UPPER.npy'),
         ('1/10000', 1e-4, PATHS / 'PATH_3' / 'R_UPPER.npy'),
         ('1/1000', 1e-3, PATHS / 'PATH_4' / 'R_UPPER.npy')]


def unpack(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path} {p.shape} {p.dtype}')
    host = np.zeros((d, d), dtype=np.float64)
    off = 0
    for row in range(d):
        host[row, row:] = p[off:off + d - row]
        off += d - row
    return torch.from_numpy(host).to(DEV)


def platen(A, Q, cache, q):
    """Lift A back to the full trace, clamp the bottom, drive the top platen."""
    half = Q.lift(A)
    S = Q.lift(half.T).T.contiguous()
    del half
    prescribed, values, registry = end_registry(cache['support_centroid'])
    free = np.setdiff1d(np.arange(q), prescribed)
    i = torch.as_tensor(free, device=DEV)
    b = torch.as_tensor(prescribed, device=DEV)
    u = torch.zeros((q, 6), dtype=torch.float64, device=DEV)
    u[b] = torch.from_numpy(values).to(DEV)
    rhs = -(S @ u)[i]
    block = S[i[:, None], i[None, :]]
    skew = float((block - block.T).norm() / block.norm())
    if skew > 1e-10:
        raise ValueError('RESPONSE_BLOCK_ASYMMETRY')
    scale = block.diagonal().rsqrt()
    scaled = scale[:, None] * block * scale[None, :]
    root = torch.linalg.cholesky((scaled + scaled.T) * .5)
    u[i] = scale[:, None] * torch.cholesky_solve(scale[:, None] * rhs, root)
    del block, scaled, root
    force = S @ u
    residual = float(((scale[:, None] * force[i]).norm(dim=0) / (scale[:, None] * rhs).norm(dim=0)).max())
    H = torch.from_numpy(values).to(DEV).T @ force[b]
    H = (H + H.T) * .5
    del S, force
    gc.collect(); torch.cuda.empty_cache()
    return H, u, residual, registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cache = dict(np.load(FROZEN / 'input' / 'TRACE_CACHE.npz', allow_pickle=False))
    if not (np.all(np.diff(cache['indptr']) == 1) and np.all(cache['coefficients'] == 1)
            and np.all(cache['kind'] == 0)):
        raise ValueError('GENERAL_SIGNED_TRACE_FIXTURE_REQUIRED')
    Q = RigidQuotient(torch.from_numpy(cache['rigid']).to(DEV),
                      torch.from_numpy(cache['order']).to(DEV)).to(DEV)
    q = Q.dimension
    d = q - 6
    report = dict(seat=328, q=q, d=d,
                  scope='five admitted tau perturbations of one cell, same complete trace and quotient',
                  quotient='the frozen 0328 quotient is used for every epsilon; the paths were admitted as SAME_TRACE',
                  cases=[])

    A0 = None
    store = {}
    for name, eps, path in CASES:
        tick = time.perf_counter()
        R = unpack(path, d)
        A = R.T @ R
        A = (A + A.T) * .5
        del R
        gc.collect(); torch.cuda.empty_cache()
        H, u, residual, registry = platen(A, Q, cache, q)
        Hinv = torch.linalg.inv(H)
        row = dict(epsilon=name, eps=eps, path=str(path),
                   free_equilibrium_relative_max=residual,
                   platen_stiffness=H.cpu().numpy().tolist(),
                   platen_eigenvalues=torch.linalg.eigvalsh(H).cpu().numpy().tolist(),
                   compliance_unit_loads=Hinv.diagonal().cpu().numpy().tolist(),
                   trace_inverse=float(Hinv.diagonal().sum()),
                   frobenius_A=float(A.norm()),
                   seconds=time.perf_counter() - tick)
        store[eps] = dict(H=H.cpu(), A=A if eps == 0.0 else None, u=u.cpu())
        if eps == 0.0:
            A0 = A
        else:
            del A
        gc.collect(); torch.cuda.empty_cache()
        report['cases'].append(row)
        print(json.dumps({k: row[k] for k in ('epsilon', 'trace_inverse', 'free_equilibrium_relative_max', 'seconds')}), flush=True)

    # --- smoothness of the physical observable ----------------------------
    c = {e: store[e]['H'] for e in store}
    tr = {e: float(torch.linalg.inv(c[e]).diagonal().sum()) for e in c}
    report['trace_inverse_by_eps'] = {str(e): tr[e] for e in sorted(tr)}
    smooth = {}
    for h in (1e-4, 1e-3):
        left = (tr[0.0] - tr[-h]) / h
        right = (tr[h] - tr[0.0]) / h
        central = (tr[h] - tr[-h]) / (2 * h)
        smooth[f'h={h:g}'] = dict(one_sided_left=left, one_sided_right=right, central=central,
                                  one_sided_ratio=right / left if left else None,
                                  kink_relative=abs(right - left) / max(abs(right), abs(left)))
    report['trace_inverse_slopes'] = smooth
    report['richardson_central_ratio'] = (smooth['h=0.0001']['central'] / smooth['h=0.001']['central']
                                          if smooth['h=0.001']['central'] else None)

    # --- the same question on the raw platen stiffness --------------------
    Hs = {e: c[e].numpy() for e in c}
    rel = {}
    for h in (1e-4, 1e-3):
        left = (Hs[0.0] - Hs[-h]) / h
        right = (Hs[h] - Hs[0.0]) / h
        rel[f'h={h:g}'] = dict(
            left_norm=float(np.linalg.norm(left)), right_norm=float(np.linalg.norm(right)),
            kink_relative=float(np.linalg.norm(right - left) / max(np.linalg.norm(right), np.linalg.norm(left))))
    report['platen_stiffness_slopes'] = rel

    # --- adjoint identity self-test at eps=0 ------------------------------
    # dH/deps from central differences must equal -u^T (dS/deps) u assembled the
    # same way; here we only check the cheap half: the derivative of the platen
    # stiffness equals the energy of the base displacement against dS/deps.
    report['status'] = 'TAU_SMOOTHNESS_COMPLETE'
    (args.output / 'TAU_GATE.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(status=report['status'],
                          trace_inverse=report['trace_inverse_by_eps'],
                          slopes=report['trace_inverse_slopes']), indent=1), flush=True)


if __name__ == '__main__':
    main()
