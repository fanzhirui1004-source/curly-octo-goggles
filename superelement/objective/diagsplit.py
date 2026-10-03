"""Where does eps_op live: the pivots, or the off-diagonals?

Codex's bucket substitution split the factor by DISTANCE (diag / same-patch /
cross-patch). This splits it by ROLE instead: the diagonal of R (the pivots, which
alone determine logdet H and which the network emits as an explicit bounded
log_pivot) versus everything above it.

Four cases per (arm, seat), all on the SAVED predicted factors - no training, no
model, nothing of Codex's touched:
  as_predicted        R_hat unchanged                     (must reproduce their numbers)
  teacher_diagonal    R_hat with the teacher's diagonal   (pivot error removed)
  teacher_offdiag     teacher off-diagonal, R_hat's diag  (off-diagonal error removed)
  teacher_all         the teacher                          (must give mu == 1)
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch

OUT = Path('/root/autodl-tmp/CLAUDE_DIAGSPLIT_20260917'); OUT.mkdir(exist_ok=True)
RUN = Path('/root/autodl-tmp/CUTFEM_M4_MULTI_20260917/RUN_TRAIN32_ACCEL_V1')
REF = Path('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS')
DEV = 'cuda:0'


def unpack(path, d):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    if p.shape != (d * (d + 1) // 2,) or p.dtype != np.float64:
        raise ValueError(f'FP64_UPPER_REQUIRED {path} {p.shape}')
    R = torch.zeros((d, d), dtype=torch.float64, device=DEV)
    off = 0
    for row in range(d):
        n = d - row
        R[row, row:] = torch.from_numpy(np.ascontiguousarray(p[off:off + n])).to(DEV)
        off += n
    return R


def spectrum(Rh, Rs, d):
    """mu = eig(R_s^-T R_h^T R_h R_s^-1). M = R_h R_s^-1 via one triangular solve."""
    Mt = torch.linalg.solve_triangular(Rs.T, Rh.T, upper=False)     # R_s^-T R_h^T
    M = Mt.T.contiguous(); del Mt
    W = M.T @ M
    trace_per_mode = float(torch.diagonal(W).sum() / d)
    logdet_gap = float(2 * (torch.log(torch.diagonal(Rh)).sum() - torch.log(torch.diagonal(Rs)).sum()))
    D_closed = trace_per_mode - logdet_gap / d - 1.0
    e_A = float((M.T @ M - torch.eye(d, dtype=torch.float64, device=DEV)).norm() /
                torch.eye(d, dtype=torch.float64, device=DEV).norm()) if d < 0 else None
    del M
    gc.collect(); torch.cuda.empty_cache()
    mu = torch.linalg.eigvalsh((W + W.T) * .5)
    del W; gc.collect(); torch.cuda.empty_cache()
    mn, mx = float(mu.min()), float(mu.max())
    pos = mu.clamp_min(1e-300)
    return dict(mu_min=mn, mu_max=mx, eps_op=max(abs(mn - 1), abs(mx - 1)),
                D_closed=D_closed, D_spectral=float((pos - torch.log(pos) - 1).mean()),
                trace_per_mode=trace_per_mode, logdet_gap=logdet_gap,
                below_0_97=int((mu < .97).sum()), above_1_03=int((mu > 1.03).sum()),
                below_0_9=int((mu < .9).sum()), above_1_1=int((mu > 1.1).sum()))


def run(arm, seat, d):
    Rs = unpack(REF / f'REFERENCE_{seat:04d}' / 'R_UPPER.npy', d)
    Rh = unpack(RUN / arm / f'EVAL_040000_{seat:04d}' / 'R_PRED_UPPER.npy', d)
    ds, dh = torch.diagonal(Rs).clone(), torch.diagonal(Rh).clone()
    pivot_log_gap = (torch.log(dh) - torch.log(ds))
    results = {'pivot_log_gap': dict(
        mean=float(pivot_log_gap.mean()), std=float(pivot_log_gap.std()),
        q01=float(pivot_log_gap.quantile(.01)), q50=float(pivot_log_gap.median()),
        q99=float(pivot_log_gap.quantile(.99)), min=float(pivot_log_gap.min()),
        max=float(pivot_log_gap.max()))}
    cases = {}
    for name in ('as_predicted', 'teacher_diagonal', 'teacher_offdiag', 'teacher_all'):
        t = time.time()
        if name == 'as_predicted':
            R = Rh
        elif name == 'teacher_diagonal':
            R = Rh - torch.diag_embed(dh) + torch.diag_embed(ds)
        elif name == 'teacher_offdiag':
            R = Rs - torch.diag_embed(ds) + torch.diag_embed(dh)
        else:
            R = Rs
        cases[name] = spectrum(R, Rs, d)
        cases[name]['seconds'] = time.time() - t
        if R is not Rh and R is not Rs:
            del R
        gc.collect(); torch.cuda.empty_cache()
        print(f'  {arm:<9} {seat:04d} {name:<17} '
              f"mu=[{cases[name]['mu_min']:.4g},{cases[name]['mu_max']:.4g}] "
              f"eps_op={cases[name]['eps_op']:.4g} D={cases[name]['D_closed']:.5g} "
              f"({cases[name]['seconds']:.0f}s)", flush=True)
    results['cases'] = cases
    del Rs, Rh; gc.collect(); torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    def dimension(path):
        n = np.load(path, mmap_mode='r', allow_pickle=False).shape[0]
        d = int((-1 + (1 + 8 * n) ** .5) / 2 + .5)
        if d * (d + 1) // 2 != n:
            raise ValueError('PACKED_LENGTH_NOT_TRIANGULAR')
        return d
    dims = {s: dimension(REF / f'REFERENCE_{s:04d}' / 'R_UPPER.npy') for s in (253, 403)}
    print('dimensions', dims, flush=True)
    out = {}
    for seat in (253, 403):
        for arm in ('baseline', 'relative'):
            print(f'=== {arm} seat {seat} d={dims[seat]} ===', flush=True)
            out[f'{arm}_{seat:04d}'] = run(arm, seat, dims[seat])
            (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1))
    print('done')
