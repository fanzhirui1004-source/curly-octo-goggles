#!/usr/bin/env python3
"""If a network predicts every entry independently, how accurate must each entry be?

The most intuitive architecture is to emit S (or its factor) entry by entry -- a shared decoder
queried once per (i, j), which costs no more weights than the decoder itself.  Whether that can
work is not a question about network size; it is a question about how per-entry relative accuracy
converts into the acceptance criterion eps_op = max|mu - 1|, and those are different norms.
Independent per-entry errors add incoherently, so the conversion carries a factor that depends on
d and has to be measured rather than argued.

Two arms, because ERROR_METRIC_20260916 showed the metric matters by 5e4:
    additive        Shat = A + E,     E symmetric, E_ij = eps * |A_ij| * n_ij
    multiplicative  Rhat = R* + E,    E_ij       = eps * |R*_ij| * n_ij
and eps_op is measured on the real whitened spectrum in both.

Read off the row where eps_op crosses 0.03: that is the per-entry accuracy a direct predictor needs.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

EPSS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    label = V.Label(rec, 0.2, 0.03, 10.0)
    label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
    d = label.d
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
    Rinv = torch.linalg.solve_triangular(R, torch.eye(d, dtype=torch.float64, device=V.DEV), upper=True)
    A = R.T @ R
    nzR = int((R != 0).sum()); nzA = int((A != 0).sum())
    print('seat 328  d %d   R* has %.3e nonzeros, A has %.3e' % (d, nzR, nzA), flush=True)
    print('a direct predictor emits one number per nonzero; the question is how accurate each must be\n', flush=True)
    print('%-10s %-14s %-14s' % ('per-entry', 'eps_op  (A)', 'eps_op  (factor R*)'), flush=True)
    torch.manual_seed(0)
    rows = []
    for eps in EPSS:
        # additive, on A, keeping symmetry
        Nz = torch.randn(d, d, dtype=torch.float64, device=V.DEV)
        Nz = torch.triu(Nz); Nz = Nz + torch.triu(Nz, 1).T
        Ah = A + eps * A.abs() * Nz
        del Nz; torch.cuda.empty_cache()
        H = Rinv.T @ Ah @ Rinv; del Ah; torch.cuda.empty_cache()
        mu = torch.linalg.eigvalsh(0.5 * (H + H.T)); del H; torch.cuda.empty_cache()
        eA = float('inf') if bool((mu <= 0).any()) else float((mu - 1).abs().max())
        del mu; torch.cuda.empty_cache()
        # multiplicative, on the factor
        Nz = torch.randn(d, d, dtype=torch.float64, device=V.DEV)
        Rh = R + eps * R.abs() * Nz
        del Nz; torch.cuda.empty_cache()
        X = Rh @ Rinv; del Rh; torch.cuda.empty_cache()
        H = X.T @ X; del X; torch.cuda.empty_cache()
        mu = torch.linalg.eigvalsh(0.5 * (H + H.T)); del H; torch.cuda.empty_cache()
        eR = float((mu - 1).abs().max()); del mu; torch.cuda.empty_cache()
        mark = ''
        if eR <= 0.03: mark += '  factor arm PASSES +-3%'
        elif eR <= 0.10: mark += '  factor arm passes +-10%'
        if eA <= 0.03: mark += '  | A arm PASSES +-3%'
        print('%-10.0e %-14.4e %-14.4e%s' % (eps, eA, eR, mark), flush=True)
        rows.append(dict(per_entry=eps, eps_op_A=eA, eps_op_R=eR))
    Path('/root/autodl-tmp/NEURAL_SCHUR/PER_ENTRY.json').write_text(json.dumps(
        dict(seat=328, d=int(d), nnz_R=nzR, nnz_A=nzA, rows=rows, seconds=time.time()-t0), indent=1))
    print('\nwritten PER_ENTRY.json (%.0f s)' % (time.time()-t0), flush=True)


if __name__ == '__main__':
    main()
