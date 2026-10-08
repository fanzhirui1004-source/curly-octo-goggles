#!/usr/bin/env python3
"""Route 4: does predicting the factor in PRECONDITIONED coordinates change what a
fixed amount of training error costs at the gate?

On 2026-09-15 a free dense factor optimised with Newton/Armijo reached
mu in [0.956, 1.047] (work gate passes) where the same start with Adam in the raw
coordinates went NaN; the column scale 1/sqrt((A^-1)_jj) was the stated key, and it is
predictable from local geometry (within-body R^2 0.981).

Note first an algebraic fact that narrows the question: an entrywise RELATIVE
perturbation commutes with column scaling, so
    (R Dg)(1 + eps N) = (R(1 + eps N)) Dg
and the per-entry relative accuracy requirement is IDENTICAL in both coordinates.
Preconditioning cannot change that. What it can change is the DYNAMIC RANGE of the
target, and therefore what Codex's actual loss - a squared error normalised by ONE
global RMS per bucket - buys in relative terms. In raw coordinates the factor's entries
span orders of magnitude, the RMS is set by the large ones, and the small ones receive
almost no supervision; measured consequence, WHERE_THE_ERROR_IS 4: median per-entry
relative error 5.725 while ||E||/||R|| is only 0.097.

So the experiment mimics the loss, not a relative perturbation:
    raw:           Rhat  = R  + delta * rms(R)  * N
    preconditioned: Rt   = R Dg,  Rthat = Rt + delta * rms(Rt) * N,  Rhat = Rthat Dg^-1
and reports eps_op(delta) in each. Equal delta = equal loss-measured error. If the
preconditioned column is far lower, the coordinate change is worth making.

Self-tests: ||R Dg (R Dg)^T - Dg A Dg|| ~ 0; diag((Dg A Dg)^-1) == 1; delta = 0 -> ~1e-13.
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor
DEV = V.DEV; F64 = torch.float64
OUT = Path('/root/autodl-tmp/CLAUDE_PRECOND_20260917'); OUT.mkdir(exist_ok=True)


def eps_op(Rh, Rs, d):
    Mt = torch.linalg.solve_triangular(Rs.T, Rh.T, upper=False); M = Mt.T.contiguous(); del Mt
    W = M.T @ M; del M; gc.collect(); torch.cuda.empty_cache()
    mu = torch.linalg.eigvalsh((W + W.T) * .5); del W; gc.collect(); torch.cuda.empty_cache()
    mn, mx = float(mu.min()), float(mu.max()); del mu
    return max(abs(mn - 1), abs(mx - 1)), mn, mx


def main():
    t0 = time.time()
    recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
    rec = [r for r in recs if int(r['seat']) == 328][0]
    rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0328')
    d = None
    R = None
    label = V.Label(rec, 0.2, 0.03, 10.0); d = label.d
    R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, DEV)
    print(f'seat 328 d={d}', flush=True)

    I = torch.eye(d, dtype=F64, device=DEV)
    Rinv = torch.linalg.solve_triangular(R, I, upper=True)        # R^-1, A^-1 = R^-1 R^-T
    ainv = (Rinv * Rinv).sum(1)                                   # diag(A^-1)
    adiag = (R * R).sum(0)                                        # diag(A) = diag(R^T R)
    del I, Rinv; gc.collect(); torch.cuda.empty_cache()
    up = torch.triu(torch.ones((d, d), dtype=torch.bool, device=DEV))

    def stats(M, name):
        v = M[up].abs(); v = v[v > 0]
        gg = torch.Generator(device=DEV).manual_seed(1)
        sm = v[torch.randint(0, v.numel(), (8_000_000,), generator=gg, device=DEV)].float()
        q = [float(torch.quantile(sm, x)) for x in (0.01, 0.5, 0.99)]
        return dict(p01=q[0], p50=q[1], p99=q[2], spread=q[2] / max(q[0], 1e-300),
                    rms=float(M[up].pow(2).mean().sqrt()), name=name)

    # Four candidate column scalings of the FACTOR: Rt = R * dg, so Atilde = Dg A Dg.
    scalings = {
        'raw':                 torch.ones_like(ainv),
        'inv_sqrt_Ainv_jj':    ainv.clamp_min(1e-300).rsqrt(),    # as REVIEW states it
        'sqrt_Ainv_jj':        ainv.clamp_min(1e-300).sqrt(),     # makes diag(Atilde^-1) = 1
        'inv_sqrt_A_jj':       adiag.clamp_min(1e-300).rsqrt(),   # makes diag(Atilde)   = 1
    }
    out = dict(seat=328, d=int(d), scalings={}, rows=[])
    print(f"  {'scaling':>20}{'range':>12}{'|R dg| spread 99/1':>22}{'diag(At^-1) max|-1|':>22}", flush=True)
    for name, dgv in scalings.items():
        Rt = R * dgv.unsqueeze(0)
        st = stats(Rt, name)
        chk = float((ainv / dgv.pow(2)).sub(1).abs().max())
        out['scalings'][name] = dict(st, diag_inv_check=chk,
                                     range=float(dgv.max() / dgv.min()))
        print(f"  {name:>20}{float(dgv.max()/dgv.min()):>12.4g}{st['spread']:>22.4e}{chk:>22.4e}",
              flush=True)
        del Rt; gc.collect(); torch.cuda.empty_cache()

    print('\n  delta is an ABSOLUTE perturbation in units of the coordinate RMS '
          '(what a bucket-normalised squared loss controls)', flush=True)
    print(f"  {'delta':>8}{'raw eps_op':>16}{'precond eps_op':>18}{'ratio':>10}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(20260917)
    hdr = f"  {'delta':>8}" + ''.join(f'{k:>20}' for k in scalings)
    print(hdr, flush=True)
    for delta in (0.0, 1e-4, 1e-3, 1e-2, 3e-2):
        N = torch.triu(torch.randn(R.shape, generator=gen, device=DEV, dtype=F64))
        row = dict(delta=delta, eps_op={})
        line = f'  {delta:>8.0e}'
        for name, dgv in scalings.items():
            rms = out['scalings'][name]['rms']
            Ph = (R * dgv.unsqueeze(0) + delta * rms * N) / dgv.unsqueeze(0)
            v, _, _ = eps_op(Ph, R, d); del Ph; gc.collect(); torch.cuda.empty_cache()
            row['eps_op'][name] = v
            line += f'{v:>20.6g}'
        del N; gc.collect(); torch.cuda.empty_cache()
        out['rows'].append(row); print(line, flush=True)
        (OUT / 'RESULT.json').write_text(json.dumps(out, indent=1))
    print(f'\ndone ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
