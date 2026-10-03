#!/usr/bin/env python3
"""Apply the storage floor, per stiffness mode, to the CURRENT GP teacher.

docs/STORAGE_FLOOR_ACCEPTANCE_20260914.md established the criterion two days ago and it has not
been connected to any acceptance number used since.  For a trace direction z,

    E = ||R z||^2                                  the energy the stored factor represents
    N = sum_r ( 0.5 eps sum_j |R_rj z_j| )^2       the energy FP64 storage of R cannot resolve
    E/N                                            how far the value stands above its own floor

with the achievable relative energy error about 1/(E/N), a measured deviation constant C = 3, and
UNJUDGEABLE = 30 -- below which C/(E/N) alone spans a +-10% band, so an out-of-band ratio says
nothing about the operator.  That doc computed it on four screen directions of the no-GP packets.
Here it is computed for EVERY stiffness eigenmode of the GP teacher's quotient operator, which is
what today's +-3%-on-every-mode gate is actually being applied to.

E_i = lam_i for a unit eigenvector, and N is one dense |R| @ |V| away, so this is two matmuls.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
sys.path.insert(0, '/root/autodl-tmp/NEURAL_SCHUR')
import v1_scaled as V
from stage_cutfem_neural_a.elimination_reference import load_upper_factor

EPS = float(np.finfo(np.float64).eps)
C_DEV, UNJUDGEABLE = 3.0, 30.0
SEATS = [int(s) for s in sys.argv[1:]] or [328]


def main():
    out = {}
    for seat in SEATS:
        t0 = time.time()
        recs = json.load(open('/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/V1_LABELS.json'))
        rec = [r for r in recs if int(r['seat']) == seat][0]
        rec.setdefault('reference', '/root/autodl-tmp/CUTFEM_SUPERELEMENT_LABELS/REFERENCE_%04d' % seat)
        label = V.Label(rec, 0.2, 0.03, 10.0)
        g = label.to_gpu(need_A=False, need_Z=False, z_dtype=torch.float32)
        d = label.d
        R = load_upper_factor(Path(rec['reference']) / 'R_UPPER.npy', d, V.DEV)
        A = R.T @ R
        lam, Vec = torch.linalg.eigh(A)          # unit eigenvectors, ascending eigenvalues
        del A; torch.cuda.empty_cache()
        absR = R.abs()
        a = absR @ Vec.abs()                     # (|R| |v|)_r for every mode at once
        del absR; torch.cuda.empty_cache()
        N = ((0.5 * EPS * a) ** 2).sum(0)
        del a; torch.cuda.empty_cache()
        E = lam.clamp(min=0)                     # E_i = v^T A v = lam_i on a unit eigenvector
        sn = torch.where(N > 0, E / N, torch.full_like(E, float('inf')))
        del Vec; torch.cuda.empty_cache()

        snc = sn.cpu().numpy(); lamc = lam.cpu().numpy()
        unj = int((snc < UNJUDGEABLE).sum())
        print('\n===== seat %04d   d = %d   (%.0f s) =====' % (seat, d, time.time()-t0), flush=True)
        print('stiffness range  lam_min %.4e   lam_max %.4e   cond %.4e' % (lamc[0], lamc[-1], lamc[-1]/lamc[0]), flush=True)
        print('modes below the storage floor (E/N < %.0f): %d of %d  (%.2f%%)'
              % (UNJUDGEABLE, unj, d, 100*unj/d), flush=True)
        print('\n%-10s %-13s %-13s %-13s %s' % ('rank', 'lam', 'E/N', 'C/(E/N)', 'judgeable?'), flush=True)
        for r in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, d//4, d//2, d-1]:
            s = snc[r]
            print('%-10d %-13.4e %-13.4e %-13.4e %s'
                  % (r, lamc[r], s, C_DEV/s if s > 0 else float('inf'),
                     'yes' if s >= UNJUDGEABLE else 'NO -- rounding alone spans the band'), flush=True)
        # how much of the spectrum is unjudgeable, and where it sits
        if unj:
            print('\nthe unjudgeable set: ranks 0..%d, stiffness up to %.4e (%.2e of lam_max)'
                  % (unj-1, lamc[unj-1], lamc[unj-1]/lamc[-1]), flush=True)
        else:
            print('\nno mode of this teacher is below the storage floor', flush=True)
        for thr in (0.03, 0.10):
            k = int((C_DEV / np.maximum(snc, 1e-300) > thr).sum())
            print('modes where rounding alone can produce a deviation > %.0f%%: %d (%.2f%%)'
                  % (100*thr, k, 100*k/d), flush=True)
        out[str(seat)] = dict(d=int(d), lam_min=float(lamc[0]), lam_max=float(lamc[-1]),
                              unjudgeable=unj, C=C_DEV, threshold=UNJUDGEABLE,
                              n_rounding_can_exceed_3pct=int((C_DEV/np.maximum(snc,1e-300) > 0.03).sum()),
                              n_rounding_can_exceed_10pct=int((C_DEV/np.maximum(snc,1e-300) > 0.10).sum()),
                              sn_percentiles={str(p): float(np.percentile(snc, p)) for p in (0, 1, 5, 25, 50, 100)},
                              lam=[float(x) for x in lamc[::max(1, d//200)]],
                              sn=[float(x) for x in snc[::max(1, d//200)]])
        np.save('/root/autodl-tmp/NEURAL_SCHUR/SN_%04d.npy' % seat, snc)
        np.save('/root/autodl-tmp/NEURAL_SCHUR/LAM_%04d.npy' % seat, lamc)
        del R, lam, N, E, sn; torch.cuda.empty_cache()
    Path('/root/autodl-tmp/NEURAL_SCHUR/FLOOR_MODES.json').write_text(json.dumps(out, indent=1))
    print('\nwritten FLOOR_MODES.json', flush=True)


if __name__ == '__main__':
    main()
