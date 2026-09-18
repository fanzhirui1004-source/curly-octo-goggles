"""--head inverse: train the existing upper-triangular head on the factor of A^-1.

Training needs nothing new.  L is upper triangular with a strictly positive diagonal,
exactly like R_*, so the head, the sampler and the loss are untouched; only the label
file changes.  Two legacy constants do have to move, because the pivots of A^-1's
factor are the reciprocals of A's: for seat 0253 the log pivots run +1.92 to +7.08
where R_*'s run -7.10 to -1.95, mirrored almost exactly (mean +3.31303 against
-3.31303, std 1.3327 against 1.3275).  The (-20, 5) window was a runaway guard, not
physics, so it becomes symmetric.

Evaluation is NOT threaded through evaluate(): the inverse head needs a different
spectrum route and a triangular inversion to recover A_hat, so it gets its own script
and these runs are launched with an empty --eval-seats.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) / 'stage_cutfem_m4'


def edit(path, old, new, count=1):
    t = path.read_text()
    n = t.count(old)
    if n != count:
        raise SystemExit(f'PATCH_ANCHOR {path.name}: expected {count}, found {n}\n---\n{old}\n---')
    path.write_text(t.replace(old, new))
    print(f'  ok {path.name}: {old.splitlines()[0][:62]}')


factors = ROOT / 'factors.py'
edit(factors, """def calibrate(samples, patch_size, seed=20260917, per_bucket=32768):""",
"""def calibrate(samples, patch_size, seed=20260917, per_bucket=32768, log_pivot_window=(-20., 5.)):""")
edit(factors, """    if log.min()<=-20 or log.max()>=5:raise ValueError('REFERENCE_OUTSIDE_LEGACY_LOG_PIVOT_BOUNDS')""",
"""    if log.min()<=log_pivot_window[0] or log.max()>=log_pivot_window[1]:
        raise ValueError('REFERENCE_OUTSIDE_LOG_PIVOT_WINDOW')""")

run = ROOT / 'run.py'
edit(run, """    ap.add_argument('--head',choices=['chol','sqrt'],default='chol')""",
"""    ap.add_argument('--head',choices=['chol','sqrt','inverse'],default='chol')""")
edit(run, """    label=factor
    if head=='sqrt':""",
"""    label=factor
    if head=='inverse':
        label=reference/'G_UPPER.npy';inv=json.loads((reference/'INVERSE_RESULT.json').read_text())
        if (inv['r_sha256']!=bindings['factor_sha256'] or inv['g_sha256']!=sha256(label)
            or int(inv['d'])!=int(receipt['dimension'])
            or inv['read_blocks_roundtrip']!=0.0):raise ValueError('FROZEN_INVERSE_LABEL_BINDING')
        bindings['inverse_label_sha256']=inv['g_sha256']
    if head=='sqrt':""")
edit(run, """        conditioning,stats=calibrate(train,args.patch_size,per_bucket=args.calibrate_pairs)""",
"""        window=(-20.,20.) if args.head=='inverse' else (-20.,5.)
        conditioning,stats=calibrate(train,args.patch_size,per_bucket=args.calibrate_pairs,
                                     log_pivot_window=window)""")
edit(run, """        model=model_for(samples[0]['context']).to('cuda:0')""",
"""        model=model_for(samples[0]['context']).to('cuda:0')
        if args.head=='inverse':
            # the pivots of A^-1's factor are the reciprocals of A's, so the runaway
            # guard has to be symmetric; it is a bound, not a modelling choice.
            model.log_pivot_lower,model.log_pivot_upper=-20.,20.""")
edit(run, """        head=args.head,diagonal_loss=args.diagonal_loss,""",
"""        head=args.head,diagonal_loss=args.diagonal_loss,
        log_pivot_window=[-20.,20.] if args.head=='inverse' else [-20.,5.],
        inverse_head_note='inverse: the target is the upper Cholesky factor L of A^-1, so '
            'A_hat^-1 = L^T L and A_hat acts by two triangular solves; the dense Schur '
            'operator is never formed. Evaluation is a separate script.',""")
print('INVERSE_PATCH_COMPLETE')
