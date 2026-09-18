"""Give the smallest pivots the same relative weight as the largest, and report g.

Measured on seat 0253's chol arm: the median relative pivot error runs 22.2% on the
smallest decile of true pivots down to 0.52% on the largest -- a 42x spread in the
wrong direction.  Two mechanisms cause it, and they multiply:

  1. the diagonal bucket residual is normalised by ONE global RMS
     (diagonal_block_rms = 0.0453) while the pivots span e^-7.10 .. e^-1.95, so the
     smallest pivot contributes (8.2e-4/0.0453)^2 ~ 3e-4 of a typical term;
  2. the head emits pivots as exp(mean + std * z), so d(pivot)/dz is proportional to
     the pivot, suppressing small-pivot gradients by the same range again.

--diagonal-loss log scores the pivots in the log domain the head already parameterises
them in, which is exactly uniform RELATIVE weight across all five decades.  The
strict-lower entries of diagonal blocks keep their existing normalisation, and the
same-patch and cross-patch buckets are untouched.

Also surfaces g = max(mu_max, 1/mu_min) in SPECTRUM.json, because eps_op = max|mu-1|
is capped at 1 on the soft side and ranked the two stage-1 arms backwards against the
measured assembled sensitivity.
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
    print(f'  ok {path.name}: {old.splitlines()[0][:66]}')


factors = ROOT / 'factors.py'
edit(factors, """def bucket_loss(prediction, target, buckets, conditioning, importance=None):
    scales=(conditioning.diagonal_block_rms,conditioning.same_patch_rms,conditioning.cross_patch_rms)
    errors=[((prediction[buckets==j]-target[buckets==j])/scales[j]).square() for j in range(3)]
    if importance is not None:
        errors=[v*importance[buckets==j,None,None] for j,v in enumerate(errors)]
    pieces=[v.mean() for v in errors]
    return torch.stack(pieces).mean(),torch.stack(pieces).detach()""",
"""def bucket_loss(prediction, target, buckets, conditioning, importance=None, diagonal_loss='absolute'):
    \"\"\"Three equally weighted buckets; the diagonal one optionally scores pivots in log.

    'absolute' is the legacy objective, unchanged to the statement. 'log' replaces the
    single global normaliser on the pivots by uniform relative weight in the log domain
    the head already parameterises them in, leaving every other entry alone.
    \"\"\"
    if diagonal_loss not in ('absolute','log'):raise ValueError('UNKNOWN_DIAGONAL_LOSS')
    scales=(conditioning.diagonal_block_rms,conditioning.same_patch_rms,conditioning.cross_patch_rms)
    errors=[((prediction[buckets==j]-target[buckets==j])/scales[j]).square() for j in range(3)]
    if importance is not None:
        errors=[v*importance[buckets==j,None,None] for j,v in enumerate(errors)]
    if diagonal_loss=='log':
        pd=prediction[buckets==0];td=target[buckets==0]
        if not bool((td.diagonal(dim1=-2,dim2=-1)>0).all()):raise ValueError('NONPOSITIVE_REFERENCE_PIVOT')
        if not bool((pd.diagonal(dim1=-2,dim2=-1)>0).all()):raise ValueError('NONPOSITIVE_PREDICTED_PIVOT')
        ratio=(pd.diagonal(dim1=-2,dim2=-1).log()-td.diagonal(dim1=-2,dim2=-1).log())
        pivot=(ratio/conditioning.pivot_log_std).square()
        lower=((torch.tril(pd,-1)-torch.tril(td,-1))/conditioning.diagonal_lower_rms).square()
        lower=lower[:,torch.tril_indices(3,3,-1,device=lower.device)[0],
                      torch.tril_indices(3,3,-1,device=lower.device)[1]]
        if importance is not None:
            w=importance[buckets==0,None];pivot=pivot*w;lower=lower*w
        errors[0]=torch.cat((pivot,lower),dim=1)
    pieces=[v.mean() for v in errors]
    return torch.stack(pieces).mean(),torch.stack(pieces).detach()""")

run = ROOT / 'run.py'
edit(run, """    ap.add_argument('--head',choices=['chol','sqrt'],default='chol')""",
"""    ap.add_argument('--head',choices=['chol','sqrt'],default='chol')
    ap.add_argument('--diagonal-loss',choices=['absolute','log'],default='absolute',
        help='log scores pivots relatively, in the domain the head already parameterises them in')""")
edit(run, """            loss,parts=bucket_loss(pred,target,bucket,conditioning,importance)""",
"""            loss,parts=bucket_loss(pred,target,bucket,conditioning,importance,args.diagonal_loss)""")
edit(run, """        head=args.head,
        head_definition=""",
"""        head=args.head,diagonal_loss=args.diagonal_loss,
        diagonal_loss_definition='absolute: legacy single-RMS normaliser on all diagonal-block '
            'entries. log: pivots scored as (log p_hat - log p_star)/pivot_log_std, uniform '
            'relative weight across the five decades of pivot; other entries unchanged.',
        head_definition=""")
edit(run, """        mu_min=float(mu[0]),mu_max=float(mu[-1]),D_per_mode=float(div['D_per_mode']),""",
"""        mu_min=float(mu[0]),mu_max=float(mu[-1]),D_per_mode=float(div['D_per_mode']),
        eps_op=max(abs(float(mu[0])-1),abs(float(mu[-1])-1)),
        under_stiff_factor=1./float(mu[0]) if float(mu[0])>0 else None,
        g=max(float(mu[-1]),1./float(mu[0])) if float(mu[0])>0 else None,
        g_definition='max_i max(mu_i, 1/mu_i); the inversion-symmetric two-sided Loewner '
            'sandwich constant. eps_op = max|mu-1| is capped at 1 on the soft side and ranked '
            'the two stage-1 arms backwards against the measured assembled sensitivity.',""")
print('LOGPIVOT_PATCH_COMPLETE')
