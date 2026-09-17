"""The sign of det(C) does not affect W = C^T C, so it must not gate the run.

The symmetric head is constrained to be symmetric with a positive diagonal, not
to be positive definite, so a predicted root may be indefinite.  A_hat = M M is
positive semidefinite either way and eps_op is well defined; only the log
determinant needed care, and log|det C| is the right quantity because
det(W) = det(C)^2.  The sign is kept as a diagnostic of the predicted root.
"""
import sys
from pathlib import Path

for root in sys.argv[1:]:
    p = Path(root) / 'stage_cutfem_m4' / 'mechanics.py'
    t = p.read_text()
    old = """    sign, gap = torch.linalg.slogdet(C)
    if float(sign) <= 0:
        raise ValueError("SINGULAR_OR_ORIENTATION_REVERSING_RELATIVE_FACTOR")
    gap = 2 * gap
    trace = C.square().sum()
    D = trace - gap - C.shape[0]
    return dict(D=D, D_per_mode=D / C.shape[0], trace=trace, logdet_gap=gap)"""
    new = """    sign, gap = torch.linalg.slogdet(C)
    if not bool(torch.isfinite(gap)):
        raise ValueError("SINGULAR_RELATIVE_FACTOR")
    gap = 2 * gap
    trace = C.square().sum()
    D = trace - gap - C.shape[0]
    return dict(D=D, D_per_mode=D / C.shape[0], trace=trace, logdet_gap=gap,
                relative_factor_determinant_sign=float(sign))"""
    if old not in t:
        if 'relative_factor_determinant_sign' in t:
            print(f'  already fixed {root}')
            continue
        raise SystemExit(f'ANCHOR_MISSING {p}')
    p.write_text(t.replace(old, new))
    print(f'  fixed {p}')

    r = Path(root) / 'stage_cutfem_m4' / 'run.py'
    t = r.read_text()
    old2 = """        head=head,divergence_consistency_tolerance="""
    new2 = """        head=head,root_determinant_sign=float(div.get('relative_factor_determinant_sign',1.)),
        divergence_consistency_tolerance="""
    if old2 in t:
        r.write_text(t.replace(old2, new2, 1))
        print(f'  surfaced sign in {r}')
    elif 'root_determinant_sign' in t:
        print(f'  sign already surfaced in {r}')
    else:
        raise SystemExit(f'ANCHOR_MISSING {r}')
print('DIVERGENCE_FIX_COMPLETE')
