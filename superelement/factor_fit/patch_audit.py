#!/usr/bin/env python3
"""Add a gap audit to v1_scaled.py.

Three runs have now stalled the same way: the rigid-correction solve residual floors just above
whatever --solve-tol is set to (1e-6 -> 1.03e-6, 1e-4 -> 1.03e-4), reject-ill fires on every
step, lr_scale walks to its floor and the model freezes.  The residual floor is not noise, it
tracks cond(L), which necessarily grows as the student stiffens, so any fixed absolute gate
stalls eventually.

But the residual may be the wrong gate entirely.  It is measured as ||S_hat U - N|| / ||N||
with U = solve_full(N), and `apply` multiplies U back through L, so the check amplifies U's
error by cond(L).  What the objective actually consumes is logdet(N^T U), which depends on U
directly, not on U pushed back through L.  If the log-det is stable while the residual is not,
the gate is measuring the wrong thing and should be replaced by a stability check on the
quantity that matters.

This patch adds --gap-audit K: every K steps, report residual and logdet(N^T U) at several
iterative-refinement counts, so the two can be compared directly.
"""
import sys
from pathlib import Path

P = Path('/root/cutfem_neural_a_20260910/superelement_v0/v1_scaled.py')
t = P.read_text()

FUNC = '''
@torch.no_grad()
def gap_audit(g, op, refines=(0, 1, 2, 4, 8)):
    """Separate the solve residual from the quantity the objective consumes.

    The rejection gate reads ||S_hat U - N|| / ||N||, which pushes U back through L and so is
    amplified by cond(L) relative to U's own error.  The objective consumes logdet(N^T U).
    This reports both at a range of refinement counts: if the log-det is stable across
    refinements while the residual sits at its floor, the residual is not evidence that the
    objective is corrupted, and gating on it is what stalls the run."""
    N = g['rigid']; nN = torch.linalg.matrix_norm(N)
    U = op.solve_full(N); rows = []; top = max(refines)
    for k in range(top + 1):
        if k in refines:
            r = float(torch.linalg.matrix_norm(op.apply(U) - N) / nN)
            sign, ld = torch.linalg.slogdet(N.T @ U)
            rows.append(dict(refine=k, residual=r, logdet=float(ld), sign=int(sign)))
        if k < top:
            U = U + op.solve_full(N - op.apply(U))
    base = rows[0]['logdet']; best = rows[-1]['logdet']
    return dict(rows=rows, logdet_drift=abs(best - base),
                drift_rel=abs(best - base) / max(abs(best), 1e-300))


'''
A1 = '@torch.no_grad()\ndef factor_conditioning(op, iters=30):'
assert A1 in t, 'anchor 1 missing'
t = t.replace(A1, FUNC.lstrip('\n') + A1, 1)

A2 = "    ap.add_argument('--reject-floor', type=float, default=1e-3, help='lower bound of the lr multiplier applied by rejections')\n"
assert A2 in t, 'anchor 2 missing'
t = t.replace(A2, A2 + "    ap.add_argument('--gap-audit', type=int, default=0, help='every K steps, log the rigid log-det and its solve residual at several refinement counts, to test whether the residual gate is measuring the quantity the objective consumes')\n", 1)

A3 = "                history.append(row); append_json(out / 'HISTORY.jsonl', row)\n"
assert A3 in t, 'anchor 3 missing'
t = t.replace(A3, "                if args.gap_audit and args.divergence_weight > 0 and step % args.gap_audit == 0:\n"
                  "                    print(json.dumps(dict(stage='gap_audit', step=step, seat=l.seat, **gap_audit(g, op))), flush=True)\n" + A3, 1)

P.write_text(t)
print('patched: gap_audit function, --gap-audit flag, call site')
import subprocess
print(subprocess.run([sys.executable, '-c', 'import ast,sys; ast.parse(open(sys.argv[1]).read())', str(P)],
                     capture_output=True, text=True).stderr or 'syntax OK')
