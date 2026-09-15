#!/usr/bin/env python3
"""Pin the rigid block, which the objective is flat along and the arithmetic is not.

A_hat = B S_hat B^T depends only on S_hat's compression to the complement of the rigid modes.
S_hat's rigid block and its rigid-to-quotient coupling are a gauge: changing them cannot change
the objective's true value.  In the identity actually used,

    logdet(B S_hat B^T) = logdet(S_hat) + logdet(N^T S_hat^-1 N) - logdet(N^T N)

a rigid eigenvalue shrinking by exp(-x) moves the two right-hand terms by -x and +x, so they
cancel exactly on paper.  They do not cancel in floating point: logdet(S_hat) is read off the
diagonals of L and C and is exact, while logdet(N^T S_hat^-1 N) comes from a solve whose
accuracy collapses as the rigid eigenvalues do.  The second term therefore lags, the
cancellation leaves a spurious descent direction, and following it makes the solve worse still.

R7 measured the result: six eigenvalues shrank by 3.2e-40 each over 300 steps, the rigid solve
residual went from 1.3e-15 to 7.2e+29, and past about step 225 the objective's value and
gradient stopped being trustworthy.

This pins the gauge by holding N^T S_hat N at the value the initial operator had.  Because the
quantity is a gauge, pinning it cannot bias the fit; it only removes the direction along which
the arithmetic degenerates.  Cost is one six-column application of S_hat per step.
"""
from pathlib import Path
import sys, subprocess

P = Path('/root/cutfem_neural_a_20260910/superelement_v0/v1_scaled.py')
t = P.read_text()

A1 = "    ap.add_argument('--gap-audit', type=int, default=0,"
assert A1 in t, 'gap-audit flag missing'
t = t.replace(A1, "    ap.add_argument('--gauge-weight', type=float, default=0.0, help=\"weight on holding N^T S_hat N at its initial value; S_hat's rigid block is a gauge of B S_hat B^T, so pinning it cannot bias the fit, but leaving it free lets the rigid eigenvalues collapse and the log-det identity degenerate with them\")\n" + A1, 1)

# loss slot: declare the gauge loss next to the other zeroed losses each step
A2 = "                    residual = 0.0\n"
assert A2 in t, 'residual init anchor missing'
t = t.replace(A2, "                    residual = 0.0; loss_g = torch.zeros((), dtype=torch.float64, device=DEV)\n"
                  "                    if args.gauge_weight > 0:\n"
                  "                        SN = op.apply(g['rigid']); G = g['rigid'].T @ SN\n"
                  "                        if gauge_ref is None: gauge_ref = G.detach().clone()\n"
                  "                        loss_g = (G - gauge_ref).square().sum() / gauge_ref.square().sum()\n", 1)

A3 = "                loss = args.energy_weight * w_e * loss_e + args.action_weight * loss_a + args.dual_weight * loss_d + args.compliance_weight * loss_c + args.divergence_weight * loss_D + args.extreme_weight * loss_x\n"
assert A3 in t, 'loss anchor missing'
t = t.replace(A3, A3.rstrip('\n') + " + args.gauge_weight * loss_g\n", 1)

A4 = "    snapshot = None; rejected = 0; reject_scale = 1.0\n"
assert A4 in t, 'state anchor missing'
t = t.replace(A4, A4.rstrip('\n') + "; gauge_ref = None\n", 1)

A5 = "                if args.divergence_weight > 0: row.update(divergence_per_mode=float(loss_D.detach()),"
assert A5 in t, 'row anchor missing'
t = t.replace(A5, "                if args.gauge_weight > 0: row.update(gauge_loss=float(loss_g.detach()))\n" + A5, 1)

P.write_text(t)
print('patched: --gauge-weight, N^T S_hat N pin, loss term, logging')
print(subprocess.run([sys.executable, '-c', 'import ast,sys; ast.parse(open(sys.argv[1]).read())', str(P)],
                     capture_output=True, text=True).stderr or 'syntax OK')
for k in ('gauge-weight', 'gauge_ref', 'loss_g', 'gauge_loss'):
    print(' ', k, t.count(k.replace('-', '_')) if '_' in k else t.count(k))
