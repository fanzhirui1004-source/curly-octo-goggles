#!/usr/bin/env python3
"""Move the gap audit before opt.step().

FreeScaledModel.forward returns self.M, the live nn.Parameter, so Operator.M IS that tensor and
opt.step() mutates it in place.  Operator.C = cholesky(I + M^T M) is built once at construction,
so after opt.step() the operator holds post-step M against pre-step C.  The audit was called
after opt.step() and so measured that inconsistent operator, not the one the loss was computed
from: its refine-0 residual read 1.3e-2 while the training path's own residual for the same step
read 1.3e-14.

Move the call to just before loss.backward(), where the operator is still the one the objective
used, and log the training path's residual alongside so the two can be checked against each
other directly."""
from pathlib import Path
import sys, subprocess

P = Path('/root/cutfem_neural_a_20260910/superelement_v0/v1_scaled.py')
t = P.read_text()

OLD = ("                if args.gap_audit and args.divergence_weight > 0 and step % args.gap_audit == 0:\n"
       "                    print(json.dumps(dict(stage='gap_audit', step=step, seat=l.seat, **gap_audit(g, op))), flush=True)\n")
assert OLD in t, 'old call site missing'
t = t.replace(OLD, '', 1)

NEW = ("                if args.gap_audit and args.divergence_weight > 0 and step % args.gap_audit == 0:\n"
       "                    print(json.dumps(dict(stage='gap_audit', step=step, seat=l.seat, train_residual=residual, **gap_audit(g, op))), flush=True)\n")
A = "                loss.backward(); gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)\n"
assert A in t, 'backward anchor missing'
t = t.replace(A, NEW + A, 1)

P.write_text(t)
print('moved gap_audit call before loss.backward(), added train_residual')
print(subprocess.run([sys.executable, '-c', 'import ast,sys; ast.parse(open(sys.argv[1]).read())', str(P)],
                     capture_output=True, text=True).stderr or 'syntax OK')
