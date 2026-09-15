# Why three capacity runs stalled: the fit succeeds and the objective breaks

Date: 2026-09-15. Runs R2, R4, R5, R7 on seat 0328.
Audit: `--gap-audit` in `v1_scaled.py`, function `gap_audit`.

## Summary

The student operator is driving its six rigid eigenvalues to zero, which is the physically
correct target: the true Schur complement of a free-floating cell is singular on rigid modes.
The log-det identity used to evaluate the objective routes through exactly those six
eigenvalues. So the better the fit, the worse the arithmetic. No tolerance setting can fix
this, and three runs were spent discovering that one tolerance at a time.

## The symptom, three times

| run | `--solve-tol` | refinement | outcome |
|---|---|---|---|
| R2 | 1e-6 | none | stalled at step 281, 45 rejections, residual floor 1.03e-6 to 1.08e-6 |
| R4 | 1e-6 | 4 | 8 rejections by step 208, residual floor about 1.1e-6 |
| R5 | 1e-4 | 4 | stalled at step 197, 427 rejections by step 710, residual 1.03e-4 to 1.19e-4 |

Each time the residual floored just above the configured tolerance, which looked like the
tolerance was the binding constraint. It was not. The residual was passing through that value
on its way up.

## What R7 shows

R7 runs the same configuration with rejection off, the rigid term never dropped, and the audit
moved before `opt.step()` so it measures the operator the loss was computed from.

| step | rigid solve residual | `logdet(N^T S^-1 N)` | D/d |
|---|---|---|---|
| 25 | 1.25e-15 | 111.229 | 0.912 |
| 150 | 1.84e-14 | 112.787 | 0.558 |
| 175 | 5.96e-13 | 130.576 | 0.524 |
| 200 | 1.23e-03 | 227.878 | 0.486 |
| 225 | 1.28e+07 | 348.384 | 0.444 |
| 250 | 4.22e+16 | 469.415 | 0.408 |
| 300 | 7.24e+29 | 656.823 | 0.345 |

The residual does not drift. It explodes, monotonically, by 44 orders of magnitude in 150
steps, while `D/d` descends smoothly the whole way.

## The eigenvalue accounting

`gap = logdet(S_hat) + logdet(N^T S_hat^-1 N) - logdet(N^T N) - logdet(A)`, so the run log
gives both halves. Taking step 50 as the baseline:

| step | change in `logdet S_hat` | change in `logdet(N^T S^-1 N)` | implied rigid | implied non-rigid | per non-rigid mode |
|---|---|---|---|---|---|
| 150 | +285.9 | +1.5 | -1.5 | +287.5 | +0.0225 |
| 200 | +1434.3 | +116.6 | -116.6 | +1550.9 | +0.1212 |
| 250 | +2616.1 | +358.2 | -358.2 | +2974.2 | +0.2325 |
| 300 | +3681.2 | +545.6 | -545.6 | +4226.8 | +0.3304 |

Six eigenvalues carry -545.6 of log-determinant, so each has shrunk by `exp(-545.6/6)` =
**3.2e-40**. The other 12792 have grown by `exp(0.33)` = 1.39x each.

The student is converging to an operator that is singular on the rigid subspace.

## Why that breaks the objective

The parameterization is `S_hat = L^T (I + M M^T)^-1 L` with `L` nonsingular, so `S_hat` is
strictly positive definite by construction. It has no rigid null space available, and the fit
approaches one anyway by sending those eigenvalues toward zero.

The label lives on the quotient: `A = B S B^T`, with `B` spanning the complement of the rigid
modes. To score the student the code needs `logdet(B S_hat B^T)`, and gets it from

```
logdet(B S_hat B^T) = logdet(S_hat) + logdet(N^T S_hat^-1 N) - logdet(N^T N)
```

Both right-hand terms carry the full rigid contribution, with opposite signs. At step 300 that
contribution is 545.6 in each, and growing without bound. Their difference is the wanted
quantity, which is finite and well behaved. So the computation is a cancellation whose
operands diverge as the fit improves.

`N^T S_hat^-1 N` is worse still: it is a solve against an operator whose condition number is
now about 1e40. The audit shows iterative refinement diverging there, refine-8 residual
9.5e+146 at step 300, and the log-det itself moving 1551 between refine 0 and refine 8 against
a `D` of about 4400. Past roughly step 225 the objective's value and its gradient are not
trustworthy.

**The physically correct fit and the numerically evaluable objective point in opposite
directions.** That is the finding.

## What this says about the rejection rule

The rule was detecting something real. An earlier version of this document argued the residual
was a phantom; that argument came from an audit placed one line after `opt.step()`, which
measured an operator holding post-step `M` against a Cholesky factor built from pre-step `M`.
Its numbers were an artifact and the conclusion drawn from them is withdrawn.

What remains true about the rule is narrower and still worth keeping: its response could not
address what it detected. It answered a growing solve residual by halving the learning rate,
which does not change a solve residual, so once triggered it could only trigger again. R5
rejected 512 consecutive steps. A guard has to be able to move the thing it gates on.

## What would actually fix it

The degeneracy is in the evaluation route, not in the fit. Three options, in order of how much
they change:

1. **Evaluate `logdet(B S_hat B^T)` directly.** With `W = L B^T` (q by d, formed by a sparse
   product and six reflector applications), `A_hat = W^T (I + M M^T)^-1 W`, and its Cholesky
   gives the log-det with no rigid solve anywhere. Cost is about `d^2 q + d^3/3`, roughly
   2.7e12 flops, which is about 2 s per step against the current 2.1 s. Exact, and the
   degeneracy disappears because the rigid directions are never inverted.
2. **Remove the rigid directions from the model.** Parameterize `S_hat = Pi L^T (I + M M^T)^-1
   L Pi` with `Pi` the projector off the rigid modes. Then `A_hat` is unchanged, because
   `Pi B^T = B^T`, and there is nothing left to collapse. The log-det still needs route 1.
3. **Stop before the boundary.** Cap the rigid collapse with a penalty. This biases the fit
   toward an operator that is not rigid-singular, which is the wrong answer, so it is only
   useful as a diagnostic to confirm the mechanism.

Route 1 is the one that keeps the model and fixes the arithmetic.

## Open

R7 continues to step 1000, where the dense evaluation computes the whitened spectrum
independently of the running objective. Comparing the two at that point measures how much the
corrupted region actually cost, and whether an operator fitted under a broken objective past
step 225 is still a good operator.
