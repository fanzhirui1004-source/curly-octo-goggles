# The rejection rule stalled three runs and could not recover by construction

Date: 2026-09-15. Runs R2, R4, R5 on seat 0328.
Audit added as `--gap-audit` in `v1_scaled.py`; function `gap_audit`.

Read the correction section before the audit numbers: the first version of this document drew
a conclusion from an invalid measurement, and that conclusion is withdrawn here.

## What happened

Three consecutive capacity runs froze the same way.

| run | `--solve-tol` | refinement | outcome |
|---|---|---|---|
| R2 | 1e-6 | none | stalled at step 281, 45 rejections, residual floor 1.03e-6 to 1.08e-6 |
| R4 | 1e-6 | 4 | 8 rejections by step 208, residual floor about 1.1e-6 |
| R5 | 1e-4 | 4 | stalled at step 197, 427 rejections by step 710, residual floor 1.03e-4 to 1.19e-4 |

The residual floor tracked the tolerance in every case. That is the tell: the runs were not
hitting a fixed numerical wall, they were hitting whatever wall was configured.

In R5 the learning-rate multiplier reached its floor of 1e-3 within about 20 rejections and
stayed there. Every step from 198 to 710 was rejected and rolled back. The model never moved
again, and the job burned GPU for another 500 steps producing nothing.

## The structural bug

The rule reads:

```
if residual > solve_tol:
    restore parameters from snapshot
    reject_scale = max(reject_scale * 0.5, reject_floor)
    continue
```

The residual is `||S_hat U - N|| / ||N||` with `U = solve_full(N)`, a property of the current
operator's conditioning. **Shrinking the learning rate does not change it.** The rule's only
response to a condition it cannot influence is to apply that response harder. Once the residual
crosses the threshold it stays crossed, so the run cannot recover by construction.

Worse, the threshold is guaranteed to be crossed eventually: the residual grows with `cond(L)`,
and `cond(L)` necessarily grows as the student fits a stiff operator. R1 saw it reach 5.1e6 from
an initial 84.7. Any fixed absolute tolerance stalls the run at some step.

## Correction: the first audit was invalid

An initial audit appeared to show the residual moving 34000x while `logdet(N^T U)` moved only
3.2e-4, which would have meant the gate was measuring a quantity the objective barely depends
on. **That measurement was wrong and the conclusion drawn from it is withdrawn.**

`FreeScaledModel.forward` returns `self.M`, the live `nn.Parameter`, so `Operator.M` is that
tensor and `opt.step()` mutates it in place. `Operator.C = cholesky(I + M^T M)` is built once at
construction. The audit call sat after `opt.step()`, so it measured an operator holding
post-step `M` against pre-step `C`. The residuals it reported were that inconsistency, not the
operator the loss was computed from.

The tell was in the same log: at step 100 the audit reported a refine-0 residual of 1.333e-2
while the training path's own residual for that step read 1.30e-14, twelve orders apart for
what should have been the same computation.

With the call moved before `loss.backward()`, the two agree and the picture is different:

| | step 25 of R7 |
|---|---|
| training-path residual | 1.2530e-15 |
| audit refine-0 residual | 1.2565e-15 |
| `logdet(N^T U)` at refine 0, 1, 2, 4 | 111.2292965885176, unchanged to all printed digits |

Early in training the solve is clean and refinement changes nothing. So the audit says nothing
yet about the regime where R5 failed. R5's residual was 1.81e-14 at step 150 and 1.04e-4 at
step 198, a real ten-order jump in 48 steps, measured on a consistent operator. What caused
that jump is still unknown. R7 runs the same configuration with the corrected audit through
that range, which is the measurement that will settle it.

## Why the earlier patches did not help

Both were responses to the symptom.

Raising `--solve-tol` from 1e-6 to 1e-4 moved the threshold. The residual grows without bound
during training, so this bought about 200 steps.

Adding iterative refinement was closer to right: the audit shows refinement contracts the
residual by roughly a factor of 4 per pass, 1.26e-2 to 3.7e-7 in eight passes. But R5 ran four
passes from a starting residual of 1.26e-2, which lands near 1e-4, exactly on the threshold it
was being judged against. Knife-edge, then permanent rejection.

Neither patch questioned whether the residual was the right thing to gate on.

## The fix, so far

Drop the rejection. R7 runs with `--reject-ill 0 --solve-tol 1e30 --rigid-refine 0
--gap-audit 25`, so the objective is the same function at every step and the audit records
both the residual and the log-det through the range where R5 died.

This is justified by the structural argument above, which stands on the R5 log alone: the rule
could not recover once triggered. It is not yet justified by a measurement showing the residual
is harmless, because that measurement has not been made. If R7 reaches the same residual regime
and the audit shows the log-det moving enough to matter against the current `D`, the answer is
refinement, which does reduce the residual, and the capacity result will have to be read with
that error budget attached.

## What to carry forward

A guard has to be able to change the thing it gates on. This one could not: it responded to a
solve residual by cutting the learning rate, which does not affect a solve residual. Once
triggered it could only trigger again, and it rejected 512 consecutive steps before the run was
killed. That is a design fault independent of whether the residual was worth worrying about.

The second lesson is about the audit itself. A diagnostic added to check an objective has to be
evaluated on the same object the objective used. Placing it one line after `opt.step()` was
enough to make it measure something else entirely, and the numbers it produced were plausible
enough to be written up before the twelve-order disagreement with the training path was noticed.
