# The rejection rule stalled three runs, and it was gating on the wrong quantity

Date: 2026-09-15. Runs R2, R4, R5 on seat 0328.
Audit added as `--gap-audit` in `v1_scaled.py`; function `gap_audit`.

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

## The gate was measuring the wrong quantity

The residual passes `U` back through `L` via `apply`, so it carries `U`'s error amplified by
`cond(L)`. The objective does not consume `U` that way. It consumes `logdet(N^T U)`, which
depends on `U` directly.

The audit computes both at a range of refinement counts. At step 25 of R6:

| refinement | residual | `logdet(N^T U)` |
|---|---|---|
| 0 | 1.256e-2 | 111.22838673 |
| 1 | 3.127e-3 | 111.22864269 |
| 2 | 8.192e-4 | 111.22869054 |
| 4 | 5.956e-5 | 111.22870526 |
| 8 | 3.680e-7 | 111.22870628 |

The residual moves by a factor of 34000. The log-det moves by 3.2e-4 absolute, 2.9e-6 relative.

That 3.2e-4 is the entire error the gate was defending against. It enters the divergence `D`
additively. `D` is about 6270 where the gate was firing, so the error was 5e-8 relative. At the
target `D/d = 1e-4`, `D` is about 1.28 and the error is 2.5e-4 relative, which moves `D/d` by
2.5e-8 against a gate of 1e-4.

**The rule rejected 427 steps to avoid an error five to eight orders of magnitude below what
matters.**

## Why the earlier patches did not help

Both were responses to the symptom.

Raising `--solve-tol` from 1e-6 to 1e-4 moved the threshold. The residual grows without bound
during training, so this bought about 200 steps.

Adding iterative refinement was closer to right: the audit shows refinement contracts the
residual by roughly a factor of 4 per pass, 1.26e-2 to 3.7e-7 in eight passes. But R5 ran four
passes from a starting residual of 1.26e-2, which lands near 1e-4, exactly on the threshold it
was being judged against. Knife-edge, then permanent rejection.

Neither patch questioned whether the residual was the right thing to gate on.

## The fix

Drop the rejection. Never truncate the rigid log-det term. R6 runs with
`--reject-ill 0 --solve-tol 1e30 --rigid-refine 0 --gap-audit 25`, so the objective is the same
function at every step and the audit reports whether that assumption holds as `cond(L)` grows.

If the audit ever shows `logdet_drift` large enough to matter against the current `D`, the
response is refinement, which demonstrably reduces the residual, not a learning-rate cut, which
demonstrably does not.

## What to carry forward

A guard should gate on the quantity the objective consumes, and its response has to be able to
change that quantity. This one failed both tests. It was added to protect the objective's
integrity and instead cost three runs and roughly four hours of GPU time.
