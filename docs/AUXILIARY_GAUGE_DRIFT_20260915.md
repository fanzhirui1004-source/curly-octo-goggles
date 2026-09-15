# The loss is distorted by a drift in an auxiliary matrix the objective does not constrain

Date: 2026-09-15. Runs R2, R4, R5, R7, R8 on seat 0328.
Audit: `--gap-audit` in `v1_scaled.py`; `superelement/factor_fit/ahat_audit.py`.

## Two matrices that must not be confused

Write the model's full-space positive definite matrix as

```
K0 = Lᵀ (I + M Mᵀ)⁻¹ L
```

What is supervised is its quotient compression `Â = B K0 Bᵀ`. The predicted Schur complement on
the complete interface is recovered as `Ŝ = Bᵀ Â B`, and **that recovery is what supplies the six
rigid zero modes.** `K0` does not need any.

The gauge is explicit: for `U` an orthonormal rigid basis and any `α > 0`,

```
K0(α) = Bᵀ A B + α U Uᵀ     ⟹     B K0(α) Bᵀ = A
```

The supervised object is identical for every `α`. The numerical state of `K0` is not.

## Measurement: the loss is distorted, by 2.5x

R7 ran 1199 steps with the rigid term never dropped. `ahat_audit.py` then scored its step-1000
checkpoint two ways: by the identity the loss uses, and directly from a Cholesky of the
materialized `Â`, which touches no inverse of `K0`.

| route | D/d | trace/d | log-det gap |
|---|---:|---:|---:|
| identity, `logdet K0 + logdet(NᵀK0⁻¹N) − logdet(NᵀN)` | 0.086341 | 0.886248 | −2559.594 |
| direct, Cholesky of `Â` | **0.034952** | 0.886248 | −1902.220 |
| exact spectrum, `eigvalsh(R*⁻ᵀÂR*⁻¹)` | 0.034952 | — | — |

The direct route and the full eigendecomposition agree to six digits. The identity overstates
`D/d` by a factor of 2.47, and the entire discrepancy is in the log-det: the trace terms are
identical to six digits. The rigid solve residual at that checkpoint is 2.9e+87 and
`σ_min(L)` is 1.9e-26, bounding `cond(K0)` at about 1.8e52.

**So the training signal was distorted, and a descending training curve past that point was not
evidence that the predicted Schur was improving.**

## Measurement: the fit was nonetheless progressing

The same checkpoint, scored only on `Â`:

| quantity | step 0 | step 1000 |
|---|---:|---:|
| true D/d | 1.11944 | 0.034952 |
| μ range | [0.0172, 275.07] | [0.0695, 1.2045] |
| `‖Â − A‖/‖A‖` | 0.8443 | 0.3047 |
| off-diagonal relative error | 0.9994 | 0.4838 |
| Cholesky of `Â` | succeeds | succeeds |
| outside the ±10% work gate | — | 4589 of 12792, 35.9% |
| outside the ±3% target gate | — | 7957 of 12792, 62.2% |

For reference, truncating the teacher's own Cholesky factor to the same radius-0.2 band gives
D/d = 0.030–0.032 with μ ∈ [0.092, 86–88]. At step 1000 of 6000 the fitted operator is already
at that divergence with a far tighter μ range, and neither gate is met.

## What was wrong in the first version of this document

Three claims are withdrawn.

1. **"The student is converging to the physically correct rigid-singular operator."** Wrong. The
   rigid null space comes from `Ŝ = BᵀÂB`, not from `K0`. Nothing in the task asks `K0`'s rigid
   block to collapse. Whether the banded `L, M` parameterization *ties* an effective fit to that
   collapse is a separate question and is not settled.
2. **"Six eigenvalues have each shrunk by 3.2e-40."** Not supported. A log-determinant gives a
   product. Attributing it to exactly six modes, equally, was an assumption, and the later solves
   feeding that number are themselves inaccurate.
3. **"Past roughly step 225 the objective's value and gradient are not trustworthy."** Too strong
   in one direction and too weak in another. The value is wrong by 2.5x at step 1000, which is
   worse than "not trustworthy" suggests; but the fit still improved on the object that matters,
   which "not trustworthy" implied it could not.

An earlier claim that the residual gate was a phantom was withdrawn before that, because the audit
behind it was placed after `opt.step()` and measured a stale operator.

## Why pinning `K0 N` is the right gauge condition

In the orthogonal basis `[Bᵀ, Ñ]`, write `K0 = [[Â, Ĉ], [Ĉᵀ, Ê]]`. Then

```
(Nᵀ K0⁻¹ N) ∝ (Ê − Ĉᵀ Â⁻¹ Ĉ)⁻¹
```

so the identity's whole ill-conditioning sits in one 6×6 Schur complement. It degenerates when
`Ê` falls toward zero or when `ĈᵀÂ⁻¹Ĉ` rises to meet it. Both `Ê` and `Ĉ` are read off `K0 N`,
which is one six-column application. Holding `K0 N` at the value the initial operator had fixes
both, and by the gauge argument above it cannot change what is supervised.

The residual risk is not bias in the objective but reachability: a banded `L` holding a fixed
rigid block may not reach the wanted `Â`. R8 runs with `--gauge-weight 1` and is read out against
R7's D/d at steps 100, 150 and 200, which were 0.628, 0.558 and 0.486 while R7's arithmetic was
still clean, and then against `ahat_audit.py` on its checkpoints, where the test is whether the
identity and direct routes agree.

## On the rejection rule

R2, R4 and R5 each stalled with the residual just above the configured `--solve-tol`, which read
as the tolerance binding. It was not: the residual was passing through on its way up. The rule's
response, halving the learning rate, cannot change a solve residual, so once triggered it could
only trigger again. R5 rejected 512 consecutive steps. That remains a design fault independent of
everything above: a guard has to be able to move the thing it gates on.

## The gauge pin holds

R8 runs the same configuration as R7 with `--gauge-weight 1`, holding the q by 6 block `K0 N`
at the value the initial operator had.

| step | R7 D/d | R8 D/d | R7 rigid residual | R8 rigid residual | R8 gauge loss |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.7731 | 0.8686 | 4.21e-15 | 3.96e-14 | 9.5e-3 |
| 100 | 0.6272 | 0.7079 | 1.25e-14 | 1.68e-13 | 9.5e-3 |
| 150 | 0.5598 | 0.6309 | 1.84e-14 | 2.62e-13 | 1.13e-2 |
| 200 | 0.4859 | 0.5558 | 1.23e-03 | 3.48e-13 | 1.30e-2 |
| 250 | 0.4093 | 0.4844 | 4.22e+16 | 4.30e-13 | 1.41e-2 |
| 444 | — | 0.3048 | — | 1.94e-12 | 1.82e-2 |

R7's residual crosses 1e-3 at step 200 and 1e+16 at step 250. R8's is still 1.9e-12 at step
444, a 28-order difference, so pinning `K0 N` does stop the drift. It creeps, 4.3e-13 to
1.9e-12 over 200 steps, which is worth watching but is not the same phenomenon.

**It is not free.** R8's D/d runs 12% to 18% above R7's at matched steps and the gap widens.
Two causes are not yet separated: the penalty diverting gradient, which at a gauge loss of
1.4e-2 against a total loss of 0.48 is about 3%, and the banded `L` being unable to reach the
same `Â` while holding a fixed rigid block. The gauge loss climbs steadily rather than settling,
which suggests the model is pushing against the pin. Note also that R7's step-200 and step-250
figures are themselves overstatements, so the true gap may be wider than the table shows.

## The pin makes the loss exact

`ahat_audit.py` on R8's step-500 checkpoint, which is the test that matters:

| route | R8 step 500, gauge weight 1 | R7 step 1000, gauge free |
|---|---:|---:|
| identity, via `logdet(NᵀK0⁻¹N)` | 0.260700 | 0.086341 |
| direct, Cholesky of `Â` | 0.260700 | 0.034952 |
| exact spectrum | 0.260700 | 0.034952 |
| identity / direct | **1.000** | **2.470** |
| log-det gap, both routes | −7101.574 / −7101.574 | −2559.594 / −1902.220 |
| rigid solve residual | 2.79e-12 | 2.93e+87 |

All three routes agree to six digits under the pin, and the log-det gap matches exactly. The
distortion is gone, not merely reduced.

What the pin costs on the true objective is a separate number, and the honest comparison is at
matched steps on the direct route. R8 step 500 is at true D/d 0.2607 with μ ∈ [0.0011, 116.96]
and 79.9% outside the work gate; R7 step 1000 is at true D/d 0.0350 with μ ∈ [0.0695, 1.2045]
and 35.9% outside. Those are different steps and R7 has no step-500 checkpoint, so the
comparison waits for R8 to reach step 1000.

A sweep over `--gauge-weight` in {0.01, 0.1, 1, 10} at 400 steps separates the two: if a much
smaller weight still bounds the residual while D/d tracks R7, the cost was the penalty.

## Open

- How much the pin costs on the true objective, measured at step 1000 against R7's 0.034952,
  and how much of that is the penalty diverting gradient versus the banded `L` not reaching the
  same `Â` under a fixed rigid block. A sweep over the weight at matched steps, each scored by
  `ahat_audit.py`, separates them.
- What the local class reaches at 6000 steps under an undistorted signal.
- A layered cost profile on a real n32 GP teacher. The packets hold only the condensed
  `S_UPPER.npy` (12798 x 12798), `TRACE.npz` and `PROBES.npz`, so the pre-condensation coupling
  graph has to be rebuilt from `SAMPLE.json` through `stage_cutfem_full_interface`. That path is
  being mapped; nothing has been assembled yet.
