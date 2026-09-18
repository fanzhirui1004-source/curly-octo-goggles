# Route 5 on the first try: predict the factor of A^-1

2026-09-18. Five arms, all seat 0253, all 20 000 steps, 8192 pairs per bucket, seed
20260917, same 5 077 241-parameter model and same sampler. Two axes: what the head
targets, and whether the diagonal bucket scores pivots absolutely or in the log domain.
Physics is the two-cell assembly with the exact adjoint, identity verified at 3e−15 to
8e−15 on every run.

| arm | `eps_op` | `mu_min` | `mu_max` | **`g`** | `e_A` | worst compliance | **worst sensitivity** |
|---|---|---|---|---|---|---|---|
| ① chol + absolute | 215.72 | 1.171e−3 | 216.72 | 854.05 | 0.0507 | 20.28 % | 25.43 % |
| ① chol + log | 154.77 | 3.044e−3 | 155.77 | 328.49 | 0.0499 | 22.41 % | 25.16 % |
| ③ sqrt + absolute | 48.03 | 3.270e−4 | 49.03 | 3057.80 | 0.0265 | 452 % | 542 % |
| ③ sqrt + log | 40.79 | 4.512e−3 | 41.79 | 221.64 | 0.0284 | 29.31 % | 38.96 % |
| **⑤ inverse + log** | **11.75** | **0.1033** | **12.75** | **12.75** | 0.4134 | **17.92 %** | **20.41 %** |

Route 5's spectrum is numerically clean: positive throughout, extreme eigenpair residual
3.07e−13, `A_hat` SPD.

## What route 5 bought

- **`g` = 12.75.** 17x better than the next best arm and **67x** better than the chol
  baseline it replaces.
- **`mu_min` = 0.1033.** 23x better than the best other arm. The soft end — the binding
  constraint in every previous arm — is nearly closed: the softest direction is 9.7x too
  soft where the baseline's was 854x.
- **Worst assembled sensitivity 20.41 %**, against 25.16 % for the best chol arm. The best
  number this project has produced from a trained network, and it came with **zero
  architecture change**: the head, the sampler and the loss are the ones already in the
  tree, pointed at a different label.

And a diagnostic worth naming: **`e_A` = 0.4134, eight times worse than chol's 0.0507**,
on the arm with the best physics. `e_A` is a stiff-weighted Frobenius error and route 5
deliberately spends its accuracy on the soft end. One more demonstration that
stiff-weighted quantities mislead here.

## The 2x2 on head times loss, and a wrong prediction of mine

| | absolute | log pivots | loss gain |
|---|---|---|---|
| chol `g` | 854.05 | 328.49 | 2.60x |
| sqrt `g` | 3057.80 | **221.64** | **13.79x** |

Strong interaction. With the absolute loss chol beats sqrt by 3.6x; with the log loss
sqrt beats chol by 1.48x — **the head ranking flips with the loss**, so neither head could
have been judged before the loss was fixed.

I had predicted the opposite, and the reasoning was explicit: a Cholesky factor localises
the soft directions in its small end-of-elimination pivots while `A^{1/2}` spreads them
across global eigenvectors, so the pivot fix should help chol more. It helped sqrt 5.3x
more. **So line ③ is not dead** — its soft-end deficit was largely a loss artifact, which
was the hypothesis the 2x2 was built to test. It is alive and behind route 5.

## `g` identifies the best and the worst and cannot discriminate between

Ranking by `g`: ⑤ 12.75 < ③log 221.64 < ①log 328.49 < ①abs 854.05 < ③abs 3057.80.
Ranking by measured sensitivity: ⑤ 20.41 % < ①log 25.16 % < ①abs 25.43 % < ③log 38.96 %
< ③abs 542 %.

`g` gets the winner right and the disaster right, and **swaps ③log with the two ① arms in
the middle**. So `g` remains what `THE_ERROR_IS_BROAD_20260918.md` concluded: a valid
composable bound that flags catastrophe, not a calibrated predictor.

## The dominant open question is now the ~20 % floor

Across five arms, `eps_op` spans 11.75 to 215.72 (18x), `g` spans 12.75 to 3057.80 (240x),
and the assembled sensitivity error sits at **20–39 % in four of the five**. Only the
collapsed `③ absolute` arm escapes the band, upward.

Something is pinning the physics near 20 % that none of these knobs moves. That is now the
thing to find, and the cheapest probe is direct: project the four load vectors onto the
whitened eigenbasis of each arm and ask which modes carry the compliance error. If it is
the **same** modes across architecturally different arms, the pinning is a property of the
encoder or the sampler, not of the target or the loss.

## Corrections in this document

- My prediction that the log-pivot loss would help line ① more than line ③: wrong, by 5.3x
  in the other direction.
- My earlier verdict that line ③ should probably be closed: premature. The 2x2 rescued it.
- `eval_inverse.py` reports `A_hat_times_A_star_minus_I` ~ 1. That is a broken diagnostic
  of mine — it computes `A_hat A_*` where the meaningful product is `A_hat A_*^-1`. Ignore
  the field; the SPD check, the positive spectrum and the 3.07e−13 eigenpair residual are
  the real numerical guards and they pass.

## Provenance

- `CLAUDE_ROUTE5_20260918/EVAL/INVERSE_EVAL.json` — route 5's spectrum
- `CLAUDE_SENSMODEL_20260918/{ROUTE5,LOGPIVOT,LOGPIVOT_SQRT}/SENS_MODEL.json` — the
  assembled compliance and adjoint sensitivities
- `CUTFEM_SUPERELEMENT_LABELS/REFERENCE_0253/INVERSE_RESULT.json` — the label self-tests
  (`L^T L A − I` = 3.40e−13, exact-label `g` = 1.0000000000006)
