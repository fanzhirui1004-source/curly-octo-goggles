# Past the floor: 20.41 % to 0.21 % assembled sensitivity error

2026-09-18, evening. `THE_FLOOR_IS_THE_SOFTEST_DECILE_20260918.md` closed with the
dominant open question: five architecturally different arms all sat at 20–39 % assembled
sensitivity error, and nothing moved them. That floor is gone. The contract — only the
assembled lattice's compliance and its design sensitivities have to be right, to 3 % — is
met with a factor of 14 to spare.

Three arms, seat 0253, 20 000 steps, 8192 pairs per bucket, seed 20260917, one variable
(what orientation the cell is shown in). Physics is the same two-cell glued assembly with
the same exact adjoint as every previous arm: 716 shared nodes, 23 508 assembled trace
dofs, four load cases, adjoint identity `(sum dc/drho + c)/c` verified at 3.1e−15 to
3.1e−14.

| arm | **worst sensitivity** | **worst compliance** | `g` | `mu_min` | `mu_max` | `e_A` |
|---|---|---|---|---|---|---|
| **equi + canonical** | **0.207 %** | **0.190 %** | **1.937** | **0.5164** | 1.772 | 0.2494 |
| **equi + identity** | **0.336 %** | 0.228 % | 2.031 | 0.4923 | 1.813 | 0.2519 |
| **equi + augment** | 0.765 % | 0.564 % | 2.014 | 0.4964 | 1.845 | 0.2704 |
| ⑤ invsqrt (`A^{-1/2}`, old encoder) | 9.01 % | 8.49 % | ~10.9 | — | — | — |
| ⑤ inverse (`L` of `A^-1`, old encoder) | 20.41 % | 17.92 % | 12.75 | 0.1033 | 12.75 | 0.4134 |
| ① chol + log (old encoder) | 25.16 % | 22.41 % | 328.49 | 3.044e−3 | 155.77 | 0.0499 |
| ① chol + absolute (old encoder) | 25.43 % | 20.28 % | 854.05 | 1.171e−3 | 216.72 | 0.0507 |

**99x better than the arm that held the record this morning**, and 123x better than the
chol baseline. The soft end is the reason: `mu_min` went from 1.17e−3 (chol) through
0.1033 (route 5) to **0.5164** — the most under-stiff direction is now 1.9x too soft where
it was 854x.

Per load, for the best arm: compliance error 0.017 %, 0.147 %, 0.066 %, 0.190 % on
compression_x, shear_y, shear_z, bending_xy; sensitivity error 0.014–0.207 % over the two
modules and four loads.

## What changed, and what this does not isolate

Two things changed together and the experiment does not separate them.

**The label.** `A = B S B^T` depends on the arbitrary Householder basis `B` of the rigid
complement (`B' = U B` gives `A' = U A U^T`), so predicting `A`'s factor is partly
predicting the compiler's elimination bookkeeping. The new label
`M_q = B^T A^{-1/2} B = ((Pi S Pi)^+)^{1/2}` with `Pi = B^T B` depends only on `S` and the
node set. It is simultaneously basis-free, symmetric PSD with exactly the rigid nullspace,
dominated by the soft directions where compliance lives, and equivariant as a nodal
tensor field. Previous routes had at most two of those four.

**The encoder.** Gone: the GRU over Morton order, the patch GRU, the elimination-order
position feature, the 18 Householder reflector entries, the patch-pair tables — every
handle that told the network *where a coordinate sits in the elimination*. A node is now
its position, its face flags and the fields at it; a pair is its two nodes, the segment
between them, and the material sampled along that segment from a U-Net encoding of the
cell. Transposition symmetry is exact by construction.

So the honest claim is that **this package** reaches 0.21 %, not that either half of it is
responsible. Separating them would need `M_q` under the old encoder, or `L` under the new
one; neither was run.

## The orientation variable, and the two memorisation probes

| arm | rotation consistency | index shuffle |
|---|---|---|
| identity | 0.1234 | 2.7e−7 |
| **augment** | **0.0152** | 3.0e−7 |
| canonical | 0.0000 | 3.6e−7 |

Rotation consistency predicts the cell rotated by `g`, maps the blocks back by `g^-1`, and
compares against the unrotated prediction. The identity arm is **12.3 % non-equivariant on
its own**; augmentation over the 48 cube elements takes that to **1.5 %, an 8x reduction,
while `g` moves by 0.8 %** — augmentation is close to free at this scale. The canonical
arm's 0.0000 is a tautology (a trivial stabiliser makes its view frame constant in `g`, so
the probe calls the same inference twice); it is not evidence of anything.

Index shuffle — relabel the nodes, predict, un-shuffle — is 3e−7 on all three arms at
q = 12 828, float32 noise. The encoder genuinely cannot see node identity.

## `g` is a bound, not a predictor — now in the favourable direction

The three arms' `g` agree to 5 % (1.937, 2.014, 2.031) while their assembled sensitivity
spans **3.7x** (0.207 %, 0.336 %, 0.765 %). `EPS_OP_IS_THE_WRONG_METRIC` and
`THE_ERROR_IS_BROAD` both found `g` uninformative in the middle of its range; that holds
here, one decade lower. And the bound is loose in the safe direction: `g = 1.94` permits
94 % compliance error on some load and the measured worst is 0.19 %, because `g` is a
maximum over 12 822 directions and the loads reach only a few of them.

Two corollaries. Acceptance still has to be measured on the assembly, not inferred from
`g`. And the rigorous 3 % gate `g <= 1.03` is far stricter than the contract needs — at
`g = 1.94` the physics is already 14x inside target.

## What this is not

- **One geometry.** All three arms trained and evaluated on seat 0253. The comparison to
  the earlier arms is apples to apples in that respect (they were also single-seat 0253,
  20 000 steps), but single-seat accuracy is memorisation, and *removing* memorisation
  handles can only cost accuracy here by construction. The decisive test is the
  multi-geometry arm on the same 55-presented / 40-unseen split where the old architecture
  failed on geometries it had trained on (`docs/data/gen_20260918`: `g` 3.5e3 to 1.1e6 on
  presented seats against 854 for single-seat memorisation). That arm needs `M_q` labels
  for the presented set and has not been run.
- **Two cells, four loads, one size.** Not a lattice. The composability argument is the
  Loewner sandwich, which `g` bounds; the measured 0.2 % is for this assembly.
- **The teacher's own equivariance is unverified.** The augment arm's labels assume
  `S(g . geometry) = P_g S P_g^T`. A census of all 273 labelled seats found **zero** with a
  tau field having a non-trivial cube stabiliser, so the witness cannot be run on existing
  labels; it would need the teacher on a constructed symmetric cell (~180 s) or a
  near-symmetric seat (the closest is 100155, deviation 0.1 % of the corner spread) for a
  loose bound.
- **The canonical arm cannot be the production choice** even though it scores best here.
  Its frame is the lexicographic-max over the 48 permuted corner vectors, which is stable
  under uniform tau scaling (verified to eps = ±0.5) but switches under a *single-corner*
  perturbation as small as 0.0018 on a tau ≈ 0.19 field — about 1 % of tau, and a random
  design direction hits a switch at median distance 0.0096 against a corner spread of
  0.031. Per-corner sensitivity is the design variable in topology optimisation, so the
  canonical model's `dc/dtau_k` is undefined wherever the argmax flips. Augmentation, or
  Phase 5's intrinsic equivariance, is the path for sensitivities.

## Corrections and pending corrections

- `factor_relative` in `train_equi.evaluate` compared a full symmetric prediction against
  a triangle-only label: a bit-exact prediction scored 0.36 instead of 0. Fixed at
  `train_equi.py:220`. The three arms above were evaluated before the fix, so their
  `factor_relative` is the floored value and is omitted from the table. Every other
  reported quantity goes through the full symmetric `M` and is unaffected; the spectrum
  identity was verified end to end in numpy (`mu = 1/eig((Mhat R^T)^T (Mhat R^T))` equals
  `eig(R^-T A_hat R^-1)` to 1e−15, exact-label `g` = 1 + 6e−15).
- **`THE_FLOOR_IS_THE_SOFTEST_DECILE_20260918.md` needs a correction.** Its §1 load-energy
  table uses `a_j^2 lambda_j` and is right. Its claim that the compliance error "attributes
  to that decile and nowhere else — `−0.0000` in every other decile" comes from
  `floor_probe.py:202`, which weights by `a_j^2 / lambda_j`. The exact first-order
  decomposition is `a_j^2 lambda_j` (with `b_j = lambda_j a_j` for a displacement-space
  `a`), so the code's weight is wrong by `lambda_j^2`, and `lambda` spans decades. On a
  synthetic 200-mode case the correct weight reproduces the exact relative compliance error
  to all printed digits (−0.111117) while the current one gives −0.169166 and inflates the
  softest decile's share of the attribution from 62 % to 95 %. The fix is a one-character
  change; `floor_probe` has to be re-run and that sentence re-stated. The qualitative
  conclusion (the soft decile dominates) survives; the "and nowhere else" does not.

## Provenance

- `docs/data/equi_20260918/A1_{identity,augment,canonical}_{SPECTRUM,PROBES,PROTOCOL,RESULT,CONDITIONING,INPUTS}.json`
- `docs/data/equi_20260918/SENS_EQUI_{identity,augment,canonical}.json`, `SENS_INVSQRT.json`
  — the assembled compliance and adjoint sensitivities, adjoint identity 3.1e−15 to 3.1e−14
- `docs/data/equi_20260918/MQ_RESULT_0253.json` — the label's self-tests (`label_g` =
  1 + 4.9e−11, rigid nullspace 9.4e−17, `read_blocks` round trip exactly 0)
- `docs/data/equi_20260918/PLATEN_LABEL_0253.json` — the exact label through the six-platen
  fixture, six compliances matching the teacher to 2.0e−13
- `docs/data/gen_20260918/` — the old architecture's 55-seat generalisation failure
- `docs/data/audit_20260918/ROUND1_FINDINGS.json` — the audit of this package, 30 findings
