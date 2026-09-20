# Condense the free cut surface out of the target

2026-09-20. Written after reading Codex's overnight package
(`/root/autodl-tmp/CLAUDE_HANDOFF_CUT_20260920`, handoff stamped 2026-09-20T13:11:43+08:00) against
our own arms. Everything below is measured; the one purely algebraic step is marked as such.

## 1. The controlled difference between "box cells pass" and "box cells fail" is augmentation

Codex ran three arms (A0/A1/A2) on 8 geometries, 64 000 steps, **8 000 visits per geometry**, and
reported that a held-out FULL cell assembles at 33.7-37.8 %, concluding one cannot say the box
family is solved. That conclusion is right about its own evidence and wrong about ours, and the
reason is not budget:

| | Codex A0 | our MULTI_AUGMENT |
| --- | --- | --- |
| parameters | 9 692 283 | 9 692 283 (bit-identical) |
| steps | 64 000 | 60 000 |
| visits per geometry | 8 000 | ~1 250 |
| cube-symmetry augmentation | **none** (`EXPERIMENT.json`: "none in all three arms") | **on** |
| seat 100051 two-cell worst compliance | 8.414-10.899 % | **1.412 %** |
| seat 100054 two-cell worst compliance | 33.734-37.810 % | **0.682 %** |

Same code (`UPSTREAM.json` records it pulled `CLAUDE_EQUI_20260918/src`), same `assemble_two.py`
sha, same four loads, same exact reference (the exact compliances agree to ~3e-13), same or larger
parameter count and step count, and 6.4x **more** exposure per geometry. The one switch that
differs is augmentation, and it is worth 6x on a presented seat and **49-55x on a held-out one**.

Seat 100054, the worst two-cell number anywhere in Codex's package, is `box_only = true`,
`kind1 = 0`. The cut plane is not its cause. So the honest statement of where we are is:

> A box cell composes inside the contract **when the arm is augmented** (WIDE_BOX 0.88-3.97 %
> compliance / 1.63-7.26 % sensitivity over 12 seats; ENSEMBLE_MW 0.57-2.25 % / 0.89-7.55 %),
> and fails at 8-38 % when it is not. Cut cells are the one family that cannot be augmented.

Three of our own statements are withdrawn by the same reading. S4_60K is *not* "a few percent":
its worst is 9.44 % (100051) and only 1 of 12 seats is inside the 3 % sensitivity line, against
11 of 12 for MULTI_AUGMENT. Our 3.91 % three-module stack is not comparable to Codex's 29-30 %,
because Codex's stack uses its own weak model for the two FULL end modules as well. And the
teacher's own conditioning kappa(A*) does **not** rank the cut family internally (Spearman +0.26,
n = 13), although the box/cut gap in kappa is real: box median 3.27e4 over a 3.6x spread, cut
median 7.75e5 over a 100x spread, read from `INVERSE_RESULT.json['condition_of_A']` for all 273
labelled seats.

## 2. Why cut cells cannot be augmented, exactly

The augmentation identity `M_q(g . cell) = G M_q G^T` is exact for a box cell and false for a cut
cell, because the cut-surface residual functionals are selected by the teacher's pivoting rule
`max_pivot_geometric_residuals_v2`, which is not rotation covariant. Forcing it costs a factor of
four (`CUT_CELL_AUGMENTATION_COSTS_FOURFOLD_20260919.md`: seat 100032 median smooth-face error
0.283 -> 1.191 %). So every cut arm has run with `g = 0` on every step - verified directly in the
step logs: `rot_cut`, `wide_cut` and `cutonly_noaug` log `"g": 0` and nothing else, while
`wide_box` spreads over the group.

But `witness_d` already measured the decisive refinement, on seat 100032 under a proper rotation
(g=5) and an improper one (g=13):

* the residual **subspace** is exactly the group image - principal-angle residual
  1.63e-15 and 2.00e-15, rank 1436 both sides;
* the **basis** is not - of 1436 selected rows only 144 (g=5) and 325 (g=13) are shared, symmetric
  difference 1848 and 856.

So the obstruction to augmenting a cut cell is a change of basis inside the cut block, nothing else.

## 3. The application does not need that block, and removing it is exact

In the acceptance configuration and in the application, a cut cell's cut surface is the part's
outer boundary: free. `ASSEMBLE_TWO.json` states it - *"two copies of one cell glued on its kind-0
coordinates; kind-1 coordinates are free surface"*. A free block can be condensed out exactly:

    T = S_BB - S_BC S_CC^-1 S_CB      on the kind-0 box coordinates only.

For any load supported on box coordinates, `T` reproduces the cell's response exactly. Measured on
the frozen teacher's own `S_UPPER.npy` for 8 cut seats, comparing compliance through the full trace
operator against compliance through `T`: relative difference **4.7e-10 to 7.6e-6** (the larger
values are the seats where the pseudo-solve itself is ill-conditioned, e.g. seat 185 at
kappa = 1.6e7).

And the algebraic step, which needs no measurement: condensation is invariant under **any** change
of basis in the eliminated block. For invertible V,

    (S_BC V) (V^T S_CC V)^-1 (V^T S_CB) = S_BC S_CC^-1 S_CB.

Combined with the witness of section 2 - the subspace is covariant, only the basis is not - this
says the condensed operator is **exactly** cubic-covariant for a cut cell. The 48-fold
augmentation that carries the box arms becomes available to the cut family. This is a prediction
with a cheap direct witness attached (section 5, step 1); it has not yet been measured end to end.

## 4. What condensation does to the target, measured

Teacher side, from `S_UPPER.npy` (8 cut seats, 2 box controls):

| seat | q | n_box | n_cut | kappa(S) | kappa(T) | ratio | CC share of \|\|S\|\|^2 | diag range S | diag range T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100000 | 711 | 285 | 426 | 1.78e6 | 3.87e5 | 4.6 | 41.5 % | 3.6e4 | 8.3e3 |
| 100032 | 10812 | 6504 | 4308 | 2.54e6 | 3.46e5 | 7.3 | 68.9 % | 1.35e5 | 1.16e4 |
| 100024 | 9426 | 2322 | 7104 | 4.93e6 | 5.53e5 | 8.9 | 93.3 % | 2.07e5 | 1.17e4 |
| 115 | 12084 | 4467 | 7617 | 2.20e6 | 3.20e5 | 6.9 | 85.0 % | 1.18e5 | 1.43e4 |
| 100028 | 9828 | 3234 | 6594 | 3.17e6 | 5.93e5 | 5.3 | 84.2 % | 1.11e5 | 1.17e4 |
| 185 | 4422 | 1251 | 3171 | 1.65e7 | 2.77e6 | 5.9 | 87.2 % | 1.01e5 | 2.34e4 |
| 100021 | 9306 | 2538 | 6768 | 6.08e6 | 2.84e5 | 21.4 | 88.1 % | 2.5e5 | 1.62e4 |
| 100079 | 13953 | 3750 | 10203 | 1.69e7 | 1.68e6 | 10.0 | 89.5 % | 1.62e5 | 1.17e4 |
| 100051 box | 12528 | 12528 | 0 | 7.58e4 | - | 1.0 | 0 | 7.7e3 | 7.7e3 |
| 100102 box | 15204 | 15204 | 0 | 2.65e5 | - | 1.0 | 0 | 1.17e4 | 1.17e4 |

Three things at once. **60-93 % of the target's Frobenius mass is in the block the answer never
reads.** The target shrinks by `(n_box/q)^2`, i.e. 7-16x, which also removes the label-shard
rotation and the forgetting it causes (the shard-return loss ratio is 1.116-1.513, and the
longer-absent shard rebounds harder in all four adjacent pairs - a dose-response Codex's
`ANALYSIS.json` contains). And the diagonal dynamic range falls from 1e5-2.5e5 to **8.3e3-2.3e4,
which is the box band** (7.7e3-1.2e4).

## 5. What condensation does to an already-trained prediction, measured

No retraining: take `A_PRED_UPPER.npy` from the finished arms, lift both teacher and prediction to
the trace with the same frozen quotient, condense both, and compare.

| arm | seat | full rel. Frobenius error | condensed | mu(full) | mu(condensed) |
| --- | --- | --- | --- | --- | --- |
| ROT_CUT | 115 | 890.6 | **0.413** | [5.67e-3, 7.90e6] | [5.94e-2, 1.08e2] |
| ROT_CUT | 100032 | 76263.2 | **0.450** | [8.99e-4, 1.36e8] | [1.06e-1, 2.86e1] |
| ROT_CUT | 100046 | 3686.9 | **0.430** | [4.52e-3, 3.13e6] | [1.25e-1, 2.94e1] |
| ROT_CUT | 100041 | 130.8 | **0.423** | [3.90e-3, 1.32e5] | [9.70e-3, 1.25e2] |
| ROT_CUT | 100066 | 35.7 | **0.426** | [1.86e-3, 6.22e5] | [4.93e-2, 4.40e2] |
| ROT_CUT | 100080 | 222.4 | **0.435** | [5.53e-3, 1.57e6] | [2.17e-2, 4.84e2] |
| WIDE_CUT | 100024 | 166.3 | **0.471** | [1.62e-3, 1.44e5] | [8.13e-2, 5.15e1] |
| WIDE_CUT | 100036 | 1079.4 | **0.476** | [2.07e-3, 1.61e6] | [7.53e-2, 8.20e1] |
| WIDE_BOX | 100051 (control, no cut block) | 0.3085 | 0.3085 | [4.17e-1, 2.27e0] | identical |

The full-trace error spans four orders of magnitude across eight cut seats and two arms; the
condensed error is 0.41-0.48 on every one of them. The count of out-of-gate directions barely moves
(0.90 -> 0.87); what moves is how wrong the worst ones are. The box control has no cut block, so
condensation is the identity there, which is the self-check.

Two readings, and the distinction matters.

*Sound:* the condensed cut error (0.41-0.48) is the same kind of quantity, and the same order, as a
box cell that **passes** the contract - WIDE_BOX seat 100051 sits at 0.3085 and assembles at 1.41 %
compliance / 1.63 % sensitivity. The unreduced cut error (35.7 to 76 263) is not that kind of
quantity at all. Condensation removes four to six orders of pathology from the learning target.

*Not yet shown:* that the remaining gap closes. The condensed pencil is still [1e-2, 29-484]
against the box control's [0.42, 2.27], so one to two orders remain, and Frobenius error does not
determine assembled compliance (section 6). The claim being made is narrower: cut cells become the
same kind of problem as box cells, which is the problem augmentation is measured to solve (6-55x).
That is the falsifiable prediction, and one arm tests it.

Note the deployment side is unchanged by this: `assemble_two` and `assemble_lattice` already glue
on kind-0 and leave kind-1 free, so the assembled response already depends on the prediction only
through its condensation. The gain here is entirely on the **learning** side.

Caveat on the spectral columns: `mu_max` and `e_A` are noisy at the resolution people compare at -
256 extra steps at lr 1e-5 move seat 100001's `mu_max` by 807x while `factor_relative` moves 9 %.
The robust column here is the Frobenius one.

## 6. What this does not fix

* Loss engineering is not the bottleneck and should not be the next move. The cheap
  deployment-aligned losses already exist in `train_equi.py` - `response_term` (line 406, sparse
  face loads) and `column_term` (line 434, point loads, per-column relative error), both with a
  self-test, both defaulted to weight 0 and never switched on in a cut arm. And their metric is
  *already satisfied* where the assembly is not: Codex's `EVAL_100032/RESULT.json` has face
  response-gate median 1.019 % with 82.5 % of loads inside 3 %, while the mixed three-cell is
  30.4 % off. Codex's expensive `L_E` (29-40 s per gradient, O(q^2) because `Chat = (B M B^T)^2`
  applies the factor twice with a dense vector between) cut its own objective by 36-48 % and made
  5 of 6 two-cell compliances worse.
* The compliance functional cancels the error it is meant to measure. Codex's `load_probe`:
  `stiffness_shift_relative` 102.9-846.6 and `relaxation_relative` 102.8-846.5 agree to 3-4
  significant figures, leaving 0.13-2.37 % local compliance error on a module whose displacement
  field is wrong by 173-329 %. Steer on the 64-load response gate and on displacement, not on the
  worst of four loads: the same seat's two-cell number moves 5.24 % -> 45.98 % across arms.
* `factor_relative` ranks the two-cell **displacement** almost perfectly (Spearman +0.924 over 30
  rows) and does **not** rank compliance (negative within a fixed geometry: rho -0.90 on seat 115,
  -1.00 on seat 133). The factor loss is aimed correctly; the compliance readout is the noisy one.
* Our own box arms have never had their displacement measured - `ASSEMBLE_TWO.json` from our runs
  carries no `relative` block. Codex's version computes it. That is a gap to close cheaply.

## 7. Solver qualification, fixed here

`assemble_lattice.py` stopped on the **recursive** residual `R <- R - alpha K P` and only checked
the true residual once at the end. On predicted cut operators the two diverge: Codex's mixed
three-cell stopped with a recursive residual under 1e-9 and a true residual of 1.863e-7 against a
1e-7 line, and its CONTROL passed the same item only at 5.343e-8. The problem is not confined to
that one run - across 34 assembly rows the predicted-side residual is 7.19e-13 to 8.32e-13 for box
cells and 1.61e-9 to 2.80e-6 for cut cells, and all six rows over the 1e-7 line are cut cells; one
of ROT_CUT's own two-cell numbers (seat 100032, 43.18 %) rides on 2.80e-6.

`--replace-every` (default 200) now recomputes `F - K U` periodically, restarts the search
direction from it (replacement breaks conjugacy with the old direction), and **requires a freshly
computed true residual** under `--rtol` to stop. Both residuals and the replacement count are
reported. This makes the gate stricter, not looser; it is not a tolerance change.

Validated on the two standing reproductions, from a copy of the arm's source with only this file
replaced (`/root/autodl-tmp/CLAUDE_SOLVER_20260920`): the 2x1x1 machinery check reproduces
`assemble_two` at `check_two = 3.175237850427948e-14`, the same value as before the change, with
`worst_compliance_rel = 0.014122046780423392` and adjoint identity 6.04e-15. A converged answer is
unchanged; only the stopping rule is.

## Provenance

* Condensation of the teacher: `/root/autodl-tmp/CLAUDE_CONDENSE_20260920/`, script `_cond2.py`.
* Condensation of trained predictions: same directory, `RUN2.log`, script `_cnd3.py`.
* Teacher conditioning and geometry descriptors for all 273 labelled seats: `_corr.py`.
* Codex's package, unpacked read-only, with `FILES_SHA256.json`.
* Arm protocols and the per-step `g` distribution: `CLAUDE_EQUI_20260918/*.log`.
