# Where to go next, given Codex's harness and the analysis side

Written 2026-09-17 after (a) reading Codex's complete `stage_cutfem_m4` source,
(b) four independent re-readings of the corpus, and (c) four new measurements on
Codex's saved factors (`docs/WHERE_THE_ERROR_IS_20260917.md`,
`docs/OBJECTIVE_FIX_20260917.md`).

## 1. What is now measured, not argued

| question | answer | how |
|---|---|---|
| Is the gate error in the pivots or the off-diagonals? | **Off-diagonals.** Exact pivots change `eps_op` 273.9 -> 275.2; exact off-diagonals cut it to 51.4. | role substitution on the saved factors |
| Is there a sub-block that carries it? | **No.** Replacing 99.3% of the factor (all 81.6M cross-patch blocks) still leaves 73.3. | distance substitution |
| How accurate must a per-entry predictor be? | **2.9e-4** for +-3%; `eps_op = 104 * eps` over four decades. | independent per-entry noise on the teacher |
| How accurate is it? | **median 3.2 (relative arm) to 5.7 (baseline)** relative per entry. Short by ~1e4. | per-entry error distribution |
| Is it a coherence problem? | **No.** Re-signing the actual error keeps magnitudes and only drops `eps_op` 273.9 -> 103. Coherence costs 2.5x. | sign randomisation |
| Is the objective's shape the bottleneck? | **No.** `eps_op` is set by `mu_max`; on the stiff side the relative loss and `D` agree within 6%. | closed form, verified numerically |
| What does the complete factor cost? | **~135 ms/cell at TF32 peak** (5.67e13 FLOP), a 236x speedup over the teacher, 135x from 1 ms. | their FLOP count, hardware peak |

Two of these correct claims made earlier the same day and should be read as
withdrawals: `D` is not "the missing gradient" for the gate, and the cost bound is
135 ms at peak rather than an implementation-bound 2 s.

## 2. What the measurements jointly imply

The complete-dense-factor head is not short by a tunable margin. It must produce
8.2e7 numbers each accurate to ~3e-4 relative, from a 5.08M-parameter network, where
each cross-patch block receives about **4.2 gradient updates in a 30000-step run**
(8192 sampled per step out of ~9.1e6 candidates). Nothing about the loss function
changes those three numbers.

This does **not** say the harness is wrong. It says the head is.

## 3. The comparison that should drive the decision

| route | size | gate reached | what kind of result |
|---|---|---|---|
| M4 complete factor (Codex) | 5.08M params -> 8.2e7 numbers | `eps_op` 264 (0253), 125-8592 across seats | a **network**, geometry-conditioned, fitting |
| Hierarchical low rank | 1.22e7 numbers (14.95% of dense) | **2.6e-2**, passes +-3%, 4 seats | **compression of the teacher**, not learning; does not transplant between seats (536-7298) |
| Collar + coarse core | **23,016** fitted parameters | **0.15**, blocked by a *proved* kinematic floor `mu_max >= 1.111` | a **fit to one seat**, not geometry-conditioned |

The middle and right columns are not the same kind of object as the left one, and
the project's own audit says so: **no neural, geometry-conditioned result below
`eps_op` ~4 has ever been produced on any seat.** Before generalisation, memorisation.

But the right column is the important one. 23,016 numbers reach 0.15 where 8.2e7
numbers reach 264, and the collar's failure is a *computable lower bound*, not a
fitting failure. That is the single strongest signal in the corpus about which
representation can hold the answer.

Its cost story is different from the factor's, and must not be oversold: the collar
still condenses at inference (0.27x the teacher's dofs, 0.076x its factorisation).
Its advantage is architectural - assemble every cell's reduced model into one global
reduced system and never form a per-cell `S` at all - not per-cell arithmetic.

## 4. Recommended order

**A. First, and before any further training spend: the Loewner floor scan.**
Smetana-Patera transfer eigenproblems over collar depth {1,2} x core {Q1, Q2,
Q2+EAS, Reissner-Mindlin} x enrichment {0,4,8,16}, on seat 0328, computed from the
fine factors we already have. It asks: **does a reduced local representation with
floor `eta <= 0.05` exist at <= 8e4 reduced dofs?** No training, no new labels, and
it decides the entire local route before anything is spent on it. If no
configuration reaches it, the local route is dead and the factor route is all there
is; if one does, it sets the architecture. This is item 0d of
`FIRST_PRINCIPLES_PATH_20260917.md` and it has never been run.

**B. In parallel, cheap, on Codex's harness.** Section 5.

**C. Only after A: the 2x2.** Codex's proposed {loss} x {CUT support features} is
well-designed, but section 1 predicts its **loss axis will move `D/d` and not
`eps_op`**. That is a falsifiable prediction; if `eps_op` moves, the analysis above
is wrong and should be discarded. The feature axis is not predicted either way.

**D. Not recommended: more steps on the current head.** 256 -> 40000 steps across
three feature sets and two losses never brought the 6-D compliance error below 0.83
on any seat, and section 2 says why.

## 5. Concrete upgrades to the harness, by file

Ordered by (value / effort). None of these requires abandoning anything.

| # | change | where | effort | why |
|---|---|---|---|---|
| 1 | **Freeze and use a holdout.** Every seat in every run is `split: train`; `run.py:selected_rows` raises `VALIDATION_TEST_HOLDOUT_NUMERIC_ACCESS_FORBIDDEN` and is never given one. There is currently **no generalisation number in the project at all**. | `run.py`, manifest | config | the headline claim is geometry -> S; it has never been tested |
| 2 | **Raise cross-block sampling, or change what is sampled.** 8192 of ~9.1e6 per step = 4.2 looks per block per 30000 steps. Either sample far more, or stop supervising individual far blocks and supervise an aggregate of them. | `factors.sample_pairs`, `GeometryMixture` | one-function | this is a hard bound on how accurate a far block can get, independent of loss |
| 3 | **Wire `D` into the loss.** It is already implemented exactly in `mechanics.divergence` and used only for reporting. `d*D = tr(H) - logdet(H) - d`: `tr(H)` is `result.square().mean()` on the tensor `relative_rows_loss` already forms, and `logdet(H) = 2(sum log R_hat_ii - sum log R_star_ii)` is exact from the predicted pivots (the complete diagonal is decoded every step, never sampled). Zero extra forward passes. | `relative_action.py` | one-function | stops `mu_min` collapsing to -1.35e-16 and `logdet` drifting +-1.4e4. **Hygiene, not the gate** - see `OBJECTIVE_FIX_20260917.md` §3 |
| 4 | **Trust only the closed form on the soft side.** Codex already computes both and reports `divergence_spectrum_absolute_difference`; that difference is 27.46 on 0353's relative arm and `None` where `mu_min < 0`. Measured: at `mu_min = 1e-18` the eigen route's `D` is 12.11% wrong. | `run.py` reporting | config | soft-side numbers on the relative arm are currently the unreliable ones |
| 5 | **Warm-restart the LR when new geometries enter.** The 6 CUT seats arrived at step 30001 with LR already decayed from 2e-4 to 3.65e-5 and no adaptation phase. Codex flags this itself and does not claim it as the root cause. | `train_multi.learning_rate` | config | it confounds every conclusion about those 6 seats |
| 6 | **Turn TF32 on for the trunk, keep mechanics in fp64.** `execution.TF32: False`. The trunk is a 768x4 MLP whose output feeds an fp64 assembly; TF32 there is ~4x and does not touch the teacher, the quotient, or the spectrum. | execution config | config | the current cost measurement is ~4x pessimistic; 2070 ms -> ~540 ms |
| 7 | **Change the head from a complete factor to a hierarchical low-rank factor.** 1.22e7 numbers instead of 8.22e7, 20 ms instead of 135 ms at peak, and it is the one representation *measured* to pass +-3%. Caveat that must be respected: SVD factors are not a legal regression target (sign and rotation ambiguity - `FIRST_PRINCIPLES §2.4`), so the head must emit a canonically-fixed parameterisation, and `LIPSCHITZ.json` shows the representation does not transplant between seats (536-7298), so smoothness in geometry must be established first. | `factors.py`, model head | structural | the only factor-shaped representation known to be sufficient |
| 8 | **Keep, unchanged**: the geometry adapter and signed-CSR pullback, the frozen Householder quotient, SPD-by-construction via `exp(log_pivot)`, the whitened-spectrum and 6-D response evaluation, `test_contract.py`, the substitution-attribution tooling, and the provenance discipline (SHAs, explicit non-claims, both-ways metric checks). This is the expensive part and it is good. | - | - | - |

## 6. The three decisions that are yours, not mine

1. **What is the cost target?** The complete factor reaches ~135 ms/cell at hardware
   peak - 236x faster than the teacher. If that is the bar, cost is solved and only
   accuracy is open. If the bar is 1 ms, no factor-shaped output reaches it and the
   contract has to change.
2. **Module-level or task-level contract?** `eps_op` on a per-cell `S` is a strong,
   composable guarantee, and it is what makes the problem hard. If the deliverable
   is instead "a global reduced system that gives the right compliance and
   sensitivities", the collar route never forms `S` and the acceptance criterion
   changes accordingly.
3. **Is an inference-time coarse solve allowed?** The collar route needs one. If it
   is not allowed, that route is out regardless of what the floor scan says.
