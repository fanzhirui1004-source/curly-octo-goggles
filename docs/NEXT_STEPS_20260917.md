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

---

## 7. Addendum: what the literature already knows (all citations verified by fetching
## the arXiv pages, not from memory)

### 7.1 Someone has already done the geometry-parameterized version of this, and it works

**Jiang, Zhan, Zhang & Wang, "Convex Neural Energy Elements: Monolithic Finite-Element
Assembly of Geometry-Parameterized Neural Operators with Stability and Error
Guarantees", arXiv:2608.02036.** A geometry-parameterized hypernetwork emits a
Cholesky-like factor so the assembled element is convex in the boundary dofs by
construction - structurally the same idea as Codex's `exp(log_pivot)` SPD head.
Reported: **0.6-1.0% relative error on unseen geometries** and 175x faster setup, on
heat conduction, extensible to 3D and to mixed element types in one assembly.

Two transfers matter more than the headline:

1. **Field-regression-then-assemble is a documented dead end.** Training an operator
   to predict fields and then assembling gives an indefinite Hessian and Newton
   converging to a wrong answer at 247% error. Anyone tempted by "predict the local
   displacement field instead" should read this first.
2. **The regularization-nullspace principle.** If the regulariser's nullspace does
   not contain the *entire* physics nullspace, there is an irreducible error floor at
   **every** value of the regularisation parameter - in plane-strain elasticity,
   omitting only the rotation mode pins the floor at 0.85%, full Tikhonov at 6.7%.
   The stated signature of this failure is **a single-signed bias**.

   Codex measures a single-signed bias: `c18_signed_error` negative on all 8 seats,
   `c6_mu_max < 1` on all 8, 8-70x too stiff. That is the signature. Our quotient `B`
   removes exactly the 6 rigid modes and the teacher's `A = B S B^T` is built the same
   way, so on the face of it the nullspace is handled exactly - **but this is now a
   specific, cheap, checkable hypothesis** rather than an open question, and it should
   be checked before anything expensive.

### 7.2 The mean-versus-max diagnosis is confirmed independently, with a proposed fix

**Oh, Lee, Darbon & Karniadakis, arXiv:2606.21828.** An operator trained to relative
L2 of O(1e-3) still produces an indefinite discrete Jacobian, "because the
mean-squared training controls error on average while leaving localized pointwise
violations of the underlying physics". Their remedy is a short **label-free
fine-tuning phase penalising the operator against the discrete energy**, which moves
the Jacobian spectrum back to positive definite; 5.4x wall-clock speedup on a 3D
6.4M-dof hyperelasticity problem.

This is section 3 of `OBJECTIVE_FIX_20260917.md` reached by a different route, and it
supplies the missing half: the fix is not a differently-shaped mean, it is an
energy-based fine-tune that acts pointwise.

### 7.3 There is a named remedy for the near/far cancellation problem

**Ling, Ying & Zhou, "PPDNO", arXiv:2606.25952.** For Dirichlet-to-Neumann operator
learning across varying domains, the geometry-independent **leading operator is a
universal Fourier multiplier** computable by FFT; the remaining geometry-dependent
correction is **smoother**. They compute the principal part exactly and train a
low-rank DeepONet on the residual only.

Our `S` is a DtN-type operator and its near field is exactly the singular,
geometry-insensitive part that is eating 97% of the factor norm. **Subtracting an
analytically known principal part and learning only the residual is directly
implementable inside Codex's existing head** and attacks the measured problem at its
root rather than reweighting around it.

### 7.4 A deterministic algorithm with our exact acceptance metric may reduce the need
### to learn at all

**Schäfer & Owhadi, "Sparse recovery of elliptic solvers from matrix-vector products",
arXiv:2110.05351** (with the earlier Schäfer-Katzfuss-Owhadi, *SIAM J. Sci. Comput.*
43(3), 2021, arXiv:2004.14455). An elliptic solution operator is recovered to accuracy
`eps` **in operator norm** - the same norm as `eps_op` - as a sparse Cholesky
factorisation with `O(N log N log^d(N/eps))` nonzeros, from only
`O(log N log^d(N/eps))` matrix-vector products, total complexity
`O(N log^2 N log^{2d}(N/eps))`.

Two consequences:

1. **It turns section 7 of `WHERE_THE_ERROR_IS` from a scan into a theorem.** For
   `N = 12822` on a 2-manifold interface at `eps = 0.03`, the predicted nonzero count
   is `~2e7` - the same order as the measured hierarchical low-rank result of
   `1.22e7` at `eps_op = 2.6e-2`. Theory and measurement agree that **an
   operator-norm-accurate factor of this `S` costs ~1e7 numbers**. At the cost
   analysis's 2486 FLOP/number ceiling that is ~20 ms, so the factor route's floor is
   tens of milliseconds and **no factor-shaped output reaches 1 ms** - now with a
   proof behind it, not just a truncation scan.
2. **It may make the learning step unnecessary for the compression half.** If the
   constants are tolerable at our `N`, the sparse factor is computable deterministically
   with a guarantee, and the only thing left to learn is whatever is cheaper to predict
   than to compute. This has never been considered in this project and it is a
   one-day check of constants, not a research programme. The honest caveat: each
   matvec needs a solve with `K_ww`, and at `N ~ 2e5`, `d = 3`, `eps = 1e-2` the
   asymptotic count is tens of thousands of matvecs, which may lose to one direct
   factorisation. **The constants decide it, and they are cheap to evaluate.**

### 7.5 The one hard negative

The exact condensed operator is **provably nonlocal**; a local multiplicative
coefficient field cannot reproduce it exactly, only up to exponential-in-radius
truncation. So "amplification is exactly 1" holds for the *exact* local model and
every finite-radius local parameterisation carries an approximation floor. The
collar's proved `mu_max >= 1.111` at depth 1 with a Q1 core is an instance of exactly
this floor - which is why section 4's item A (the Smetana-Patera floor scan over
depth and enrichment) is the right experiment and not an optional one.

## 8. Revised priority, after the addendum

| | action | cost | decides |
|---|---|---|---|
| **A0** | Check the Schäfer-Owhadi constants for our `N` and interface dimension. | ~1 day, no training | whether a deterministic operator-norm-accurate factor is cheaper than learning one |
| **A** | Smetana-Patera Loewner floor scan (§4A). | no training, existing fine factors | whether any reduced local representation can reach `eta <= 0.05`; kills or sets the local route |
| **A1** | Test the regularization-nullspace hypothesis (§7.1) against our `B`. | hours | whether the single-signed 8-70x stiffness has a known, structural cause |
| **B** | Harness upgrades §5, plus **principal-part subtraction (§7.3)** as a new item between #3 and #7, and **energy fine-tuning (§7.2)** as the objective change that §3 says a differently-shaped mean cannot deliver. | days | - |
| **C** | Codex's 2x2, with the falsifiable prediction of §4C. | weeks of GPU | - |
