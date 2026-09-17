# The gate is 0.15, not 0.03 - measured on the deliverable

2026-09-17. Four experiments run after the user set the contract at the task level
(only the assembled lattice's compliance and optimisation sensitivities must be
right), the cost target at 100 ms per cell, and allowed an inference-time coarse
solve. Scripts in `superelement/objective/`, data in
`docs/data/codex_probe_20260917/`.

## 1. E1 - what the deliverable actually needs

Two copies of seat 0328 bonded on their 696 matching face nodes, 23,508 assembled
trace dofs, four self-equilibrated load cases, rigid modes removed by projection.
Design variable: a uniform stiffness multiplier on each module - the standard SIMP
knob - so the exact adjoint is `dc/drho_m = -u_m^T S_m u_m`. The optimiser only ever
has the approximate model, so it computes `-uhat_m^T Shat_m uhat_m`; that is what is
compared. Exact-arm identity `sum_m dc/drho_m + c = 0` holds to 1e-14..1e-16.

Modules are the hierarchical factor approximation at four tolerances:

| module `eps_op` | numbers | worst displacement | worst compliance | **worst sensitivity** |
|---|---|---|---|---|
| **0.152** | 8.58e6 | 4.09e-2 | 3.09e-3 | **3.74e-3** |
| 0.081 | 1.02e7 | 2.33e-2 | 3.30e-3 | **4.11e-3** |
| 0.026 | 1.22e7 | 9.41e-3 | 2.54e-3 | **2.59e-3** |
| 0.011 | 1.41e7 | 3.64e-3 | 1.04e-3 | **1.11e-3** |

**A module that fails the +-10% work gate by 1.5x delivers compliance to 0.31% and
optimisation sensitivities to 0.37%.**

`ASSEMBLY_TOLERANCE_20260916.md` predicted the sensitivity would be *first* order in
the operator error, like displacement (4.1%), and named this as the experiment that
should set the gate. It is not: it comes out at 0.37%, tracking compliance rather
than displacement. The reason is that `dc/drho_m = -u_m^T S_m u_m` is itself an
energy, and `u` is stationary for the energy, so the `u`-error enters at second order
exactly as it does for compliance. The worry was well posed and the answer is
favourable.

The sensitivity column is **not monotone** (3.74e-3, 4.11e-3, 2.59e-3, 1.11e-3), so
like the compliance column it has hit this construction's own floor near 2.5e-3. The
true requirement is looser than this test can resolve.

**Gate: module `eps_op` ~0.15 for sub-1% compliance and sensitivities.** That is 10x
looser than the +-3% target gate that has driven every decision in this project, and
1.5x looser than the +-10% work gate.

Scope, honestly: one seat, a two-cell strip, four loads, and a design variable that
scales a module uniformly. A *shape* variable (TPMS thickness tau) gives
`-u_m^T (dS/dtau) u_m`, still an energy and still second order in `u`, but it needs
`dS/dtau` to be right - a quantity this test does not exercise. That is the one
remaining gap before "0.15" can be called the gate without qualification.

## 2. E1b - where Codex's actual network lands on that gate

The same strip, seat 0253, modules replaced by Codex's **actual predicted factors** at
40,000 steps:

| arm | module `eps_op` | worst displacement | worst compliance | **worst sensitivity** |
|---|---|---|---|---|
| baseline | 273.9 | 2.51 | 4.63e-1 | **4.82e-1** |
| relative | 264.0 | 1.42 | 3.15e-1 | **4.08e-1** |

**41-48% sensitivity error.** A topology optimiser given gradients that wrong does not
converge to anything meaningful. The relaxed gate does not rescue the current network:
264 against 0.15 is still a factor of ~1700.

## 3. E3 - the failure is concentrated on the softest directions

Seat 0253, baseline arm. For `H v = mu v` the physical direction is `x = R*^-1 v`,
whose predicted/true energy ratio is `mu` and whose true stiffness is `1/|R*^-1 v|^2`.
Deciles by true stiffness, softest first:

| decile | true stiffness | median `mu` | max `mu` | % > 1.1 | % < 0.9 |
|---|---|---|---|---|---|
| 0 (softest) | 3.08e-6 | 4.473 | **274.9** | 98.7% | 1.3% |
| 1 | 1.45e-5 | 1.641 | 2.926 | 54.0% | 46.0% |
| 2 | 2.62e-5 | 0.577 | 1.840 | 48.1% | 51.9% |
| 3 | 4.83e-5 | 0.690 | 1.451 | 38.5% | 61.5% |
| 4 | 8.44e-5 | 0.796 | 1.285 | 42.1% | 57.9% |
| 5 | 1.41e-4 | 0.862 | 1.167 | 28.2% | 54.8% |
| 6 | 2.12e-4 | 0.905 | 1.108 | 0.2% | 48.2% |
| 7 | 2.84e-4 | 0.976 | 1.066 | 0.0% | 9.0% |
| 8 | 3.42e-4 | 0.969 | **1.038** | 0.0% | **0.0%** |
| 9 (stiffest) | 3.90e-4 | 0.969 | **1.029** | 0.0% | **0.0%** |

All 20 modes driving `eps_op` sit in the bottom 7% of true stiffness. **The stiffest
20% of directions already pass the +-10% work gate**, and the stiffest decile is
within 2.9%.

This vindicates the user's long-standing objection that the extreme-soft directions
are not where the physics is. It does **not** say only the softest decile is wrong:
deciles 2-6 still have 30-50% of their modes below 0.9, i.e. typically 10-42% too
soft. Accuracy degrades monotonically with softness; `eps_op` reads only the end of
that curve. And E1b shows the assembled error is still 41-48%, so the middle of the
curve matters too.

## 4. E2 - Schaefer/Owhadi does not apply here, measured

Seat 0328. Maximin (farthest-point) ordering of the 4,264 trace nodes, radius-`rho`
truncation of the reordered Cholesky factor:

| ordering | `rho` | nnz | % of dense | `eps_op` |
|---|---|---|---|---|
| maximin | 8 | 2.39e6 | 2.92% | 206.6 |
| maximin | 16 | 5.78e6 | 7.06% | 129.7 |
| reverse maximin | 8 | 8.53e6 | 10.43% | 68.6 |
| reverse maximin | 16 | 1.92e7 | 23.44% | 13.0 |

Against, on the same seat: hierarchical low rank **2.6e-2 at 1.22e7 numbers (14.95%)**,
and even plain magnitude truncation **1.15 at 12.2%**. The maximin radius pattern is
~500x worse than the hierarchical partition at comparable budget and is beaten by
magnitude truncation.

`FIRST_PRINCIPLES_PATH_20260917.md:195` already stated the reason - the screening
theory does not apply to **boundary-restricted** operators. This makes that a
measurement rather than an assertion, and closes the line. One variant is untested:
the screening theory is about the factor of a *Green's function*, so it may apply to
`A^-1` rather than `A`. A sparse factor of `A^-1` gives cheap *solves*, not the *apply*
that assembly needs, so it would only help under an iterative global solve.

## 5. What this all means

| | before today | after today |
|---|---|---|
| teacher cost | "31.85 s" | **39-95 s** (`TEACHER_COST_20260917.md`) |
| cost target | undefined | 100 ms - and the complete factor is 135 ms at TF32 peak, the H-matrix 20 ms. **Cost is solved.** |
| accuracy gate | `eps_op` 0.03 asserted | **`eps_op` ~0.15 measured on the deliverable** |
| H-matrix budget at the gate | 1.22e7 numbers | **8.58e6 numbers** |
| collar + coarse core (`eps_op` 0.15, 23,016 params) | "blocked by a proved floor `mu_max >= 1.111`" | the floor is **0.111 < 0.15**, so **it is no longer blocked** |
| Codex's network | `eps_op` 264 | 264, and **41-48% sensitivity error** when assembled |

The single largest change: **the collar route's proved kinematic floor (0.111) now
sits inside the gate (0.15) instead of outside the gate (0.03/0.10).** The
construction that was declared dead - "no positive-definite material can pass +-10%"
- is alive under the contract the user has chosen, and it hits the gate with 23,016
parameters where the complete factor needs 8.2e7 numbers and misses by 1700x.

And the learning problem it poses is smaller in both directions: ~23,000 outputs
instead of 8.2e7 (3,500x fewer), and - if the claimed 1:1 mapping from per-parameter
relative error to `eps_op` holds, which is asserted in `FIRST_PRINCIPLES §2.3` and
only partly measured - a per-number tolerance of a few percent instead of 3e-4.

## 6. What should happen next

1. **Verify the 1:1 amplification claim for the collar parameterisation** by direct
   perturbation, the way E3 of `WHERE_THE_ERROR_IS` measured `eps_op = 104 * eps` for
   the dense factor. If it is 1:1, the collar learning problem is ~100x looser per
   number and 3,500x smaller than the one Codex is solving.
2. **Close the shape-sensitivity gap in §1** - rerun E1 with a `tau` design variable
   using the teacher's own `dS/dtau` (Codex's F6 already computes a central-difference
   derivative at `2.31e-4` relative Frobenius agreement).
3. **Then, and only then, the memorisation test**: can a network predict the 23,016
   collar parameters for one seat? The project has never produced a geometry-conditioned
   neural result below `eps_op` ~4 on any seat; 0.15 is the target and the collar
   parameterisation is the first one where that target is not obviously out of reach.
4. Keep Codex's harness - adapter, frozen quotient, SPD-by-construction, spectrum and
   response evaluation, contract tests, provenance. Swap the head. The eight upgrades
   in `NEXT_STEPS_20260917.md` §5 still apply, with #1 (a real holdout) now the most
   urgent, because nothing in this project has ever been evaluated out of sample.
