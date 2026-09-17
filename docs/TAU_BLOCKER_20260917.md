# The tau-sensitivity gate needs new teacher labels, and why I did not fake it

2026-09-17. The user set the design variable to TPMS thickness `tau`. This records
what that requires, why the existing dataset cannot supply it, and what was run
instead.

## 1. What the tau gate needs

With `c` the assembled compliance and `tau_m` the thickness of module `m`,

        dc/dtau_m = - u_m^T (dS_m/dtau) u_m

`THE_REAL_GATE_20260917.md` measured the version of this with a uniform stiffness
multiplier, where `dS/drho = S` and the answer is 0.37% at module `eps_op` 0.152.
With `tau` the quadratic form is still an energy, so the `u`-error still enters at
second order - but the result now depends on **`dS/dtau` being right**, which that
test does not exercise. Getting `dS/dtau` needs the teacher's `S` at `tau - delta`,
`tau`, `tau + delta` for the *same* cell.

## 2. The dataset cannot supply it

The cell geometry is fully determined by 12 numbers: `tau_corners` (8, trilinear
thickness field) and `cut_plane` (4). The TPMS function itself is universal
(`phi = sum cos(2 pi x)`, `adapter.geometry_values`). So the 33-seat dataset is a
12-parameter family and in principle spans `tau`.

Measured: `tau` mean runs from 0.194 (seat 0328) to 0.578 (seat 0483), and **32 of
33 seats share the identical cut plane [1,0,0,2]** (only seat 0366 differs). But no
two seats are the same thickness *field* shifted. Closest pairs by mean thickness,
with the non-uniform residual `max|d tau - mean(d tau)|`:

| pair | d(mean tau) | non-uniform residual |
|---|---|---|
| 0226 -> 0196 | 0.00027 | 0.0836 |
| 0220 -> 0895 | 0.00032 | 0.0807 |
| **0253 -> 0328** | **0.00053** | **0.0186** |
| 0399 -> 0434 | 0.00046 | 0.2471 |

The best pair (0253/0328) still has 97% of its difference in the *shape* of the
thickness field rather than a uniform offset, and the two cells have different
interface sizes (q = 12,828 vs 12,798) so their operators cannot even be subtracted.
**A finite difference in `tau` is not available from this dataset.**

## 3. What it would take

Two extra teacher labels at `tau +- delta` for one seat. The compute is minutes
(39-95 s each). The work is driving the frozen CutFEM pipeline
(`CUTFEM_GP_TASK_V2_20260915_R1/FROZEN_FE_6624dc8/scripts/`, several hundred
scripts, with `cutfem_full_interface_stage.py` only a wrapper taking `--run-id` and
a REMAINDER command). A wrong quadrature policy, ghost-penalty face set or ordering
would produce a confident and wrong derivative, which is worse than no answer. The
correct order of work is: **first reproduce an existing label bit-for-bit, then
perturb `tau`.** That is a few hours, not a few minutes, and it has not been done.

## 4. What was run instead

The other half of the proposal, which is independent of `tau` and decides whether
the 23,016-number route is worth pursuing at all: **does a per-parameter relative
error on the collar's core material map 1:1 to `eps_op`?**

The parameterisation turns out to be exactly linear, which makes the test exact:

| artefact | content |
|---|---|
| `B_ZERO_R1/K_COLLAR1.npz` | 62,088 x 62,088 assembled collar + coarse core, **true material, zero learned parameters** (`ASSEMBLY.json`: `parameters = 0`), 13,232,536 nonzeros |
| `B_CORE_COMPILE_R1/ENERGY_COEFFICIENTS.npy` | (1096, 21, 24, 24) - core cell `c`'s stiffness is `sum_p theta[c,p] E[c,p]` |
| `B_CORE_COMPILE_R1/CORE_DOFS.npy` | (1096, 24) scatter targets |
| `ASSEMBLY.json` | trace 12,798 + internal 49,290 = 62,088; `potential_core_material_parameters = 23016` = 1096 x 21 |

So `K(theta) = K_true + sum_c scatter(sum_p delta_theta[c,p] E[c,p])` exactly.
Perturb `theta` by independent relative noise `eps`, condense the 49,290 interior
dofs, map to the quotient, whiten against the teacher's factor, report `eps_op(eps)`.

Two self-tests, both mandatory:

1. **Quotient round trip** `A -> S -> A`, which caught a real error in the first
   draft (`project` is `lift(self(x))`, an orthogonal projection `(q,n)->(q,n)`, not
   the reduction; the reduction is `quotient(x)` itself). Now **3.263e-17, exact**.
2. **`eps = 0` must reproduce the recorded zero-learning spectrum**
   `mu in [1.000, 1.855]` (`REVIEW_20260916` stage-2 table). `mu >= 1` is a theorem
   here - the collar space is a subspace of the fine space carrying the same energy -
   so `mu_min < 1` means the assembly or the dof ordering is wrong and the run is
   void. The script aborts rather than reporting perturbation numbers.

Script: `superelement/objective/amp_collar.py`.
