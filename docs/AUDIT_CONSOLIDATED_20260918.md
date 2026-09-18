# The audit of the equi package, consolidated

2026-09-18. Two rounds over `superelement/equi/` (six dimensions, then four targeted
questions), 40 raw findings, deduplicated to 18 actionable items. All are applied
(`1130197`); the three seat-0253 arms are being re-run because four of them move numbers.

Raw findings: `docs/data/audit_20260918/ROUND1_FINDINGS.json`.

## 1. What was checked and found CORRECT

This half matters as much as the defects: it is what the next person does not have to
re-derive. Every number below is a measurement someone ran, not a reading of the code.

**The augmentation pairing is correct, and now for a stated reason.** The training step
rotates the input context by `g` (positions move, node indices do not) and rotates the
target blocks by the same `g` with no node permutation. Derivation: `M_q` is basis-free, so
it depends only on `Pi = B^T B`, i.e. only on `span(rigid)`; `span(rigid)` is
`G`-invariant (translations map to translations; the rotation mode `w` maps to the rotated
cell's mode with axis `det(Q) Q w`, which stays in the 6-space for reflections too);
therefore `M_q(g . cell) = G M_q G^T` block-wise with no permutation. Verified at
**1.98e-15 over all 48 elements**, with a *different* Householder `order` on the rotated
side — which is what makes the basis-independence a measurement rather than a docstring.
No permutation is needed because the label is read at the original index pair and handed to
a context in which that index has moved: the binding is inherited, not re-derived.
`cubic_group.node_permutation` is for comparing against a teacher run of the rotated
geometry, and training never calls it.

**`rotate_target_blocks` agrees bitwise (0.0) with the directly built rotated label**, over
all 48 elements on a genuine symmetric `M_q`. Two things fall out that were not obvious:
the symmetrise → rotate → re-mask on the diagonal block is *mandatory*, because for a
signed permutation `(Q M Q^T)[a,d] = sign(a) sign(d) M[pi(a), pi(d)]` draws from **both**
triangles; and the rotated diagonal pivots stay strictly positive (min 0.0498), which the
log-pivot loss requires. The five calibrated conditioning scales are rotation-invariant to
2.2e-16, so calibrating on unrotated labels is sound.

**`--proper-only` is not needed.** Nothing in the feature set is a pseudo-scalar or
pseudo-vector: the node scalars are true scalars, every vector channel is the gradient of a
scalar (hence polar), faces and corners permute, and the label's blocks are a true rank-2
tensor field of a matrix function. The identity holds over the 24 improper elements at the
same 1.98e-15. The flag is kept only as an escape hatch if the teacher should turn out to
break reflection symmetry more than rotation symmetry, which is unmeasured either way.

**The resampling really does commute with the group — including the one I expected to
fail.** `avg_pool3d(.,2)`: under `i -> n-1-i` with `n` even, block `{2m, 2m+1}` maps to
block `{n/2-1-m}`, so the partition is preserved and the induced action on the pooled grid
is exactly `cell_source_index(n/2, g)`; 0 failures of 48 at every level of 32→16→8→4, max
error 3.3e-16. **Nearest upsampling by 2 also commutes exactly, error identically 0.0** —
it is the transpose of the partition that pooling preserves, so it inherits the property.
Both are now asserted in `model.selftest` (1.2e-07 float32 and exactly 0.0).

**The feature rewrite lost no physical information.** `(tau-phi >= 0) & (tau+phi >= 0)` is
**exactly** `tau >= |phi|` — 0 disagreements in 600 300 adversarial samples including exact
ties, in both float64 and float32, with no sign caveat (and `tau` is in fact always
positive: the trilinear weights are non-negative and sum to 1 to 2.2e-16). Dropping the two
signed channels loses nothing because `tau` and `phi` are both kept, so both are recoverable
to 0.0. `cut_margin = 2 - p_x >= 0` identically on the unit cube for the inert plane, so
that channel was constant. `field_values` reproduces the frozen `adapter.geometry_values`
to **0.0** (not 1e-15) at both node points and cell centres, and the corner index convention
`4bx+2by+bz` is byte-identical across `context.trilinear`, `adapter.geometry_values` and
`cubic_group.CORNER_BITS`. Under the box-only contract the adapter's dropped statistics
(`signed_sum`, `abs_sum`, coefficient norm, `log1p(lengths)`, `signed_moment`, `variance`,
`kind`) are constant or exact functions of retained channels.

**The physics chain is the validated one, matrix for matrix.** `platen.py`'s
`lift(lift(A).T).T` equals `run.py:six_response`'s construction of `S = B^T A B`
(`||S_run - S_platen|| / ||S_run|| = 4.69e-15`); `S @ u` is literally
`F.full_action(R, Q, u) = B^T A B u`; the prescribed/free split, the `rsqrt` rescaling, the
Cholesky and the residual check are identical. The exact label reproduces `run.py`'s
reference compliances to 1.3e-15 with the same `trace(H^-1)`.

**The label and the evaluation chain** (checked by hand with a numpy port of
`RigidQuotient`): `B B^T = I` at 4.4e-16, `B @ rigid = 0` at 3.1e-16, `Qt(Qt(M_q).T)`
recovers `A^{-1/2}` at 4.8e-16, and `mu = 1/eig((Mhat R^T)^T (Mhat R^T))` equals
`eig(R^-T A_hat R^-1)` at 1.1e-15 with exact-label `g = 1 + 6.4e-15`. Writing `R` instead
of `R^T` gives `g = 1.72` on the same small case — the route-5 trap, correctly avoided.
`predict_full_q`'s triangular index inversion is exact at every triangular number.

## 2. What was wrong, and is now fixed

### Moves numbers (hence the re-run)

| # | where | defect | measured |
|---|---|---|---|
| 1 | `train_equi.py:220` | `factor_relative` compared a full symmetric prediction against a triangle-only label | a bit-exact prediction scored 0.36, not 0 |
| 2 | `floor_probe.py:202` | compliance attribution weighted `a^2/lambda`; the exact first-order weight is `a^2 lambda` | reported −0.169166 where the exact error is −0.111117; softest-decile share 95 % instead of 62 % |
| 3 | `context.py:70` | a hard `solid = 1[margin >= 0]` volume channel: a step function of tau, discontinuous where the design derivative is needed, and redundant with `margin` | 4 voxels flip at eps = −1e−3; the artefact is 34x the smooth signal |
| 4 | `context.py` | volume and node fields unnormalised | `grad_phi` held 97 % of the input variance and does not depend on tau at all; `tau` held 0.0024 %. Now max share 0.278 |
| 5 | `train_equi.py:399` | the group draw consumed the pair-sampling generator, so the three arms saw different pair sequences | the A1 comparison was not single-variable |

Item 2 is a correction to a published document: see §3. Items 3 and 4 change the model's
input width, so the A1 checkpoints are not loadable by the new code; that is accepted, they
are a finished experiment.

### Guards for things that were unguarded

6. `train_equi.prepare` and `build_mq_label.build` now assert that `span(cache['rigid'])` is
   the rigid-body space of the node positions — the one algebraic assumption the
   augmentation label rests on. Both arrays were already in hand and nothing checked it.
7. `build_mq_label` binds the trace cache's sha256. **Every self-test in the builder passes
   with the wrong cache**, because it is self-consistent in whatever basis it is handed; this
   is the only thing between a mismatched `--trace-cache` and a silently wrong label. Phase 3
   builds five labels from five caches, so this was load-bearing.
8. `build_mq_label` now **gates** on its six self-tests (`label_g`, `round_trip_to_M`,
   `MMG_residual`, `back_squared_times_A_minus_I`, `rigid_nullspace`, `rigid_span_residual`)
   rather than only recording them, and writes `MQ_FAILURE.json` on a breach. Essential
   before a 95-seat batch build. `sym_residual_before_average` is now recorded *before* the
   symmetrising average, so it can fail; `packed_symmetry_exact` was a tautology and is gone.
9. `near_radius` is a registered buffer. It decided both which head runs and which normaliser
   scales the output, and was a bare class attribute outside the `state_dict`.
10. `--proper-only` raises without `--augment` (it was silently ignored, so a run could claim
    `proper_only: true` having applied no rotation at all).
11. `--canonical` raises on a seat whose corner stabiliser is non-trivial. There the
    lexicographic max is a coset, not an element, so the cell would be presented in several
    orientations while the protocol claims one. 144 `AFFINE` seats are candidates.
12. The far-pair rejection loop is bounded, and an empty far bucket raises at `prepare` time
    instead of spinning forever with no output.
13. `platen.py` refuses a non-box-only trace (`run.py` has this guard; `platen.py` depended on
    the same assumption twice without it) and records provenance — both sha256s and the tau
    corners — since five Phase-3 runs differ only by which file is passed.
14. `model.selftest` covers the resampling claim and forces diagonal-block coverage: the
    shipped seed drew **zero** diagonal pairs with 94 % probability, so the whole pivot branch
    (`smooth_interval_bound`, `exp(log_pivot)`, `diag_embed`) was never executed.
    `GroupTables`' scalar/vector split and `EquiModel`'s input width now derive from
    `context.VOLUME_*_NAMES` instead of being hard-coded 4 and 10.
15. The canonical arm's rotation-consistency probe is vacuous — with a trivial stabiliser
    `view_frame` is constant in `g`, so the probe re-ran a bit-identical inference and
    reported 0.0 whatever the model does. It now writes that note instead of paying for six
    full 9.14M-block inferences.

## 3. Corrections to published documents

- **`THE_FLOOR_IS_THE_SOFTEST_DECILE_20260918.md`.** Its §1 load-energy table uses
  `a_j^2 lambda_j` and is right (re-run with the fixed code reproduces 0.9610, 0.9447,
  0.9448, 0.9411 exactly). Its claim that the compliance error "attributes to that decile and
  nowhere else — `−0.0000` in every other decile" came from the `a^2/lambda` weight and
  overstates the concentration. The qualitative conclusion survives; "and nowhere else" does
  not.
- **`PAST_THE_FLOOR_20260918.md` and `HANDOVER_EQUI_20260918.md`: the "2.0e-13 against the
  teacher" claim is wrong.** `RESPONSE.json['reference']` is `six_response(R_*)` where `R_*`
  is the reference *factor*, never the teacher's `S`. So the 2e-13 establishes that
  `MQ_UPPER` packing, `B M_q B^T = A^{-1/2}`, `(B M_q B^T)^{-2} = R_*^T R_*` and the platen
  solve are mutually consistent to 1e-13 — a round trip of the quotient algebra, which is
  worth having, but not an independent physics validation. Nothing in the repository compares
  a platen response against `S_UPPER` directly.
- **The rigid-nullspace leak is provably invisible to every single-cell metric.** Since
  `B B^T = I` and `B N = 0`, `B (I - Pi) = 0`, so `B E B^T = B (Pi E Pi) B^T` exactly for any
  error `E`. The leak measures precisely the component that the sandwich discards, so
  `Pi M_q Pi` is a bit-level no-op for `g`, `mu_min`, `e_A` and the assembled physics. Drop
  it from the Phase-0 triage; keep it as a loss diagnostic (capacity spent on a subspace no
  reported metric scores), and it does matter for the Phase-6 assembled solver, where the
  cells are not quotiented.
- **Handover Phase 3's instruction to edit `metadata.geometry.tau_corners` sets nothing.**
  `compile_equi_inputs` takes tau from `cache['tau_corners']` in the TRACE_CACHE, exactly as
  the frozen adapter does; `metadata['geometry']` is read only for `representation` and
  `box_max`. Any Phase-3 shim must write the perturbed corners into the npz, or assert the
  two agree.
- **Handover Phase 5's G-CNN recipe is under-specified in a way that would not work.** The
  27 offsets of a 3x3x3 kernel fall into 4 orbits under the 48 elements (centre, faces,
  edges, corners), so a kernel literally tied as `w[Qd] = w[d]` is isotropic with 4 free
  weights — "48 朝向权值共享" has to mean a regular-representation lift with 48 orientation
  channels per base channel, not an invariant kernel. The input layer is a *lifting* layer
  for the reducible representation `rho = I_3 (+) Q (+) Q` (3 scalar and 2 vector volume
  channels), so its kernel must satisfy `w[Qd] = rho(g) w[d] rho(g)^T`, not just
  `w[Qd] = w[d]`. And after the lift, `GroupNorm` and the global `mean`/`amax` pools stop
  being equivariant unless they also reduce over the orientation axis. Separately, the
  handover's claim that the convolution kernels are the *only* learned non-equivariance is
  true of the volume encoder alone: the node and pair encoders and the three 9-component
  heads are non-equivariant by construction and are 62.5 % of the parameters. Phase 5
  replaces both halves.

## 4. Open, not defects

- **The teacher's own equivariance is still unverified**, and it is the premise of the augment
  arm: the labels assume `S(g . geometry) = P_g S P_g^T`. A census of all 273 labelled seats
  found **zero** with a non-trivial cube stabiliser, so the witness cannot be run on existing
  labels; the closest seat is 100155 at 0.1 % of the corner spread, which is a loose bound
  given that a 0.01 % thickness change is known to move `mu_max` by 60 % when a cut cell is
  born. A clean witness needs the teacher (~180 s) on a constructed symmetric cell. If the
  teacher breaks symmetry at relative `epsilon`, the augment arm's achievable `g` is floored
  near `1 + O(epsilon)`: harmless at the expected 1e−10, fatal to the comparison at 1e−2.
- **`build_mq_label` holds about seven simultaneous `q x q` float64 buffers**: ~9.2 GB at
  q = 12 828 and ~22.4 GB at q = 20 000. That fits the 32 GB card only when it is otherwise
  empty, which is exactly the condition that produced 18 consecutive OOMs during the
  LABELS95 build. Sequence the 95-seat build against an empty GPU, or run it on CPU.
