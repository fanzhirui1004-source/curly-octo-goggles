# The cut cell's operator is covariant; only the teacher's coordinates were not

2026-09-20. Teacher-side only, no training. This is the measurement the cut-cell programme was
missing, and it comes out positive.

## 1. The number

Seat 100000 (19 active cells, q = 711, 142 of its 237 coordinate rows are cut-surface residual
functionals), driven through the pinned teacher on the seat and on three of its rotated images:

| element | operation | plane after `g` | relative covariance residual |
| --- | --- | --- | --- |
| 1 | mirror z, `diag(1,1,-1)` | `(1, b, 0)`, unchanged | **9.51e-15** |
| 20 | 90 deg about z | `(-b, 1, 0)` | **8.59e-15** |
| 10 | 90 deg about x | `(1, 0, b)` - a z-tilt the dataset contains nowhere | **2.74e-14** |

The quantity is `||G S_tilde G^T - S_tilde(g . cell)||_F / ||S_tilde||_F` with `G = P_g (x) Q_g`,
where `S_tilde` is the trace operator pulled back to the background-node displacements of the cut
support (section 2). The Frobenius norm ratio is 1 to 15 digits in all three cases, the active node
set of each rotated cell is exactly the image of the base one, and the number of cut coordinates is
the same 142 - even though the *basis* of those 142 differs, which is the whole point.

So the CutFEM discretisation of a cut cell - its cut-surface quadrature, its ghost penalty, its
stabilisation - **is equivariant under the cube group to machine precision**. Nothing was assumed
here; g = 10 rotates the cut plane into a `z`-tilt that no seat in the dataset has, and the teacher
still returns the rotated operator.

## 2. Why the same operator looked non-covariant before

The teacher's cut coordinates are residual functionals chosen by a greedy pivot
(`max_pivot_geometric_residuals_v2`). The pivot order is not equivariant: for a rotated cell the
selected subspace is the same (measured earlier: subspace residual 1.6e-15) but only 144 of 1436
basis *rows* are shared. And a non-orthogonal change of basis does not commute with a matrix square
root, so the label `M_q = B^T A^{-1/2} B` cannot be covariant even when the operator is. That is
the whole of the cut family's symmetry problem, and it is a bookkeeping problem, not physics.

Write `q_c = C u` for the cut coordinates as functionals of the background-node displacements `u`
(`C` is the teacher's own CSR trace basis, and the box rows are single-node selections in the same
matrix). Pull the trace form back to `u`:

    S_tilde = (C (x) I_3)^T S (C (x) I_3).

Under any change of cut basis `q'_c = V q_c` the teacher's blocks move as `C' = V C`,
`S'_cc = V^-T S_cc V^-1`, `S'_bc = S_bc V^-1`, so

    C'^T S'_cc C' = C^T V^T V^-T S_cc V^-1 V C = C^T S_cc C,        S'_bc C' = S_bc C.

**The pullback is exactly invariant to the pivot choice.** It is therefore covariant iff the
discretisation is, and section 1 says the discretisation is. The teacher's data has been covariant
all along; we were reading it in a gauge that hid it.

## 3. It keeps the cut surface loadable, and route B buys nothing on top

The user's constraint is that the cut surface must stay loadable, which is what ruled out
condensing it away. The pullback keeps it, and exactly:

* a surface traction `t` on the cut surface enters as the nodal load `l_a = int_Gamma t . phi_a`;
* if `u` is in `ker C` then `sum_a u_a phi_a` vanishes on Gamma, so `l . u = int_Gamma t . 0 = 0`;
* hence `l` is orthogonal to `ker C`, i.e. `l` lies in `range(C^T)` = the range of the pullback.

So **every physical surface traction is representable**, and the rank deficiency of `S_tilde`
(rank `3(n_b + n_c)` inside `3 n_support`) is precisely the subspace of nodal patterns that are
invisible on the cut surface - patterns whose energy the trace operator legitimately does not carry.

That settles the route A / route B comparison stated to the user earlier. Codex's route B -
re-running the teacher's numeric stage while retaining the cut-support background nodes as unknowns
- lands on **the same coordinates**, costs a fresh numeric stage per seat (207-282 s measured, so
8-11 h for the 139 in-domain cut seats), yields a *different* operator that no longer matches the
frozen dataset, and its only extra content is the response to nodal loads in `ker C`, which no
traction can produce. Route A reaches the same place from the packets already on disk, with zero
teacher time. Route B is dominated and is not being run.

## 4. What it costs

Measured on all 184 cut seats from the frozen trace caches alone (`docs/data/cut_rotation_20260920/support.py`):

| | min | median | max |
| --- | --- | --- | --- |
| nodal dofs / teacher q | 1.002 | **1.309** | 1.406 |
| dense storage, square of the above | 1.00 | **1.71** | 1.98 |

The cut support and the box nodes are disjoint in every seat, so the layout is clean: box nodes then
cut-support nodes. Seat 100000: q = 711 becomes 891 nodal dofs on 297 nodes, growth 1.25, and the
teacher's own `body_dimension` for that cell is 891 - the pullback is exactly the cell's active node
space, which is why nothing has to be invented to index it.

## 5. What this changes about training, and what it does not

The lever this unlocks is the only one in the project measured to convert a fit into
generalisation: the 48-fold cube symmetry, which cut cells have never been able to use. Section 1
says the symmetry is there in the truth. To use it the *target* has to be gauge-invariant, because
the network's own output is expressed in whichever cut basis the presented cell happens to have:

* an entrywise loss on `M_q` is gauge-dependent and is what every cut arm so far has descended;
* the pullback `S_tilde` is gauge-invariant, and so is the assembly-free response - nodal loads
  transform as `l -> G l` and nodal displacements as `u -> G u`, with no basis question anywhere.

So the change is to the loss and the augmentation target, not to the architecture or the output
size. That is the next piece of work, and it is the first cut-cell arm with a reserved held-out set
declared before it runs (see `THE_CUT_FAMILY_IS_TWO_PARAMETERS_20260920.md` section 6: only 2 of 184
cut seats were never presented anywhere, so the reservation has to be made when the new target is
built).

What is **not** established: this is one seat, and a small one (19 active cells), with 3 of the 48
elements. The same witness has to run on a large seat - 100032 is 3651 active cells, q = 10812 - and
on an improper element other than a mirror, before the result is quoted as a property of the family
rather than of one cell. The wrapper takes `--elements`, so that is a launch, not new code.

## Provenance

`superelement/objective/cut_rotation_witness.py` (geometry / numeric / compare).
`docs/data/cut_rotation_20260920/` carries `support.py`, `run_rotation_witness.sh` and
`COVARIANCE_100000.json` with the pins: seat 100000, `SAMPLE.json`
`22c3d40e...`, trace cache `e6b1e0a7...`, `S_UPPER.npy` `ca9366a2...`, teacher commit
`6624dc86706bbd8e3cc16b4b3adbfb043eff08ec`, wrapper `04611f66...`.
Remote outputs: `/root/autodl-tmp/CLAUDE_CUT_ROT_20260920/`.

## 6. A hypothesis this raised and then killed: the coordinates are NOT under-resolved

Section 5 says the augmentation target has to be gauge-invariant. That prompted a sharper
hypothesis, which would have been the whole answer if true: that a cut coordinate is not resolvable
from what the model is handed, so the training target is not a function of the training input, and
no amount of data or capacity could fix the cut family.

The model sees a coordinate through two channels: explicit aggregates (centroid, kind, signed sum,
`|c|` sum, `||c||`, log nnz, support radius, face membership, field values, first and second
moments) and the pullback of a learned volume field, `sum_a c_ia f(x_a)`, with `f` trilinearly
sampled from a 32^3 grid. The second channel is linear in the grid values, so it is exactly a weight
vector `w_i` over the 32^3 cells, and two coordinates with equal `w` and equal aggregates are
indistinguishable *whatever the network learns*.

Computed exactly (`docs/data/cut_rotation_20260920/injectivity.py`; the analytic trilinear map
reproduces `torch.grid_sample` to 4.4e-16), on seat 100000:

| | closest relative `w` distance | that pair's aggregate distance | that pair's operator-row distance |
| --- | --- | --- | --- |
| cut coordinates | 0.216 | 0.338 | 2.93 |
| box coordinates (control) | 0.707 | 0.026 | 1.09 |

**Refuted.** The closest pair of cut coordinates is 22 % apart in the field-pullback channel and 34 %
apart in the explicit aggregates - far from degenerate. And the control cuts the other way: box
coordinates, which do generalise, sit at aggregate distance 0.026 with operator rows already 1.09
apart, so "near-identical descriptor, different row" is the normal state of a cell that works. The
cut family's failure is not an input-resolution defect, and section 5's claim is narrowed
accordingly: the pullback buys a gauge-free target and one uniform coordinate type, not a repair of
an ill-posed map.
