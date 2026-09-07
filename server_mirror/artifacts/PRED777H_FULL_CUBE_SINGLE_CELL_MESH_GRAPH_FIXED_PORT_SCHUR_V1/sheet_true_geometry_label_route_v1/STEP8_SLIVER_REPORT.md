# Step 8: the sliver artefact in the label operator, root-caused and removed

Sheet route, 2026-09-05.  Evidence scripts: `scripts/pred777h_full_cube_v1/diagnostics/`.

## 1. The symptom

The largest eigenvalue of the fixed-port operator is not a property of the cell.  For the same geometry at three mesh
tiers (carrier 1/32 fixed, q identical):

| cell | fast | production | reference | fast / ref | prod / ref |
|---|---|---|---|---|---|
| pop_uncut_0658 | 0.3893 | 0.0362 | 0.0343 | 11.3 | 1.06 |
| pop_cut_1215 | 0.2330 | 0.0389 | 0.0365 | 6.4 | 1.07 |
| pop_cut_0473 | 0.1358 | 0.0404 | 0.0405 | 3.4 | 1.00 |
| pop_uncut_0529 | 0.0366 | 0.0656 | 0.0335 | 1.09 | 1.96 |

The eigenvector of lambda_max in the bad labels puts 97 % to 100 % of its mass on the four corners of ONE carrier
square (a healthy one spreads over 10 or more nodes).  Over the 286-label production sample, 26 % of the labels have
90 % or more of that mode on one square, 6.6 % have 99 % or more.  The six uniform-strain stiffnesses are unaffected
(0.03 % to 0.2 % across tiers): the mode is numerically huge and physically inert, which is the worst case for a
training target, because it is the largest entry scale of the matrix and it varies tenfold between meshes.

None of the operator gates (rigid residual, positive semidefiniteness, nonmanifold facets) sees it, and the mesh's
minimum dihedral angle correlates with it at 0.22 only.

## 2. The mechanism, step by step

1. **Every sub-degree tetrahedron stands on a needle CAP triangle.**  In five meshes (three tiers, cut and uncut)
   every tetrahedron below 1 degree has three vertices on a box-face cap triangle and one interior vertex; 41 of 41
   such tetrahedra in three production meshes stand on a cap triangle of angle below 1 degree (0.0025 to 0.6 degrees).
   A tetrahedron on a needle base has shape-function gradients of 1 / width for the needle's vertices whatever its
   height, so nothing done to the volume mesh can repair it.
2. **The needle's apex is a carrier support point.**  22 of 22 in pop_uncut_0658: `carrier_active_set` forces one
   material sample into every carrier triangle the band touches (so that every active carrier node has stiffness), at
   the deepest sample of that triangle.  Where the band only clips the triangle's corner the deepest sample is a hair
   from the chain (level depth 0.0007 to 0.0075 against a median of 0.19), and Triangle, which may not put Steiner
   points on the chain (`YY`), builds the needle (support point, chain vertex, chain vertex) on that chain edge; the
   flip pass protects chain edges, so the needle survives.
3. **Some needles are bombs and some are not.**  A needle whose three vertices see an affine carrier field strains
   affinely and is harmless (pop_uncut_0658 production had 23 needles and lambda_max / reference = 1.06); one whose
   vertices straddle a kink of the carrier field, or whose apex is an interior vertex a hair above the face, is not.

## 3. What was tried and rejected

* **CGAL tetrahedral remeshing with the boundary locked** (`cpp/pred777h_sliver_remesh`, CGAL 5.4): built and run.
  It cannot flip next to a locked boundary and left the min dihedral at 0.003 degrees.  A `Triangulation_3` cannot
  even hold the material alone, because the sheet's boundary has genus 5 and the infinite vertex's link must be a
  sphere (`manifold_check`).  CORRECTION (2026-09-07): the statement that CGAL's exude and perturb cannot take our
  boundary was too strong.  `pred777h_cgal_polyhedral_mesh3` builds a Mesh_3 domain from our own
  `SHEET_SOLID_SURFACE.off` and its optimisers do apply.  What it cannot do is PRESERVE that boundary: Mesh_3
  remeshes the domain surface (19 410 facets in the complex from 9 168 input triangles on pop_cut_0473), so the
  carrier nodes do not survive and the fixed port is destroyed.  That is the real reason this path cannot produce
  labels, and it is also what makes it a valid independent convergence reference (see
  `cgal_convergence_reference_v1/BRIEF.md`).
* **Boundary-locked sliver perturbation** (`smooth_slivers`): moves the interior vertex of every sliver to maximise
  the star's minimum dihedral angle.  Min dihedral 0.003 to 4.6 degrees, sub-degree tetrahedra to zero, but the
  operator got WORSE where the cap was already clean: pop_uncut_0658 production lambda_max / reference 1.13 with it,
  1.01 without.  It trades slivers on sound bases, harmless, for flat tetrahedra on the port face, which are not.
  Kept as an opt-in diagnostic (`--sliver-smooth`), off by default.

## 4. The fix of record

All in the surface stage, where the needle is made:

* `carrier_active_set(support_min_depth=0.05)`: no support point in a triangle where the band is shallower than
  0.05 in level units (about one feature size from the chain at |grad| <= 2 pi sqrt 3).
* support points are never feature-size snap targets (a chain vertex snapped onto one made the support point a
  chain vertex, pop_uncut_0529).
* `_repair_cap_needles`, before the flip pass: a needle below 2 degrees whose apex or long-edge endpoint is a vertex
  the cap owns (Steiner point, unshared support point) is removed by collapsing that vertex onto another vertex of
  the needle, under the 2D link condition, orientation of every fan triangle and no duplicate triangles; an owned
  apex that cannot collapse is moved into its fan.  Chain vertices, carrier nodes and crossings never move.
* the receipt reports `surface.cap_needles` (remaining, min angle, collapsed, moved) and
  `schur.top_mode_mass_on_4_nodes` / `top_mode_nodes_above_1pct`, the diagnostic of record for this artefact.

## 5. Result

Five labels re-produced with the fix (`fixed4/`), against the labels produced before it and against the reference
tier produced before it (the reference carries the artefact too, mildly: its own top mode sits 65 % on four nodes).
"top mode on 4" is the share of lambda_max's eigenvector on its four heaviest carrier nodes; "physics" is the largest
relative difference of the six uniform-strain stiffnesses.

| label | cap needles before / after | lambda_max before / after | lambda_max / reference before / after | top mode on 4 before / after | physics vs reference before / after |
|---|---|---|---|---|---|
| pop_uncut_0658 fast | 22 / 0 | 0.3893 / 0.0403 | 11.34 / 1.18 | 1.000 / 0.928 | 2.4e-3 / 3.3e-3 |
| pop_uncut_0658 production | 23 / 2 | 0.0362 / 0.0355 | 1.056 / 1.036 | 0.808 / 0.708 | 6.8e-4 / 9.9e-4 |
| pop_uncut_0529 fast | 20 / 5 | 0.0366 / 0.0365 | 1.092 / 1.091 | 0.848 / 0.772 | 2.0e-3 / 3.0e-3 |
| pop_uncut_0529 production | 12 / 0 | 0.0656 / 0.0352 | 1.959 / 1.050 | 0.972 / 0.785 | 9.2e-4 / 1.5e-3 |
| pop_cut_0009 production | 18 / 2 | 0.5683 / 0.0379 | (no reference) | 1.000 / 0.843 | 3.3e-4 vs the old label |

Every production-tier label now sits within 1 % to 5 % of the reference's lambda_max with its top mode spread over 10
or 11 carrier nodes; the fast tier keeps a 9 % to 18 % excess, which is the coarse mesh, not an artefact (the same cell's
production tier is at 1.04).  The physics moved by at most 1e-3, the size of the port-support change itself (fewer
forced support points).  The remaining needles (0 to 5 per label, 0.07 to 0.6 degrees) are pairs of chain vertices a
few 1e-5 apart with a third vertex, harmless to the operator (pop_uncut_0529 fast: top mode 0.77 on four nodes with
five of them); merging such near-duplicate chain vertices is a follow-up.

Per-cell cost of the fix: none measurable (the surface stage does a few collapses; the volume mesh is unchanged in
size).  The 24 tests of the sheet surface, cut carrier and sliver repair pass.


**Assembly.**  The 2 x 2 x 2 coarse panel (`sheet_block_validation`, n_per_unit 32, remesh 0.06) re-run with the
repaired cell labels: the monolithic and the constrained-monolithic compliances are unchanged to every printed digit
(the block mesh does not go through the cell-label surface path), and the assembled-versus-constrained difference
on the service loads fell from 8.9e-4 / 8.6e-4 / 1.1e-3 (tension, shear, bending) to 2.7e-4 / 5.1e-5 / 5.1e-6.  A
good part of what was booked as "assembly error" was this artefact.  The 8H and 4H sinusoidal port loads moved from
2.7e-3 / 1.3e-2 to 7.3e-3 / 1.5e-2, same order, at the coarse preset.

**Block validation script.**  Its assembled carrier operator was a dense matrix with a dense eigendecomposition (65 GB
for 3 x 3 x 2), which took the machine down twice; it is now block sparse, factorised by symmetric Pardiso on the upper
triangle (in-core when it fits), with the four smallest eigenvalues by shift-invert.


## 7. Track D: interface error on panel-shaped blocks (coarse preset)

`sheet_block_validation` at n_per_unit 32, remesh 0.06, size_max 0.09, with the repaired cell labels; loads on the
x = N face, x = 0 fixed.  "asm / constr" is the assembled-labels compliance against the constrained-monolithic
control (same carrier coupling, one body), the assembly error proper; "constr / free" is the contract cost at this
coarse preset; "interface L2" is the relative L2 difference of the displacement on the internal x-interfaces,
assembled against constrained.

| load | 2 x 2 x 2 (8 cells): asm / constr, constr / free, interface L2 | 3 x 3 x 2 (18 cells): asm / constr, constr / free, interface L2 |
|---|---|---|
| tension x | +2.7e-4, -8.2e-3, 6.0e-3 | +7.6e-4, -5.7e-3, 4.0e-3 |
| shear y | +5.1e-5, -2.1e-3, 2.2e-3 | +4.8e-4, -1.6e-3, 1.6e-3 |
| bending z | +5.1e-6, -9.2e-3, 5.0e-3 | +6.4e-4, -6.6e-3, 2.1e-3 |
| wave 16H | +1.3e-3, -2.0e-2, 1.5e-2 | +1.8e-3, -1.7e-2, 1.1e-2 |
| wave 8H | +7.3e-3, -1.5e-1, 1.5e-1 | +7.8e-3, -1.5e-1, 1.1e-1 |
| wave 4H | +1.5e-2, -3.0e-1, 2.5e-1 | +1.8e-2, -3.0e-1, 1.7e-1 |

The assembly error on service loads stays below 1e-3 from 8 to 18 cells and the interface displacement error does not
grow; the short-wave port loads cost 1 % to 2 % of assembly error at any size, against a 15 % to 30 % contract cost
at this coarse preset (0.4 % to 4.5 % at the production preset, step 7).  Sizes: 3 x 3 x 2 is 1.04e6 Tet10 dof
monolithic (factorised in 25 s) and 85 365 assembled carrier dof with 4.2e8 upper nonzeros (symmetric Pardiso, 44 s).
The 4 x 4 x 2 panel follows.


## 8. Re-production of the samples with the repaired pipeline

The 300-cell stratified production sample and the 20-cell track A set were re-produced (`repro/`).  Track B: 297 PASS,
3 MESH_FAIL (two HXT nonmanifold boundary recoveries, one Gmsh crash, all on pre-existing 1e-4 features the surface
carried before the fix as well; fail closed, no label).  Track A: 28 of 28 after two reference-tier re-runs.

| statistic (300-cell sample) | before | after |
|---|---|---|
| top mode with >= 99 % of its mass on one carrier square | 19 (6.6 %) | 0 |
| >= 95 % | 39 (13.6 %) | 5 (1.7 %) |
| >= 90 % | 74 (25.9 %) | 31 (10.4 %) |
| exact null modes beyond 6 + 3 x unsupported (30-cell sample) | 5 of 5 cells, +2 to +15 | 0 of 30 |
| first eigenvalue above the null space, over lambda_max | 1e-16 to 1e-17 | median 8e-5, min 6e-7 |
| guards (rigid residual, PSD, nonmanifold) | all pass | all pass, rigid residual median 2.4e-15 |
| zero-row fraction, median | 13.8 % | 13.6 % (geometric) |
| power-law exponents, normal / shear | 1.337 / 1.205 | 1.350 / 1.209 |
| storage projected for 1949 cells | 193 GB | 165 GB |

The support points in shallow band intrusions were also the source of the extra null modes (two columns sharing one
vertex): with them gone the null space is exactly the combinatorial lower bound on every sampled cell and a real
spectral gap appears.  The remaining lambda_max tail belongs to near-empty cells: the 14 cells with material volume
below 0.01 have a median lambda_max of 0.136 against 0.038 for the other 283, with the top mode spread over 10 or
more nodes (a small piece of material pinned between ports, physical).  Whether cells below 1 % material enter the
training set is a population decision, recorded as open.

**Single-cell error, metric of record (A1 v4).**  Three metrics were tried and rejected on the repaired labels: the
generalized eigenvalues on the reference's range (v1) and on a declared energy floor from the top (v2) are dominated
by single-node modes whose value is the local mesh around one carrier node (bounds 2 to 30); the softest modes in
the mass norm (v3) are the hairline modes one mesh resolves and the other does not (bounds 4 to 180).  Neither end
of the spectrum is what a neighbouring cell transmits.  The subspace of record is the span of the k lowest
eigenvectors of the carrier Laplace-Beltrami operator in the carrier mass norm, both shipped with the label and
functions of the geometry only, with the six rigid modes projected out: smooth port fields.  Production against
reference, worst relative strain energy over that subspace:

| cell | 6 modes | 30 | 90 (wavelength >= 4 spacings) | 300 | 900 | six uniform strains |
|---|---|---|---|---|---|---|
| pop_uncut_0529 | 0.9 % | 1.3 % | 2.2 % | 3.1 % | 8.4 % | 0.13 % |
| pop_uncut_0297 | 0.24 % | 0.33 % | 0.43 % | 0.64 % | 2.4 % | 0.05 % |
| pop_uncut_0658 | 0.50 % | 0.79 % | 1.1 % | 1.7 % | 6.3 % | 0.08 % |
| pop_cut_1215 | 0.24 % | 0.33 % | 0.75 % | 1.3 % | 3.5 % | 0.05 % |
| pop_cut_0122 | 0.34 % | 0.52 % | 1.1 % | 1.7 % | 6.4 % | 0.10 % |
| pop_cut_1118 | 0.68 % | 1.2 % | 2.1 % | 4.1 % | 12.3 % | 0.11 % |
| pop_cut_0473 | 0.56 % | 0.86 % | 1.4 % | 3.3 % | 12.8 % | 0.14 % |
| pop_cut_0418 | 0.26 % | 0.40 % | 0.61 % | 0.96 % | 3.4 % | 0.06 % |

The fast tier is 2x to 3x the production tier in every column (monotone convergence with the mesh).  Reading: on the
smooth port fields assembly transmits, down to four carrier spacings, a production label is within 0.4 % to 2.2 % of
the reference in every direction; the error grows towards the carrier's own resolution limit, as it must.

## 6. Consequences for the route

* lambda_max is not the operator scale; the "operator scale dynamic range 11.2x" of the production statistics was
  this artefact.  Normalise with the carrier mass matrix (shipped in the label), never with lambda_max.
* The A1 energy floors must not be anchored on lambda_max; the revised metric uses common support and the carrier
  mass matrix.
* Every label produced before this fix carries the artefact with probability about one in four; the production
  sample (286 labels) and the track A labels must be re-produced before any statistics are quoted or any training
  starts.

## 9. Thin cut faces, and what the cut-face index set really is (2026-09-06)

**The index set has two families, not one.**  A cut-face carrier node is either the crossing of the cut plane with a
background grid LINE (or a grid node lying in the plane), or the exact CENTROID of a polygon the symmetric fan rule
cannot fan from a vertex.  Measured on the layout itself: a 45-degree half-cut carries 1073 nodes = 561 grid nodes +
0 line crossings + 512 fan centroids, and every one of those 512 is the centroid of its ring to 1e-9; a 26.6-degree
corner cut carries 229 = 99 + 66 + 64.  The second family is still a deterministic function of the plane (its
polygon's vertices are first-family points), so it is shared by the two cells of an internal interface and does not
depend on the volume mesh, but an index scheme built on "3267 grid lines" alone does not cover it.

**A vanishing corner does not degenerate the port.**  Shaving the corner (1,1,*) with `a x + b y <= d`, `t = a+b-d`
shrinking, at three azimuths:

| cut-face width | nodes | grid nodes + line crossings + fan centroids | triangles | max aspect ratio | merged | refused |
|---|---|---|---|---|---|---|
| 5.7 h | 293 | 165 + 0 + 128 | 512 | 5.7 | 0 | no |
| 1.4 h | 98 | 66 + 0 + 32 | 128 | 5.7 | 0 | no |
| 0.71 h | 98 | 0 + 66 + 32 | 128 | 5.7 | 0 | no |
| 0.23 h | 98 | 0 + 66 + 32 | 128 | 17.7 | 0 | no |
| 0.045 h | 98 | 0 + 66 + 32 | 128 | 88 | 0 | no |
| 0.005 h | 98 | 0 + 66 + 32 | 128 | 884 | 0 | no |

The node set saturates at 98 (two boundary columns of 33 crossings, one centroid per layer) below one carrier
spacing and never changes again, identically at every azimuth: the cut-face port dimension is 294 however thin the
corner.  The merge pass never fires (it merges grid nodes near the cut, not the two edges of the strip), and the
degeneracy screen correctly does not refuse (the plane contains no cell edge).  What degrades is the triangle aspect
ratio, as about 4 h / width.

**In the population.**  Minimum caliper width of the cut face over all 1302 cut cells: min 0.087 h, 12 cells (0.9 %)
below h, 4 below h/4, 54 (4.1 %) below 3 h.  The six thinnest were produced at the production preset:

| cell | width / h | status | lambda_max | top mode on 4 nodes | null dim vs lower bound | material volume |
|---|---|---|---|---|---|---|
| pop_cut_0548 | 0.087 | PASS | 0.0370 | 0.893 (10 nodes) | 714 = 714 | 0.248 |
| pop_cut_1157 | 0.141 | PASS | 0.0375 | 0.865 (8) | 1269 = 1269 | 0.322 |
| pop_cut_0027 | 0.188 | EMPTY | | | | the retained sliver holds no material |
| pop_cut_0215 | 0.196 | PASS | 0.0323 | 0.909 (7) | 1185 = 1185 | 0.138 |
| pop_cut_1235 | 0.259 | EMPTY | | | | idem |
| pop_cut_0035 | 0.262 | PASS | 0.0358 | 0.731 (12) | 789 = 789 | 0.274 |

Every produced one has lambda_max at the population median (0.038), a top mode spread over 7 to 12 carrier nodes, a
null space exactly at the combinatorial lower bound, and rigid residual 2e-15 to 7e-15.  The needle cap triangles the
thin strip must carry did NOT become stiffness bombs: the worst tetrahedra of these meshes sit on the box faces
(pop_cut_0548 at (0.31, 0.0007, 0.72), pop_cut_0035 at (0.003, 0.71, 0.30)), the ordinary residual family of
near-duplicate chain vertices, not on the cut face.  A cut face thin enough to be a problem carries so little
material trace that the volume mesher builds almost nothing on it.

A thin cut is also the case where only a sliver is RETAINED: two of the six are EMPTY, screened before meshing by the
exact level-set volume estimate, which is the intended behaviour.

`geometry.cut_face.min_width` and `min_width_over_carrier_spacing` are now recorded in every receipt (pure geometry,
no meshing), so this family can be filtered downstream without re-deriving it.  No gate is imposed: the measured
labels are clean.

## 10. Dataset policy of record (2026-09-06)

The population is produced in full; nothing is filtered out at production time.  Two families are extreme but
physical, and both are RECORDED so a training run can select on them without re-deriving anything:

| family | how it is identified | size | decision |
|---|---|---|---|
| near-empty cells | `mesh.material_volume` (and `geometry.material_volume_estimate` before meshing) | volume < 0.01: 14 of 297 in the validation sample (about 5 %) | INCLUDED.  Their operators are physical: a small piece of material pinned between two ports is genuinely stiff, and lambda_max is about 3.5x the rest (median 0.136 against 0.038), with the top mode spread over many carrier nodes.  A loss that normalises with the carrier mass matrix, not with lambda_max, handles them. |
| thin cut faces | `geometry.cut_face.min_width` and `min_width_over_carrier_spacing` | below one carrier spacing: 12 of 1302 cut cells (0.9 %); below h/4: 4 | RECORDED, NOT EXCLUDED.  The port does not degenerate (section 9): the node set saturates at 98 and the six thinnest produced labels are clean. |

A cell whose retained material falls below `EMPTY_VOLUME = 1e-6` produces status EMPTY and no operator; that screen is
an exact level-set volume estimate and runs before any meshing.  A cell whose volume mesh is nonmanifold or whose
mesher crashes is retried ONCE at an 8 % smaller interior size and then recorded as unproducible; in the validation
sample this was 3 of 300 (1 %) before any retry.
