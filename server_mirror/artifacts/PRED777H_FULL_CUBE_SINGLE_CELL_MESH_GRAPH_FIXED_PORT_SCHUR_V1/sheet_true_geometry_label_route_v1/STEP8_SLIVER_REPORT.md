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
  It cannot flip next to a locked boundary and left the min dihedral at 0.003 degrees.  CGAL's own exude / perturb
  need a Mesh_3 regular triangulation and cannot take our boundary; a Triangulation_3 cannot even hold the material
  alone, because the sheet's boundary has genus 5 and the infinite vertex's link must be a sphere (`manifold_check`).
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

## 6. Consequences for the route

* lambda_max is not the operator scale; the "operator scale dynamic range 11.2x" of the production statistics was
  this artefact.  Normalise with the carrier mass matrix (shipped in the label), never with lambda_max.
* The A1 energy floors must not be anchored on lambda_max; the revised metric uses common support and the carrier
  mass matrix.
* Every label produced before this fix carries the artefact with probability about one in four; the production
  sample (286 labels) and the track A labels must be re-produced before any statistics are quoted or any training
  starts.
