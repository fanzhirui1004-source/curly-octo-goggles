# Step 5: population census, eight-cell assembly, rotation action (2026-09-04)

## 1. Population of record
`make_sheet_population.py --n-uncut 700 --n-cut 1300 --seed 20260904` (2009 cells incl. 9 anchors): corners in
[0.1755, 0.8775] (rho 0.10-0.50), mean uniform, gradient magnitude 0.47 u^2 with 10 % forced into [0.40, 0.47],
random direction, higher trilinear modes up to 20 % of the gradient, corner spread capped at 0.47 (rejection, 40-48 %
of draws); cut cells: theta uniform in [0, 45 deg], offset uniform over the cell; the remaining orientations are
reached by the 48 cube symmetries.  Coverage bins (mean 5 x gradient 3 x theta 3 x offset 4) in COVERAGE.json.

## 2. Census (production preset 1/64 + 0.02, surface + Gmsh volume, thickness scaling on)
Every cell of the population built end to end (surface gates plus the Gmsh volume) at the label module of record,
one JSON line per cell in `population/CENSUS.jsonl` (the module's commit in `CENSUS_MODULE_COMMIT.txt`).

| status | cells | share |
|---|---|---|
| OK (surface gates and Gmsh volume) | 1946 | 96.9 % |
| EMPTY (the cut leaves no band) | 59 | 2.9 % |
| SURFACE_BAD / SURFACE_FAIL | 3 | 0.15 % |
| GEOMETRY_DEGENERATE (cut through a cell edge) | 1 | 0.05 % |

The 59 EMPTY cells are all cut cells with offset fraction below 0.096: the retained sliver holds no band.  The three
failures (theta 4.7 deg / offset 0.064, theta 38.2 deg / offset 0.789, theta 1.6 deg / offset 0.988) were diagnosed
to one mechanism and removed at the source (section 2.3); the census was then repeated at that module (section 2.4).
The degenerate cell is the 45 degree anchor (section 2.2).

Mesh quality and cost of the 1946 labels:

| quantity | p5 | p50 | p95 |
|---|---|---|---|
| sheet triangle quality (min per cell) | 0.0012 | 0.023 | 0.130 |
| minimum dihedral angle (deg) | 0.005 | 0.073 | 0.686 |
| tetrahedra below 5 deg | 27 | 141 | 272 |
| triangles | 2127 | 32306 | 39399 |
| tetrahedra | 3994 | 92641 | 157200 |
| surface seconds | 1.5 | 9.8 | 11.8 |
| Gmsh seconds | 0.9 | 12.4 | 21.9 |

Ten cells needed a retry factor, 296 were refined by the thickness rule (up to 1/94), 8 needed a T-junction repair,
10 have more than one component, none had a cap-classification conflict and none lost a floating fragment.  The
meshed volume agrees with the exact level-set estimate to 0.3 % median and 1 % worst over the 1771 cells whose
volume exceeds 0.02 (below that the 120^3 sampling estimate is itself the less accurate number).

### 2.1 The four failures of the first pass and their root causes (all fixed in the mesher, no population rule)
The first pass (module before this step, 742 cells) failed on four cut cells.  Every one was diagnosed to a
mechanism and fixed at the source; no case is excluded from the population.

| cell | theta | offset | symptom | root cause | fix |
|---|---|---|---|---|---|
| pop_cut_0155 | 3.0 deg | 0.99 | folds on every retry, level residual 0.07 | constrained snap | (a) |
| pop_cut_0328 | 2.4 deg | 0.98 | folds on every retry, level residual 0.14 | constrained snap | (a) |
| pop_cut_0254 | 2.9 deg | 0.96 | 5 open edges on the x = 1 cap, level residual 0.30 | constrained snap, then cap polygon | (a) + (b) |
| pop_cut_0076 | 7.1 deg | 0.69 | one edge with four triangles (sheet pinched onto the cut cap) | tangent patch | (c) |

(a) `snap_near_plane_constrained` keeps a vertex on every plane it already lies on, so a vertex on the x = 1 face
within the snap distance eps = 0.2 h of the cut plane was projected onto the intersection LINE of the two planes: a
move of eps / sin(theta) along the face, 19 snap distances (0.076, almost four mesh sizes) at 3 deg.  Every face
vertex within 0.075 of the line was dragged onto it: the boundary strip folded, and vertices on two planes could
not return to the level set (residuals 0.07-0.30).  Worse, in pop_cut_0254 the band does not even reach the
retained strip of the x = 1 face: the dragged vertices were part of the sheet in the wedge the cut removes, and the
snap moved them INTO the retained region.  The joint correction is now capped at 2 eps (dihedral angles above 30 deg
unchanged); a vertex farther from the line is left to the exact clip, which handles it correctly.
(b) A port face the band does not reach has no carrier constraints, so its cap was built with the whole cube face as
boundary polygon (the code path meant for the cut plane).  The chain on that face ends at the cut line, where
nothing bounds the flood fill, so cap triangles grew past the cut line up to the far cube corner.  The boundary
polygon of every cap is now the facet of the retained region (box face or cut polygon clipped by all the other
planes).
(c) Where the sheet is tangent to the cut plane (here beside the y = 1 edge) a patch of sheet triangles lies within
the snap distance of the plane with the material between them and the plane.  The plain snap moved two of its
vertices onto the plane and not their neighbours (0.0044 vs eps 0.004), so the sheet touched the cap along an
interior edge: an edge with two sheet and two cap triangles.  Such patches (all vertices within 0.5 h, band level
rising away from the plane) are now flattened onto the plane as a whole and dropped with the flat triangles; the
cap closes the solid there (a material sliver thinner than h/2 becomes part of the cap).

Verification: unit tests for (a) and (b) added (12 tests pass); the four cells build watertight, self-intersection
free surfaces and Gmsh volumes at the first retry factor; a 20-cell regression panel (14 cut cells with theta below
30 deg where (a) changes the behaviour, 3 at 45 deg, 3 uncut anchors) stays OK with triangle counts within 1 %,
sheet quality equal or better (rho 0.20 anchor 0.063 -> 0.100, rho 0.10 0.0009 -> 0.0015) and volumes within
3e-4 relative.  Label-level effect of the change: labels built with the module before and after
the fix at the production preset (same layout, same active set, compared in the carrier low-frequency norm):
pop_cut_0003 (theta 18 deg, where (a) changes the snapping) 0.40 %, uniform rho 0.20 anchor 0.37 % (raw 5-6 %,
median mode-energy ratio -0.04 / -0.06 %), i.e. within the mesh convergence error of 0.6-1.6 % measured in STEP4.
The tangent-patch rule flattens about 250 triangles per cell (material slivers thinner than h/2 along the box
faces), which is where the 1 % triangle-count drop comes from.


### 2.5 Label census (38 cells, production preset, module 3e8c450)
The surface census does not run the Schur, so 38 cells were labelled end to end with the production command: the
seven uncut anchors and one graded uncut cell, two cut cells in every (theta, offset) bin, the two smallest
retained pieces, two cells that needed a retry factor and two with more than one component (list in
`population/LABEL_CENSUS_CASES.txt`, receipts in `population/label_census/`).

| quantity | result |
|---|---|
| status | 38 PASS, 0 failures |
| rigid-body residual, worst | 3.2e-14 |
| smallest eigenvalue relative to the largest, worst | -2.3e-15 (positive semidefinite to round-off) |
| mesh port support outside the geometric active set | 0 in every cell |
| active carrier nodes without mesh support (zero rows) | 6 to 25 % of the active set, median 13 % |
| operator scale (largest eigenvalue) | 1.4e-2 to 3.8e-1 (27x across the sample) |
| q | 57 to 10584, median 4240 |
| wall per label (12 or 20 threads) | 10 s to 413 s, median 123 s; thick anchor rho 0.50: 413 s, of which Schur 273 s |
| storage | 1.84 GB for the 38 (fp32 upper triangles) |

No cell of the sample fails an operator gate; the label pipeline is sound across the design space, including the
slivers.  Two properties of the target itself stand out for the training design (section 5): the zero rows, and
the 27x range of operator scale.

### 2.2 The degenerate cut family (found by the census, 2026-09-04)
One cell produced no record at all: `pop_anchor_cut_theta45`, whose plane x + y = 1 contains two vertical edges of
the cell.  The exact chart compiler refuses such a plane ("patch box_x_min has coincident semantic edges") during
geometry compilation, before the mesher's error handling, so the tools died with a traceback instead of recording a
status.  A sweep of vertical planes shows the rule exactly: the compiler refuses when the plane contains a cell edge
(local offset d in {0, a, b, a + b}) and accepts every other offset.  `sheet_label_pipeline.cut_contains_a_cell_edge`
now decides this by exact rational arithmetic before any meshing; the production command writes a receipt with
status GEOMETRY_DEGENERATE and the census tool reports the same status.  The remedy for a real structure is to move
the WORLD cut plane by one exact rational epsilon: every cell keeps the same plane, so shared faces still agree.
This matters because a cut aligned with the lattice (a 45 degree plane through lattice points, an axis-aligned plane
at an integer coordinate) hits the degenerate set for many cells at once, while random offsets essentially never do.


### 2.3 The last three failures: the near-plane treatment, redesigned
All three came from the same step of the remesh-and-clip stage, the blanket "move every vertex within eps of a
plane onto it", and from the tangent-patch pass that had been added on top of it in section 2.1(c):

| cell | what the diagnosis showed |
|---|---|
| pop_cut_0475 | the box_x_min cap stopped 0.002 short of the cut line: sheet vertices that merely pass near the face with the VOID in between had been pulled onto it, fabricating a contact where the exact level set has a gap (the band does not reach that strip of the face), and the cap beside it had no carrier support and stayed open |
| pop_cut_0620 | one edge with four triangles on the cut plane: both endpoints of an interior sheet edge had been snapped while their neighbours stayed 0.004 off the plane, the same pinch as pop_cut_0076 |
| pop_cut_0625 | 13 fold pairs, all 0.004 to 0.011 from the cut plane: the tangent-patch pass moved vertices by up to half a mesh size and folded triangles beside the grazing cut |

The distance-only rule was replaced by one that reads the local topology and the material side
(`_snap_crossings_and_flatten_patches`).  A near vertex whose one-ring reaches clearly beyond the plane on both
sides is a crossing: it is moved onto the plane so the clip boundary runs through vertices.  Every other near vertex
belongs to a patch that runs alongside the plane; patches are handled by connected component: with the material
between patch and plane the whole patch is moved onto the plane and the flat-triangle drop turns the thin slab into
cap, with the void between nothing moves (a gap outside the solid is harmless).  The invariant behind this, "no
interior sheet edge lies in a cap plane with material below it", is enforced after every step that moves or merges
vertices (`_flatten_tangent_edges`): the two triangles of such an edge are flattened unless the move would fold a
neighbouring triangle; the guard is the fold itself, not a distance threshold, because a threshold always has a case
just beyond it (pop_cut_0076's off-plane neighbour sat at 1.1 eps).

Result on the 27-cell panel (the seven cells that ever failed plus the 20-cell regression set): all OK, triangle
counts within 1 %, volumes within 2e-3 relative, sheet quality equal or better.  Label-level effect at the
production preset, old rule against new on the same layout and active set: pop_cut_0003 0.40 %, uniform rho 0.20
anchor 0.37 % in the carrier low-frequency norm, median mode-energy ratio +0.04 / +0.07 %, i.e. mesh noise.
Commit 3e8c450.

### 2.4 Census of record, second pass (module 3e8c450)
All 2009 cells rebuilt at commit 3e8c450 (56 single-thread workers, the surface stage is single-threaded):

| status | cells |
|---|---|
| OK | 1949 |
| EMPTY | 59 |
| GEOMETRY_DEGENERATE (the 45 degree anchor) | 1 |
| surface failures | 0 |

Against the first pass at ff9d147: the three failures are gone, cells needing a retry factor drop from 10 to 2, cells
needing a T-junction repair from 8 to 1, sheet quality and cost unchanged (median surface 12 s).  This is the
census of record for production: no case in the population is excluded by a rule, and the only refusal is the
exactly degenerate anchor, which is a status with a receipt.

## 3. Eight-cell assembly (2 x 2 x 2 block of science_00's parent field, fast preset 1/48 + 0.03, carrier 1/32)
Eight independent cell labels assembled by carrier coordinates (12 internal interfaces, active-set mismatch 0 on all)
against the monolithic Tet10 block (1.45 M dof, free interfaces, free external faces), with the constrained-monolithic
control (external port faces tied to the carrier, interfaces free).  Loads on x = 2, x = 0 fixed:

| load | assembled vs free | external constraint alone | internal interfaces alone | interface displacement vs free |
|---|---|---|---|---|
| tension x | -0.68 % | -0.65 % | -0.03 % | 0.23 % |
| shear y | -0.29 % | -0.27 % | -0.02 % | 0.23 % |
| bending z | -0.76 % | -0.76 % | +0.004 % | 0.31 % |
| wave 16H (one period over the block face) | -1.6 % | -1.7 % | +0.12 % | 0.52 % |
| wave 8H | -8.9 % | -9.5 % | +0.68 % | 1.1 % |
| wave 4H | -25 % | -27 % | +2.2 % | 1.7 % |

Same picture as the 2 x 1 x 1 block of STEP3: the internal interfaces cost 0.02-0.12 % on smooth loads, and what the
assembly loses is the kinematic tie of the EXTERNAL faces to the 1/32 carrier (a property of how the block is loaded,
not of the cell labels).  The assembled carrier system has 11283 nodes of which 943 carry no stiffness (active fringe
nodes with no fine support; dropped) and its smallest eigenvalues are 2e-8 (no spurious mechanism).


Also verified for the cut case (2026-09-04): two neighbouring cells sharing ONE world cut plane produce identical
carrier traces on their shared face, including when the cut line clips that face (924 nodes and 1728 triangles on
both sides, empty difference both ways, at three offsets).  Cut cells are therefore assemblable at the carrier
level; the physics comparison for a block of cut cells is still open (section 5).

### 3.1 Cut cells assembled end to end (2026-09-04)
`sheet_block_validation.py` now carries one WORLD cut through every cell of the block (each cell gets the exact local
plane from the base cell's spec, its own clipped carrier trace, faces the cut removes are skipped, cells the cut
empties leave the assembly).  2 x 1 x 1 block of science_00's parent field with the plane 3x + 10y <= 9: it crosses
the internal interface x = 1 at y = 0.6, cuts the fixed face x = 0 at y = 0.9 and leaves a strip y <= 0.3 of the
loaded face x = 2 (fast preset):

| load | assembled vs free monolith | external constraint alone | internal interface alone |
|---|---|---|---|
| tension x | -0.82 % | -0.84 % | +0.03 % |
| shear y | -0.37 % | -0.39 % | +0.01 % |
| bending z | -4.50 % | -4.56 % | +0.06 % |
| wave 16H | -4.59 % | -4.64 % | +0.05 % |
| wave 8H | -10.5 % | -10.7 % | +0.23 % |
| wave 4H | -22.8 % | -23.5 % | +0.91 % |

Internal interface active-set mismatch 0 (the cut line crosses it); the interface itself costs 0.01 to 0.06 % on
smooth loads, exactly as in the uncut blocks; the larger external-constraint numbers for bending and the waves come
from the loaded face being a strip.  The assembled carrier system (2230 nodes, 165 without stiffness) has smallest
eigenvalues 8e-7: no mechanism.  Cut cells are assemblable from independently produced labels.

## 4. Rotation action, first measurement (frozen carrier; superseded by section 7)
`sheet_rotation_check.py`: a cell and its symmetry image (corner values permuted, cut plane transformed about the
cell centre, exact rationals) are labelled independently; the prediction T S T^T (T = carrier-node permutation x R)
is compared with the image's label on the common active nodes in the carrier low-frequency norm.

Finding that changed the label definition: the carrier triangulation is not invariant under the cube symmetries (one
diagonal per square; clipped, re-triangulated traces on cut faces), so the earlier rule "vertices of the carrier
triangles the band reaches" gave a cell and its image active sets differing by about 100 nodes.  The label's active
set is now the symmetric square rule (a node is active iff a closed carrier square containing it is reached by the
band on a symmetric cell-centred lattice) united with the vertices of the cap-constraint triangles (the barycentric
rule, unchanged, so the surfaces are byte-identical to STEP4); the union adds a handful of hairline nodes whose rows
are zero.  A cell and its image now have corresponding node sets up to those zero rows.

Cut cell pop_cut_0003 (theta 18 deg, offset 0.53, corners 0.29-0.66), each label built independently at the fast
preset (1/48 + 0.03, interior 0.05); relative Frobenius norms on the common active nodes, low-frequency = the
carrier P1 Laplacian modes below the 1/32 grid's own cut-off:

| image | q(original) / q(image) | active-set mismatch | raw | low-frequency |
|---|---|---|---|---|
| ID (same manifest, rebuilt) | 7014 / 7014 | 0 | 0 (bit-identical) | 0 |
| INV (inversion) | 7014 / 7023 | 3 zero rows | 21 % | 2.8 % |
| R05 (proper, x->z->y) | 7014 / 7026 | 4 zero rows | 51 % | 4.5 % |
| M01 (reflection) | 7014 / 7026 | 4 zero rows | 25 % | 2.8 % |

Production preset (1/64 + 0.02, interior 0.04) on the same cell: INV 2.75 %, R05 2.82 % (raw 20 %, 21 %).  Uniform
anchor cell rho 0.20 (no cut, exactly symmetric geometry, fast preset): INV 3.07 %, R05 3.27 %, M01 3.15 %, active-set
mismatch 0.  In every comparison the median mode-energy ratio is -1.0 to -1.1 %.

Reading: the residual is not mesh noise (the finer preset leaves it unchanged, and it is the same for the
symmetric uniform cell), it is the label pipeline's own orientation dependence: the carrier triangulation (one
diagonal per square, cut traces re-triangulated) and the mesh (marching cubes, remesher, Gmsh) are not invariant
under the cube group, so a cell and its image get slightly different discretisations of the same operator.
About 3 % in the carrier low-frequency norm, consistent in sign.  This is the size of the "label noise" that the
plan "vertical cuts only, other orientations by the 48 symmetries" introduces.  Section 8 traces it to the carrier
triangulation and removes it; the numbers in this section are the BEFORE state and are kept as the measurement that
motivated the change.


## 5. What the route does NOT yet cover (assessment 2026-09-04)
Recorded here so it is decided before 2000 labels are produced, not after.

1. **Cut orientation family.**  The population holds vertical cuts only, extended by the 48 cube symmetries.  A cube
   symmetry maps the normal (cos theta, sin theta, 0) to a signed permutation of it, so the reachable set is exactly
   the planes PARALLEL TO A COORDINATE AXIS.  A general oblique plane, normal (1,1,1)/sqrt(3) for instance, is
   outside the training distribution.  If production needs arbitrary cut orientations the population needs a polar
   angle dimension and has to be regenerated.
2. **One cut plane per cell.**  The manifest and the geometry carry a single `cut_plane`.  A cell at a concave corner
   of the cut region sees two or three planes and is not supported.
3. **Label census: DONE** (section 2.5): 38 of 38 PASS, gates satisfied to round-off, zero rows 6 to 25 %, operator
   scale over a 27x range.
4. **Cut cells assembled end to end: DONE** (section 3.1): interface mismatch 0, interface cost 0.01 to 0.06 %.
5. **Zero rows in the target.**  About 15 % of the active carrier nodes of a normal cut cell carry no mesh support
   (68 % in a sliver cell), and which rows are zero depends on the mesh, not the geometry.  That is noise in the
   learning target; either the geometric active rule tightens (level tolerance 0.05 is generous) or the target is
   defined on the supported set.
6. **Slivers dominate one bucket.**  118 of the 1239 OK cut cells (9.5 %) hold less than 1 % material volume, and
   their operators are two to three orders of magnitude smaller (measured: a cell of volume 1e-5 with 31 tetrahedra
   gives a valid label, q = 57, rigid residual 8.9e-15, largest eigenvalue 6.5e-3).  They are numerically sound but
   they widen the target's dynamic range and half of the cut bucket carries almost no physics.

## 6. Production timing and plan
See PRODUCTION_PLAN.md (same directory).  Measured with the memory-budgeted queue (`run_sheet_label_queue.py`,
commits a5587de and 5819d12) on the 64-core / 128 GiB box: 60 random population labels in two batches, no failures,
steady state 165 to 185 labels per hour with the lighter mix and about 100 per hour with a heavy one (8 thick cells
of 30), cgroup memory at most 83 GB.  The queue is bound by the memory budget of the Schur stages, not by threads
(Schur time per unit work identical at 8 and 12 threads).  2000 labels: about 12 to 20 h on this box alone.

## 7. Decisions needed before production (state at the end of 2026-09-04)
Resolved today: the cut orientation family stays vertical for now (decision of record, other orientations later);
cut cells assemble end to end (3.1); the label census passes across the design space (2.5); the residual surface
failures are gone at the source (2.3, 2.4); the degenerate cut is a status (2.2).  Still open, and decisions rather
than defects: the zero rows and the operator-scale range of the training target (section 5, items 5 and 6), and
the single cut plane per cell (item 2).
1. Rotation augmentation: RESOLVED by the symmetric carrier of section 8.  The residual was the carrier
   triangulation, not the mesh; with the parity diagonal the rotation action is exact at machine precision on a
   fixed mesh and the independent-mesh residual falls to 0.8 %, unbiased.  The augmentation "vertical cuts only,
   other orientations by the 48 cube symmetries" is sound as stated.
2. Nearly empty cut cells.  Cut offsets below ~0.1 of the range leave a sliver without band: EMPTY cells (about
   6 % of the cut draws).  They are correct (skipped with a receipt) but spend population slots; whether to resample
   them is a sampling design choice, not a robustness one.  No rule was added.
3. Storage of fp32 upper triangles (agreed) and the 1/32 carrier (agreed) stand; q grew by 5-6 % with the union
   active-set rule (e.g. G0 5643 -> 5976) and is unchanged by the symmetric carrier.

## 8. Where the orientation error comes from, and the symmetric carrier (2026-09-04)

### 8.1 The measurement that settles it
The rotation check labels a cell and its symmetry image independently, so its residual mixes two causes: the mesh
(marching cubes, remesher, Gmsh are not equivariant) and the carrier P1 space (the port constraint).  A second
experiment separates them: the ORIGINAL cell's Tet mesh is rotated rigidly and labelled against the image cell's
carrier, so the fine discretisation is identical on both sides and only the carrier can differ.

| experiment (uniform anchor rho 0.20, R05) | low-frequency residual |
|---|---|
| independent meshes, frozen carrier | 3.27 % |
| same mesh rotated, frozen carrier | 3.49 % |

The residual does not drop when the mesh is made identical, so it is not mesh noise: it is the carrier.

### 8.2 The defect in the frozen carrier
Every carrier square is split into two triangles by fanning from one corner, and the corner is chosen as the
smallest NODE ID, which is a hash of the exact coordinates (`_initial_carrier`, frozen vendor snapshot).  The
diagonal of each square is therefore drawn pseudo-randomly, and the pattern has no relation to the cube group: of
the 48 symmetries, 47 map the triangulation of the uniform cell onto a different triangulation.  The port P1 space
is what the Schur complement is expressed in, so a cell and its image are labelled in two different spaces.

### 8.3 The symmetric rule (Union Jack by lattice parity)
Join the two corners of each carrier square whose lattice coordinate sum is even.  Each square has exactly two such
corners and they are diagonally opposite (the four corner sums are s, s+1, s+1, s+2), so the rule always defines a
diagonal, and it produces the Union Jack pattern (diagonals alternating with the square's parity).  It is exactly
invariant: a cube symmetry maps a lattice coordinate c to c or n - c, which preserves each coordinate's parity when
n is even (n = 32 here), so the even-sum pair of a square maps to the even-sum pair of its image.  Measured on the
uniform cell: 0 of 48 symmetries broken, against 47 of 48 for the frozen fan.

### 8.4 The decisive test
With the symmetric carrier, the same-mesh experiment (identical fine discretisation, so the carrier is the only
possible source of a difference) gives a low-frequency residual of 7.5e-15 against 3.49 % for the frozen carrier:
the rotation action on the label is EXACT at machine precision, not approximate.  What remains in the
independent-mesh comparison is the mesh noise alone.

| uniform anchor rho 0.20, R05 | frozen carrier | symmetric carrier |
|---|---|---|
| same mesh rotated (carrier only) | 3.49 % | 7.5e-15 |
| independent meshes (carrier + mesh) | 3.27 % | 0.78 % |

Independent meshes, uniform anchor rho 0.20 (fast preset), all three symmetries measured:

| symmetry | frozen carrier | symmetric carrier |
|---|---|---|
| INV (inversion) | 3.07 % (median mode energy -1.09 %) | 0.81 % (+0.005 %) |
| R05 (proper rotation) | 3.27 % (-1.10 %) | 0.78 % (+0.006 %) |
| M01 (reflection) | 3.15 % (-1.13 %) | 0.81 % (+0.003 %) |

The systematic bias of about -1.1 % in the mode energies is gone; what is left is unsigned mesh noise at the
convergence level of the preset (0.6 to 1.6 % measured in STEP4).  Active-set mismatch is 0 in every case, and the
label size q is unchanged (the rule moves diagonals, not nodes).

Cut cell pop_cut_0003 at the production preset, where the cut line crosses two port faces and their carrier traces
are clipped (the worst case for the rule, since clipped patches keep the frozen fan): INV 2.75 % -> 0.55 %, R05
2.82 % -> 0.49 %, and the active-set mismatch, which was 3 to 4 nodes under the frozen carrier, becomes 0.  The label size q is identical
between a cell and its image in every case (5544 for the uniform anchor, 7023 for the cut cell): the rule moves
diagonals, not nodes, so nothing about the label's size or storage changes.

Regression: the 24-cell panel of section 2.1 rebuilds with the symmetric carrier without a single failure, triangle
counts within 0.5 % and volumes unchanged (the carrier triangles enter the surface only as the cap constraint
polygons, so the sheet itself is untouched).

Neighbouring cells still agree on a shared face: a neighbour's origin is one full cell away, that is n lattice steps
with n even, so the parity of every lattice coordinate is preserved and both cells split the shared squares the same
way.  Measured on the 2 x 1 x 1 block of section 3's parent field: the internal interface has active-set mismatch 0 (as
under the frozen carrier), the assembled compliances move by less than 0.1 % of themselves (tension -0.74 % ->
-0.79 % against the free monolith, shear -0.31 % -> -0.32 %, bending -1.20 % -> -1.32 %, wave 16H -6.1 % ->
-6.8 %), and the assembled carrier system keeps a clean spectrum (smallest eigenvalues 8e-7, no mechanism).  Its
node count grows from 3428 to 3632 because the block's port constraints are the touched cap triangles, whose set
depends on the diagonals; the single-cell label size q is unchanged (5544 and 7023 above).

Implementation: `carrier_triangulation.unionjack_layout` rebuilds the triangulation of the complete lattice squares
of an already built layout and leaves everything else untouched (node identity, node count, local-to-global maps,
and the clipped patches on faces the cut plane crosses, whose nodes are off-lattice).  The vendored generator is a
frozen snapshot bound by SHA-256, so the rule is applied as a post-processing step in this route's own package
rather than by editing the vendor.
