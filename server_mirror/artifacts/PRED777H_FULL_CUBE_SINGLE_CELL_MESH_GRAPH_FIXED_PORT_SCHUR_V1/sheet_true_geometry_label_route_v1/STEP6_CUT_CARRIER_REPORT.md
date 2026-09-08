# Step 6: cut-face carrier conditioning (fixed index set, bounded aspect ratio, symmetric fans)

Sheet route, 2026-09-04.  Module `cut_carrier.py`, wired into `sheet_label_pipeline.build_carrier_layout` (constant
`CUT_CARRIER_MERGE = True`), gate in `produce_sheet_label.py`, contract field `cut_carrier_merge`.

## 1. Findings that motivated the change

* **Index set.**  Every cut-face carrier node is a crossing of the cut plane with a line of the 1/32 background grid,
  for any plane orientation: 160 random planes (polar angle 10.9° to 89.7°), 182 587 cut-face nodes, 0 off a grid
  line.  The fixed index set of the cut port is therefore the 3·33·33 = 3267 grid lines (2·33·33 = 2178 for the c = 0
  family, which is a prism: a 2-D polyline × 33 z-levels); the only continuous data per node is its crossing
  parameter, a function of the plane.  Observed node counts: c = 0 population min 66 / median 1188 / max 2046;
  tilted min 70 / median 1292 / max 2336.
* **Conditioning.**  The frozen fan degenerates when the plane passes near a grid node (crossings nearly coincide)
  or when a face cut line runs close to a face grid line (thin strips):

  | population | metric | frozen carrier |
  |---|---|---|
  | c = 0, 240 real cells | cut-face max aspect ratio: median / p90 / worst | 19 / 94 / 6.3e3 |
  | | box-face max aspect ratio: median / p90 / worst | 66 / 297 / 2.4e4 |
  | | cells with a triangle above 100 / 1000 | 91 / 6 |
  | tilted, 160 random planes | cut-face: median / p90 / worst | 854 / 6361 / 4.0e5 |
  | | box-face: median / p90 / worst | 266 / 1472 / 1.9e4 |
  | | cells with a triangle above 100 / 1000 | 152 / 71 |

  The carrier slivers do not reach the volume mesh (the sub-5° tetrahedra of the census, median 0.19 %, are the same
  in uncut cells and come from the sheet surface), so the damage is confined to the port basis: the carrier mass
  whitening, the active-set rule and the training target.

## 2. The pass (plan A)

With h = 1/32 and eps = 1/10:

1. **In-plane merge.**  For every grid node N near the cut (interior N: dist(N, plane) < eps h; N on one box face:
   distance to that face's cut line < eps h; N on a cell edge: plane/edge crossing within eps h of N), the
   crossings of the grid lines through N within √3 eps h of N, and N itself when it is a carrier node, are moved to
   one canonical point: the projection of N on the plane / the nearest point of the face cut line / the plane-edge
   crossing.  The point is on the plane and on every box face the merged nodes were on, so the trace stays exactly
   planar and conforming with the box faces; the rule is a function of the global plane and grid only, so
   neighbouring cells make the same decision (test: shared-face keys, coordinates and triangles identical).  A
   carrier lattice node keeps its key (its slot moves by < eps h); other merged nodes are keyed by the exact world
   coordinate of the canonical point.
2. **Symmetric fans.**  Every cut-face polygon (plane ∩ grid cell) and every box-face polygon is retriangulated by
   isometry-invariant rules: lattice squares by the Union Jack parity diagonal; other polygons by the fan whose sorted
   angle list is lexicographically largest; a tied quadrilateral by the diagonal joining opposite vertices of even grid
   parity; a remaining tie (a symmetric polygon: the c = 0 rectangles with mixed line families, the regular hexagons of
   a body-diagonal plane) by a centroid node keyed by the exact centroid.  For the c = 0 rectangles no invariant
   diagonal exists (the x↔y mirror maps each diagonal's labels to the other's), so the centroid there is forced.
3. **Gate.**  Two families no in-face move can fix: the plane within eps h of a cell corner, and the plane nearly
   containing a cell edge (a face cut line within eps h of the edge while the plane/edge crossing is not).  Their
   strips have aspect ≈ h / gap.  A cell whose smallest such gap is below 1/50 h (aspect above 50) is refused with
   status GEOMETRY_DEGENERATE and the existing remedy (translate the world plane by one exact rational epsilon).

## 3. Results

| population | metric | before | after |
|---|---|---|---|
| c = 0, 240 | cut-face max aspect: median / p90 / worst | 19.2 / 94.3 / 6325 | 7.00 / 11.3 / 22.9 |
| | box-face max aspect: median / p90 / worst | 65.8 / 297 / 24400 | 7.65 / 12.4 / 40.8 |
| | cells with a triangle above 30 / 100 | 189 / 91 | 1 / 0 |
| | carrier nodes after / before (median, max) | | 1.057, 1.97 |
| | cut-face polygons: parity quads / centroids (median) | | 672 / 288 |
| tilted, 160 | cut-face max aspect: median / p90 / worst | 854 / 6361 / 403000 | 8.02 / 9.72 / 18.0 |
| | box-face max aspect: median / p90 / worst | 266 / 1472 / 18800 | 9.03 / 9.94 / 45.6 |
| | cells with a triangle above 30 / 100 | 160 / 152 | 1 / 0 |
| | carrier nodes after / before (median, max) | | 0.951, 0.998 |
| | cut-face polygons: anchor fans / centroids (median) | | 1124 / 0 |

Shared-edge trace audit: 400 / 400 PASS.  Prolongation on interior samples of every rebuilt triangle: OK.  Merge time
median 1.1 s per cell.  Residual worst triangles are needles of the unfixable families (edge lengths h, 0.02 h, h).

**Gate frequency (full population, gap < 0.02 h):** c = 0 population, 1301 cells: 236 (18.1 %) carry an unfixable family within eps h, 11 (0.85 %) fall below the 0.02 h gate (smallest residual gap 0.0034 h; p10 0.030 h; median 0.071 h); carrier nodes after / before median 1.057, p95 1.175, max 1.97.  Tilted, 160 planes: 70 (43.8 %) carry a family, 2 (1.25 %) gated (smallest gap 0.0126 h); nodes after / before median 0.951, max 0.998.  One c = 0 cell is the exact degenerate case already refused by the chart compiler.

**Tests** (`tests/pred777h_full_cube_v1/test_cut_carrier.py`, carrier 1/16): aspect bound and exact planarity for a
near-degenerate vertical and a near-degenerate tilted plane; near-corner plane reported and gated; uncut cell untouched;
two neighbouring cells agree on the shared face (keys, world coordinates, triangles); merged carrier invariant under all
48 cube symmetries (node set and triangle set).  20 tests pass with the existing sheet-surface suite.

**48-symmetry invariance at carrier 1/32, real cells:** 0 of 48 symmetries broken for pop_cut_0623 (4563 nodes, 192 centroids, 231 clusters), pop_cut_0003 (6387 nodes, 416 centroids, 132 clusters), tilt60_mid (5334 nodes, 0 centroids, 271 clusters) and tilt45_mid (4147 nodes, 0 centroids, 258 clusters): node set and triangle set of the merged layout of every symmetry image equal the image of the merged layout.

**End-to-end labels through the merged pipeline** (fast preset):

| cell | status | q | rigid residual | min eigenvalue | tets < 5° | clusters | carrier nodes |
|---|---|---|---|---|---|---|---|
| pop_cut_0623 (worst c = 0 sliver, aspect 6325) | PASS | 3066 | 1.7e-13 | −4.7e-17 | 115 | 231 | 4478 → 4563 |
| pop_cut_0003 | PASS | 7116 | 1.0e-14 | −2.2e-17 | 260 | 132 | 6041 → 6387 |
| tilt60_mid (polar 49.6°) | PASS | 4512 | 4.8e-15 | −2.1e-17 | 154 | 271 | 5749 → 5334 |
| tilt45_mid (polar 49.1°) | PASS | 3549 | 9.0e-15 | −2.2e-17 | 84 | 258 | 4591 → 4147 |

**Operator continuity across merge events** (pop_cut_0003, plane offset swept in 11 steps of 1/1500, each offset
labelled independently at the fast preset; energies of 9 fixed smooth port fields, 6 uniform strains and 3 quadratics):
the relative step between consecutive offsets is 0.9e-3 to 1.2e-3 at every step; the four steps at which the cluster
count changes (165→132, 132→99, 99→132, 132→99) give 1.20e-3, 1.16e-3, 0.89e-3, 1.12e-3, indistinguishable from the
steps without an event (0.89e-3 to 1.12e-3).  The threshold rule's discontinuity is below the geometric and mesh
variation of one step; q was 7116 at all 11 offsets.

## 4. What this settles and what it does not

* The cut port has a fixed index set for any orientation (grid lines, plus grid-cell centroids where a fan tie is
  forced); its conditioning is bounded (aspect ≤ ~25 on the cut face, ≤ ~50 on box faces at eps = 1/10) except for the
  gated families; the rules are cube-symmetric and cross-cell consistent.
* Not settled here: the cut face is still excluded from the port (task 2); the global cut-plane triangulation and the
  EMPTY-cell geometry for the skin coupling (task 3); the residual families are gated, not fixed, and their frequency
  is reported above.

## 5. The cut face as a port (task 2)

The cut face was excluded from the port at two places (`carrier_active_set` / `equivariant_active_nodes` dropped
`cut_0` from the carrier shell; `compile_port` hid the cut plane from the mesh adapter).  Now: the cut cap is built on
the `cut_0` carrier trace like a box-face cap (constraint points and carrier edges), the mesh adapter classifies cut
facets as the port `cut_0`, P interpolates them on the cut trace, the active set on the cut face is the geometric
triangle rule (the trace is cube-symmetric by construction), a fragment attached to the cut face alone is kept (it is
load bearing through the skin), and the rotation check matches carrier nodes by exact coordinates.  The contract
`domain` now reads "the cut surface is a port (cut_0 carrier trace, skin coupling), like the box faces".

**Mesher.**  With the cut cap constrained to the carrier, Gmsh Delaunay (algorithm 1) produced a volume mesh with 31
nonmanifold facets (up to 14 tetrahedra sharing a facet, volumes 1e-7 to 1e-22) on pop_cut_0003: a boundary-recovery
failure on a sliver-rich cap.  The cut caps are not worse than the box caps (59 cut cells: cut-cap minimum quality median
0.0028, fraction below 0.3 median 0.197; box caps 0.0012 and 0.160), so this is Gmsh, not the cut trace.  HXT on the
same surface gives 0 nonmanifold facets.  `mesh_sheet_solid` now counts nonmanifold volume facets, rejects such a mesh
itself (not downstream in the adapter) and retries once with HXT, the same deterministic retry already used for a
Gmsh crash; the receipt records `algorithm3d` and `nonmanifold_facets`.

**End-to-end labels with the cut port** (fast preset):

| cell | status | ports | q (cut free → cut port) | rigid residual | min eigenvalue | algorithm3d |
|---|---|---|---|---|---|---|
| pop_cut_0003 | PASS | 6 box + cut_0 | 7116 → 8868 | 7.9e-15 | −3.2e-17 | 10 (HXT retry) |
| pop_cut_0623 | PASS | 5 box + cut_0 | 3066 → 4224 | 1.9e-13 | −5.0e-17 | 1 |
| tilt60_mid | PASS | 5 box + cut_0 | 4512 → 6804 | 3.9e-15 | −2.9e-17 | 1 |
| pop_anchor_rho20 (uncut control) | PASS | 6 box | 5544 → 5544 | 2.9e-15 | −1.6e-17 | 1 |

**2 × 1 × 1 cut block (plane 3x + 10y ≤ 9), cut face as a port in both cells and in the constrained monolithic control**
(the block's cut cap conforms to the union of the two cut traces; the control ties the cut-face fine nodes of each
cell to that cell's cut trace):

| load | assembled vs constrained (cut port) | assembled vs constrained (cut free, before) | constrained vs free (cut port) | constrained vs free (cut free, before) |
|---|---|---|---|---|
| tension x | 4.0e-4 | 2.7e-4 | −0.99 % | −0.84 % |
| shear y | 6.7e-4 | 1.3e-4 | −0.87 % | −0.39 % |
| bending z | 8.7e-4 | 6.0e-4 | −5.2 % | −4.6 % |
| wave 16H | 8.5e-4 | 5.3e-4 | −5.3 % | −4.6 % |
| wave 8H | 4.1e-3 | 2.3e-3 | −12.8 % | −10.7 % |
| wave 4H | 1.25e-2 | 9.1e-3 | −26.2 % | −23.5 % |

Interface active-set mismatch 0; interface displacement vs the constrained control 0.0010 to 0.0033.  The assembly of the
two labels reproduces the constrained monolithic model of the same block to the same order as before (the residual is
the independent meshing of the two cells and the block).  The constrained-vs-free column is larger than before by 0.2 to
2.7 points: it now includes the Ritz restriction of the cut face itself, which is exactly the coupling contract (the cut
face displacement is a carrier P1 field).  Global carrier nodes of the block: 3264 (2230 with the cut free).

## 6. The coupling deliverable (task 3)

Every label directory, PASS or EMPTY, now carries `CUT_FACE_TRACE.npz` (`sheet_label_pipeline.write_cut_face_trace`):
the cell's `cut_0` carrier trace in world coordinates (`node_keys`, `node_xyz_world`, `triangles`, `on_box_face_ring`,
`plane_world`, the placement, the layout signature).  Node keys are the global carrier keys, identical in neighbouring
cells for the ring nodes on a shared face, so the union over a block is one conforming triangulation of the cut plane.

Checks: the 2 × 1 × 1 cut block stitches to 3182 nodes / 6144 triangles with 0 key-or-coordinate conflicts, every edge
in exactly 2 triangles except the 218 outer-boundary edges, all of which lie on the block's outer box faces (the 33
shared ring nodes of the internal face are the vertical cut line's 33 z-levels); a 2 × 2 × 1 block with two cells
entirely on the discarded side of the plane skips those cells (no material, no cut face) and stitches identically.
EMPTY cell pop_cut_0015 (volume 0) exports 228 nodes / 384 triangles / 70 ring nodes with status EMPTY.  Test:
`test_cut_face_trace_export_stitches_across_neighbours`.

**Cost of the cut port**: q grows by ×1.25 to ×1.51 on the cells above (the cut trace is 20 to 45 % of the carrier
nodes and its band-touched fraction is comparable to the box faces); dense storage grows by the square.  The
production plan's storage and throughput figures must be re-measured on the new definition (open item).

## 7. Root causes instead of gates (revision of sections 2.3 and 5)

The gate on the residual sliver families and the HXT retry on a nonmanifold Gmsh mesh were both removed.  Both
problems were root-caused; the fixes are upstream of the mesher, plus one decision on the mesher itself.

**7.1 The Gmsh failure was a feature-size problem, not a sliver problem.**  On pop_cut_0003 the frozen carrier
(slivers of aspect 1e3, minimum cap quality 0.0016) meshes cleanly with Delaunay; the merged carrier at eps = 1/10 h
fails with 31 nonmanifold facets, and a size floor on the interior field (Mesh.MeshSizeMin) changes nothing.  The merge
at 1/10 h left carrier edges of 0.003 to 0.005 next to sheet triangles of 0.03; the cap, a constrained triangulation of
the carrier, is refined to that size (cap triangles of area 1e-5, hub vertices with 8 to 16 incident triangles, all of
them the nonmanifold facets' vertices) and Delaunay's boundary recovery overlaps tetrahedra while grading from 0.003 to
0.03.  A long thin sliver is a single constrained cap triangle that nothing refines, which is why the census carried
box-face strips of aspect 2.4e4 without a failure.  Frequency at eps = 1/10: 11 of 55 labelled cut cells (20 %) needed
the HXT retry.

*Fix:* the merge radius is set by the surface scale: eps = 1/3 h = 0.0104 (≥ 0.5 remesh size; the Labelle-Shewchuk
snapping range), so every carrier edge is ≥ 1/3 h (test).  Failing cell: minimum carrier edge 0.0046 → 0.0138,
cut-face aspect 12.2 → 2.7, Delaunay nonmanifold facets 31 → 0.  Survey at eps = 1/3 (240 c = 0 cells / 160 tilted
planes): cut-face maximum aspect median 2.69 / 3.13 (was 7.0 / 8.0), box faces 3.80 / 4.28, cells with any triangle
above 10: 5 / 4 (was 70 / 24), carrier nodes after/before median 0.992 / 0.856 (the tilted cut face loses 44 % of its
nodes), audits 400 / 400 PASS, 48-symmetry invariance 0 / 48 broken on the four real cells.

**7.2 The cap carried features far below the sheet scale.**  A sheet chain vertex a hair from a carrier node or edge,
and chain/carrier-edge crossings a few 1e-3 from a chain vertex (the relative 30 % snap of a short chain edge), gave
cap triangles of area 1e-9 to 1e-5 and the 1e-22-volume tetrahedra of every mesh; on pop_cut_0645 they made HXT leave
flat tetrahedra that Netgen's optimiser turned into 4 nonmanifold facets.  *Fix:* a feature-size invariant in the cap
PSLG (`triangulate_plane_region(feature_size = 0.2 remesh)`): a chain vertex within that distance of a carrier node is
moved onto it, one within that distance of a carrier edge is projected onto it, and a crossing within that distance of
a chain vertex is snapped to the chain vertex (absolute, on top of the relative rule).  Rules that keep the sheet a
manifold, each found by a failure: carrier nodes never move; a carrier node on another clip plane (a ring node, already
a vertex of that plane's cap) is never a target and a chain vertex on another plane never moves (pinch of two caps:
the uncut fixture, 4 non-manifold edges); one chain vertex per target and none adjacent on the chain to an assigned
vertex; a fold check on the incident sheet triangles; and crossings are never snapped to a carrier node (that
duplicates the node on the sheet: pop_cut_0165 and pop_cut_1125, 1 and 5 non-manifold edges; the A/B test isolated
it).  Effect on the four probe cells: 434 to 663 chain vertices moved, minimum surface quality 0.0006 → 0.0027 to
0.0031, minimum dihedral angle of the volume mesh 0.04° → 0.13° to 0.15°, sub-5° tetrahedra 379 / 151 / 289 → 129 /
76 / 75 (Delaunay, before 7.3).

**7.3 The volume mesher.**  After 7.1 and 7.2, Gmsh's legacy Delaunay still overlapped four tetrahedra on an ordinary
cut-cap triangle of quality 0.15 (pop_cut_0245, edges 0.017 to 0.031, nothing small anywhere near it) and segfaulted
on two of 60 cut cells (pop_cut_0305, pop_cut_0745).  HXT meshed every one of those surfaces with the same quality
(pop_cut_0003: minimum dihedral 0.149° vs 0.149°, 128 vs 129 sub-5° tetrahedra).  HXT's mesh depends on the thread
count and on nothing else (two runs at the same count are byte-identical; 1, 8 and 16 threads differ), so HXT is the
algorithm of record with the generation thread count pinned to 1 (`MESH_GENERATION_THREADS`; 12 s instead of 8 s on
65k tetrahedra), the crash retry is deleted and a nonmanifold mesh fails closed with a MESH_FAIL receipt.  This is a
decision about which mesher to trust, made on the failure analysis, not a retry.

**7.4 The residual families are harmless; the gate is gone.**  Sweep with the gate disabled (pop_cut_0003 base, fast
preset, energies of 9 smooth port fields): the strip family (plane nearly containing the vertical edge (1, 0, z), gap
0.2 h → 0.002 h) passes every guard at every gap (rigid residual ≤ 2.8e-14, q 8088 to 8328) and the operator
converges smoothly, relative energy steps 5.1e-3, 2.6e-3, 1.6e-3, 5.4e-4, 2.5e-4, 1.7e-4 as the gap shrinks (the
geometry itself changes by the gap); the corner family (tilted plane cutting off corner (1, 1, 1) within 0.2 h → 0.002
h) gives the identical operator at every gap (q 7779, energies equal to all digits): the corner region is void, its
box-face slivers are zero rows.  The receipt keeps the residual gap as a diagnostic (`cut_carrier_residual_strip`);
nothing is refused.

**7.5 Re-validation under the final code** (eps = 1/3, cap feature size 0.2 remesh, HXT at 1 thread, no retry, no gate;
fast preset):

| cell | status | q | rigid residual | nonmanifold | tets < 5° | min dihedral | surface min quality |
|---|---|---|---|---|---|---|---|
| pop_cut_0645 (HXT + Netgen failed before 7.2) | PASS | 2319 | 1.3e-15 | 0 | 152 | 0.305° | 0.0052 |
| pop_cut_0245 (Delaunay overlapped) | PASS | 4395 | 1.0e-14 | 0 | 83 | 0.050° | 0.0011 |
| pop_cut_0305 (Delaunay segfault) | PASS | 9381 | 2.4e-15 | 0 | 125 | 0.096° | 0.0035 |
| pop_cut_0745 (Delaunay segfault) | PASS | 4395 | 3.0e-15 | 0 | 115 | 0.141° | 0.0029 |
| pop_cut_0165 (pinched by the crossing snap) | PASS | 222 | 2.4e-15 | 0 | 16 | 0.700° | 0.0138 |
| pop_cut_1125 (pinched by the crossing snap) | PASS | 8100 | 2.5e-15 | 0 | 108 | 0.173° | 0.0033 |

60-cell run (the same 60 cut cells as the eps = 1/10 baseline): 60 cells, 58 PASS + 2 EMPTY, 0 failures (the
eps = 1/10 baseline had 11 of 55 needing the HXT retry and, after 7.1 and 7.2 alone, 1 hard failure and 2 crashes).
Volume mesh: minimum dihedral angle min 0.0037° median 0.103°, sub-5° tetrahedra median 90 max 162; surface minimum
quality min 0.0001 median 0.0022.  q median 4977, max 10194; total time per cell median 143 s, max 358 s (fast preset,
10 workers).

2 × 1 × 1 cut block (cut face a port, HXT): assembled vs constrained monolithic −4.5e-5 (tension),
6.5e-6 (shear), −5.7e-5 (bending), −1.2e-5 (16H), 1.9e-3 (8H), 4.5e-3 (4H); interface active-set mismatch 0.  The
assembly now reproduces the constrained monolithic model of the same block to five significant figures on the smooth
loads (it was 1.4e-4 to 6.4e-3 before 7.1 to 7.3).  Block mesh: minimum dihedral angle 0.494°, 57 sub-5° tetrahedra
(140 at eps = 1/10 with Delaunay).

Test suites: 21 tests pass (test_cut_carrier, test_sheet_solid_surface).


## 8. The null space of the operator (2026-09-05)

The formula `dim Null(S) = 6 x components + 3 x unsupported carrier nodes` is a LOWER bound.  Measured on five
production labels, the exact null space (eigenvalues below 1e-14 lambda_max) is larger by 2 to 15, and there is no
spectral gap: 17 to 88 further modes sit between 1e-14 and 1e-4 of lambda_max.

**Mechanism, proven.**  `S = P' S_bb P` with `P = scalar (x) I3`, `scalar` the P1 prolongation from the active carrier
columns to the fine boundary-trace vertices, so `Null(S) = null(scalar) (x) R^3 + rotations`:
`dim Null(S) = 3 dim null(scalar) + 3 x components` (matches on every cell where the threshold is unambiguous).
`null(scalar)` holds one indicator per column the trace never reaches, PLUS one vector per pair of neighbouring
carrier columns whose entire material trace is the SAME single mesh vertex (identical columns, cosine 1.000000000000;
four such pairs in pop_cut_0236, one in pop_cut_0136).  The prolongation's null space is clean (the same integer at
every threshold from 1e-4 to 1e-14); the staircase above it belongs to `S_bb`: material attached by a hairline has a
genuinely tiny stiffness, and no threshold separates it.

**Where the zero columns come from.**  Of the zero-support columns, 80 % to 92 % have a patch that provably holds no
material at any mesh resolution (the active-set rule's 0.05 level margin), 8 % to 18 % hold material this mesh missed
(the trace is thinner than the local element).  The second family makes the RANK mesh dependent while q is not (10
cells x 3 tiers: q identical, null dimension 926 / 917 / 895 for pop_uncut_0529).  Deleting columns by mesh support
would therefore give the same geometry three different q and would discard real material; deleting by geometric proof
is safe on the production tier but not on the fast tier (4 proven-empty columns of pop_cut_0473 are used by its
coarser surface).  Decision of record: the active set is unchanged; the label ships the support mask and the
numerical rank at a declared threshold (1e-14 and 1e-10 lambda_max, both recorded); the decoder and the loss never
assume full rank; no Cholesky, no S^-1, no whitening by S; the carrier mass matrix is the normalisation; the
geometric material coverage of each carrier patch (`decide_tri`) is an input feature, not a learned output.
