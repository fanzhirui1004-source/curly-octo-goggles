# Step 4: surface robustness on near-degenerate cells (G5 class), 2026-09-04

Scope: the open item of STEP3 section 6 (G5, rho = 0.019, failed at the 0.02 and 0.015 levels), generalized to the
whole population the production run will meet: thin walls (rho = 0.1), vertical cuts at any angle and offset,
grazing cuts (cut plane tangent to a wall), knife-edge wedges, cells that fall into several pieces, and empty cells.

## 1. What failed and why (diagnosis on the old pipeline)
| case | symptom | root cause |
|---|---|---|
| G5 at 0.02 | Gmsh "a segment and a facet intersect" | a 0.06h edge left by the remesher was folded by the final Newton projection into two self-intersecting slivers (verified: 1 proper pair) |
| G5 at 0.015 | 4 non-manifold edges | the remesher moved boundary vertices off the planes; the re-snap produced degenerate triangles; the cap then merged two chain vertices and created a duplicate sheet triangle |
| thin cut cells (rho 0.1) | caps covering both sides of a chain, volume 3x too large | the material side of a chain edge was read from the level set next to the edge; where the cut plane is tangent to a wall (an X-crossing of the contour, present in every cut cell) the level is quadratically small and chain vertices sit off the level set, so the side test flips |
| knife-edge wedges | Gmsh "overlapping facets" | material thinner than the snap distance next to the plane gives facets that Gmsh treats as coincident |
| cut leaving two pieces | Gmsh "no tetrahedra" | one surface loop was built for several closed shells |
| coincident vertices after projection | zero-area triangles that block their own collapse | the fold-over guard rejected every collapse of an already degenerate triangle |

## 2. Changes (sheet_solid_surface.py, tet10_label.py, produce_sheet_label.py)
1. Remesh first, clip exactly afterwards (`_remesh_and_clip`): the sheet is clipped 1.5 h outside every plane, remeshed
   (pymeshlab isotropic), projected back onto the level set, then clipped EXACTLY by the planes.  Every boundary
   vertex lies on its planes to round-off, box corners are exact, and the remesher's own boundary handling is out of
   the loop.  `resnap_boundary_to_planes` / `restore_plane_corners` are no longer on the production path.
2. Slivers from the exact clipping: near-plane vertices are snapped onto the plane (0.2 h, joint least squares when a
   vertex already lies on other planes), short edges (< 0.3 h) are collapsed anywhere on the sheet under three guards
   (a vertex only merges into a neighbour lying on every plane it lies on, link condition, no fold-over, no new
   needle below quality 0.05), needles are repaired by collapse or edge flip, coincident vertices are merged.
3. Projection guard: in-plane Newton steps are capped at 0.5 h and skipped where the in-plane gradient is below 25 %
   of the full gradient (the sheet tangent to the plane), so a snapped vertex never flies along the plane.
4. Cap classification from the sheet orientation: the open sheet is oriented outward before the caps are built and
   the material side of every chain edge is -n x t (combinatorial, exact also at tangencies).  The level set is only
   sampled as a diagnostic (`cap_level_disagreements`, 0 on every cell so far) and label conflicts are counted.
5. Verified self-intersection gate: pymeshlab candidates are re-tested pairwise with an exact triangle/triangle test
   (touching is not intersecting; the pymeshlab coplanar test over-reports on the caps), plus fold-over pairs on the
   open sheet.  A stage result with intersections, non-manifold edges or boundary vertices off their planes is retried
   with the remesh size scaled by 0.9, 1.12, 0.8; when every factor fails the build raises (fail closed).  The closed
   surface is gated again before Gmsh and `produce_sheet_label.py` refuses a self-intersecting surface.
6. General T-junction repair on the closed surface (any vertex strictly inside any edge splits that edge).
7. Resolution follows the thinnest wall: t_min = 2 tau_min / max|grad phi| = 0.184 tau_min; the remesh size is
   capped at 0.6 t_min and the marching-cubes grid refined to at least 3 cells per t_min.  For rho = 0.1
   (tau = 0.1755, t_min = 0.032) this means 1/94 and 0.0194 at every preset; cells with tau >= 0.18 are unchanged.
8. One Gmsh volume per closed shell (a cut can leave several pieces of material).
9. Empty cells raise `EmptyCellError` from the remesh stage instead of a pymeshlab error.
10. Schur: carrier columns without fine support (about a third of the layout columns on the true geometry) are
    skipped in the Pardiso solve.  Byte-identical operator on G0 (max |dS| = 0), Schur time 75 s -> 33 s at the fast
    preset.
11. After the in-plane projections, vertices that drifted to within the snap distance of the cut plane (a box-face
    vertex sliding toward the cut line) are snapped onto it, and every sheet edge whose endpoints lie on the same two
    planes (a snapped sliver along a plane line, which both caps would claim) is collapsed.
12. Cap regions that no chain edge touches are void on port faces (their boundary is a triangulation-hull edge, not a
    physical one; a material cusp below the mesh resolution must not get a cap without a sheet), and follow the level
    on the cut plane (whose boundary is the cube polygon).
13. Simple cap quality pass (STEP3 item 3, kept simple): after classification, the longest edge of every cap
    triangle below quality 0.3 is flipped when it is neither a chain edge nor a PSLG segment and the flip improves the
    pair.  Slivers whose long edge is a chain or carrier edge remain (a chain vertex almost on a carrier line): those
    need a Steiner point on the segment and a matching sheet split, which is not done.
14. Gmsh runs in a child process (`mesh_sheet_solid(isolate=True)`): a Gmsh crash becomes a RuntimeError with the
    log tail, not a dead production worker.  (One stress cell crashed Gmsh in-process after pymeshlab and meshes
    normally in isolation.)

## 3. Evidence
### 3.1 The five convergence cells, final code (carrier 1/32, low-frequency norm, wavelength >= 4H)
Surface 1/48 + remesh 0.03 -> 1/64 + 0.02 -> 1/96 + 0.015 (the sheet surfaces of these runs were rebuilt with the
committed module and are byte-identical to the panel's):

| cell | rho | cut | 0.03 vs 0.02 | 0.02 vs 0.015 | whitened 0.02 vs 0.015 | min dihedral at 0.03 / 0.02 / 0.015 (deg) |
|---|---|---|---|---|---|---|
| G0 | 0.229 | no | 0.63% | 0.26% | 7.3% | 0.013 / 0.019 / 0.15 |
| G1 | 0.114 | yes | 1.61% | 0.31% | 8.2% | 0.008 / 0.092 / 0.078 |
| G3 | 0.229 | no | 0.64% | 0.24% | 7.2% | 0.080 / 0.009 / 0.042 |
| G4 | 0.217 | yes | 1.30% | 0.28% | 7.4% | 0.0008 / 0.032 / 0.18 |
| G5 | 0.019 | yes | 0.72% | 0.39% | 36% | 0.24 / 0.19 / 1.3 |

G5's row is now complete (it failed at both finer levels before).  The 0.02 -> 0.015 step is 0.24-0.39% on every
cell, as in STEP3.  The fast level (0.03) is worse than before on the two cut cells (G1 1.61%, G4 1.30%; STEP3 had
0.85% and 0.65%): the knife-edge truncation at 0.2 h is coarser there and the boundary slivers give tetrahedra with
dihedral angles below 0.01 deg, whose stiffness is ill-conditioned (the whitened distance 0.03 -> 0.02 is 1.3-1.6
against 0.07 for 0.02 -> 0.015).  The production preset (0.02) is unaffected; the fast preset should not be used for
cut cells.  Cap-triangle quality (STEP3 item 3, "simple, not mainline") is the lever for the sliver tetrahedra.

Old pipeline vs new pipeline on G0 at the fast preset (same carrier, same support): 0.31% low-frequency, i.e. inside
the mesh-noise band (mesher reproducibility 0.12%, one refinement step 0.63%).

### 3.2 Thin uniform cell, rho = 0.1 (tau = 0.1755, wall 0.032 thick), thickness-scaled resolution
| levels | low-frequency distance | tets | Schur seconds (8 threads) |
|---|---|---|---|
| 1/94 + 0.0194 (production, scaled) vs 1/96 + 0.015 | 0.33% | 68k vs 126k | 122 vs 277 |
| 1/96 + 0.015 vs 1/128 + 0.011 | 0.17% | 126k vs 282k | 277 vs 557 |

Same convergence band as the tau = 0.4 cells; the production preset is adequate for the thinnest walls of the
population, and cheaper than G0 because the sheet area is the same while the wall is thinner.

### 3.3 Sixteen pilot cells at the production preset (final code)
14 cells watertight, zero verified self-intersections, zero cap-classification conflicts, one component each, volumes
unchanged to 5 digits (0.22859 for the eight uncut tau = 0.4 cells); science_08 and science_15 report EMPTY with a
clean message.  Three independent passes gave identical statistics (the pipeline is deterministic).

### 3.4 Schur column skip
G0, fast preset, same surface and mesh (md5 identical): operator identical to the bit (max |dS| = 0);
Schur 75 s -> 33 s.

### 3.5 Stress population (342 manifests, production preset, surface + Gmsh volume)
Six thickness fields (uniform rho 0.1 / 0.23 / 0.5 and three graded fields, one of them spanning tau 0.1755 to
0.8775 inside a single cell) x {uncut, vertical cuts at 0 / 15 / 30 / 45 deg x 12 offsets across the cell} plus 24
random graded cells (cut and uncut), `scripts/pred777h_full_cube_v1/make_sheet_stress_population.py`; every cell
through `sheet_surface_check.py --mesh` (surface + verified gates + Gmsh volume), results in `robustness/`.

| outcome | cells |
|---|---|
| PASS (watertight, no verified self-intersection, no cap conflict, Gmsh volume) | 326 |
| EMPTY (the sheet does not enter the retained region) | 15 |
| MESH_FAIL | 1 |

The one failure (`mid_grad_diag_a45_d03`, graded rho 0.1-0.5, 45 deg cut) has a clean surface but a cap sliver of
quality 8e-5 (a chain vertex almost on a carrier edge) on which Gmsh fails with both algorithms; it is the case for
the Steiner-point cap repair noted under change 13.  Among the 326 passes: one cell needed a remesh retry, five cells
are two or more separate pieces of material (each piece a Gmsh volume), no cap-classification conflict and no
T-junction repair fired.  Surface build 8 s median / 46 s max, Gmsh 9 s median / 71 s max, 64k tetrahedra median /
178k max (thick cells).  Surface volume against a 120^3 point-grid estimate: 0.16 % median; the largest deviations
are the near-empty slivers (volume ~1e-4) where the grid estimate itself is coarse.

The pipeline is deterministic: the 16 pilot cells were rebuilt three times with identical statistics, and the panel
surfaces rebuilt with the committed module are byte-identical.

## 4. Production consequences
- Production preset unchanged (1/64 + 0.02 + interior 0.04, carrier 1/32); cells with tau_min < 0.18 are refined
  automatically (1/94 + 0.0194 for rho = 0.1) and cost less than a rho = 0.23 cell.
- Fail-closed statuses a census will count: EMPTY, SURFACE_FAIL (every remesh retry failed), SURFACE_BAD
  (watertightness or verified self-intersection), MESH_FAIL (Gmsh, now in a child process, with one HXT retry).
- The fast preset (0.03) is not suitable for cut cells (1.3-1.6 % against 0.3 % at 0.02).

## 5. Open items carried forward
- Cap slivers whose long edge is a chain or carrier edge (Steiner point on the segment + matching sheet split):
  affects Gmsh min dihedral (0.001-0.1 deg) and the one stress failure.
- The 2x2x2 assembly validation and the population census run on this pipeline next.
