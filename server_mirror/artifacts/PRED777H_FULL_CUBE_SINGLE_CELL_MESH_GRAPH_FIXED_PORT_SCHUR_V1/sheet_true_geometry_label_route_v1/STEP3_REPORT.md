# Step 3: labels on the TRUE sheet-TPMS geometry (no port collars) — 2026-09-03/04

Owner decisions recorded here: carrier grid 1/32 per face; cells with zero material are skipped; sheet fragments not
connected to any port face are removed; narrow port bands are labeled as they are (no threshold); the world cut surface
is a free surface, never a port.

## 0. Why the geometry changed
The cell geometry used through step 2 carried a 1/32-thick solid plate on every box face ("port collar"): 46% of the G0
material (0.179 of 0.387) sat within 1/32 of a face. The intended object is the sheet solid |phi| <= tau clipped by the
box and the world cut. The collars were a meshing convenience (they made the box faces trivially cappable) and they
changed the physics: they turned the lattice into "sheet + solid diaphragms".

## 1. Volume validation against the literature
Hao Xiaozheng, "Optimization design of carrying heat dissipation performance of TPMS gradient lattice structure"
(Dalian University of Technology, 2023), Table 2.2: for Schwarz P with material  -C < phi < C,  C = 1.755 rho  (R > 0.99,
Monte-Carlo fit). Our thickness field plays the role of C.

| quantity | value |
|---|---|
| tau (mean over the eight cell corners, G0) | 0.4000 |
| rho predicted by the paper fit, C/1.755 | 0.2279 |
| rho by exact integration of our level set (400^3 samples) | 0.2284 (+0.23%) |
| rho of our Tet4/Tet10 mesh | 0.2288 (+0.16% vs exact) |

The pipeline reproduces the published density law to 0.2%, i.e. within the paper's own fit residual. For reference, the
collar geometry gave rho = 0.387, which by the same law would require C = 0.68 instead of the specified 0.40.

## 2. Pipeline (module `sheet_solid_surface.py`)
1. marching cubes of |phi| - tau on a padded, box-aligned grid (h = 1/48 production, 1/96 for references);
2. exact Newton projection of every vertex onto the level set (residual 1.5e-15), plane-constrained for boundary vertices;
3. exact clipping by the six box planes and the cut plane, with plane snapping;
4. short-boundary-edge collapse (fold-over guarded), pymeshlab isotropic remeshing, re-projection, box-corner restoration,
   removal of in-plane sheet triangles;
5. per plane: the material cross-section is triangulated by constrained Delaunay (Triangle, run in a subprocess) whose
   constraints are the carrier nodes AND carrier edges of every carrier triangle that meets the band, plus one interior
   material support point per such triangle; segment arrangement (crossings and collinear overlaps split) before the call;
6. region classification by confidence-weighted flood fill seeded from the contour edges (material side read from the
   clipped level a small offset to each side);
7. T-junction repair on shared lines, duplicate-vertex merge, floating-component removal, outward orientation;
8. Gmsh volume mesh (Delaunay or HXT + Netgen, boundary preserved), Tet10 promotion, fixed-port Schur (MKL Pardiso).

Active carrier set (mesh independent): the vertices of every carrier triangle the material band reaches within a
tolerance. This set is a SUPERSET of any mesher's support, so the port interpolation P is never modified: partition of
unity, affine and rigid-body reproduction stay exact. Verified on all five cases: `support_outside_active = 0` and the
Schur rigid residual is 5e-15 to 1.2e-14 (an earlier clip-and-renormalize variant cost 1e-6 and was removed).
Active nodes that no fine node touches (28-152 per cell, the fringe of the band) carry exactly zero rows and columns in
the operator; the assembler drops them.

## 3. Census of the 16 pilot cells on the true geometry (carrier 1/32)
| cell | rho | cut | faces with material | active carrier nodes | surface components / dropped |
|---|---|---|---|---|---|
| science_00..07 (uncut) | 0.2283-0.2284 | no | 6 | 1705-1722 | 1 / 0 |
| science_13 | 0.2284 | yes | 6 | 1705 | 1 / 0 |
| science_14 | 0.2168 | yes | 5 | 1431 | 1 / 0 |
| science_12 | 0.1137 | yes | 5 | 890 | 1 / 0 |
| science_11 | 0.0567 | yes | 5 | 415 | 1 / 0 |
| science_10 (thin wall + cut) | 0.019 | yes | 3 | 262 | 1 / 0 |
| science_09 | 0.0057 | yes | 1 | 287 | 1 / 0 |
| science_08, science_15 | 0.0000 | yes | - | - | EMPTY: the sheet does not enter the retained wedge |
All non-empty cells are watertight with a single connected component; no floating fragment occurred in the pilot set.
The old "near_empty_nonempty" stratum disappears with the collars: those cells are empty, not near-empty.

## 4. Single-cell accuracy (carrier 1/32, low-frequency norm, wavelength >= 4H)
Mesher reproducibility (Gmsh HXT vs Delaunay, same surface):

| case | G0 | G1 | G3 | G4 | G5 |
|---|---|---|---|---|---|
| reproducibility | 0.12% | 0.15% | 0.12% | 0.14% | 0.56% |

Three-level mesh convergence (surface 1/48 + volume 0.03 -> 1/64 + 0.02 -> 1/96 + 0.015), same norm:

| case | 0.03 vs 0.02 | 0.02 vs 0.015 | order | extrapolated error at 0.03 | at 0.02 |
|---|---|---|---|---|---|
| G0 | 0.63% | 0.26% | 1.5 | 1.4% | 0.7% |
| G1 | 0.85% | 0.29% | 1.7 | 1.6% | 0.7% |
| G3 | 0.63% | 0.26% | 1.5 | 1.4% | 0.7% |
| G4 | 0.65% | 0.27% | 1.5 | 1.4% | 0.7% |
| G5 | - | - | - | - | - |

G0 timings: 82 s / 253 s / 434 s of Schur at 255k / 554k / 1030k Tet10 dof.
G5 (rho = 0.019, the thinnest cell) does not survive the finer settings: at 0.02 Gmsh reports "a segment and a facet
intersect", at 0.015 the surface has 4 non-manifold edges. Its 0.03 label is fine (rigid residual 4.7e-15). Robustness of
the surface pipeline on near-degenerate cells is an open item (section 6).

Mesh size dominates; mesher noise is 5-10x smaller. Recommended production setting: surface 1/64, volume 0.02
(~260 s per cell, ~0.7%); 0.03 (~100 s, ~1.4%) is the fast setting.

## 5. 2x1x1 block validation (two cells with their own graded thickness; loads on x=2, x=0 fixed)
Free-interface monolithic Tet10 vs the fixed-port assembly, with a constrained-monolithic control that ties only the
external port faces to the carrier:

| load | carrier 1/16: assembled vs free | external constraint alone | internal interface alone | carrier 1/32: assembled vs free | external alone | interface alone |
|---|---|---|---|---|---|---|
| tension x | -2.9% | -3.0% | +0.07% | -0.73% | -0.82% | +0.09% |
| shear y | -1.6% | -1.6% | +0.01% | -0.32% | -0.38% | +0.06% |
| bending z | -3.7% | -3.8% | +0.06% | -1.18% | -1.30% | +0.13% |
| wave 16H | -13.1% | -13.5% | +0.4% | -5.96% | -6.35% | +0.42% |
| wave 8H | -20.5% | -21.1% | +0.7% | -8.47% | -8.98% | +0.55% |
| wave 4H | -35.0% | -37.1% | +3.4% | -24.6% | -26.3% | +2.4% |

The internal interface — the thing the operator library must get right — is exact to 0.1% for smooth loads at both carrier
resolutions (it was 1-2.5% with collars). Everything else is the P1 constraint on the loaded EXTERNAL face, and it falls
by 3-4x when the carrier goes from 1/16 to 1/32. A few carrier nodes per block are active but tied to no fine node
(3 at 1/32, 7 at 1/16); they carry zero stiffness and are dropped at assembly.

## 6. Production command and reproducibility (2026-09-04)
`scripts/pred777h_full_cube_v1/produce_sheet_label.py --case-id ... --geometry-manifest ... --output-dir ... [--resolution fast|production|reference]`

- create-only output; presets fast (surface 1/48, remesh 0.03, interior 0.05), production (1/64, 0.02, 0.04),
  reference (1/96, 0.015, 0.03); carrier 1/32; Gmsh Delaunay by default.
- fail-closed: material volume below 1e-6 -> status EMPTY and no operator; port support outside the geometric active
  set, rigid residual above 1e-10, or min eigenvalue below -1e-8 lambda_max -> error.
- writes `FIXED_PORT_SCHUR_TET10.npz` with the operator on the active set, carrier ids / coordinates / port membership,
  the rigid basis, the carrier mass and Laplace-Beltrami matrices (the training loss needs them) and the material
  volume; `mesh.mesh`; `SHEET_SOLID_SURFACE.off`; SHA-256 receipts in `LABEL_RECEIPT.{json,md}` with the git head, the
  resolution and the execution settings (thread counts).
- measured on G0: fast 161 s (surface+mesh 13 s, port 29 s, Schur 81 s), production 298 s on G3, 120 s on G1 (rho 0.11).
- certificates on every label so far: partition of unity 0.0, affine 1e-16, rigid residual 3e-15 to 1.2e-14,
  min eigenvalue > -2.1e-17.

Reproducibility (item 5):

| repeat | mesh.mesh | surface OFF | operator npz |
|---|---|---|---|
| same settings, same threads | identical | identical | identical |
| 8 vs 4 threads | identical | identical | differ in bytes, agree to 2.1e-14 |

The geometry is fully deterministic; only the Schur back-substitution depends on MKL's summation order, so the receipt
records the thread counts. MKL_CBWR=COMPATIBLE does not make the operator byte-identical across thread counts (the
chunked BLAS products are affected too); pin the thread count if byte-level reproduction is required.

A defect the new unit tests (`tests/pred777h_full_cube_v1/test_sheet_solid_surface.py`, 5 tests) caught: the carrier
node -> surface vertex map was built before fragment removal and vertex merging renumbered the vertices, so the
"Gmsh moved a carrier node" check was verifying the wrong vertices. The map is now rebuilt by coordinate after every
topology change and is the cap conformity certificate: on G0, 1086 carrier nodes lie on the surface and all 1086 keep
their exact coordinates through the volume mesher. Labels were not affected (the port map is compiled geometrically).

## 7. Open items before production
1. DONE (2026-09-04): the active set is a superset of every mesher's support; P is unmodified; rigid residual 1e-14.
2. DONE for G1/G3/G4 (table in section 4). G5 fails at the two finer settings; make the surface pipeline robust on
   near-degenerate cells (rho ~ 0.02 and below) and then finish its convergence row.
3. Census over the full population, not just the 16 pilot cells; redefine the strata (the near-empty stratum is gone,
   an EMPTY class appears) and pick new representative cases.
4. DONE (2026-09-04): `produce_sheet_label.py`, section 6.
5. DONE (2026-09-04): determinism table in section 6.
5b. Cap-triangle quality: 50-230 tets below 5 degrees per cell; measure the effect on the operator.
6. DONE (2026-09-04): the carrier mass and Laplace-Beltrami matrices ship inside every label npz.
7. A 2x2x2 assembly check.
8. MMG (Gmsh Algorithm3D=7) fails on sheet surfaces ("unable to set tetrahedron"); HXT and Delaunay are the production pair.
