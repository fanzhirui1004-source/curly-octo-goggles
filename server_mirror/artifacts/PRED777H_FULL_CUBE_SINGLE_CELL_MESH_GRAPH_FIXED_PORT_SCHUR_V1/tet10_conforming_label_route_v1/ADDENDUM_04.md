# Addendum 04: label route change (Tet10, boundary-conforming fixed ports, carrier-weighted gates)

Status: FROZEN 2026-09-03. Applies to PRED777H_FULL_CUBE_SINGLE_CELL_MESH_GRAPH_FIXED_PORT_SCHUR_V1 label production.
Evidence: `_claude_diag/tet10_gates/STEP1_REPORT.md`, `_claude_diag/step2/STEP2_REPORT.md` (v4), per-case
`step2s/<case>/gates/TET10_LABEL_GATES.json`, `sweep/`, `labels/`.

## 1. Topology gate semantics (patch 0001)
- `nonmanifold_boundary_edge_count` moves from the hard-zero set to a bounded field (limit 8). A boundary pinch left
  by Mesh_3's perturber at an acute protected feature edge keeps the tetrahedral complex valid; the fixed-port
  kinematic, carrier-rank and FE gates certify physics independently (G2/A1 passes all of them with one pinch).
- The C++ polyhedral wrapper reports the count and no longer aborts. Duplicate boundary facets remain fatal.
- The perturber stays ON: no-perturb meshes contain 0-degree slivers that the FE assembler rejects.

## 2. Element: Tet10 (straight-sided P2) replaces Tet4 for every label
- Tet4 is locked by 15-18% (median energy ratio) on TPMS cells in the assembly-relevant norm and does not converge
  within reach (A1 still 13.8% from Tet10 A2). Tet10 at A1-level sizing is ~1% (Richardson over A0/A1/A2).
- Promotion (`tet10_label.promote_mesh_artifact_to_tet10`): one global midpoint per edge; port-edge midpoints inherit
  the average carrier row (partition of unity exact, dyadic weights). Constant-strain patch test: Tet10 and Tet4
  Schur operators agree to 1e-10 under affine boundary data (unit test).

## 3. Port meshing: boundary-conforming (Gmsh) replaces CGAL Mesh_3 for label production
- Every active port face is triangulated by the deterministic carrier trace layout and midpoint-refined k times; the
  fine port nodes are nested in the carrier, P has dyadic weights, affine reproduction error is 0. No interpolation
  certificate is needed.
- Cavity surfaces come from the coherent polyhedral domain OFF; their triangulation is improved by tangential vertex
  smoothing with projection onto the ORIGINAL polyhedron (deviation at round-off, cavity volume change 4e-4 relative);
  crease vertices (dihedral > 30 deg) are frozen. G0: surface quality min 0.23 (was 0.004), volume mesh min dihedral
  10.5 deg with zero tets below 10 deg (was 0.31 deg, 387 below 5 deg).
- Interior mesher: Gmsh HXT (Algorithm3D=10) with Netgen optimization. Netgen optimization is MANDATORY: without it
  near-empty cells get almost no interior nodes and the operator moves by 2% (G2, k=1).
- CGAL Mesh_3 remains a challenger route only.

## 4. Gates and norms
- Norm: carrier mass-whitened, restricted to Laplace-Beltrami modes of the carrier surface with wavelength >= 4H.
  Raw Frobenius is reported but never gated: the short-wavelength part of S is a boundary layer of depth 1/k that
  no practical mesh resolves and that does not enter assembled structural response.
- Reproducibility gate: max pairwise distance of Tet10 Schur operators over the three Netgen-optimized interior
  algorithms (HXT, Delaunay, Frontal) <= 0.5%. Result on the six-case panel: 0.23-0.37% (all PASS). The HXT-without-
  Netgen variant is reported as degraded-mesher sensitivity (0.32-0.50% on full cells, 0.73% on G2 at k=2).
- Convergence gate: k -> k+1 (port refinement + interior 0.05 -> 0.035) <= 3%. Result: G0 0.82%, G1 0.85%,
  G3 0.82%, G4 0.82%, G5 0.77%, G2 (k2 -> k3) 0.65%.

## 5. Label resolution per population stratum (protocol decision)
| stratum | port refine k | interior size | Tet10 dof (observed) | solver |
|---|---|---|---|---|
| uncut_nonempty, shallow_cut, moderate_cut (full cells) | 2 | 0.035 | 0.6-0.8M | SuperLU (Pardiso above 1M internal dof) |
| deep_cut, thin_wall_ligament | 1 | 0.05 | 0.06-0.4M | SuperLU |
| near_empty_nonempty | 2 | 0.035 | < 0.05M | SuperLU |

Production command: `scripts/pred777h_full_cube_v1/produce_tet10_label.py --stratum {full,thin,near_empty}` (create-only,
SHA-256 receipts, rigid-residual and PSD checks; deterministic: G5 twice bit-identical). Regression: the six-case panel
(`run_tet10_label_gates.py` + `combine_tet10_label_gates.py`) must pass both gates before any population run.

## 6. Physics facts recorded with this route
- 2x1x1 assembly of G0 labels vs free-interface monolithic Tet10: compliance deficit 6.4% (tension), 0.7% (shear),
  8.8% (bending). A constrained-monolithic control attributes 4.7% / 0.1% / 6.5% of it to the trace constraint on the
  loaded EXTERNAL face alone; the internal interface (both traces tied + one-sided collars) costs 0.7-2.5% for smooth
  loads and 3-7% for waves, with interface displacements within 1% (smooth) and 5% (8H wave). External free/loaded
  faces need a decision (tpms_free labeling, richer external carrier, or accept).
- Cut-position sweep (x <= d, d = 26/64..34/64): q jumps by 192 at grid lines 28/64 and 32/64, the operator seen from
  the opposite face moves 0.17-0.21% per 1/64 step everywhere, including across the grid lines. No numerical jump.
