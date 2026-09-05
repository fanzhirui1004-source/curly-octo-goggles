# Step 2 report: boundary-conforming fixed ports (Gmsh route), v4 (2026-09-03)

NOTE (2026-09-03, after the panel): the cell geometry carries 1/32 solid plates on all box faces (46% of the G0 material),
which are not part of the intended sheet-TPMS design. All numbers below are pipeline properties on that geometry; see
ADDENDUM_04 section 0 for the consequences and the route to the true geometry.

Construction: port faces triangulated by the carrier trace layout, midpoint-refined k times (k=1: 1/32, k=2: 1/64);
cavity surfaces from the coherent polyhedral OFF with polyhedron-preserving tangential smoothing (deviation 3e-16,
cavity volume change 4e-4 relative; G0 volume mesh min dihedral 10.5 deg, no tets below 10 deg). Gmsh volume mesh,
boundary preserved. P exactly nested (dyadic weights, affine error 0.0). Production command deterministic: two runs of
produce_tet10_label.py on G5 give byte-identical meshes and bit-identical Schur operators.

## 1. Six-case panel (smoothed cavities, Tet10, carrier low-frequency norm, wavelength >= 4H)

Reproducibility = max pairwise distance over interior meshers at k=1 (size 0.05); convergence = max distance of the
k=1 operators from the k=2 reference (size 0.035, Delaunay). Thresholds: 0.5% / 3%.

| case | stratum | q | lowfreq modes | repro, 3 Netgen-optimized meshers (HXT, Delaunay, Frontal) | repro incl. HXT without Netgen | convergence k1 -> k2 |
|---|---|---|---|---|---|---|
| G0 | uncut full cell | 4614 | 759 | 0.35% PASS | 0.50% (at threshold) | 0.82% PASS |
| G1 | shallow cut | 3078 | 504 | 0.31% PASS | 0.50% PASS | 0.85% PASS |
| G3 | moderate cut | 4614 | 759 | 0.35% PASS | 0.50% PASS | 0.82% PASS |
| G4 | deep cut | 4422 | 711 | 0.37% PASS | 0.49% PASS | 0.82% PASS |
| G5 | thin wall | 1476 | 207 | 0.23% PASS | 0.32% PASS | 0.77% PASS |
| G2 (k=1) | near-empty wedge | 153 | 24 | 0.37% PASS | 2.11% FAIL | 2.10% (nong) / 0.44% |
| G2 (k=2, ref k=3) | near-empty wedge | 153 | 24 | 0.33% PASS | 0.73% | 1.09% (nong) / 0.65% PASS |

Reading. Every maximum is the pair that contains the HXT-without-Netgen mesh; among the three Netgen-optimized meshers
the spread is 0.23-0.37% on all six cases. On the near-empty wedge the unoptimized mesh has almost no interior Tet4
nodes at k=1 (198 nodes vs 261-282), so its Tet10 operator is a different discretization, 2% away and 2% from the k=2
reference. Decision: Netgen optimization is a mandatory part of the label mesher (it is the production default); the
reproducibility gate is defined over the three Netgen-optimized algorithms; the unoptimized variant is reported as a
degraded-mesher sensitivity, not gated. Near-empty cells are labeled at k=2 (their meshes are tiny: 1k-4k nodes).

Comparison with step 1 (CGAL Mesh_3 interior, non-nested ports): G0 1.05% / 3.9%, G5 1.04% / 2.9% -> both FAIL; the
conforming route reduces mesher noise by 3x and convergence error by 4x at 2-6x fewer dof (G0: 386k Tet10 dof at k=1).

## 2. Physics validation (2x1x1 block of G0, Tet10, k=1 labels; consistent loads f_c = P^T f_fine)

| load | assembled vs free-interface monolithic | trace constraint only (constrained monolithic vs free) | assembled vs constrained monolithic (= internal interface) | interface displacement (x=1) rel. L2 |
|---|---|---|---|---|
| uniform tension x | -6.4% | -4.7% | -1.8% | 0.9% |
| shear y | -0.7% | -0.1% | -0.7% | 0.8% |
| bending (z-linear) | -8.8% | -6.5% | -2.5% | 0.8% |
| wave, wavelength 16H | -26.5% | -21.3% | -6.6% | 2.4% |
| wave, wavelength 8H | -33.6% | -30.1% | -5.0% | 5.1% |
| wave, wavelength 4H | -51.7% | -50.3% | -2.8% | 1.2% |

Reading. The "constrained monolithic" control is the same monolithic mesh with only the loaded x=2 trace tied to the
P1 carrier; its deficit is the price of the trace constraint on a loaded EXTERNAL face alone: 4.7% (tension), 6.5%
(bending), 21-50% (waves; the loads shorter than 8H are at or beyond the carrier's resolution). What remains between
the constrained monolithic and the fixed-port assembly is the internal interface (both cells' x=1 traces tied to the
carrier plus the one-sided 1/32 collars): 0.7-2.5% for smooth loads and 3-7% for waves, with interface displacements
within 1% (smooth) and 5% (8H wave). The physically relevant mechanism on the loaded face is that the one-sided
collar skin is bending-dominated and the P1 trace forces kinks at 1/16 spacing.

Implication for the library: cells whose port faces are true external boundaries (free or loaded) carry a 6-9%
systematic over-stiffness at H=1/16 for smooth loads; internal interfaces do not. Options: (a) do not declare a
free external face as a port (label it tpms_free); (b) a richer carrier on external faces only; (c) accept and
document for the application's loading scenarios.

## 3. Physical-jump sweep (G0 geometry cut at x <= d, d = 26/64 ... 34/64, Tet10 labels, thin preset k=1)

Distance between consecutive labels restricted to the uncut x=0 face block (289 carrier nodes, 156 low-frequency
vector modes), relative, low-frequency norm:

| d (64ths) | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 |
|---|---|---|---|---|---|---|---|---|---|
| q | 2886 | 2886 | 2886 | 3078 | 3078 | 3078 | 3078 | 3270 | 3270 |
| step from previous d | - | 0.21% | 0.19% | 0.20% | 0.20% | 0.19% | 0.21% | 0.17% | 0.20% |

The carrier dimension q jumps by 192 when the cut crosses the grid lines 28/64 and 32/64 (a new column of port
nodes activates), but the operator seen from the opposite face moves by the same 0.2% per 1/64 as everywhere else:
no numerical jump accompanies the combinatorial jump. The fixed-port label is Lipschitz in the cut position at this
resolution, which is the property the NN needs.

## 4. Production
- `produce_tet10_label.py --stratum {full,thin,near_empty}`: create-only output, SHA-256 receipts, rigid residual and
  PSD checks, Pardiso above 1M internal dof. G2 (near_empty) 3 s; G5 (thin) bit-identical across two runs;
  G0 (full, k=2, size 0.035): k=2 reference label PASS (753k Tet10 dof, rigid residual 5e-14, 105 min with SuperLU); the production setting for full cells is k=1 (decision 2026-09-03, see section 5).
- Pardiso path verified against SuperLU on G2 k=2: max |dS|/max|S| = 5e-15 (Tet10), 1e-15 (Tet4).
- Unit tests: tests/pred777h_full_cube_v1/test_tet10_label.py (6), plus the patch-0001 gate tests (37 passed).
