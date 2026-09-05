# Mirror of the Tet10 boundary-conforming fixed-port label route

Snapshot of the files committed on the AutoDL server repository
`cut_control_tpms_v1_full_cube_single_cell_v1` (branch `codex/pred777h-full-cube-single-cell-v1`,
commits `2c2c385` and `843f7b4`, 2026-09-03). The server is the source of truth; this copy exists so the
reports can be read without a server connection.

- `artifacts/.../tet10_conforming_label_route_v1/ADDENDUM_04.md`: frozen protocol (Tet10, conforming ports, gates, per-stratum resolution).
- `.../STEP1_REPORT.md`, `.../STEP2_REPORT.md`: CGAL-route diagnosis and the conforming-route results (six-case panel, physics validation, cut-position sweep).
- `.../panel/`, `.../sweep/`, `.../validation/`, `.../labels/`: gate JSON/MD, mesh reports, sweep receipts and analysis, 2x1x1 assembly validation, production label receipts.
- `src/.../tet10_label.py`, `src/.../conforming_port_mesh.py`, `scripts/pred777h_full_cube_v1/*.py`, `tests/.../test_tet10_label.py`: the new modules, scripts and unit test.
- `patch0001_bounded_topology.patch`: the CGAL topology-gate change (nonmanifold boundary edges bounded instead of hard-zero) as a git patch.


## Update of 2026-09-05: the sheet-TPMS true-geometry label route

Snapshot of every file changed on the server between commits `843f7b4` and `17b0fdd` (22 commits, the true-geometry
sheet route: no port collars, carrier-conforming caps, the cut face as a port, the block-with-skin validation, and
the sliver root cause).  The server remains the source of truth.

- `artifacts/.../sheet_true_geometry_label_route_v1/STEP6_CUT_CARRIER_REPORT.md`: cut-face carrier conditioning, the
  three root-caused mechanisms, and (section 8) the null space of the operator.
- `.../STEP7_SKIN_REPORT.md`: the coupling contract proven end to end on a block with a bonded skin.
- `.../STEP8_SLIVER_REPORT.md`: lambda_max was a sliver artefact (a needle cap triangle from a forced support point),
  root-caused and removed; what was tried and rejected (CGAL remeshing, boundary-locked perturbation).
- `src/.../sheet_solid_surface.py`, `cut_carrier.py`, `sheet_label_pipeline.py`, `sheet_contract.py`,
  `scripts/pred777h_full_cube_v1/produce_sheet_label.py`, `sheet_block_validation.py`, `sheet_skin_validation.py`,
  `scripts/pred777h_full_cube_v1/diagnostics/`, `cpp/pred777h_sliver_remesh/`, `tests/pred777h_full_cube_v1/`.
- `patches/0001..0022`: the 22 server commits as git patches, in order.
