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
