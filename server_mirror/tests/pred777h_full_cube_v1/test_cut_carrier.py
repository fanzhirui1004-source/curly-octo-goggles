"""Cut-face carrier conditioning (cut_carrier): bounded aspect ratio, exact conformity, cross-cell agreement and
cube-symmetry invariance of the merged carrier."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import conforming_port_mesh as cpm
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import cut_carrier as cc
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import rotations as rot
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import sheet_label_pipeline as pipe
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.full_cube_backend import compile_full_cube_geometry_inputs
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.full_cube_geometry import (
    CanonicalRationalPlane,
    make_cell_spec,
    make_parent_p1_field,
)
from cctpms.port.global_cell_patch_trace_layout import (
    audit_global_cell_patch_shared_edge_trace_agreement,
    build_global_cell_patch_p1_prolongation,
)

N = 16
H = Fraction(1, N)


def spec_with(a, b, c, d, origin=(0, 0, 0)):
    field = make_parent_p1_field(field_id="CUT_CARRIER_TEST", base=Fraction(2, 5))
    return make_cell_spec(thickness_field=field, cell_world_origin=tuple(Fraction(o) for o in origin), cell_id="cut_carrier",
                          parent_world_cut_id="CUT", world_cut=CanonicalRationalPlane("cut_0", "cut_0", Fraction(a), Fraction(b), Fraction(c), Fraction(d)))


def base_layout(spec):
    charts, *_ = compile_full_cube_geometry_inputs(spec)
    return pipe.build_carrier_layout(charts, spec, carrier_n=N, cut_merge=False)


def aspect(layout, faces=None):
    ports = tuple(str(t.source_id) for t in layout.local_traces)
    coords, _, per_face = cpm.carrier_shell(layout, ports)
    out = {}
    for src, tris in per_face:
        T = coords[tris]; area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        e = np.stack([np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 1], axis=1), np.linalg.norm(T[:, 0] - T[:, 2], axis=1)], 1)
        out[src] = float((e.max(1) ** 2 / np.maximum(2 * area, 1e-300)).max())
    return out


# a vertical plane through the neighbourhood of the grid line (1/2, 1/2, z), and a tilted plane through the
# neighbourhood of the grid node (1/2, 1/2, 1/2): the crossings of the grid lines through those nodes nearly coincide
NEAR_VERTICAL = (9, 4, 0, Fraction(13, 2) + Fraction(1, 1000))
NEAR_TILTED = (9, 4, 7, Fraction(10, 1) + Fraction(1, 1000))
# 9x + 4y + 5z = 9.001 also passes within 1e-4 h of the corners (1, 0, 0) and (0, 1, 1): no in-face move can remove
# the slivers of a corner, so the pass reports the residual gap and the gate refuses the cell
NEAR_CORNER = (9, 4, 5, Fraction(9, 1) + Fraction(1, 1000))


@pytest.mark.parametrize("plane", [NEAR_VERTICAL, NEAR_TILTED])
def test_merge_bounds_the_aspect_ratio_and_keeps_the_trace_exact(plane) -> None:
    spec = spec_with(*plane); base = base_layout(spec)
    before = aspect(base)
    merged, report = cc.merge_cut_carrier(base, spec, carrier_n=N)
    after = aspect(merged)
    assert before["cut_0"] > 100, "the fixture must exercise a near-degenerate crossing"
    assert after["cut_0"] < 30 and max(after.values()) < 60
    # the carrier has no feature below the merge radius (the surface-scale invariant behind the volume mesher's robustness)
    coords, tris, _ = cpm.carrier_shell(merged, tuple(str(t.source_id) for t in merged.local_traces))
    E = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]]); L = np.linalg.norm(coords[E[:, 0]] - coords[E[:, 1]], axis=1)
    assert L.min() >= float(cc.CUT_CARRIER_MERGE_EPS * H) * 0.999
    assert report["clusters"] >= 1 and report["merged_nodes"] >= 1
    # the trace is still one conforming P1 surface: shared edges agree, every triangle is invertible for the prolongation
    assert audit_global_cell_patch_shared_edge_trace_agreement(merged)["status"] == "PASS"
    n = np.asarray([float(v) for v in merged.local_trace("cut_0").chart.plane_normal_exact]); d = float(merged.local_trace("cut_0").chart.plane_offset_exact)
    for trace in merged.local_traces:
        X = np.asarray(trace.node_coordinates); T = np.asarray(trace.triangles)
        build_global_cell_patch_p1_prolongation(trace, (X[T[:, 0]] + 2 * X[T[:, 1]] + 3 * X[T[:, 2]]) / 6.0)
        if str(trace.source_id) == "cut_0":
            assert np.abs(X @ n - d).max() < 1e-12, "merged cut-face nodes stay exactly on the plane"
    # global table and traces agree
    coords = np.asarray(merged.global_scalar_node_coordinates)
    for trace in merged.local_traces:
        M = merged.local_to_global_scalar[str(trace.source_id)].tocsr()
        assert np.allclose(coords[M.indices], np.asarray(trace.node_coordinates), atol=1e-12)
        assert tuple(merged.global_scalar_node_keys[i] for i in M.indices) == tuple(trace.global_node_ids)


def test_near_corner_cut_is_reported_not_refused() -> None:
    """A plane through a cell corner leaves box-face slivers no in-face move can remove; they are reported in the
    receipt (diagnostic) and the trace is still a valid conforming carrier: the family is harmless (sweep 2026-09-05)."""
    spec = spec_with(*NEAR_CORNER); base = base_layout(spec)
    merged, report = cc.merge_cut_carrier(base, spec, carrier_n=N)
    assert report["skipped"].get("corner", 0) >= 1
    assert report["min_unfixed_gap_over_h"] < 0.01
    diag = cc.cut_carrier_gate(merged.diagnostics["cut_carrier_merge"])
    assert diag is not None and "corner" in diag["skipped"]
    assert audit_global_cell_patch_shared_edge_trace_agreement(merged)["status"] == "PASS"
    assert cc.cut_carrier_gate(cc.merge_cut_carrier(base_layout(spec_with(*NEAR_TILTED)), spec_with(*NEAR_TILTED), carrier_n=N)[1] | {"applied": True}) is None


def test_uncut_cell_is_untouched() -> None:
    field = make_parent_p1_field(field_id="CUT_CARRIER_TEST", base=Fraction(2, 5))
    spec = make_cell_spec(thickness_field=field, cell_world_origin=(Fraction(0), Fraction(0), Fraction(0)), cell_id="uncut", parent_world_cut_id="NO_CUT", world_cut=None)
    charts, *_ = compile_full_cube_geometry_inputs(spec)
    layout = pipe.build_carrier_layout(charts, spec, carrier_n=N, cut_merge=False)
    merged, report = cc.merge_cut_carrier(layout, spec, carrier_n=N)
    assert merged is layout and report["applied"] is False


def test_neighbouring_cells_agree_on_the_shared_face() -> None:
    """Two cells sharing the face x = 1 of the first, cut by one tilted world plane crossing that face near a grid
    node: the merged nodes and triangles on the shared face are identical (keys and world coordinates)."""
    a, b, c = 9, 4, 5; d = Fraction(9) + Fraction(4, 2) + Fraction(5, 2) + Fraction(1, 1000)     # near (1, 1/2, 1/2)
    left = spec_with(a, b, c, d); right = spec_with(a, b, c, d, origin=(1, 0, 0))
    ml, rl = cc.merge_cut_carrier(base_layout(left), left, carrier_n=N); mr, rr = cc.merge_cut_carrier(base_layout(right), right, carrier_n=N)
    assert rl["clusters"] >= 1 and rr["clusters"] >= 1

    def face(layout, src, shift):
        coords, _, per_face = cpm.carrier_shell(layout, (src,))
        keys = layout.global_scalar_node_keys
        nodes = {keys[g]: tuple(np.round(coords[g] + shift, 12).tolist()) for g in np.unique(per_face[0][1])}
        tris = {tuple(sorted(keys[g] for g in t)) for t in per_face[0][1]}
        return nodes, tris
    nl, tl = face(ml, "box_x_max", np.zeros(3)); nr, tr = face(mr, "box_x_min", np.array([1.0, 0.0, 0.0]))
    assert nl == nr, "shared-face node keys and world coordinates differ between the two cells"
    assert tl == tr, "shared-face triangulations differ between the two cells"


def test_merged_carrier_is_invariant_under_the_48_cube_symmetries() -> None:
    """Uniform thickness, so a cube symmetry acts on the plane alone: the merged layout of the image equals the
    image of the merged layout, node set and triangle set (the merge rule and every fan rule are isometry invariant)."""
    a, b, c, d = NEAR_TILTED
    n = np.array([a, b, c], dtype=float)
    spec0 = spec_with(a, b, c, d); m0, _ = cc.merge_cut_carrier(base_layout(spec0), spec0, carrier_n=N)
    C0, F0, _ = cpm.carrier_shell(m0, tuple(str(t.source_id) for t in m0.local_traces))
    key = lambda p: tuple(np.round(np.asarray(p) * 1e9).astype(np.int64).tolist())
    proper = rot.proper_cubic_rotations()
    matrices = [r.matrix for r in proper] + [np.diag([-1, 1, 1]).astype(np.int8) @ r.matrix for r in proper]
    centre = np.array([0.5, 0.5, 0.5])
    for R in matrices:
        Rn = np.rint(R.astype(float) @ n).astype(int); dd = Fraction(d) - Fraction(int(n @ (2 * centre))) / 2 + Fraction(int(Rn @ (2 * centre))) / 2
        spec1 = spec_with(int(Rn[0]), int(Rn[1]), int(Rn[2]), dd)
        m1, _ = cc.merge_cut_carrier(base_layout(spec1), spec1, carrier_n=N)
        C1, F1, _ = cpm.carrier_shell(m1, tuple(str(t.source_id) for t in m1.local_traces))
        RC = rot.rotate_points(C0, R); idx1 = {key(p): i for i, p in enumerate(C1)}
        perm = np.array([idx1.get(key(p), -1) for p in RC])
        assert (perm >= 0).all() and len(C1) == len(C0), f"node set not invariant under {R.tolist()}"
        assert {tuple(sorted(int(perm[v]) for v in t)) for t in F0} == {tuple(sorted(int(v) for v in t)) for t in F1}, f"triangulation not invariant under {R.tolist()}"


def test_cut_face_trace_export_stitches_across_neighbours(tmp_path) -> None:
    """The coupling deliverable: two neighbouring cells export their cut_0 traces in world coordinates with global keys;
    the ring nodes on the shared face have the same keys and coordinates in both files."""
    a, b, c = 9, 4, 5; d = Fraction(9) + Fraction(4, 2) + Fraction(5, 2) + Fraction(1, 1000)
    infos = {}; files = {}
    for tag, origin in (("left", (0, 0, 0)), ("right", (1, 0, 0))):
        spec = spec_with(a, b, c, d, origin=origin); charts, *_ = compile_full_cube_geometry_inputs(spec)
        layout = pipe.build_carrier_layout(charts, spec, carrier_n=N)
        infos[tag] = pipe.write_cut_face_trace(tmp_path / f"{tag}.npz", layout, spec); files[tag] = np.load(tmp_path / f"{tag}.npz")
    assert infos["left"]["nodes"] > 0 and infos["right"]["nodes"] > 0
    L, R = files["left"], files["right"]
    lk = {str(k): tuple(np.round(x, 9).tolist()) for k, x in zip(L["node_keys"], L["node_xyz_world"])}
    rk = {str(k): tuple(np.round(x, 9).tolist()) for k, x in zip(R["node_keys"], R["node_xyz_world"])}
    shared = set(lk) & set(rk)
    assert len(shared) >= 3, "the cut polygons of the two cells share their ring on the common face"
    assert all(lk[k] == rk[k] for k in shared), "shared ring nodes have identical world coordinates in both exports"
    assert all(abs(lk[k][0] - 1.0) < 1e-9 for k in shared), "the shared ring lies on the common face x = 1"
    n, dd = L["plane_world"][:3], L["plane_world"][3]
    assert np.abs(L["node_xyz_world"] @ n - dd).max() < 1e-9 and np.abs(R["node_xyz_world"] @ n - dd).max() < 1e-9
