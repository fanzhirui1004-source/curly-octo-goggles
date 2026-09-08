from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import sheet_solid_surface as sss
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.cgal_mesh3_adapter import CgalGeometryParameters
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.full_cube_backend import (
    build_full_cube_trace_layout,
    compile_full_cube_geometry_inputs,
)
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.full_cube_geometry import (
    make_cell_spec,
    make_parent_p1_field,
)

TAU = 0.4


@pytest.fixture(scope="module")
def fixture():
    field = make_parent_p1_field(field_id="SHEET_SURFACE_TEST", base=Fraction(2, 5))
    spec = make_cell_spec(thickness_field=field, cell_world_origin=(Fraction(0), Fraction(0), Fraction(0)),
                          cell_id="sheet_unit", parent_world_cut_id="NO_CUT", world_cut=None)
    charts, *_ = compile_full_cube_geometry_inputs(spec)
    layout = build_full_cube_trace_layout(spec, charts=charts)
    geometry = CgalGeometryParameters(tau_corners=(TAU,) * 8, cut_plane=None)
    return geometry, layout


def test_sheet_level_is_the_collar_free_band(fixture) -> None:
    geometry, _ = fixture
    # material is the band |phi| <= tau with phi = sum cos(2 pi x_i): thin sheet, void at the cell centre and corners
    points = np.asarray([[0.5, 0.5, 0.5], [0.25, 0.25, 0.2], [0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    level = sss.sheet_level(geometry, points)
    assert level[0] == pytest.approx(3.0 - TAU)   # cell centre: phi = -3
    assert level[1] < 0.0                         # inside the band: |phi| = |cos(0.4 pi)| = 0.31 < tau
    assert level[2] == pytest.approx(3.0 - TAU)   # cube corner: phi = +3
    assert level[3] > 0.0                         # phi = 2, outside the band


def test_clip_by_plane_keeps_the_retained_side_and_shares_cut_vertices() -> None:
    V = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    F = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    n = np.asarray([1.0, 0.0, 0.0]); d = 0.5
    V2, F2 = sss.clip_by_plane(V, F, n, d)
    s = V2 @ n - d
    assert s.max() <= 1.0e-12                              # nothing survives beyond the plane
    on = np.abs(s) <= 1.0e-12
    assert on.sum() == 3                                   # three edges cross the plane
    assert len(np.unique(np.round(V2, 12), axis=0)) == len(V2)   # cut vertices are shared, never duplicated
    area = 0.5 * np.linalg.norm(np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]]), axis=1).sum()
    assert abs(area - 0.5) < 1.0e-12


def test_projection_lands_on_the_level_set_and_respects_planes(fixture) -> None:
    geometry, _ = fixture
    planes = sss.clip_planes(geometry)
    rng = np.random.default_rng(0)
    V = rng.uniform(0.05, 0.95, size=(200, 3))
    V[:20, 0] = 0.0                                        # vertices pinned to the x = 0 plane
    out = sss.project_to_sheet_in_planes(geometry, V, planes, iters=20)
    residual = np.abs(sss.sheet_level(geometry, out))
    assert np.quantile(residual, 0.9) < 1.0e-9             # Newton converges except where the gradient vanishes
    assert np.abs(out[:20, 0]).max() <= 1.0e-12            # plane-pinned vertices stayed in their plane


def test_surface_is_watertight_carrier_conforming_and_has_the_published_density(fixture) -> None:
    geometry, layout = fixture
    result = sss.build_sheet_solid_surface(geometry, layout, n_per_unit=48, remesh_size=0.04, carrier_n=16)
    report = result.report
    assert report["watertight"]["non_two_manifold_edges"] == 0
    assert report["components"]["dropped"] == 0
    # Schwarz P sheet density law C = 1.755 rho (Hao 2023): tau = 0.4 -> rho = 0.228
    assert abs(report["volume"] - TAU / 1.755) < 0.01
    # every carrier node the band reaches is a surface vertex, at its exact coordinate
    coords = np.asarray(layout.global_scalar_node_coordinates)
    for face, mapping in result.carrier_nodes.items():
        for global_id, vertex in mapping.items():
            assert np.allclose(result.V[vertex], coords[global_id], atol=1.0e-12)
    # boundary vertices of the sheet part all lie on a clip plane
    sheet = result.F[result.labels == "tpms_free"]
    boundary = {v for edge in sss._directed_boundary(sheet) for v in edge}
    planes = sss.clip_planes(geometry)
    for v in boundary:
        assert min(abs(result.V[v] @ n - d) for _, n, d in planes) <= 1.0e-9


def test_active_carrier_set_is_geometric_and_covers_the_band(fixture) -> None:
    geometry, layout = fixture
    ports = ("box_x_min", "box_x_max", "box_y_min", "box_y_max", "box_z_min", "box_z_max")
    coords, per_face, support_points = sss.carrier_active_set(geometry, layout, ports, carrier_n=16)
    active = {src: np.unique(tris) for src, tris in per_face}
    assert all(len(v) > 0 for v in active.values())
    for src, tris in per_face:
        # every support point is material and inside its triangle's plane
        for k, point in support_points[src].items():
            assert sss.sheet_level(geometry, point[None])[0] < 0.0
        # the six faces of an uncut cell carry the same band up to sampling: node counts agree within 5%
        assert abs(len(np.unique(tris)) - len(np.unique(per_face[0][1]))) <= 0.05 * len(np.unique(per_face[0][1]))


def test_tri_tri_proper_distinguishes_touching_from_crossing() -> None:
    A = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # shares the edge (0,0,0)-(1,0,0) at a right angle: touching only
    B = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert not sss.tri_tri_proper(A, B)
    # pierces A through its interior
    C = np.asarray([[0.2, 0.2, -0.5], [0.3, 0.2, 0.5], [0.2, 0.3, 0.5]])
    assert sss.tri_tri_proper(A, C)
    # coplanar and adjacent along an edge (a T-junction free tiling): not an intersection
    D = np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert not sss.tri_tri_proper(A, D)
    # coplanar and overlapping
    E = np.asarray([[0.1, 0.1, 0.0], [0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    assert sss.tri_tri_proper(A, E)


def test_split_all_t_junctions_repairs_a_hanging_vertex() -> None:
    # two triangles sharing edge 0-1, plus a vertex 4 in the middle of that edge used only by a third triangle
    V = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, -1.0, 0.0], [0.5, 0.0, 0.0], [0.5, -0.2, 0.3]])
    F = np.asarray([[0, 1, 2], [1, 0, 3], [4, 1, 5]], dtype=np.int64)
    F2, labels, n = sss.split_all_t_junctions(V, F, ["a", "b", "c"])
    assert n == 1 and len(F2) == 5
    assert not any(0 in f and 1 in f for f in F2)          # the long edge no longer exists
    assert sorted(labels) == ["a", "a", "b", "b", "c"]


def test_collapse_never_creates_a_needle() -> None:
    # a short edge whose collapse would flatten the neighbouring triangle into a needle is refused
    V = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.5, 0.001, 0.0], [0.0, -0.5, 0.0], [-0.5, 0.2, 0.0]])
    F = np.asarray([[0, 1, 2], [1, 0, 3], [0, 2, 4], [3, 2, 1]], dtype=np.int64)
    planes = []
    V2, F2, done = sss._collapse_edges(V, F, planes, [(0, 1)])
    assert done == 1 and len(F2) == 2                       # the needle and its twin vanish with the collapsed edge
    assert sss.tri_quality(V2, F2).min() >= 0.05
    # a collapse whose only admissible direction (vertex 3 is pinned on two planes) would turn the good triangle
    # (2, 6, 5) into the needle (3, 6, 5) with the same orientation is refused by the quality guard
    V3 = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.3, 0.0], [0.5, 0.002, 0.0], [0.5, -1.0, 0.0], [0.0, 0.0015, 0.0], [1.0, 0.002, 0.0]])
    F3 = np.asarray([[0, 3, 2], [3, 1, 2], [0, 4, 3], [3, 4, 1], [2, 6, 5]], dtype=np.int64)
    planes3 = [("a", np.asarray([0.0, 1.0, 0.0]), 0.002), ("b", np.asarray([1.0, 0.0, 0.0]), 0.5)]
    _, _, done3 = sss._collapse_edges(V3, F3, planes3, [(2, 3)])
    assert done3 == 0


def test_thin_cut_cell_surface_is_watertight_and_has_the_right_volume() -> None:
    """rho = 0.1 sheet cut by an oblique vertical plane: knife edges, tangencies and a thin wall in one cell."""
    from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.full_cube_geometry import CanonicalRationalPlane
    tau = Fraction(1755, 10000)
    field = make_parent_p1_field(field_id="SHEET_THIN_CUT_TEST", base=tau)
    cut = CanonicalRationalPlane("cut_0", "cut_0", Fraction(1), Fraction(5774, 10000), Fraction(0), Fraction(7227, 10000))
    spec = make_cell_spec(thickness_field=field, cell_world_origin=(Fraction(0), Fraction(0), Fraction(0)),
                          cell_id="sheet_thin_cut", parent_world_cut_id="CUT_TEST", world_cut=cut)
    charts, *_ = compile_full_cube_geometry_inputs(spec)
    layout = build_full_cube_trace_layout(spec, charts=charts)
    geometry = CgalGeometryParameters(tau_corners=(float(tau),) * 8, cut_plane=(1.0, 0.5774, 0.0, 0.7227))
    result = sss.build_sheet_solid_surface(geometry, layout, n_per_unit=48, remesh_size=0.03, carrier_n=16)
    report = result.report
    assert report["resolution_effective"]["remesh_size"] < 0.03           # thickness-scaled: the wall is 0.032 thick
    assert report["watertight"]["non_two_manifold_edges"] == 0
    assert report["self_intersecting_triangles"] == 0
    assert report["cap_classification_conflicts"] == 0
    # exact material volume from a point grid (retained side of every plane, inside the band)
    n = 100; xs = (np.arange(n) + 0.5) / n
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = sss.sheet_level(geometry, grid) < 0.0
    for _, nn, dd in sss.clip_planes(geometry):
        inside &= (grid @ nn - dd) <= 0.0
    assert abs(report["volume"] - inside.mean()) < 0.03 * inside.mean()


def test_equivariant_active_set_is_a_superset_and_rotation_invariant(fixture) -> None:
    """Node-distance active set: contains every vertex of every carrier triangle the band reaches, and maps onto itself
    under a proper rotation and a reflection of a graded, cut cell (the triangle rule does not)."""
    from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import rotations as rot
    geometry, layout = fixture
    ports = ("box_x_min", "box_x_max", "box_y_min", "box_y_max", "box_z_min", "box_z_max")
    _, per_face, _ = sss.carrier_active_set(geometry, layout, ports, carrier_n=16)
    tri_nodes = {int(g) for _, tris in per_face for g in np.unique(tris)}
    eq_nodes = sss.equivariant_active_nodes(geometry, layout, ports, carrier_n=16)
    assert tri_nodes <= eq_nodes
    assert len(eq_nodes) <= 1.3 * len(tri_nodes)
    # graded, cut cell: compare the node set of the cell with the rotated node set of its symmetry image
    coords = np.asarray(layout.global_scalar_node_coordinates)
    corners = np.asarray([0.2, 0.35, 0.3, 0.5, 0.25, 0.45, 0.4, 0.6])           # order (i,j,k) for i,j,k in {0,1}
    cut = (1.0, 0.5, 0.0, 0.95)
    for R in (np.asarray([[0, -1, 0], [0, 0, -1], [1, 0, 0]]), np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])):
        idx = lambda i, j, k: (i * 2 + j) * 2 + k
        rc = np.empty(8)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = np.array([i, j, k]) - 0.5; xi = R.T @ x + 0.5; rc[idx(i, j, k)] = corners[idx(*[int(round(v)) for v in xi])]
        n_new, d_new = rot.transform_plane(np.asarray(cut[:3]), cut[3], R)
        g0 = CgalGeometryParameters(tau_corners=tuple(corners), cut_plane=cut); g1 = CgalGeometryParameters(tau_corners=tuple(rc), cut_plane=(*n_new, d_new))
        a0 = sss.equivariant_active_nodes(g0, layout, ports, carrier_n=16); a1 = sss.equivariant_active_nodes(g1, layout, ports, carrier_n=16)
        key = lambda p: tuple(np.rint(np.asarray(p) * 16).astype(int).tolist())          # the fixture layout is the 1/16 grid
        lookup = {key(p): g for g, p in enumerate(coords)}
        image = {lookup[key(rot.rotate_points(coords[g][None], R)[0])] for g in a0}
        assert image == a1


def test_constrained_snap_does_not_drag_vertices_along_a_shallow_face() -> None:
    """A vertex on the x face within the snap distance of a 3 deg cut plane used to be moved along the face by
    eps / sin(3 deg) (nineteen snap distances) onto the intersection line; the joint correction is now capped."""
    theta = np.radians(3.0); n_cut = np.array([np.cos(theta), np.sin(theta), 0.0]); d_cut = 1.0372
    planes = [("box_x_max", np.array([1.0, 0.0, 0.0]), 1.0), ("cut_0", n_cut, d_cut)]
    eps = 0.004; y_line = (d_cut - np.cos(theta)) / np.sin(theta)
    V = np.array([[1.0, y_line - 0.5 * eps / np.sin(theta), 0.5],     # on the face, 0.5 eps inside the cut plane, 9.5 eps from the line
                  [1.0, y_line - 0.5 * eps, 0.5],                     # on the face, 0.5 eps from the line: snapped onto it
                  [0.5, 0.5, 0.5]])
    W = sss.snap_near_plane_constrained(V, n_cut, d_cut, eps, planes)
    assert np.allclose(W[0], V[0]), "vertex far from the line must stay where the exact clip can handle it"
    assert np.linalg.norm(W[1] - V[1]) <= 2.0 * eps + 1e-12
    assert abs(W[1] @ n_cut - d_cut) < 1e-9 and abs(W[1][0] - 1.0) < 1e-9, "near the line the vertex lands on both planes"
    assert np.allclose(W[2], V[2])


def test_retained_facet_polygon_stops_at_the_cut_line() -> None:
    n_cut = np.array([0.9987, 0.0508, 0.0]); n_cut = n_cut / np.linalg.norm(n_cut); d_cut = 1.0074
    planes = [("box_x_min", np.array([-1.0, 0, 0]), 0.0), ("box_x_max", np.array([1.0, 0, 0]), 1.0), ("box_y_min", np.array([0, -1.0, 0]), 0.0),
              ("box_y_max", np.array([0, 1.0, 0]), 1.0), ("box_z_min", np.array([0, 0, -1.0]), 0.0), ("box_z_max", np.array([0, 0, 1.0]), 1.0), ("cut_0", n_cut, d_cut)]
    poly = sss._retained_facet_polygon(planes, "box_x_max")
    y_line = (d_cut - n_cut[0]) / n_cut[1]
    assert np.allclose(poly[:, 0], 1.0) and poly[:, 1].max() <= y_line + 1e-9 and poly[:, 1].max() >= y_line - 1e-9
    assert (poly @ n_cut - d_cut).max() <= 1e-9
    area = 0.5 * abs(sum(poly[i][1] * poly[(i + 1) % len(poly)][2] - poly[(i + 1) % len(poly)][1] * poly[i][2] for i in range(len(poly))))
    assert abs(area - y_line) < 1e-9
    full = sss._retained_facet_polygon(planes, "box_y_min")            # the cut does not reach y = 0 inside the cell
    assert len(full) == 4 and abs(0.5 * abs(np.cross(full[1] - full[0], full[3] - full[0])[1]) * 2 - 1.0) < 1e-9
    cut_poly = sss._retained_facet_polygon(planes, "cut_0")
    assert len(cut_poly) >= 4 and (cut_poly.min(axis=0) >= -1e-9).all() and (cut_poly.max(axis=0) <= 1 + 1e-9).all()


def test_unionjack_carrier_triangulation_is_cube_symmetric(fixture) -> None:
    """The frozen carrier fans each square from the corner with the smallest node id (a hash of the coordinates), so
    the port P1 space depends on the cell's orientation; the parity rule is invariant under all 48 symmetries."""
    from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import carrier_triangulation as ct
    from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import conforming_port_mesh as cpm
    from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import rotations as rot

    _, layout = fixture
    n = 16
    ports = tuple(sorted(str(t.source_id) for t in layout.local_traces))
    unionjack = ct.unionjack_layout(layout, carrier_n=n)

    def triangle_set(lay):
        coords, tris, _ = cpm.carrier_shell(lay, ports)
        return {tuple(sorted(tuple(np.rint(coords[i] * n).astype(int).tolist()) for i in t)) for t in tris}

    frozen_set, union_set = triangle_set(layout), triangle_set(unionjack)
    assert len(union_set) == len(frozen_set), "the retriangulation keeps the triangle count of each square"
    proper = rot.proper_cubic_rotations()
    matrices = [r.matrix for r in proper] + [np.diag([-1, 1, 1]).astype(np.int8) @ r.matrix for r in proper]
    broken_frozen = 0
    for M in matrices:
        image = {tuple(sorted(tuple(np.rint(p).astype(int).tolist()) for p in rot.rotate_points(np.asarray(t, dtype=float) / n, M) * n)) for t in union_set}
        assert image == union_set, "the parity diagonal must be invariant under every cube symmetry"
        image_frozen = {tuple(sorted(tuple(np.rint(p).astype(int).tolist()) for p in rot.rotate_points(np.asarray(t, dtype=float) / n, M) * n)) for t in frozen_set}
        broken_frozen += image_frozen != frozen_set
    assert broken_frozen >= len(matrices) - 1, "the frozen fan is not symmetric (this is the defect being fixed)"
    # node identity is untouched: only which diagonal splits each square changes
    assert unionjack.global_scalar_node_keys == layout.global_scalar_node_keys


def test_degenerate_cut_through_a_cell_edge_is_detected_before_meshing() -> None:
    """The exact chart compiler refuses a cut plane containing a cell edge; the route detects it by exact arithmetic
    so the case gets a status and a receipt instead of an uncaught error (population census 2026-09-04)."""
    from fractions import Fraction

    from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import sheet_label_pipeline as pipe
    from generated_cell.pred768_canonical_geometry_spec import CanonicalRationalPlane

    field = make_parent_p1_field(field_id="DEGENERATE_TEST", base=Fraction(2, 5))

    def spec_with(a, b, d, origin=(Fraction(0), Fraction(0), Fraction(0))):
        return make_cell_spec(thickness_field=field, cell_world_origin=origin, cell_id="degenerate",
                              parent_world_cut_id="CUT", world_cut=CanonicalRationalPlane("cut_0", "cut_0", Fraction(a), Fraction(b), Fraction(0), Fraction(d)))

    # x + y <= 1 passes through the vertical edges at (1,0) and (0,1): refused by the compiler
    assert pipe.cut_contains_a_cell_edge(spec_with(1, 1, 1)) is not None
    assert pipe.cut_contains_a_cell_edge(spec_with(1, 1, 0)) is not None
    assert pipe.cut_contains_a_cell_edge(spec_with(1, 1, 2)) is not None
    assert pipe.cut_contains_a_cell_edge(spec_with(2, 1, 3)) is not None
    # generic offsets are admissible
    assert pipe.cut_contains_a_cell_edge(spec_with(1, 1, Fraction(53, 50))) is None
    assert pipe.cut_contains_a_cell_edge(spec_with(2, 1, Fraction(3, 2))) is None
    assert pipe.cut_contains_a_cell_edge(spec_with(1, 0, Fraction(1, 2))) is None
    # the test is in the CELL's frame: one world plane can be degenerate for one cell and generic for its neighbour,
    # so the check belongs to the cell, not to the cut
    world = (Fraction(1), Fraction(3), Fraction(3))          # x + 3y = 3 contains the edge (0,1) of the cell at the origin
    assert pipe.cut_contains_a_cell_edge(spec_with(*world)) is not None
    assert pipe.cut_contains_a_cell_edge(spec_with(*world, origin=(Fraction(1), Fraction(0), Fraction(0)))) is None
