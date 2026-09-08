from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.conforming_port_mesh import (
    carrier_shell,
    refine_shell,
    watertight_report,
)
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.fixed_port_adapter import (
    compile_fixed_port,
)
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.full_cube_backend import (
    build_full_cube_trace_layout,
    compile_full_cube_geometry_inputs,
)
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.full_cube_geometry import (
    make_cell_spec,
    make_parent_p1_field,
)
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.mesh_artifact import (
    MeshArtifact,
)
from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1.tet10_label import (
    build_carrier_norms,
    compare_operators,
    dense_fixed_port_schur,
    promote_mesh_artifact_to_tet10,
)


def _geometry_and_layout():
    field = make_parent_p1_field(field_id="TET10_LABEL_TEST")
    specification = make_cell_spec(
        thickness_field=field,
        cell_world_origin=(Fraction(0), Fraction(0), Fraction(0)),
        cell_id="tet10_label_unit",
        parent_world_cut_id="NO_CUT",
        world_cut=None,
    )
    charts, _, _, _ = compile_full_cube_geometry_inputs(specification)
    layout = build_full_cube_trace_layout(specification, charts=charts)
    return specification, layout


def _corner_tet_split_at_centroid(specification) -> MeshArtifact:
    """Corner tetrahedron of the unit cube split at its centroid: three port faces, one free face, one interior node."""

    nodes = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.25, 0.25, 0.25),
        ),
        dtype=np.float64,
    )
    tets = np.asarray(((0, 1, 2, 4), (0, 3, 1, 4), (0, 2, 3, 4), (1, 3, 2, 4)), dtype=np.int64)
    artifact = MeshArtifact(
        mesh_nodes=nodes,
        tet_connectivity=tets,
        element_family="TET4",
        element_order=1,
        boundary_facets=np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64),
        boundary_patch_membership=(("tpms_free",), ("box_x_min",), ("box_y_min",), ("box_z_min",)),
        node_tags={"boundary": np.asarray((1, 1, 1, 1, 0), dtype=np.int8)},
        element_tags={"material": np.zeros(4, dtype=np.int64)},
        node_attributes={},
        element_attributes={},
        material_attributes={"young_modulus": np.ones(4, dtype=np.float64)},
        thickness_attributes={"thickness": np.full(5, 0.4, dtype=np.float64)},
        mesh_quality={"volume": np.full(4, 1.0 / 24.0, dtype=np.float64)},
        feature_edges=np.empty((0, 2), dtype=np.int64),
        source_geometry_hash=specification.spec_hash,
        mesher_name="unit-test",
        mesher_version="1",
        mesher_parameters={},
        canonicalization_metadata={"index_base": 0},
    )
    artifact.validate()
    return artifact


@pytest.fixture(scope="module")
def fixture():
    specification, layout = _geometry_and_layout()
    mesh = _corner_tet_split_at_centroid(specification)
    port = compile_fixed_port(mesh, specification, layout)
    return mesh, specification, layout, port


def test_tet10_promotion_keeps_partition_of_unity_and_marks_only_port_edges(fixture) -> None:
    mesh, _, _, port = fixture
    promo = promote_mesh_artifact_to_tet10(mesh, port)
    assert promo.parent_node_count == 5
    assert promo.elements.shape == (4, 10)
    # 10 edges in total (6 corner-corner + 4 corner-centroid); only the 6 corner-corner edges lie on faces.
    assert len(promo.node_coordinates) == 5 + 10
    # Every corner-corner edge lies on one of the three port faces (the free face shares its edges with them);
    # the four corner-centroid edges are interior: 6 port-edge midpoints, 4 interior midpoints.
    assert promo.port_midpoint_count == 6
    assert len(promo.port_node_indices) == len(port.fine_port_node_ids) + 6
    row_sums = np.asarray(promo.scalar_prolongation.sum(axis=1)).ravel()
    assert np.max(np.abs(row_sums - 1.0)) <= 1.0e-12
    # Midpoint rows are the average of the endpoint rows: their weights are dyadic halves of corner weights.
    corner_rows = promo.scalar_prolongation[: len(port.fine_port_node_ids)].toarray()
    mid_rows = promo.scalar_prolongation[len(port.fine_port_node_ids):].toarray()
    for row in mid_rows:
        matched = False
        for i in range(len(corner_rows)):
            for j in range(i + 1, len(corner_rows)):
                if np.allclose(row, 0.5 * (corner_rows[i] + corner_rows[j])):
                    matched = True
        assert matched


@pytest.mark.parametrize("element", ["TET4", "TET10"])
def test_dense_fixed_port_schur_is_symmetric_psd_and_annihilates_rigid_modes(fixture, element) -> None:
    mesh, _, _, port = fixture
    schur, timing = dense_fixed_port_schur(mesh, port, element=element, workers=1, column_chunk_size=64)
    q3 = 3 * len(port.active_global_carrier_ids)
    assert schur.shape == (q3, q3)
    assert np.allclose(schur, schur.T)
    scale = float(np.abs(schur).max())
    assert scale > 0.0
    assert np.abs(schur @ port.rigid_basis).max() <= 1.0e-9 * scale
    eigenvalues = np.linalg.eigvalsh(schur)
    assert eigenvalues.min() >= -1.0e-9 * scale
    assert timing["fine_dof"] > 0.0


def test_tet10_and_tet4_agree_on_the_constant_strain_patch(fixture) -> None:
    # Four non-coplanar corner nodes fix an affine field; the P1 carrier lift forces every port midpoint onto the
    # affine trace, so the exact solution has constant strain and both elements must reproduce it exactly.
    mesh, _, _, port = fixture
    s4, _ = dense_fixed_port_schur(mesh, port, element="TET4", workers=1)
    s10, _ = dense_fixed_port_schur(mesh, port, element="TET10", workers=1)
    assert np.abs(s10 - s4).max() <= 1.0e-10 * np.abs(s4).max()


def test_carrier_norms_and_operator_comparison(fixture) -> None:
    _, _, layout, port = fixture
    norms = build_carrier_norms(layout, port, wavelength_cutoff=0.25)
    q = len(port.active_global_carrier_ids)
    assert norms.mass_scalar.shape == (q, q)
    assert np.linalg.eigvalsh(norms.mass_scalar).min() > 0.0
    ones = np.ones(q)
    assert np.abs(norms.laplacian_scalar @ ones).max() <= 1.0e-10 * np.abs(norms.laplacian_scalar).max()
    assert 0 < norms.lowfreq_vector.shape[1] < 3 * q
    assert norms.lowfreq_vector.shape[1] % 3 == 0
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3 * q, 3 * q))
    S = A @ A.T + np.eye(3 * q)
    same = compare_operators(S, S, norms)
    assert same["raw_frobenius"] == 0.0
    assert same["lowfreq_frobenius"] == 0.0
    scaled = compare_operators(1.1 * S, S, norms)
    assert abs(scaled["raw_frobenius"] - 0.1) <= 1.0e-12
    assert abs(scaled["whitened_frobenius"] - 0.1) <= 1.0e-12
    assert abs(scaled["lowfreq_frobenius"] - 0.1) <= 1.0e-12
    assert abs(scaled["lowfreq_energy_ratio_p50"] - 0.1) <= 1.0e-12


def test_carrier_shell_refinement_is_nested_and_watertight_per_face() -> None:
    _, layout = _geometry_and_layout()
    active = ("box_x_min", "box_y_min", "box_z_min")
    coords, tris, per_face = carrier_shell(layout, active)
    assert tris.shape[1] == 3
    assert {src for src, _ in per_face} == set(active)
    refined_coords, refined_faces = refine_shell(coords, per_face, 1)
    # Nested: every original carrier node keeps its coordinate and index.
    assert np.array_equal(refined_coords[: len(coords)], coords)
    total = 0
    for (src, before), (src_r, after) in zip(per_face, refined_faces, strict=True):
        assert src == src_r
        assert len(after) == 4 * len(before)
        total += len(after)
        # Faces stay planar: refined nodes lie on the same coordinate plane as the parent face.
        axis = {"box_x_min": 0, "box_y_min": 1, "box_z_min": 2}[src]
        assert np.abs(refined_coords[np.unique(after)][:, axis]).max() <= 1.0e-12
    # New nodes are edge midpoints (dyadic): coordinates are multiples of H/2 = 1/32.
    new = refined_coords[len(coords):]
    assert np.abs(new * 32.0 - np.round(new * 32.0)).max() <= 1.0e-12
    # The three port faces meeting at the origin corner form an open shell: each face is 2-manifold inside,
    # boundary edges (shell rim) have one facet.
    report = watertight_report(np.vstack([f for _, f in refined_faces]))
    assert report["edges"] > 0
    assert report["non_two_manifold_edges"] > 0
