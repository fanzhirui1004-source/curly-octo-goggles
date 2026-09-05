"""Root-cause fixes of the sliver artefact (2026-09-05): cap needle repair and boundary-locked sliver perturbation."""
from __future__ import annotations

import numpy as np

from pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1 import sheet_solid_surface as sss


def test_repair_cap_needles_removes_a_steiner_apex_beside_a_chain_edge() -> None:
    # a cap region: chain edge A-B on the bottom, Steiner apex P a hair above it, a fan of two triangles to W1, W2
    A, B, W1, W2 = (0.0, 0.0), (0.02, 0.0), (0.0, 0.015), (0.02, 0.015)
    P = (0.011, 3e-7)                                              # 3e-7 off the chain edge: a needle of ~0.002 degrees
    pts = np.asarray([A, B, W1, W2, P], dtype=np.float64)          # inputs first, the Steiner point last
    tris = np.asarray([[0, 1, 4], [0, 4, 2], [4, 1, 3], [4, 3, 2]], dtype=np.int64)
    before = sss._tri_min_angle_2d(pts, tris).min()
    assert before < 0.01
    out, rep = sss._repair_cap_needles(pts, tris, n_input=4)
    assert rep["cap_needle_apex_collapsed"] + rep["cap_needle_apex_moved"] >= 1
    assert rep["cap_needles_remaining"] == 0 and rep["cap_min_angle_degrees"] > 2.0
    # inputs untouched, orientation kept, region still covered (same total area)
    assert np.allclose(pts[:4], [A, B, W1, W2])
    a2 = lambda T: (pts[T[:, 1], 0] - pts[T[:, 0], 0]) * (pts[T[:, 2], 1] - pts[T[:, 0], 1]) - (pts[T[:, 1], 1] - pts[T[:, 0], 1]) * (pts[T[:, 2], 0] - pts[T[:, 0], 0])
    assert np.all(a2(out) > 0) and abs(a2(out).sum() - a2(tris).sum()) < 1e-12


def test_repair_cap_needles_never_touches_an_input_apex() -> None:
    pts = np.asarray([(0.0, 0.0), (0.02, 0.0), (0.011, 3e-7), (0.0, 0.015), (0.02, 0.015)])
    tris = np.asarray([[0, 1, 2], [0, 2, 3], [2, 1, 4], [2, 4, 3]])
    out, rep = sss._repair_cap_needles(pts, tris, n_input=5)     # the apex is an input vertex (a chain vertex)
    assert rep["cap_needle_apex_collapsed"] == 0 and rep["cap_needle_apex_moved"] == 0 and len(out) == 4


def test_smooth_slivers_lifts_an_interior_apex_off_a_cap_and_keeps_the_surface() -> None:
    # a cube [0,1]^3 corner block: cap triangle on z=0 with an interior apex 1e-6 above it, closed by surface vertices
    V = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],          # z = 0 cap
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],          # top
                    [0.4, 0.4, 1e-6]], dtype=np.float64)                # interior apex, a sliver over (0,1,2)
    # tets: the sliver plus a fan closing the volume around the apex (all other vertices are on the surface)
    T = np.asarray([[0, 1, 2, 8], [1, 3, 2, 8], [0, 2, 6, 8], [0, 6, 4, 8], [0, 4, 5, 8], [0, 5, 1, 8],
                    [1, 5, 7, 8], [1, 7, 3, 8], [3, 7, 6, 8], [3, 6, 2, 8], [4, 6, 7, 8], [4, 7, 5, 8]], dtype=np.int64)
    P = V[T]; vol = np.einsum("ij,ij->i", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))
    T[vol < 0, 1], T[vol < 0, 2] = T[vol < 0, 2].copy(), T[vol < 0, 1].copy()
    d0 = sss._tet_min_dihedral(V[T]).min(); assert d0 < 0.01
    V2, rep = sss.smooth_slivers(V, T)
    assert rep["inverted"] == 0 and rep["vertices_moved"] == 1
    assert rep["after"]["min_dihedral_degrees"] > 5.0 and rep["after"]["min_dihedral_degrees"] > d0
    assert np.array_equal(V2[:8], V[:8])                                # every surface vertex exactly where it was
    P2 = V2[T]; assert abs(np.abs(np.einsum("ij,ij->i", P2[:, 1] - P2[:, 0], np.cross(P2[:, 2] - P2[:, 0], P2[:, 3] - P2[:, 0]))).sum() / 6 - 1.0) < 1e-12
