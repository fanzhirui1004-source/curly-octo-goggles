"""Frozen constants of the sheet-TPMS true-geometry label route (no port collars).

This is a separate contract from ``contract.FullCubeContract`` (the V1 collar route, still frozen for its own
artifacts).  Every production receipt carries ``SheetSolidContract().as_dict()``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SheetSolidContract:
    schema: str = "pred777h_sheet_solid_label_contract_v1"
    material: str = "abs(phi) <= tau(x), phi = sum cos(2 pi x_i) (Schwarz P), tau trilinear from the global P1 thickness field"
    domain: str = "unit cube clipped by the world cut plane; the cut surface is a port (cut_0 carrier trace, skin coupling), like the box faces"
    port_collar_enabled: bool = False
    carrier_grid_per_face: int = 32                      # P1 carrier 1/32 on every box face (decision of record 2026-09-03)
    carrier_triangulation: str = "union jack: each carrier square is split by the diagonal joining its two corners of even lattice coordinate sum (exactly invariant under the 48 cube symmetries for an even carrier grid)"
    cut_carrier_merge: str = "cut cells: grid nodes within 1/3 carrier spacing of the cut have their crossings (and the node, on a box face) merged to one canonical point on the plane (no carrier feature below 1/3 h = 0.0104, i.e. below half the sheet remesh size); every cut-face and clipped polygon is fanned by an isometry-invariant rule (lattice parity square, angle-list anchor, parity quad, exact centroid); residual strips of a plane nearly containing a cell edge or corner are reported, never refused (harmless: see cut_carrier)"
    cap_feature_size: str = "cap PSLG feature-size invariant: a sheet chain vertex within 0.15 remesh size of a carrier node or carrier edge is moved onto it and a chain/carrier-edge crossing within that distance of the carrier node is snapped to it (carrier nodes never move); a nonmanifold volume mesh fails closed"
    carrier_active_rule: str = "geometric superset: vertices of carrier triangles the band reaches within level tolerance 0.05"
    empty_volume: float = 1.0e-6                        # material volume below which a cell carries no label (status EMPTY)
    floating_fragments: str = "connected components touching no port face are removed"
    resolution_presets: tuple = (("reference", 96, 0.015, 0.03), ("production", 64, 0.02, 0.04), ("fast", 48, 0.03, 0.05))
    thickness_scaling: str = "t_min = 0.184 tau_min; remesh size <= 0.6 t_min; marching-cubes grid >= 3 cells per t_min"
    surface_gates: str = "watertight, zero verified self-intersections, zero cap-classification conflicts, boundary on planes"
    element: str = "TET10 (straight-sided promotion of the Gmsh Tet4 mesh), E = 1, nu = 0.3"
    volume_mesher: str = "Gmsh HXT (Algorithm3D 10) at a pinned generation thread count of 1 (byte-identical meshes), Netgen optimisation; a nonmanifold or crashed mesh fails closed, no retry (legacy Delaunay overlapped tetrahedra on well-shaped cap triangles and segfaulted on 2 of 60 cut cells)"
    cap_needle_repair: str = "cap triangulation needle repair (2026-09-05): a Steiner apex that Triangle (q20, no Steiner points on the chains) leaves a hair beside a chain edge is collapsed onto an endpoint of that edge or moved into its fan before the flip pass; input vertices never move; the receipt reports remaining needles below 2 degrees and the cap minimum angle (every sub-degree tetrahedron of the 2026-09-05 production sample stood on such a triangle)"
    sliver_perturbation: str = "NOT applied (opt-in diagnostic only): moving interior vertices to maximise the minimum dihedral angle trades slivers on sound cap triangles, which are harmless to the port operator, for flat tetrahedra on the port face, which are not (pop_uncut_0658 production: lambda_max / reference 1.13 with the perturbation, 1.01 without); the needle repair of the cap is the remedy of record"
    operator_top_mode: str = "diagnostic of record: the share of lambda_max's eigenvector on its four heaviest carrier nodes (one carrier square); a sliver on a port face puts 0.97 to 1.00 of it there at up to 11x the converged operator scale, a physical port stiffness spreads over 10 or more nodes; lambda_max is therefore NOT the operator scale (normalise with the carrier mass matrix)"
    schur_solver: str = "MKL Pardiso on the internal block, supported carrier columns only"
    operator_gates: str = "rigid-body residual <= 1e-10, min eigenvalue >= -1e-8 lambda_max, mesh port support inside the geometric active set"
    density_law_reference: str = "Hao 2023 Table 2.2: C = 1.755 rho (tau = 0.4 -> rho = 0.228; measured 0.2286)"

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["resolution_presets"] = {name: {"n_per_unit": n, "remesh_size": h, "size_max": s} for name, n, h, s in self.resolution_presets}
        return payload


__all__ = ["SheetSolidContract"]
