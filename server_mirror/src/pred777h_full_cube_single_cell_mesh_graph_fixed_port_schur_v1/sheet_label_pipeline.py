"""Shared label stages of the sheet route: Gmsh mesh -> fixed port on the geometric active set -> Tet10 Schur.

Used by the production command (produce_sheet_label.py) and the prototype driver, so the two cannot drift apart.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path

import numpy as np

from . import carrier_triangulation as ct
from . import cut_carrier as cc
from . import cgal_mesh3_adapter as adapter
from . import fixed_port_adapter as fpa
from . import sheet_solid_surface as sss
from . import tet10_label as t10

CARRIER_TRIANGULATION = "unionjack"
CUT_CARRIER_MERGE = True     # cut_carrier.merge_cut_carrier on every cut cell (decision of record 2026-09-04)

MESHER_PARAMETERS = {"domain_contract": "SHEET_SOLID_NO_COLLAR", "optimization_enabled": False, "semantic_plane_tolerance": 1e-10}


def build_carrier_layout(charts, spec, *, carrier_n: int = 32, triangulation: str = CARRIER_TRIANGULATION, cut_merge: bool | None = None):
    """The P1 carrier layout of one cell, with the route's carrier triangulation.

    Every entry point of the sheet route builds its layout here so the label definition cannot drift between the
    production command, the prototype driver and the checks.  triangulation="unionjack" replaces the vendored fan
    (which picks each square's diagonal by a hash of the node coordinates, hence not equivariant) by the parity
    diagonal, which is exactly invariant under the 48 cube symmetries; "frozen" keeps the vendored rule.  cut_merge
    (default CUT_CARRIER_MERGE) applies the cut-face merge and symmetric fans of cut_carrier to cut cells."""
    from cctpms.port.global_cell_patch_trace_layout import build_global_cell_patch_p1_trace_layout

    layout = build_global_cell_patch_p1_trace_layout(
        charts, background_lower=(0.0, 0.0, 0.0), background_upper=(1.0, 1.0, 1.0), background_shape=(carrier_n,) * 3,
        world_identity_rotation=spec.placement.rotation, world_identity_translation=spec.placement.translation)
    if triangulation == "unionjack":
        layout = ct.unionjack_layout(layout, carrier_n=carrier_n)
    elif triangulation != "frozen":
        raise ValueError(f"unknown carrier triangulation {triangulation!r}")
    # cut cells: merge the near-degenerate crossings and refan every cut-face / clipped polygon by the symmetric rules
    # (cut_carrier); the merge radius is the contract constant CUT_CARRIER_MERGE_EPS = 1/10 carrier spacing
    if CUT_CARRIER_MERGE if cut_merge is None else cut_merge:
        layout, _ = cc.merge_cut_carrier(layout, spec, carrier_n=carrier_n)
    return layout


def write_cut_face_trace(path: Path, layout, spec) -> dict | None:
    """The cell's cut_0 carrier trace in world coordinates: the coupling deliverable of the skin (decision of record
    2026-09-04: the cut face is a port).  Node keys are the global carrier keys (identical in neighbouring cells for
    shared ring nodes), so the union over a block is one conforming triangulation of the cut plane.  Written for
    EMPTY cells too (the plane crosses them; the skin has no hole there).  Returns a summary or None without a cut."""
    traces = {str(t.source_id): t for t in layout.local_traces}
    if "cut_0" not in traces:
        return None
    tr = traces["cut_0"]; R = np.asarray(spec.placement.rotation, dtype=np.float64); T = np.asarray([float(v) for v in spec.placement.translation])
    local = np.asarray(tr.node_coordinates, dtype=np.float64); world = local @ R.T + T
    ring = np.zeros(len(local), dtype=bool)
    for src, other in traces.items():                       # nodes shared with a box face lie on the boundary ring of the cut polygon
        if src != "cut_0":
            ring |= np.isin(np.asarray(tr.global_node_ids), np.asarray(other.global_node_ids))
    chart = tr.chart; n = np.asarray([float(v) for v in chart.plane_normal_exact]); d = float(chart.plane_offset_exact)
    n_world = R @ n; d_world = d + float(n_world @ T)
    np.savez(path, node_keys=np.asarray(tr.global_node_ids), node_xyz_world=world, node_xyz_local=local, triangles=np.asarray(tr.triangles, dtype=np.int64),
             on_box_face_ring=ring, plane_world=np.asarray([*n_world, d_world]), placement_translation=T, placement_rotation=R,
             carrier_triangulation=np.asarray([layout.diagnostics.get("carrier_triangulation", "")]), layout_semantic_signature=np.asarray([layout.semantic_signature]))
    return {"path": str(path), "nodes": int(len(local)), "triangles": int(len(tr.triangles)), "ring_nodes": int(ring.sum())}


@dataclass
class PortStage:
    artifact: object
    port: object
    active: np.ndarray            # geometric active mask over the layout carrier scalar nodes
    check: dict
    summary: dict
    seconds: float


def compile_port(mesh_path: Path, geometry, spec, layout, *, carrier_n: int = 32) -> PortStage:
    """Fixed port of a sheet-solid mesh on the geometric active carrier set (P is never modified).  Raises when the
    mesh port support escapes the geometric active set.  The cut face is a port (its facets are classified as cut_0
    and interpolated by the cut_0 carrier trace), so a skin attached to it couples through the carrier."""
    t0 = time.perf_counter()
    art = adapter.mesh_artifact_from_cgal_medit(mesh_path, geometry=geometry, mesher_version="5.4",       # cut_0 is a port
                                                mesher_parameters=MESHER_PARAMETERS, plane_tolerance=1e-10, source_geometry_hash=spec.spec_hash)
    port = fpa.compile_fixed_port(art, spec, layout)
    active = sss.geometric_active_mask(geometry, layout, port, carrier_n=carrier_n)
    check = sss.verify_active_superset(port, active)
    if check["support_outside_active"]:
        raise RuntimeError(f"mesh port support escapes the geometric active set: {check}")
    cert = port.certificates
    summary = {"carrier_scalar_nodes": int(len(active)), "active_carrier_nodes": int(active.sum()), "q_active": int(3 * active.sum()),
               "fine_port_nodes": int(port.scalar_fine_to_carrier.shape[0]), "active_set_check": check, "active_ports": list(port.active_global_port_ids),
               "partition_of_unity_error": float(cert["maximum_partition_of_unity_error"]),
               "affine_reproduction_error": float(cert["maximum_affine_reproduction_error"]),
               "deterministic_replay_sha256": cert["deterministic_replay_sha256"]}
    return PortStage(artifact=art, port=port, active=active, check=check, summary=summary, seconds=time.perf_counter() - t0)


@dataclass
class SchurStage:
    schur_active: np.ndarray      # q_active x q_active, symmetric
    rigid_active: np.ndarray      # q_active x 6
    eigenvalues: np.ndarray
    rigid_residual: float
    inactive_block_max: float
    timing: dict
    seconds: float
    top_mode_mass_on_4_nodes: float = 0.0   # share of lambda_max's eigenvector on its four heaviest carrier nodes (1.0 = one carrier square: a sliver)
    top_mode_nodes_above_1pct: int = 0


def schur_label(stage: PortStage, *, workers: int = 8, column_chunk_size: int = 1024, rigid_tolerance: float = 1.0e-10,
                eigenvalue_tolerance: float = 1.0e-8, top_mode_gate: float | None = None) -> SchurStage:
    """Tet10 fixed-port Schur restricted to the active carrier nodes, with the operator gates (rigid-body residual,
    positive semidefiniteness).  Raises on a gate failure."""
    t0 = time.perf_counter()
    S, timing = t10.dense_fixed_port_schur(stage.artifact, stage.port, element="TET10", workers=workers, column_chunk_size=column_chunk_size)
    keep = np.repeat(stage.active, 3)
    S_active = np.ascontiguousarray(S[np.ix_(keep, keep)]); rigid = np.ascontiguousarray(stage.port.rigid_basis[keep])
    rigid_residual = float(np.linalg.norm(S_active @ rigid) / max(np.linalg.norm(S_active), 1e-300))
    eig, vecs = np.linalg.eigh(S_active)
    # localisation of the stiffest mode: a sliver tetrahedron on a port face puts all of it on one carrier square
    # (four nodes) at up to 11x the converged operator scale (2026-09-05); a physical port stiffness is spread out
    part = np.sort(np.linalg.norm(vecs[:, -1].reshape(-1, 3), axis=1) ** 2)[::-1]
    top4 = float(part[:4].sum()); n1 = int((part > 0.01).sum()); del vecs
    inactive = float(np.abs(S[np.ix_(~keep, ~keep)]).max()) if (~keep).any() else 0.0
    if rigid_residual > rigid_tolerance:
        raise RuntimeError(f"rigid-body residual {rigid_residual:.2e} exceeds {rigid_tolerance:g}")
    if eig[0] < -eigenvalue_tolerance * eig[-1]:
        raise RuntimeError(f"Schur operator is not positive semidefinite (min eigenvalue {eig[0]:.3e})")
    if top_mode_gate is not None and top4 > top_mode_gate:
        raise RuntimeError(f"stiffest mode is localised on one carrier square ({top4:.3f} of its mass on 4 nodes): a sliver artefact")
    return SchurStage(schur_active=S_active, rigid_active=rigid, eigenvalues=eig, rigid_residual=rigid_residual, inactive_block_max=inactive,
                      timing=timing, seconds=time.perf_counter() - t0, top_mode_mass_on_4_nodes=top4, top_mode_nodes_above_1pct=n1)


def carrier_norms(layout, port, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P1 mass and Laplace-Beltrami matrices of the ACTIVE carrier complex (the training loss is defined with them)."""
    M, L = t10.carrier_mass_and_laplacian(layout, port)
    idx = np.where(active)[0]
    return M[np.ix_(idx, idx)], L[np.ix_(idx, idx)]


def load_schur(npz_path: Path) -> np.ndarray:
    """Full symmetric float64 operator from a production label (upper-triangle float32 storage, or the older full array)."""
    z = np.load(npz_path)
    if "schur_upper_f32" in z.files:
        q = int(z["q"][0]); S = np.zeros((q, q), dtype=np.float64); iu = np.triu_indices(q)
        S[iu] = z["schur_upper_f32"].astype(np.float64); S = S + np.triu(S, 1).T
        return S
    return np.asarray(z["schur"], dtype=np.float64)


def load_carrier_norms(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Dense P1 mass and Laplace-Beltrami matrices of the active carrier complex from a production label."""
    z = np.load(npz_path)
    if "carrier_mass_coo" in z.files:
        n = int(z["carrier_scalar_count"][0]); out = []
        for name in ("carrier_mass", "carrier_laplacian"):
            rc = z[f"{name}_coo"]; M = np.zeros((n, n)); np.add.at(M, (rc[0], rc[1]), z[f"{name}_values"]); out.append(M)
        return out[0], out[1]
    return np.asarray(z["carrier_mass"]), np.asarray(z["carrier_laplacian"])


__all__ = ["MESHER_PARAMETERS", "PortStage", "SchurStage", "carrier_norms", "compile_port", "load_carrier_norms", "load_schur", "schur_label"]


class MemoryBudget:
    """Cooperative memory budget for concurrent label jobs on one machine.

    Memory, not CPU, bounds how many labels a machine can run at once: the Schur stage of a thick cell peaks above
    30 GB while the surface and port stages need well under 1 GB.  Jobs therefore run freely through the cheap
    stages and reserve their estimated Schur memory here before the expensive one, so a 128 GiB cgroup carries
    ten or more jobs in flight with only as many Schurs as fit.  Reservations are files `res_<pid>` in one
    directory, summed under a directory lock; a reservation whose process is gone counts for nothing, so a killed
    job cannot leak budget.  Estimate: 36 MB per 1000 fine Tet10 dof; the peak RSS of the Schur stage measured on 60
    production labels is 33 MB per 1000 dof (2026-09-04), so the margin is about 10 percent."""

    def __init__(self, directory: Path, budget_gb: float, poll_seconds: float = 3.0):
        self.directory = Path(directory); self.budget_gb = float(budget_gb); self.poll = float(poll_seconds); self.mine = None
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def estimate_gb(tet10_dof: int) -> float:
        return 0.036 * tet10_dof / 1000.0

    def _reserved(self) -> float:
        import os
        total = 0.0
        for f in self.directory.glob("res_*"):
            try:
                pid = int(f.name[4:]); os.kill(pid, 0)
            except (ValueError, ProcessLookupError):
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
                continue
            except PermissionError:
                pass
            try:
                total += float(f.read_text())
            except (ValueError, FileNotFoundError):
                pass
        return total

    def reserve(self, gb: float) -> float:
        """Block until `gb` fits under the budget; returns the seconds waited."""
        import fcntl, os
        gb = float(gb); waited = 0.0; lock = self.directory / "lock"
        while True:
            with open(lock, "w") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    if self._reserved() + gb <= self.budget_gb or (gb > self.budget_gb and self._reserved() == 0.0):
                        self.mine = self.directory / f"res_{os.getpid()}"; self.mine.write_text(f"{gb:.3f}"); return waited
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            time.sleep(self.poll); waited += self.poll

    def release(self) -> None:
        if self.mine is not None:
            try:
                self.mine.unlink()
            except FileNotFoundError:
                pass
            self.mine = None


def cut_contains_a_cell_edge(spec) -> dict | None:
    """The exact chart compiler refuses a world cut plane that CONTAINS one of the cell's box edges: the cut's trace
    on a port face then coincides with an edge of that face ("coincident semantic edges"), and for a plane through
    an edge of the retained side the half-space intersection degenerates.  Measured 2026-09-04: a vertical plane
    a x + b y <= d is refused exactly when d is one of {0, a, b, a+b} in the cell's local frame, that is when the
    plane passes through a vertical edge of the cell, and accepted otherwise.

    This is not a hazard of the mesher but of the geometry: it is decided by exact rational arithmetic before any
    meshing, so it is reported as a status rather than raised as a crash.  A world cut that hits it can be moved by
    one exact rational epsilon GLOBALLY (all cells keep the same plane, so the shared-face traces still agree).

    Returns None when the cut is admissible, else a dict describing the coincidence."""
    from fractions import Fraction

    cut = getattr(spec, "cut_plane_local", None)           # make_cell_spec already carries the world cut in cell coordinates
    if cut is None:
        return None
    a, b, c, d_local = (Fraction(cut.a), Fraction(cut.b), Fraction(cut.c), Fraction(cut.d))
    coefficients = (a, b, c)
    for axis in range(3):                                   # edges along `axis`: the other two coordinates are 0 or 1
        if coefficients[axis] != 0:
            continue                                        # a plane not parallel to this edge direction cannot contain such an edge
        others = [k for k in range(3) if k != axis]
        for u in (Fraction(0), Fraction(1)):
            for v in (Fraction(0), Fraction(1)):
                if coefficients[others[0]] * u + coefficients[others[1]] * v == d_local:
                    return {"reason": "the cut plane contains a cell edge (the exact chart compiler refuses it)",
                            "edge_direction_axis": axis, "edge_coordinates": {f"x{others[0]}": str(u), f"x{others[1]}": str(v)},
                            "plane_local": [str(a), str(b), str(c), str(d_local)],
                            "remedy": "move the WORLD cut plane by one exact rational epsilon (all cells share it, so the shared-face traces still agree)"}
    return None
