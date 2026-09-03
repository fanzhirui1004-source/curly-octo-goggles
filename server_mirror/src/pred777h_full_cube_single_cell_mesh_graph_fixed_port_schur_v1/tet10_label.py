"""Tet10 fixed-port label helpers and carrier-weighted norms.

The Tet4 :class:`MeshArtifact` is promoted to straight-sided Tet10 (one global
midpoint per edge).  Physical-port midpoints inherit the average of their
endpoint carrier rows, so the fixed carrier is unchanged: ``u_fine = P q`` on
every actual Tet10 port node.  Carrier mass / Laplace-Beltrami operators are
assembled on the deterministic trace layout so that Schur operators can be
compared in a mass-whitened norm and on a long-wavelength subspace, instead of
the raw Frobenius norm that is dominated by unresolvable short-wavelength port
boundary layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, kron, vstack

from .fixed_port_adapter import FixedPortMap
from .frozen_backend import activate_frozen_backend
from .mesh_artifact import MeshArtifact
from .teacher_contract_calibration import build_fixed_carrier_constraint, chunked_fixed_carrier_schur


activate_frozen_backend()

from cctpms.fem.assembly import assemble_global_stiffness  # noqa: E402
from cctpms.fem.tet10 import assemble_global_tet10_stiffness  # noqa: E402
from cctpms.port.tet10_nodal_schur import TET10_EDGE_LOCAL_NODES  # noqa: E402


@dataclass(frozen=True)
class Tet10Promotion:
    node_coordinates: np.ndarray
    elements: np.ndarray
    port_node_indices: tuple[int, ...]
    scalar_prolongation: csr_matrix
    parent_node_count: int
    port_midpoint_count: int


def promote_mesh_artifact_to_tet10(mesh: MeshArtifact, port: FixedPortMap) -> Tet10Promotion:
    nodes = np.asarray(mesh.mesh_nodes, dtype=np.float64)
    tets = np.asarray(mesh.tet_connectivity[:, :4], dtype=np.int64)
    edges: dict[tuple[int, int], int] = {}

    def edge_id(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in edges:
            edges[key] = len(edges)
        return edges[key]

    elements = np.empty((len(tets), 10), dtype=np.int64)
    elements[:, :4] = tets
    for row, tet in enumerate(tets):
        for slot, (first, second) in enumerate(TET10_EDGE_LOCAL_NODES):
            elements[row, 4 + slot] = len(nodes) + edge_id(int(tet[first]), int(tet[second]))
    midpoints = np.zeros((len(edges), 3), dtype=np.float64)
    for (a, b), index in edges.items():
        midpoints[index] = 0.5 * (nodes[a] + nodes[b])
    port_ids = tuple(int(v) for v in port.fine_port_node_ids)
    position = {node: i for i, node in enumerate(port_ids)}
    scalar4 = port.scalar_fine_to_carrier
    port_mid: dict[int, tuple[int, int]] = {}
    for facet, membership in zip(np.asarray(mesh.boundary_facets), mesh.boundary_patch_membership, strict=True):
        if membership[0] == "tpms_free":
            continue
        a, b, c = (int(v) for v in facet[:3])
        for u, v in ((a, b), (b, c), (a, c)):
            key = (u, v) if u < v else (v, u)
            port_mid[len(nodes) + edges[key]] = key
    mid_nodes = sorted(port_mid)
    rows = [scalar4.getrow(position[n]) for n in port_ids]
    for m in mid_nodes:
        a, b = port_mid[m]
        rows.append(0.5 * (scalar4.getrow(position[a]) + scalar4.getrow(position[b])))
    scalar10 = vstack(rows, format="csr")
    row_sums = np.asarray(scalar10.sum(axis=1)).ravel()
    if float(np.max(np.abs(row_sums - 1.0))) > 1.0e-12:
        raise ValueError("Tet10 carrier lift loses partition of unity")
    return Tet10Promotion(
        node_coordinates=np.vstack([nodes, midpoints]),
        elements=elements,
        port_node_indices=tuple(list(port_ids) + mid_nodes),
        scalar_prolongation=scalar10,
        parent_node_count=len(nodes),
        port_midpoint_count=len(mid_nodes),
    )


def pardiso_fixed_port_schur(stiffness: csr_matrix, node_count: int, port_node_indices: tuple[int, ...], vector_prolongation: csr_matrix,
                             q: int, column_chunk_size: int) -> np.ndarray:
    """Dense fixed-carrier Schur with the internal block factorized by MKL Pardiso (large internal systems)."""

    import pypardiso  # optional dependency, only needed above the SuperLU memory range

    constraint, _ = build_fixed_carrier_constraint(node_count=node_count, port_node_indices=port_node_indices, vector_prolongation=vector_prolongation)
    Kc = (constraint.T @ stiffness @ constraint).tocsr(); Kc = (0.5 * (Kc + Kc.T)).tocsr()
    Kqq = Kc[:q, :q].toarray(); Kqi = Kc[:q, q:].tocsr(); Kii = Kc[q:, q:].tocsr()
    solver = pypardiso.PyPardisoSolver(); solver.set_iparm(1, 1); solver.set_iparm(2, 3); solver.factorize(Kii)
    S = Kqq.copy(); B = Kqi.T.tocsc()
    for c0 in range(0, q, column_chunk_size):
        S[:, c0:c0 + column_chunk_size] -= Kqi @ solver.solve(Kii, B[:, c0:c0 + column_chunk_size].toarray())
    solver.free_memory(everything=True)
    return S


def dense_fixed_port_schur(mesh: MeshArtifact, port: FixedPortMap, *, element: str, workers: int = 8, column_chunk_size: int = 128,
                           pardiso_above: int = 1_000_000) -> tuple[np.ndarray, dict[str, float]]:
    """Exact dense fixed-carrier Schur (q x q) with Tet4 or Tet10 displacement.

    The internal block is factorized by SuperLU (chunked_fixed_carrier_schur) up to ``pardiso_above`` internal dof and by
    MKL Pardiso above it (SuperLU runs out of memory near 2M dof).
    """

    timing: dict[str, float] = {}
    started = time.perf_counter()
    if element.upper() == "TET4":
        nodes = np.asarray(mesh.mesh_nodes, dtype=np.float64)
        stiffness = assemble_global_stiffness(nodes, mesh.tet_connectivity[:, :4], E=1.0, nu=0.3).tocsr()
        node_count = len(nodes)
        port_nodes = tuple(int(v) for v in port.fine_port_node_ids)
        vector = port.fine_to_carrier
    elif element.upper() == "TET10":
        promo = promote_mesh_artifact_to_tet10(mesh, port)
        timing["tet10_promotion"] = time.perf_counter() - started
        stiffness = assemble_global_tet10_stiffness(promo.node_coordinates, promo.elements, E=1.0, nu=0.3, workers=int(workers)).tocsr()
        node_count = len(promo.node_coordinates)
        port_nodes = promo.port_node_indices
        vector = kron(promo.scalar_prolongation, eye(3, format="csr"), format="csr")
    else:
        raise ValueError("element must be TET4 or TET10")
    timing["assembly"] = time.perf_counter() - started
    q = 3 * len(port.active_global_carrier_ids)
    internal_dof = 3 * (node_count - len(port_nodes))
    if internal_dof > int(pardiso_above):
        schur = pardiso_fixed_port_schur(stiffness, node_count, tuple(int(v) for v in port_nodes), vector, q, column_chunk_size)
        timing["solver"] = 2.0  # 2 = mkl_pardiso
    else:
        result = chunked_fixed_carrier_schur(
            fine_stiffness=stiffness, node_count=node_count, port_node_indices=port_nodes,
            vector_prolongation=vector, carrier_coordinates=port.carrier_coordinates, column_chunk_size=column_chunk_size,
        )
        schur = np.asarray(result.schur, dtype=np.float64)
        timing["solver"] = 1.0  # 1 = superlu
    timing["schur_total"] = time.perf_counter() - started
    timing["fine_dof"] = float(3 * node_count)
    timing["internal_dof"] = float(internal_dof)
    return 0.5 * (schur + schur.T), timing


def carrier_mass_and_laplacian(layout, port: FixedPortMap) -> tuple[np.ndarray, np.ndarray]:
    """Scalar P1 mass and Laplace-Beltrami matrices on the active carrier surface complex."""

    n = len(port.active_global_carrier_ids)
    compact = dict(port.global_to_compact)
    M = np.zeros((n, n)); Lp = np.zeros((n, n))
    for trace in layout.local_traces:
        if str(trace.source_id) not in port.active_global_port_ids:
            continue
        # local trace node -> global scalar column (one nonzero per row) -> compact active index
        l2g = layout.local_to_global_scalar[str(trace.source_id)].tocsr()
        ids = []
        for local in range(l2g.shape[0]):
            cols = l2g.indices[l2g.indptr[local]:l2g.indptr[local + 1]]
            if len(cols) != 1:
                raise ValueError("local trace node does not map to exactly one global carrier node")
            ids.append(compact[str(layout.global_scalar_node_keys[int(cols[0])])])
        X = np.asarray(trace.node_coordinates, dtype=np.float64)
        for tri in np.asarray(trace.triangles, dtype=np.int64):
            p = X[tri]; e1 = p[1] - p[0]; e2 = p[2] - p[0]
            normal = np.cross(e1, e2); area = 0.5 * float(np.linalg.norm(normal))
            if area <= 0.0:
                continue
            g = [ids[int(t)] for t in tri]
            # P1 gradients on the triangle plane
            nrm = normal / (2.0 * area)
            grads = [np.cross(nrm, p[(k + 2) % 3] - p[(k + 1) % 3]) / (2.0 * area) for k in range(3)]
            for a in range(3):
                for b in range(3):
                    M[g[a], g[b]] += area / 12.0 * (2.0 if a == b else 1.0)
                    Lp[g[a], g[b]] += area * float(grads[a] @ grads[b])
    return M, Lp


@dataclass(frozen=True)
class CarrierNorms:
    mass_scalar: np.ndarray
    laplacian_scalar: np.ndarray
    whitener_vector: np.ndarray          # M^{-1/2} kron I3
    lowfreq_vector: np.ndarray           # M-orthonormal low-frequency modes kron I3
    lowfreq_eigenvalues: np.ndarray
    wavelength_cutoff: float


def build_carrier_norms(layout, port: FixedPortMap, *, wavelength_cutoff: float) -> CarrierNorms:
    M, Lp = carrier_mass_and_laplacian(layout, port)
    w, V = np.linalg.eigh(M)
    if w.min() <= 0:
        raise ValueError("carrier mass matrix is not positive definite")
    Mih = V @ np.diag(w ** -0.5) @ V.T
    mu, U = np.linalg.eigh(Mih @ Lp @ Mih)        # generalized: Lp u = mu M u, u = Mih U
    kmax = 2.0 * np.pi / wavelength_cutoff
    keep = mu <= kmax ** 2
    U = Mih @ U[:, keep]                           # M-orthonormal modes
    I3 = np.eye(3)
    return CarrierNorms(M, Lp, np.kron(Mih, I3), np.kron(U, I3), mu[keep], wavelength_cutoff)


def compare_operators(S_a: np.ndarray, S_b: np.ndarray, norms: CarrierNorms) -> dict[str, float]:
    """Distances of S_a from S_b (reference) in raw, mass-whitened and low-frequency norms."""

    W = norms.whitener_vector; V = norms.lowfreq_vector
    D = S_a - S_b
    out = {"raw_frobenius": float(np.linalg.norm(D) / np.linalg.norm(S_b))}
    Wb = W @ S_b @ W; Wd = W @ D @ W
    out["whitened_frobenius"] = float(np.linalg.norm(Wd) / np.linalg.norm(Wb))
    Vb = V.T @ S_b @ V; Vd = V.T @ D @ V
    out["lowfreq_frobenius"] = float(np.linalg.norm(Vd) / np.linalg.norm(Vb))
    lam, Q = np.linalg.eigh(Vb); keep = lam > 1e-9 * lam[-1]
    ratios = np.einsum("ij,ij->j", Q[:, keep], (V.T @ S_a @ V) @ Q[:, keep]) / lam[keep] - 1.0
    out["lowfreq_energy_ratio_p50"] = float(np.median(ratios))
    out["lowfreq_energy_ratio_p95abs"] = float(np.quantile(np.abs(ratios), 0.95))
    out["lowfreq_energy_ratio_max_abs"] = float(np.max(np.abs(ratios)))
    out["lowfreq_mode_count"] = int(V.shape[1])
    return out


__all__ = [
    "CarrierNorms", "Tet10Promotion", "build_carrier_norms", "carrier_mass_and_laplacian",
    "compare_operators", "dense_fixed_port_schur", "pardiso_fixed_port_schur", "promote_mesh_artifact_to_tet10",
]
