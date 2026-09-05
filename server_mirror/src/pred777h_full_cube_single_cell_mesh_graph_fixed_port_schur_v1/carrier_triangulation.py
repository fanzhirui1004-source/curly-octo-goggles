"""Symmetric (Union Jack) retriangulation of the P1 carrier trace.

The frozen carrier layout fans every carrier square from the corner with the smallest node id, and that id is a
hash of the exact coordinates: the diagonal of each square is effectively drawn at random.  The P1 space on the
port faces therefore depends on the cell's orientation, and a cell and its image under a cube symmetry get
different port constraints even when the geometry is exactly symmetric (rotation check 2026-09-04: about 3 % of
the label in the carrier low-frequency norm, independent of mesh resolution).

This module rebuilds the triangulation of the full carrier squares with a rule that IS invariant under the 48 cube
symmetries when the carrier resolution n is even: join the two corners whose lattice coordinate sum is even.  Each
square has exactly two such corners and they are diagonally opposite (the corner sums are s, s+1, s+1, s+2), and
every cube symmetry maps a lattice coordinate c to c or n - c, which preserves each coordinate's parity for even n,
so the even-sum pair of a square maps to the even-sum pair of its image.  The result is the Union Jack pattern
(diagonals alternating with the parity of the square).

Only complete lattice squares are rebuilt.  Patches clipped by the cut plane keep the frozen fan: their nodes are
off-lattice and no symmetric rule applies to them (the cut plane is not part of the symmetric carrier anyway).
Node identity, node count and the local-to-global maps are untouched, so the label's carrier and its size are
unchanged; only which P1 basis functions sit on the faces changes.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from fractions import Fraction

import numpy as np

__all__ = ["unionjack_triangles", "unionjack_layout", "triangulation_report"]


def _cross2(a, b, c) -> Fraction:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def unionjack_triangles(node_xyz: np.ndarray, triangles: np.ndarray, *, carrier_n: int) -> tuple[np.ndarray, dict]:
    """Retriangulate every complete carrier square of one face; other triangles are kept verbatim.

    node_xyz: (m, 3) coordinates of the face's carrier nodes; triangles: (t, 3) indices into them.
    Returns (triangles, report)."""
    if carrier_n % 2:
        raise ValueError(f"the symmetric carrier rule needs an even carrier resolution, got {carrier_n}")
    X = np.asarray(node_xyz, dtype=np.float64); T = np.asarray(triangles, dtype=np.int64)
    lat = X * carrier_n; on_lattice = np.abs(lat - np.rint(lat)).max(axis=1) <= 1e-9
    key = {tuple(np.rint(lat[i]).astype(int).tolist()): i for i in range(len(X)) if on_lattice[i]}
    axes = [k for k in range(3) if np.ptp(X[:, k]) > 1e-12]      # the two in-plane axes of this port face
    if len(axes) != 2:
        return T, {"squares": 0, "retriangulated": 0, "kept": int(len(T)), "reason": "face is not planar in two axes"}
    a, b = axes
    squares: list[tuple[int, int, int, int]] = []
    for lattice, i00 in key.items():
        step_a = list(lattice); step_a[a] += 1; step_b = list(lattice); step_b[b] += 1
        step_ab = list(lattice); step_ab[a] += 1; step_ab[b] += 1
        i10 = key.get(tuple(step_a)); i01 = key.get(tuple(step_b)); i11 = key.get(tuple(step_ab))
        if i10 is None or i01 is None or i11 is None:
            continue
        squares.append((i00, i10, i11, i01))     # corners in cyclic order
    # a triangle belongs to the square whose minimum lattice corner it shares, provided its three nodes are on the
    # lattice and span exactly one lattice step in each in-plane axis (O(triangles), no pairwise search)
    square_of_corner = {tuple(np.rint(lat[sq[0]]).astype(int).tolist()): k for k, sq in enumerate(squares)}
    covered: dict[int, list[int]] = {}
    tri_square = np.full(len(T), -1, dtype=np.int64)
    for t, tri in enumerate(T):
        idx = [int(v) for v in tri]
        if not all(on_lattice[i] for i in idx):
            continue
        L = np.rint(lat[idx]).astype(int)
        if np.ptp(L[:, a]) != 1 or np.ptp(L[:, b]) != 1:
            continue
        k = square_of_corner.get(tuple(L.min(axis=0).tolist()))
        if k is None:
            continue
        if set(idx) <= set(squares[k]):
            tri_square[t] = k; covered.setdefault(k, []).append(t)
    out: list[tuple[int, int, int]] = [tuple(int(v) for v in T[t]) for t in range(len(T)) if tri_square[t] < 0]
    kept = len(out); rebuilt = 0
    for k, sq in enumerate(squares):
        if len(covered.get(k, ())) != 2:         # not a square of this triangulation (clipped, or refined): keep
            continue
        parity = {i: int(np.rint(lat[i]).astype(int).sum()) % 2 for i in sq}
        even = [i for i in sq if parity[i] == 0]
        if len(even) != 2:
            raise ValueError("a carrier square must have exactly two corners of even lattice parity")
        p, q = even; other = [i for i in sq if i not in even]
        out.append((p, q, other[0])); out.append((q, p, other[1])); rebuilt += 1
    return np.asarray(out, dtype=np.int64).reshape(-1, 3), {"squares": len(squares), "retriangulated": rebuilt, "kept": kept}


def unionjack_layout(layout, *, carrier_n: int):
    """A copy of the trace layout whose port faces carry the symmetric (Union Jack) triangulation."""
    traces = []; report: dict[str, dict] = {}
    for trace in layout.local_traces:
        tris, rep = unionjack_triangles(np.asarray(trace.node_coordinates, dtype=np.float64), trace.triangles, carrier_n=carrier_n)
        uv = trace.uv_coordinates_exact
        fixed = []
        for t in tris:                            # keep the frozen orientation convention (positive in the chart uv)
            i, j, k = (int(v) for v in t)
            fixed.append((i, j, k) if _cross2(uv[i], uv[j], uv[k]) > 0 else (i, k, j))
        traces.append(dc_replace(trace, triangles=np.asarray(fixed, dtype=np.int64)))
        report[str(trace.source_id)] = rep
    diagnostics = dict(layout.diagnostics); diagnostics["carrier_triangulation"] = "unionjack_even_lattice_parity"
    diagnostics["carrier_triangulation_report"] = report
    return dc_replace(layout, local_traces=tuple(traces), diagnostics=diagnostics)


def triangulation_report(layout) -> dict:
    return {"mode": layout.diagnostics.get("carrier_triangulation", "frozen_fan_by_node_id"),
            "triangles": {str(t.source_id): int(len(t.triangles)) for t in layout.local_traces}}
