"""Cut-face carrier conditioning: in-plane merge of near-degenerate crossings and a symmetric polygon fan.

The frozen carrier puts a cut-face node wherever the cut plane crosses a line of the 1/n background grid (and nowhere
else), so every cut-face node is keyed by a grid line and the index set is fixed for any plane.  Two things degrade
it.  When the plane passes within a small distance of a grid NODE, the crossings of the lines through that node
nearly coincide; and when the cut line of a box face runs close to a face grid line, the polygons between them are
thin strips.  Both give unbounded aspect ratios (survey 2026-09-04: above 1000 in 44 % of tilted planes, worst 4e5).

This pass, with h = 1/carrier_n and a merge radius eps h:

1. for every grid node N near the cut (interior N: dist(N, plane) < eps h; N on one box face: distance to that
   face's cut line < eps h; N on a cell edge: the plane crosses the edge within eps h of N), relocates N (when it
   is a carrier node) and the crossings of the grid lines through N within sqrt(3) eps h of N to ONE canonical
   point: the projection of N on the plane, the point of the face cut line nearest to N, or the plane/edge
   crossing.  The point lies on the plane and on every box face the merged nodes lie on, so the trace stays
   conforming with the box faces and with the neighbouring cell (the rule is a function of the global plane and
   the global grid only).  A carrier lattice node keeps its key (its lattice slot moves by less than eps h);
   other merged nodes are keyed by the exact world coordinate of the canonical point;
2. rebuilds the triangulation of every polygon (cut face: plane ∩ grid cell; box face: square ∩ retained half
   space) by rules that are invariant under the 48 cube symmetries: a lattice square (corners on lattice slots)
   gets the Union Jack parity diagonal; any other polygon is fanned from the vertex whose fan has the
   lexicographically largest sorted angle list; an exact tie of a quadrilateral is split along the diagonal joining
   its two vertices of even grid parity when they are opposite; any remaining tie (a symmetric polygon, e.g. the
   regular hexagons of a body-diagonal plane) gets a centroid node keyed by the exact centroid.

The vendored generator is not modified; this is a post-processing pass over its layout like the Union Jack pass.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from fractions import Fraction
import math

import numpy as np
from scipy.sparse import csr_matrix

__all__ = ["merge_cut_carrier", "polygon_fan", "cut_carrier_gate", "CUT_CARRIER_MERGE_EPS", "CUT_CARRIER_MIN_GAP"]

# Merge radius in carrier spacings h = 1/carrier_n.  1/3 h = 0.0104 is set by the SURFACE scale, not by the sliver
# bound alone: the cap of a port face is a constrained triangulation of the carrier, so a carrier edge much shorter than
# the sheet remesh size (0.02 to 0.03) is refined by the cap and forces the volume mesh to grade from that size to the
# sheet size.  At eps = 1/10 (edges down to 0.003) Gmsh Delaunay's boundary recovery overlapped tetrahedra on 11 of 55
# cut cells; at 1/3 (edges >= 0.0104, i.e. >= 0.5 remesh) the same cells mesh cleanly, the carrier aspect ratio bound
# drops to ~3 and the node count falls (2026-09-05).  1/3 is also the Labelle-Shewchuk snapping range.
CUT_CARRIER_MERGE_EPS = Fraction(1, 3)
CUT_CARRIER_MIN_GAP = Fraction(1, 50)        # diagnostic threshold: a residual strip narrower than this is reported (never refused)
_TIE = 1e-9                                  # relative tie tolerance of the angle-list anchor rule


def cut_carrier_gate(report: dict | None, *, min_gap: Fraction = CUT_CARRIER_MIN_GAP) -> dict | None:
    """DIAGNOSTIC (not a refusal): the residual families no in-face move can fix, the plane within eps h of a cell
    corner or nearly containing a cell edge (a face cut line within eps h of the edge while the plane/edge crossing is
    not).  Their strips have aspect ratio ~ h / gap.  They are harmless to the operator: a sweep of the strip family
    to gap 0.002 h (pop_cut_0003, 2026-09-05) keeps every guard and the operator converges smoothly (energy steps
    5e-3 -> 1.7e-4 as the gap shrinks), and the corner family lies in void (identical operator at every gap).  Their
    long thin carrier triangles are single constrained cap triangles (no refinement, no size gradation) which the
    volume mesher handles; the census carried strips of aspect 2.4e4 without a failure.  Kept for the receipt."""
    if not report or not report.get("applied"):
        return None
    gap = report.get("min_unfixed_gap_over_h")
    if gap is None or gap >= float(min_gap):
        return None
    return {"reason": f"near-degenerate cut: an unfixable carrier strip of width {gap:.2e} h (< {float(min_gap):g} h) at a cell corner or edge "
                      f"({', '.join(f'{k}: {v}' for k, v in report.get('skipped', {}).items())})",
            "remedy": "translate the world cut plane by one exact rational epsilon (the whole block shares the plane)",
            "min_unfixed_gap_over_h": gap, "skipped": report.get("skipped", {})}


def _dot(a, b) -> Fraction:
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _cross2(a, b, c) -> Fraction:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _dist2(p, q) -> Fraction:
    return sum(((x - y) ** 2 for x, y in zip(p, q)), Fraction(0))


def _canonical_point(N, faces, plane, h, eps):
    """(canonical point, kind) for grid node N, or (None, reason).  Exact.  reason "far" = nothing to do."""
    n, d = plane
    delta = _dot(n, N) - d; r2 = eps * eps * h * h
    near_plane = delta * delta < r2 * _dot(n, n)
    if len(faces) == 0:
        if not near_plane:
            return None, "far"
        s = -delta / _dot(n, n)
        return tuple(N[k] + s * n[k] for k in range(3)), "interior"
    if len(faces) == 1:
        k = faces[0]
        m = tuple(n[j] if j != k else Fraction(0) for j in range(3))       # the cut line of the face x_k = N_k
        if _dot(m, m) == 0:
            return None, ("plane_near_parallel_face" if near_plane else "far")
        if delta * delta >= r2 * _dot(m, m):                                 # |delta| / |m| = distance to the line
            return None, "far"
        s = -delta / _dot(m, m)
        return tuple(N[j] + s * m[j] for j in range(3)), "face"
    if len(faces) == 2:
        k = [j for j in range(3) if j not in faces][0]                      # the cell edge through N runs along axis k
        if n[k] == 0:
            return None, ("plane_near_parallel_edge" if near_plane else "far")
        t = -delta / n[k]
        if t * t < r2:
            P = list(N); P[k] = N[k] + t
            return tuple(P), "edge"
        for f in faces:                                                      # a face cut line within eps h of N but the
            m = tuple(n[j] if j != f else Fraction(0) for j in range(3))     # plane/edge crossing is not: a strip along
            if _dot(m, m) and delta * delta < r2 * _dot(m, m):               # the edge that no in-face move can remove
                return None, "edge_strip_unfixed"
        return None, "far"
    return None, ("corner" if near_plane else "far")


def _angles(P, tri) -> list[float]:
    A, B, C = (P[i] for i in tri); out = []
    for X, Y, Z in ((A, B, C), (B, C, A), (C, A, B)):
        u = Y - X; v = Z - X; nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        out.append(0.0 if nu == 0 or nv == 0 else math.acos(max(-1.0, min(1.0, float(u @ v) / (nu * nv)))))
    return out


def polygon_fan(P: np.ndarray, order: list[int]) -> tuple[list[tuple[int, int, int]] | None, float]:
    """Fan of the convex polygon `order` (indices into P, cyclic) from the anchor with the lexicographically largest
    sorted angle list (an isometry invariant).  Returns (triangles, min angle), or (None, min angle) when two
    DIFFERENT fans tie (a symmetric polygon)."""
    m = len(order)
    if m == 3:
        return [tuple(order)], min(_angles(P, tuple(order)))
    seen: dict[frozenset, tuple] = {}
    for a in range(m):
        tris = [(order[a], order[(a + j) % m], order[(a + j + 1) % m]) for j in range(1, m - 1)]
        key = frozenset(tuple(sorted(t)) for t in tris)
        if key not in seen:
            seen[key] = (sorted(x for t in tris for x in _angles(P, t)), tris)
    cands = sorted(seen.values(), key=lambda c: c[0], reverse=True)
    if len(cands) > 1 and all(abs(x - y) <= _TIE * math.pi for x, y in zip(cands[0][0], cands[1][0])):
        return None, cands[0][0][0]
    return cands[0][1], cands[0][0][0]


def _cyclic_order(uv: np.ndarray, nodes: list[int]) -> list[int]:
    c = uv[nodes].mean(axis=0)
    ang = np.arctan2(uv[nodes, 1] - c[1], uv[nodes, 0] - c[0])
    return [nodes[i] for i in np.argsort(ang, kind="stable")]


def merge_cut_carrier(layout, spec, *, carrier_n: int, eps: Fraction = CUT_CARRIER_MERGE_EPS):
    """A copy of the layout with the cut-face merge and the symmetric polygon fans applied; (layout, report)."""
    if not any(str(t.source_id) == "cut_0" for t in layout.local_traces):
        return layout, {"applied": False, "reason": "no cut face"}
    if (layout.diagnostics.get("cut_carrier_merge") or {}).get("applied"):
        raise ValueError("merge_cut_carrier applied twice: the pass is not idempotent (build_carrier_layout already merges)")
    # the vendored generator is importable once the backend has registered it (same lazy idiom as the pipeline)
    from cctpms.port.global_cell_patch_trace_layout import _world_to_uv_exact
    from cctpms.port.global_planar_surface_p1 import _canonical_hash, _point_payload, _world_identity_point
    eps = Fraction(eps); h = Fraction(1, carrier_n); slot_tol = eps * h * Fraction(101, 100)
    cut = layout.local_trace("cut_0"); chart = cut.chart
    n = tuple(Fraction(v) for v in chart.plane_normal_exact); d = Fraction(chart.plane_offset_exact); plane = (n, d)
    transform = (tuple(tuple(int(v) for v in row) for row in spec.placement.rotation), tuple(Fraction(v) for v in spec.placement.translation))

    # ---- exact coordinates and trace membership of every global node ----
    G = len(layout.global_scalar_node_keys); keys = list(layout.global_scalar_node_keys)
    exact: list = [None] * G; l2g: dict[str, np.ndarray] = {}
    for tr in layout.local_traces:
        src = str(tr.source_id); M = layout.local_to_global_scalar[src].tocsr()
        loc2glob = np.asarray([int(M.indices[M.indptr[i]]) for i in range(M.shape[0])], dtype=np.int64); l2g[src] = loc2glob
        for i, (u, v) in enumerate(tr.uv_coordinates_exact):
            g = int(loc2glob[i])
            if exact[g] is None:
                exact[g] = tuple(Fraction(x) for x in tr.chart.map_uv_exact(u, v))
    cut_nodes = set(int(g) for g in l2g["cut_0"])
    exact_index = {p: g for g, p in enumerate(exact)}

    def lattice_of(p):
        lat = [x / h for x in p]
        return tuple(Fraction(round(x)) * h for x in lat)

    def parity_of_point(p) -> int | None:
        ints = [int(x / h) for x in p if (x / h).denominator == 1]
        return sum(ints) % 2 if len(ints) >= 2 else None

    # ---- clusters: grid nodes near the cut ----
    candidates: set = set()
    for g in cut_nodes:
        candidates.add(lattice_of(exact[g]))
    for tr in layout.local_traces:                                  # box-face lattice nodes near the face cut line
        if str(tr.source_id) == "cut_0":
            continue
        for g in l2g[str(tr.source_id)]:
            p = exact[int(g)]
            if all((x / h).denominator == 1 for x in p):
                candidates.add(p)
    candidates = {N for N in candidates if all(0 <= x <= 1 for x in N)}
    target: dict[int, int] = {}; new_exact: list = []; new_keys: list[str] = []; parity: dict[int, int] = {}
    moved_key: dict[int, tuple] = {}                                    # kept-key nodes whose coordinate changes
    report = {"clusters": 0, "merged_nodes": 0, "lattice_nodes_moved": 0, "skipped": {}, "eps": str(eps)}

    def node_for(point, keep: int | None = None):
        if keep is not None:
            moved_key[keep] = point; return keep
        g = exact_index.get(point)
        if g is not None and g not in target:
            return g
        payload = {"kind": "cut_carrier_merged_node", "world_exact_xyz": _point_payload(_world_identity_point(point, transform)),
                   "identity_rule": "exact_world_coordinate_only"}
        key = f"CELL_PATCH/SURFACE_P1_MERGED/{_canonical_hash(payload)[:24]}"
        g = G + len(new_exact); new_exact.append(point); new_keys.append(key); exact_index[point] = g
        return g

    lines_through: dict[tuple, list[int]] = {}
    for g in cut_nodes:                                                  # crossings by the lattice node they are nearest to
        lines_through.setdefault(lattice_of(exact[g]), []).append(g)
    min_gap = None                                                       # smallest residual gap / h of an unfixable family
    for N in sorted(candidates):
        faces = [k for k in range(3) if N[k] == 0 or N[k] == 1]
        point, kind = _canonical_point(N, faces, plane, h, eps)
        if point is None:
            if kind != "far":
                report["skipped"][kind] = report["skipped"].get(kind, 0) + 1
                # the residual strip width: the distance from N to the nearest face cut line through its faces
                gaps = []
                for f in faces:
                    m = tuple(n[j] if j != f else Fraction(0) for j in range(3))
                    if _dot(m, m):
                        gaps.append(abs(float(_dot(n, N) - d)) / math.sqrt(float(_dot(m, m))))
                if gaps:
                    g = min(gaps) / float(h); min_gap = g if min_gap is None else min(min_gap, g)
            continue
        near = [g for g in lines_through.get(N, ()) if _dist2(exact[g], N) <= 3 * eps * eps * h * h]
        lattice_node = exact_index.get(N)                                # N itself, when it is a carrier node
        members = list(near) + ([lattice_node] if lattice_node is not None else [])
        if not members:
            continue
        if len(members) == 1 and exact[members[0]] == point:
            continue
        g_new = node_for(point, keep=lattice_node)
        for g in members:
            if g != g_new:
                target[g] = g_new
        parity[g_new] = int(sum(int(x / h) for x in N)) % 2
        report["clusters"] += 1; report["merged_nodes"] += sum(1 for g in members if g != g_new)
        report["lattice_nodes_moved"] += int(lattice_node is not None)
        report.setdefault("kinds", {})[kind] = report.setdefault("kinds", {}).get(kind, 0) + 1

    # ---- new global node table (the fans are rebuilt even when nothing merged: the frozen fan is not symmetric) ----
    all_exact = list(exact) + new_exact; all_keys = keys + new_keys
    for g, p in moved_key.items():
        all_exact[g] = p
    alive = [g for g in range(len(all_exact)) if g not in target]
    old2new = {g: i for i, g in enumerate(alive)}
    remap = lambda g: old2new[target.get(g, g)]
    coords = np.asarray([[float(x) for x in all_exact[g]] for g in alive], dtype=np.float64)

    def slot_of(p):
        """(lattice slot, True) when p is within slot_tol of a lattice node, else (None, False)."""
        N = lattice_of(p)
        return (N, True) if _dist2(p, N) <= slot_tol * slot_tol else (None, False)

    traces = []; l2g_out: dict[str, tuple] = {}; fan_report: dict[str, dict] = {}
    for tr in layout.local_traces:
        src = str(tr.source_id); loc2glob = l2g[src]
        T = np.asarray(tr.triangles, dtype=np.int64); X = np.asarray(tr.node_coordinates, dtype=np.float64)
        # polygons: the triangles of one grid cell (cut face) or one grid square (box face), on the frozen geometry
        cen = X[T].mean(axis=1) * carrier_n
        if src == "cut_0":
            cell_key = [tuple(np.floor(c - 1e-9).astype(int).tolist()) for c in cen]
        else:
            ax = [k for k in range(3) if np.ptp(X[:, k]) > 1e-12]
            cell_key = [tuple(np.floor(cen[t, ax] - 1e-9).astype(int).tolist()) for t in range(len(T))]
        polys: dict[tuple, set[int]] = {}
        for t in range(len(T)):
            polys.setdefault(cell_key[t], set()).update(int(v) for v in T[t])
        new_glob_of_local = np.asarray([remap(int(g)) for g in loc2glob], dtype=np.int64)
        glob_list = sorted(set(int(g) for g in new_glob_of_local)); gl2loc = {g: i for i, g in enumerate(glob_list)}
        uv_exact = [_world_to_uv_exact(tr.chart, all_exact[alive[g]]) for g in glob_list]
        uv = np.asarray([[float(u), float(v)] for u, v in uv_exact], dtype=np.float64)
        P3 = coords[np.asarray(glob_list, dtype=np.int64)]
        rep = {"polygons": len(polys), "lattice_squares": 0, "anchor_fans": 0, "parity_quads": 0, "centroids": 0, "dropped": 0}
        new_tris: list[tuple[int, int, int]] = []; min_angle = math.pi
        for key_, locals_ in polys.items():
            nodes = sorted({gl2loc[int(new_glob_of_local[v])] for v in locals_})
            if len(nodes) < 3:
                rep["dropped"] += 1; continue
            order = _cyclic_order(uv, nodes); m = len(order)
            pex = [all_exact[alive[glob_list[i]]] for i in order]
            # (a) lattice square: Union Jack parity diagonal
            if m == 4 and src != "cut_0":
                slots = [slot_of(p) for p in pex]
                if all(ok for _, ok in slots):
                    S = [tuple(int(x / h) for x in N) for N, _ in slots]
                    even = [i for i, s in enumerate(S) if sum(s) % 2 == 0]
                    span = [max(s[k] for s in S) - min(s[k] for s in S) for k in range(3)]
                    if len(even) == 2 and (even[1] - even[0]) == 2 and sorted(span) == [0, 1, 1] and len(set(S)) == 4:
                        p, q = order[even[0]], order[even[1]]; o1, o2 = order[(even[0] + 1) % 4], order[(even[0] + 3) % 4]
                        new_tris.extend([(p, q, o1), (q, p, o2)]); rep["lattice_squares"] += 1; continue
            # (b) anchor fan by the angle-list rule
            tris, ang = polygon_fan(P3, order)
            if tris is not None and all(_cross2(uv_exact[i], uv_exact[j], uv_exact[k]) != 0 for i, j, k in tris):
                new_tris.extend(tris); min_angle = min(min_angle, ang); rep["anchor_fans"] += 1; continue
            # (c) tied quadrilateral: diagonal between opposite vertices of even grid parity
            if m == 4:
                par = []
                for i, p in zip(order, pex):
                    g = alive[glob_list[i]]
                    par.append(parity.get(g, parity_of_point(p)))
                even = [i for i, pv in enumerate(par) if pv == 0]
                if None not in par and len(even) == 2 and even[1] - even[0] == 2:
                    p, q = order[even[0]], order[even[1]]; o1, o2 = order[(even[0] + 1) % 4], order[(even[0] + 3) % 4]
                    if _cross2(uv_exact[p], uv_exact[q], uv_exact[o1]) != 0 and _cross2(uv_exact[q], uv_exact[p], uv_exact[o2]) != 0:
                        new_tris.extend([(p, q, o1), (q, p, o2)]); rep["parity_quads"] += 1; continue
            # (d) centroid node keyed by the exact centroid of the polygon
            cpt = tuple(sum((p[k] for p in pex), Fraction(0)) / m for k in range(3))
            gc = node_for(cpt)
            if gc >= len(all_exact):
                all_exact.append(cpt); all_keys.append(new_keys[-1]); alive.append(gc); old2new[gc] = len(alive) - 1
                coords = np.vstack([coords, [[float(x) for x in cpt]]])
            gcn = old2new[target.get(gc, gc)]
            if gcn not in gl2loc:
                gl2loc[gcn] = len(glob_list); glob_list.append(gcn); uv_c = _world_to_uv_exact(tr.chart, cpt); uv_exact.append(uv_c)
                uv = np.vstack([uv, [[float(uv_c[0]), float(uv_c[1])]]]); P3 = np.vstack([P3, coords[gcn]])
            c = gl2loc[gcn]
            new_tris.extend((c, order[j], order[(j + 1) % m]) for j in range(m)); rep["centroids"] += 1
        fixed = []
        for i, j, k in new_tris:
            o = _cross2(uv_exact[i], uv_exact[j], uv_exact[k])
            if o == 0:
                raise ValueError(f"degenerate carrier triangle on {src} after the cut-face merge")
            fixed.append((i, j, k) if o > 0 else (i, k, j))
        rep["min_fan_angle_deg"] = None if min_angle == math.pi else round(math.degrees(min_angle), 3)
        fan_report[src] = rep
        traces.append(dc_replace(tr, global_node_ids=tuple(all_keys[alive[g]] for g in glob_list), uv_coordinates_exact=tuple(uv_exact),
                                 node_coordinates=P3.copy(), triangles=np.asarray(fixed, dtype=np.int64).reshape(-1, 3)))
        l2g_out[src] = (np.arange(len(glob_list), dtype=np.int64), np.asarray(glob_list, dtype=np.int64))
    out_keys = tuple(all_keys[g] for g in alive); Gn = len(alive)
    l2g_csr = {src: csr_matrix((np.ones(len(r), dtype=np.float64), (r, c)), shape=(len(r), Gn)) for src, (r, c) in l2g_out.items()}
    report["fans"] = fan_report; report["nodes_before"] = G; report["nodes_after"] = Gn
    report["min_unfixed_gap_over_h"] = min_gap
    signature = _canonical_hash({"base": layout.semantic_signature, "cut_carrier_merge": {"eps": str(eps), "rule": "grid_node_canonical_point_v2",
                                 "fan": "lattice_parity|angle_list_anchor|parity_quad|exact_centroid"}, "node_keys": list(out_keys)})
    diagnostics = dict(layout.diagnostics); diagnostics["cut_carrier_merge"] = {**report, "applied": True}
    diagnostics["global_scalar_master_node_count"] = Gn; diagnostics["global_vector_master_dof_count"] = 3 * Gn
    diagnostics["global_triangle_count"] = int(sum(len(t.triangles) for t in traces))
    new_layout = dc_replace(layout, local_traces=tuple(traces), local_to_global_scalar=l2g_csr, global_scalar_node_keys=out_keys,
                            global_scalar_node_coordinates=coords, semantic_signature=signature, diagnostics=diagnostics)
    return new_layout, report
