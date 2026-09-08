"""True sheet-TPMS solid (no port collars) as a closed, carrier-conforming surface, plus its Gmsh volume mesh.

Material = { |phi(x)| <= tau(x) } clipped by the unit box and the world cut plane.  Steps:
1. marching cubes of |phi| - tau on a padded grid aligned with the box (skimage);
2. isotropic remeshing of the sheet clipped with a margin outside the planes, then exact clipping by every box plane
   and the cut plane (Sutherland-Hodgman per triangle, shared intersection vertices, near-plane vertices snapped onto
   the plane), short-edge collapse and needle repair, re-projection onto the level set within the planes; a
   self-intersection or non-manifold result is retried with a scaled remesh size (fail-closed);
3. the clipped surface's boundary loops on each plane bound the material cross-section (bands); on active port faces the
   band is re-triangulated by constrained Delaunay (`triangle`) with the carrier trace nodes and edges as constraints
   (so every fine port node lies on a carrier node, a carrier edge or inside a carrier triangle); on the cut plane the
   band is triangulated without constraints;
4. consistent outward orientation per closed shell, watertightness and verified self-intersection checks;
5. Gmsh volume mesh (discrete surfaces, boundary preserved), MEDIT output.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .conforming_port_mesh import _components, _min_dihedral, carrier_shell, signed_volume, tri_quality, watertight_report

_BOX = (("box_x_min", 0, 0.0, -1.0), ("box_x_max", 0, 1.0, 1.0), ("box_y_min", 1, 0.0, -1.0),
        ("box_y_max", 1, 1.0, 1.0), ("box_z_min", 2, 0.0, -1.0), ("box_z_max", 2, 1.0, 1.0))


def sheet_level(geometry, points: np.ndarray) -> np.ndarray:
    """|phi| - tau: negative inside the sheet.  No collar terms."""
    tpms_signed, _ = geometry.material_band_components(np.asarray(points, dtype=np.float64))
    return np.asarray(tpms_signed, dtype=np.float64)


def clip_planes(geometry) -> list[tuple[str, np.ndarray, float]]:
    """(name, n, d) with the retained side n.x - d <= 0; box normals point outward."""
    planes = []; box_max = np.asarray(getattr(geometry, "box_max", (1.0, 1.0, 1.0)), dtype=np.float64)
    for name, axis, value, sign in _BOX:
        n = np.zeros(3); n[axis] = sign; planes.append((name, n, sign * (value * box_max[axis])))
    if geometry.cut_plane is not None:
        p = np.asarray(geometry.cut_plane, dtype=np.float64); s = np.linalg.norm(p[:3])
        planes.append(("cut_0", p[:3] / s, float(p[3]) / s))
    return planes


def marching_sheet(geometry, n_per_unit: int = 128, pad: int = 4) -> tuple[np.ndarray, np.ndarray]:
    from skimage.measure import marching_cubes
    h = 1.0 / n_per_unit; box_max = np.asarray(getattr(geometry, "box_max", (1.0, 1.0, 1.0)), dtype=np.float64)
    axes = [np.arange(-pad, int(round(box_max[k] * n_per_unit)) + pad + 1, dtype=np.float64) * h for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    vol = sheet_level(geometry, np.stack([X, Y, Z], axis=-1))
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(h, h, h), allow_degenerate=False)
    verts = np.asarray(verts, dtype=np.float64) + np.array([axes[0][0], axes[1][0], axes[2][0]])
    return verts, np.asarray(faces, dtype=np.int64)


def project_to_sheet(geometry, V: np.ndarray, iters: int = 4, fd: float = 1e-6) -> tuple[np.ndarray, float]:
    """Newton projection of points onto the exact level set |phi| - tau = 0 along the numerical gradient."""
    V = np.asarray(V, dtype=np.float64).copy(); res0 = np.abs(sheet_level(geometry, V)).max()
    for _ in range(iters):
        f = sheet_level(geometry, V); g = np.zeros_like(V)
        for k in range(3):
            e = np.zeros(3); e[k] = fd
            g[:, k] = (sheet_level(geometry, V + e) - sheet_level(geometry, V - e)) / (2 * fd)
        g2 = np.einsum("ij,ij->i", g, g); step = (f / np.maximum(g2, 1e-300))[:, None] * g
        V -= step
    return V, float(res0)


def snap_near_plane_constrained(V: np.ndarray, n: np.ndarray, d: float, eps: float, planes, tol: float = 1e-9, allow: np.ndarray | None = None,
                                max_move: float | None = 2.0) -> np.ndarray:
    """snap_near_plane that keeps a vertex on every clip plane it already lies on (joint least-squares correction), so
    snapping onto an oblique cut plane does not lift box-face vertices off their face.  `allow` (bool per vertex)
    restricts snapping to vertices where the sheet crosses the plane transversally: where the sheet is tangent to the
    plane (knife edge) a snapped vertex would leave the level set with no nearby in-plane point to return to.
    The joint correction moves a vertex by eps / sin(dihedral angle): where the plane meets a face the vertex lies on
    at a shallow angle this would drag it along the face by many mesh sizes onto the intersection line (population
    census 2026-09-04: a 3 deg cut against the x face folded the boundary strip and stranded vertices off the level
    set).  A correction longer than max_move * eps is not applied; the exact clip handles that vertex instead."""
    V = V.copy(); s = V @ n - d; near = np.abs(s) < eps
    if allow is not None:
        near &= allow
    near = np.where(near)[0]
    if not len(near):
        return V
    onp = np.stack([np.abs(V[near] @ nn - dd) <= tol for _, nn, dd in planes], axis=1) if planes else np.zeros((len(near), 0), bool)
    for row, i in enumerate(near):
        rows = [(n, d)] + [(nn, dd) for k, (_, nn, dd) in enumerate(planes) if onp[row, k] and abs(abs(nn @ n) - 1.0) > 1e-12]
        if len(rows) == 1:
            V[i] -= s[i] * n; continue
        A = np.stack([r[0] for r in rows]); rhs = np.array([r[1] for r in rows]) - A @ V[i]
        corr, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        if max_move is not None and np.linalg.norm(corr) > max_move * eps:
            continue
        V[i] += corr
    return V


def clip_by_plane(V: np.ndarray, F: np.ndarray, n: np.ndarray, d: float, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Keep the part with n.x - d <= 0.  Intersection vertices are shared between neighbouring triangles."""
    V = np.asarray(V, dtype=np.float64).copy(); s = V @ n - d
    on = np.abs(s) <= tol
    V[on] -= np.outer(s[on], n); s[on] = 0.0
    inside = s <= 0.0
    keep_all = inside[F].all(axis=1); drop_all = (~inside[F]).all(axis=1)
    out_tris = [F[keep_all]]
    new_pts: list[np.ndarray] = []; cache: dict[tuple[int, int], int] = {}; base = len(V)

    def cut_point(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            t = s[a] / (s[a] - s[b]); p = V[a] + t * (V[b] - V[a]); p = p - (p @ n - d) * n
            cache[key] = base + len(new_pts); new_pts.append(p)
        return cache[key]

    mixed = np.where(~keep_all & ~drop_all)[0]; tris = []
    for f in F[mixed]:
        poly = []
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            if s[a] <= 0.0:
                poly.append(a)
            if s[a] * s[b] < 0.0:
                poly.append(cut_point(a, b))
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
    if tris:
        out_tris.append(np.asarray(tris, dtype=np.int64))
    F2 = np.vstack(out_tris) if out_tris else np.empty((0, 3), dtype=np.int64)
    V2 = np.vstack([V, np.asarray(new_pts)]) if new_pts else V
    return compact(V2, F2)


def compact(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return V[used], inv.reshape(F.shape).astype(np.int64)


def _directed_boundary(F: np.ndarray) -> list[tuple[int, int]]:
    edges = set()
    for a, b, c in F:
        edges.update(((int(a), int(b)), (int(b), int(c)), (int(c), int(a))))
    return [e for e in edges if (e[1], e[0]) not in edges]


def boundary_chains_on_plane(V: np.ndarray, F: np.ndarray, n: np.ndarray, d: float, tol: float = 1e-9) -> tuple[list[list[int]], list[bool]]:
    """Boundary chains (vertex id sequences) of the surface lying on the plane, following the surface orientation.
    Closed chains are loops; open chains end where the boundary leaves the plane (band reaching a box edge)."""
    onp = np.abs(V @ n - d) <= tol
    nxt: dict[int, list[int]] = defaultdict(list); indeg: dict[int, int] = defaultdict(int)
    for a, b in _directed_boundary(F):
        if onp[a] and onp[b]:
            nxt[a].append(b); indeg[b] += 1
    chains: list[list[int]] = []; closed: list[bool] = []; used_edges = set()
    starts = [v for v in nxt if indeg[v] == 0] + [v for v in nxt if indeg[v] > 0]
    for start in starts:
        while any((start, b) not in used_edges for b in nxt.get(start, [])):
            chain = [start]; cur = start
            while True:
                cands = [b for b in nxt.get(cur, []) if (cur, b) not in used_edges]
                if not cands:
                    break
                b = cands[0]; used_edges.add((cur, b)); chain.append(b); cur = b
                if cur == start:
                    break
            if len(chain) >= 2:
                if chain[-1] == chain[0]:
                    chains.append(chain[:-1]); closed.append(True)
                else:
                    chains.append(chain); closed.append(False)
    return chains, closed


def _triangulate_isolated(pslg: dict, opts: str) -> dict:
    """Run Triangle in a subprocess: the C library aborts the whole process on some inputs, and its global state is not
    reliable across repeated in-process calls.  Falls back to a plain constrained triangulation ('pYY')."""
    import subprocess, sys, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in.npz"); outp = os.path.join(td, "out.npz")
        np.savez(inp, vertices=np.asarray(pslg["vertices"], dtype=np.float64), segments=np.asarray(pslg["segments"], dtype=np.int32))
        code = ("import numpy as np, triangle as tr, sys\n"
                "z = np.load(sys.argv[1]); out = tr.triangulate({'vertices': z['vertices'], 'segments': z['segments']}, sys.argv[3])\n"
                "np.savez(sys.argv[2], vertices=out['vertices'], triangles=out['triangles'])\n")
        for o in (opts, "pYY"):
            r = subprocess.run([sys.executable, "-c", code, inp, outp, o], capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and os.path.exists(outp):
                z = np.load(outp); return {"vertices": z["vertices"], "triangles": z["triangles"]}
        raise RuntimeError(f"Triangle failed on a cap PSLG ({len(pslg['vertices'])} points, {len(pslg['segments'])} segments): {r.stderr[-200:]}")


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a); u /= np.linalg.norm(u); v = np.cross(n, u)
    return u, v  # u x v = n


def _seg_intersections(p: np.ndarray, q: np.ndarray, A: np.ndarray, B: np.ndarray, eps: float = 1e-12):
    """Parameters (t on pq, s on AB) of proper intersections of segment pq with segments A[i]B[i]."""
    r = q - p; e = B - A
    den = r[0] * e[:, 1] - r[1] * e[:, 0]
    ok = np.abs(den) > eps
    w = A - p
    t = np.full(len(A), np.nan); s = np.full(len(A), np.nan)
    t[ok] = (w[ok, 0] * e[ok, 1] - w[ok, 1] * e[ok, 0]) / den[ok]
    s[ok] = (w[ok, 0] * r[1] - w[ok, 1] * r[0]) / den[ok]
    return t, s


def _cube_plane_polygon(n: np.ndarray, d: float, box_max=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Ordered polygon (3D points) of the intersection of plane n.x = d with the box [0, box_max]."""
    bx, by, bz = box_max
    corners = np.array([(x, y, z) for x in (0.0, bx) for y in (0.0, by) for z in (0.0, bz)])
    pts = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) != 1:
                continue
            si, sj = corners[i] @ n - d, corners[j] @ n - d
            if si * sj < 0:
                t = si / (si - sj); pts.append(corners[i] + t * (corners[j] - corners[i]))
            elif si == 0:
                pts.append(corners[i])
    if not pts:
        return np.empty((0, 3))
    P = np.unique(np.round(np.asarray(pts), 12), axis=0)
    u, v = _plane_basis(n); c = P.mean(axis=0)
    ang = np.arctan2((P - c) @ v, (P - c) @ u)
    return P[np.argsort(ang)]


def _clip_polygon_by_halfspace(poly: np.ndarray, n: np.ndarray, d: float, tol: float = 1e-12) -> np.ndarray:
    """Sutherland-Hodgman: the part of the ordered 3D polygon with n.x - d <= 0."""
    if len(poly) == 0:
        return poly
    out = []; s = poly @ n - d
    for i in range(len(poly)):
        j = (i + 1) % len(poly); pi, pj = poly[i], poly[j]; si, sj = s[i], s[j]
        if si <= tol:
            out.append(pi)
        if (si < -tol and sj > tol) or (si > tol and sj < -tol):
            t = si / (si - sj); out.append(pi + t * (pj - pi))
    return np.asarray(out).reshape(-1, 3)


def _retained_facet_polygon(planes, name: str, box_max=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Ordered polygon of the facet of the retained region (box cut by every plane) lying on plane `name`: the box face
    or the cut plane's cube polygon, clipped by all the other retained half-spaces.  A cap built on the whole box face
    of a face the cut crosses would extend past the cut line, where nothing bounds the flood fill (census 2026-09-04:
    a face the band does not reach, hence without carrier constraints, grew cap triangles up to the far cube corner)."""
    n, d = next((nn, dd) for nm, nn, dd in planes if nm == name)
    bx, by, bz = box_max; corners = np.array([(x, y, z) for x in (0.0, bx) for y in (0.0, by) for z in (0.0, bz)])
    if name.startswith("box_"):
        on = np.abs(corners @ n - d) <= 1e-12; P = corners[on]
        u, v = _plane_basis(n); c = P.mean(axis=0); ang = np.arctan2((P - c) @ v, (P - c) @ u); poly = P[np.argsort(ang)]
    else:
        poly = _cube_plane_polygon(n, d, box_max)
    for nm, nn, dd in planes:
        if nm != name:
            poly = _clip_polygon_by_halfspace(poly, nn, dd)
    return poly


def triangulate_plane_region(V: np.ndarray, chains: list[list[int]], closed: list[bool], n: np.ndarray, d: float, is_material, *, level_fn=None,
                             constraint_points: np.ndarray | None = None, constraint_segments: np.ndarray | None = None,
                             max_area: float | None = None, quality: str = "q20", steiner_on_chains: bool = False, snap_relative: float = 0.3,
                             orientation_sign: float = 1.0, classification_report: dict | None = None, unseeded_material: bool = True,
                             cap_flip_quality: float = 0.3, feature_size: float | None = None, fixed_vertices=None, can_move=None, fixed_targets=None,
                             removable_points=None):
    """Constrained Delaunay triangulation of the material region on the plane.

    chains: boundary chains (global vertex ids) of the already-built surface on this plane (closed or open);
    constraint_points/segments: extra 3D points and 3D segments (pairs of points) that must appear in the triangulation
    (carrier trace on a port face, the cube polygon on the cut plane).  Every chain edge crossed by a constraint is split
    and the split is reported so the neighbouring surface triangles can be split identically.
    feature_size: PSLG feature-size invariant.  A chain vertex within feature_size of a constraint point is moved onto
    it and one within feature_size of a constraint segment is projected onto it, and a chain/constraint-segment crossing
    within feature_size of a chain vertex is snapped to that chain vertex (the constraint segment is rerouted through
    it, as the relative snap_relative rule already does); constraint points never move, chain
    vertices in `fixed_vertices` (global ids on another clip plane) never move, constraint points in `fixed_targets`
    (on another clip plane: ring nodes, already vertices of that plane's cap) are never targets, one chain vertex per
    target, no vertex adjacent on the chain to a vertex already assigned to that target, and `can_move` (the caller's
    fold check on the incident sheet triangles) must agree.  V is updated IN PLACE for moved chain vertices.  Snapping
    chain/constraint-segment CROSSINGS to a carrier node is not done: it duplicates the node on the sheet and pinches
    the surface (pop_cut_0165 / pop_cut_1125, 2026-09-05).  Without the invariant the cap carries features of 1e-4 and
    smaller (a sheet vertex a hair from a carrier node) that Triangle refines and the volume mesher must grade to.
    Returns (new vertex xyz, triangles in global ids, global ids of constraint points, chain-edge splits)."""
    sheet_level_2d = level_fn if level_fn is not None else (lambda P3: np.where(is_material(P3), -1.0, 1.0))
    classification_conflicts = [0]
    if classification_report is None:
        classification_report = {}
    u, v = _plane_basis(n); o = n * d
    to2 = lambda P: np.column_stack([(np.asarray(P) - o) @ u, (np.asarray(P) - o) @ v])
    to3 = lambda P2: o + P2[:, :1] * u + P2[:, 1:2] * v
    chain_ids = sorted({i for ch in chains for i in ch}); lid = {g: k for k, g in enumerate(chain_ids)}
    pts2 = to2(V[chain_ids]) if chain_ids else np.empty((0, 2))
    if len(pts2) > 1:  # near-duplicate chain vertices (< 1e-7 apart) collapse onto one PSLG vertex
        from scipy.spatial import cKDTree
        for i2, j2 in sorted(cKDTree(pts2).query_pairs(1e-7)):
            for g, k in lid.items():
                if k == j2:
                    lid[g] = i2
    moved_vertices = 0
    if feature_size is not None and constraint_points is not None and len(pts2):
        # feature-size invariant: chain vertices closer than feature_size to a carrier node / carrier edge are moved
        # onto it (the sheet boundary shifts by at most feature_size, within the near-plane snap budget).  Rules that
        # keep the sheet a manifold: one chain vertex per carrier node (the nearest), never a vertex adjacent on the
        # chain to one already assigned to that node (no chain edge collapses onto a node), never a vertex that another
        # PSLG vertex already occupies within 1e-7 after the move, never a vertex whose move folds an incident sheet
        # triangle (can_move callback of the caller), never a vertex on another clip plane.
        cpts2 = to2(constraint_points); fixed_local = {lid[g] for g in (fixed_vertices or ()) if g in lid}
        segs_c = np.asarray(constraint_segments, dtype=np.int64).reshape(-1, 2) if constraint_segments is not None else np.zeros((0, 2), np.int64)
        # a carrier node on another clip plane (a ring node) already is a vertex of that plane's cap: a chain vertex
        # of this plane moved onto it would merge two surface vertices of different caps (a pinch); such nodes and
        # the ring segments between them are never targets
        ft = set(int(i) for i in (fixed_targets or ()))
        dd_big = np.full(len(cpts2), np.inf)
        if len(segs_c):
            segs_c = segs_c[~(np.isin(segs_c[:, 0], list(ft)) & np.isin(segs_c[:, 1], list(ft)))]
        nbr: dict[int, set[int]] = defaultdict(set)
        for ch, cl in zip(chains, closed):
            m = len(ch) if cl else len(ch) - 1
            for k in range(m):
                a_, b_ = lid[ch[k]], lid[ch[(k + 1) % len(ch)]]; nbr[a_].add(b_); nbr[b_].add(a_)
        local_to_global = {}
        for g, k in lid.items():
            local_to_global.setdefault(k, g)
        # candidates to carrier nodes, nearest first, one vertex per node
        cand = []
        for k, g in local_to_global.items():
            if k in fixed_local:
                continue
            dd = np.linalg.norm(cpts2 - pts2[k], axis=1)
            if ft:
                dd[list(ft)] = np.inf
            j = int(np.argmin(dd))
            if dd[j] <= feature_size:
                cand.append((float(dd[j]), k, j))
        taken: dict[int, int] = {}; moved_local: dict[int, np.ndarray] = {}
        for dist_, k, j in sorted(cand):
            if j in taken:
                continue
            if any(taken.get(x) == j for x in nbr[k]):
                continue
            target = cpts2[j]
            if np.any(np.linalg.norm(pts2 - target, axis=1) <= 1e-7):
                continue                         # another chain vertex already sits on this node
            if can_move is not None and not can_move(local_to_global[k], to3(target[None])[0]):
                continue
            taken[j] = k; moved_local[k] = target
        # remaining vertices: project onto a nearby carrier edge (never onto a node another vertex took)
        for k, g in local_to_global.items():
            if k in fixed_local or k in moved_local or not len(segs_c):
                continue
            Pa = cpts2[segs_c[:, 0]]; Pb = cpts2[segs_c[:, 1]]; e = Pb - Pa; L2 = np.einsum("ij,ij->i", e, e)
            w = pts2[k] - Pa; t = np.einsum("ij,ij->i", w, e) / L2; dist = np.abs(w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.sqrt(L2)
            inner = (t > 1e-9) & (t < 1 - 1e-9) & (dist <= feature_size)
            if not inner.any():
                continue
            i = int(np.argmin(np.where(inner, dist, np.inf))); target = Pa[i] + t[i] * e[i]
            if np.any(np.linalg.norm(pts2 - target, axis=1) <= 1e-7) or (can_move is not None and not can_move(g, to3(target[None])[0])):
                continue
            moved_local[k] = target
        for k, target in moved_local.items():
            pts2[k] = target; V[local_to_global[k]] = to3(target[None])[0]; moved_vertices += 1
        for g, k in lid.items():                    # collapsed duplicates (the 1e-7 pass above) follow their representative
            if g != local_to_global[k] and k in moved_local:
                V[g] = V[local_to_global[k]]
    if classification_report is not None:
        classification_report["chain_vertices_moved_by_feature_size"] = int(moved_vertices)
    segs = []
    for ch, cl in zip(chains, closed):
        m = len(ch) if cl else len(ch) - 1
        for k in range(m):
            segs.append((lid[ch[k]], lid[ch[(k + 1) % len(ch)]]))
    extra2: list[np.ndarray] = []; tol = 1e-7

    def all_pts():
        return np.vstack([pts2, np.asarray(extra2)]) if extra2 else pts2

    def add_point(p2: np.ndarray) -> int:
        ap = all_pts()
        if len(ap):
            dd = np.linalg.norm(ap - p2, axis=1); j = int(np.argmin(dd))
            if dd[j] <= tol:
                return j
        extra2.append(np.asarray(p2, dtype=np.float64)); return len(pts2) + len(extra2) - 1

    seg_arr = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
    A = pts2[seg_arr[:, 0]] if len(seg_arr) else np.empty((0, 2)); B = pts2[seg_arr[:, 1]] if len(seg_arr) else np.empty((0, 2))
    chain_splits: dict[int, list[tuple[float, int]]] = defaultdict(list)
    cons_segments: list[tuple[int, int]] = []; cons_ids: list[int] = []

    def point_on_segments(p2: np.ndarray):
        """indices of chain edges whose interior contains p2, with the parameter."""
        if not len(seg_arr):
            return []
        e = B - A; w = p2 - A; L2 = np.einsum("ij,ij->i", e, e)
        t = np.einsum("ij,ij->i", w, e) / L2; dist = np.abs(w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.sqrt(L2)
        return [(int(i), float(t[i])) for i in np.where((dist <= tol) & (t > 1e-9) & (t < 1 - 1e-9))[0]]

    if constraint_points is not None:
        c2 = to2(constraint_points); cid = []
        for k in range(len(c2)):
            j = add_point(c2[k]); cid.append(j)
            for i, t in point_on_segments(c2[k]):
                chain_splits[i].append((t, j))
        cons_ids = cid
        if constraint_segments is not None:
            for a, b in constraint_segments:
                p, q = c2[a], c2[b]
                splits = [(0.0, cid[a]), (1.0, cid[b])]
                if len(seg_arr):
                    t, s = _seg_intersections(p, q, A, B)
                    hit = np.where(np.isfinite(t) & (t > 1e-9) & (t < 1 - 1e-9) & (s > 1e-9) & (s < 1 - 1e-9))[0]
                    for i in hit:
                        # crossings close to a chain vertex are snapped onto it (avoids sliver splits of the sheet)
                        Lch = float(np.linalg.norm(B[i] - A[i]))
                        near_a = s[i] < snap_relative or (feature_size is not None and s[i] * Lch <= feature_size)
                        near_b = s[i] > 1 - snap_relative or (feature_size is not None and (1 - s[i]) * Lch <= feature_size)
                        if near_a and near_b:                       # a chain edge shorter than two feature sizes: the nearer vertex
                            near_a, near_b = (s[i] <= 0.5), (s[i] > 0.5)
                        if near_a:
                            splits.append((float(t[i]), int(seg_arr[i, 0])))
                        elif near_b:
                            splits.append((float(t[i]), int(seg_arr[i, 1])))
                        else:
                            j = add_point(p + t[i] * (q - p)); splits.append((float(t[i]), j)); chain_splits[int(i)].append((float(s[i]), j))
                    # chain vertices in the interior of the constraint segment
                    w = pts2 - p; r = q - p; L2 = r @ r
                    tt = (w @ r) / L2; dist = np.abs(w[:, 0] * r[1] - w[:, 1] * r[0]) / np.sqrt(L2)
                    for k in np.where((dist <= tol) & (tt > 1e-9) & (tt < 1 - 1e-9))[0]:
                        splits.append((float(tt[k]), int(k)))
                splits = sorted(set(splits))
                for k in range(len(splits) - 1):
                    if splits[k][1] != splits[k + 1][1]:
                        cons_segments.append((splits[k][1], splits[k + 1][1]))
    final_segs: list[tuple[int, int]] = []; edge_splits_global = []
    for i, (a, b) in enumerate(seg_arr):
        inner = sorted({(t, j) for t, j in chain_splits.get(i, []) if j != int(a) and j != int(b)})
        chain = [(0.0, int(a))] + inner + [(1.0, int(b))]
        for k in range(len(chain) - 1):
            if chain[k][1] != chain[k + 1][1]:
                final_segs.append((chain[k][1], chain[k + 1][1]))
        if inner:
            edge_splits_global.append((int(a), int(b), [j for _, j in inner]))  # local ids for now
    # origin bookkeeping: chain pieces remember their original chain edge so later splits reach the sheet triangles
    origin = {}
    for i, (a, b) in enumerate(seg_arr):
        inner = sorted({(t, j) for t, j in chain_splits.get(i, []) if j != int(a) and j != int(b)})
        chain = [(0.0, int(a))] + inner + [(1.0, int(b))]
        for k in range(len(chain) - 1):
            if chain[k][1] != chain[k + 1][1]:
                origin[(min(chain[k][1], chain[k + 1][1]), max(chain[k][1], chain[k + 1][1]))] = i
    final_segs += cons_segments
    final_segs = sorted({(min(a, b), max(a, b)) for a, b in final_segs if a != b})
    # planar arrangement: no two segments may properly cross (Triangle crashes or overlaps on crossing PSLG segments)
    for _pass in range(8):
        allp = all_pts(); segs_np = np.asarray(final_segs, dtype=np.int64)
        if len(segs_np) < 2:
            break
        A2 = allp[segs_np[:, 0]]; B2 = allp[segs_np[:, 1]]; found = None
        for si in range(len(segs_np)):
            t, s_ = _seg_intersections(A2[si], B2[si], A2, B2)
            hit = np.where(np.isfinite(t) & (t > 1e-7) & (t < 1 - 1e-7) & (s_ > 1e-7) & (s_ < 1 - 1e-7))[0]
            hit = [h for h in hit if h != si and len({int(x) for x in segs_np[si]} & {int(x) for x in segs_np[h]}) == 0]
            if hit:
                found = (si, int(hit[0]), float(t[hit[0]])); break
        if found is None:
            break
        si, sj, t = found; p2 = A2[si] + t * (B2[si] - A2[si]); j = add_point(p2)
        new_segs = []
        for k, (a2, b2) in enumerate(final_segs):
            if k in (si, sj) and j not in (a2, b2):
                new_segs += [(min(a2, j), max(a2, j)), (min(j, b2), max(j, b2))]
                oi = origin.get((a2, b2))
                if oi is not None:  # split of a chain piece: register on the original chain edge
                    e = B[oi] - A[oi]; tt = float(((p2 - A[oi]) @ e) / (e @ e)); chain_splits[oi].append((tt, j))
                    origin[(min(a2, j), max(a2, j))] = oi; origin[(min(j, b2), max(j, b2))] = oi
            else:
                new_segs.append((a2, b2))
        final_segs = sorted(set(new_segs))
    # points lying in the interior of a segment split that segment (collinear overlaps, touching segments)
    for _pass in range(8):
        allp = all_pts(); segs_np = np.asarray(final_segs, dtype=np.int64)
        if not len(segs_np):
            break
        A2 = allp[segs_np[:, 0]]; B2 = allp[segs_np[:, 1]]; e = B2 - A2; L2 = np.einsum("ij,ij->i", e, e)
        new_segs = []; changed = False
        for k, (a2, b2) in enumerate(final_segs):
            w = allp - A2[k]; tt = (w @ e[k]) / L2[k]; dist = np.abs(w[:, 0] * e[k, 1] - w[:, 1] * e[k, 0]) / np.sqrt(L2[k])
            on = np.where((dist <= 1e-9) & (tt > 1e-9) & (tt < 1 - 1e-9))[0]; on = [int(q) for q in on if q not in (a2, b2)]
            if not on:
                new_segs.append((a2, b2)); continue
            changed = True; chain = [(0.0, a2)] + sorted((float(tt[q]), q) for q in on) + [(1.0, b2)]
            oi = origin.get((a2, b2))
            for m in range(len(chain) - 1):
                p_, q_ = chain[m][1], chain[m + 1][1]
                if p_ != q_:
                    new_segs.append((min(p_, q_), max(p_, q_)))
                    if oi is not None:
                        origin[(min(p_, q_), max(p_, q_))] = oi
            if oi is not None:
                for t_, q in chain[1:-1]:
                    e0 = B[oi] - A[oi]; chain_splits[oi].append((float(((allp[q] - A[oi]) @ e0) / (e0 @ e0)), q))
        final_segs = sorted(set(new_segs))
        if not changed:
            break
    # recompute chain-edge splits (the arrangement may have added some)
    edge_splits_global = []
    for i, (a, b) in enumerate(seg_arr):
        inner = sorted({(t, j) for t, j in chain_splits.get(i, []) if j != int(a) and j != int(b)})
        if inner:
            edge_splits_global.append((int(a), int(b), [j for _, j in inner]))
    allp = all_pts()
    opts = "p" + quality + ("Y" if steiner_on_chains else "YY") + (f"a{max_area:.12g}" if max_area is not None else "")
    # sanitize: merge points closer than 1e-7 (union-find via KD-tree pairs), drop degenerate/duplicate segments
    from scipy.spatial import cKDTree
    parent = np.arange(len(allp))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i2, j2 in cKDTree(allp).query_pairs(1e-7):
        ri, rj = find(i2), find(j2)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    root = np.array([find(i) for i in range(len(allp))]); uniq = np.unique(root); old2new = {int(r): k for k, r in enumerate(uniq)}
    new_of_old = np.array([old2new[int(r)] for r in root]); allp_s = allp[uniq]
    segs_s = sorted({(min(new_of_old[a], new_of_old[b]), max(new_of_old[a], new_of_old[b])) for a, b in final_segs if new_of_old[a] != new_of_old[b]})
    pslg = {"vertices": allp_s, "segments": np.asarray(segs_s, dtype=np.int32).reshape(-1, 2)}
    import os
    if os.environ.get("SHEET_DEBUG_DIR"):
        dd = Path(os.environ["SHEET_DEBUG_DIR"]); dd.mkdir(parents=True, exist_ok=True); k = len(list(dd.glob("cap_*.npz")))
        np.savez(dd / f"cap_{k}.npz", vertices=allp, segments=pslg["segments"], n=n, d=d, opts=opts, n_chain=len(chain_ids))
    out = _triangulate_isolated(pslg, opts)
    pts_out = np.asarray(out["vertices"], dtype=np.float64); tris = np.asarray(out["triangles"], dtype=np.int64)
    if not np.allclose(pts_out[:len(allp_s)], allp_s, atol=1e-12):
        raise RuntimeError("triangle reordered input vertices")
    # map the triangulation back to the un-merged local ids (representative = smallest old id of each merged group)
    rep_old = {int(new_of_old[i]): i for i in range(len(allp) - 1, -1, -1)}
    n_new = len(allp_s); n_steiner = len(pts_out) - n_new
    back = np.array([rep_old[k] for k in range(n_new)] + list(range(len(allp), len(allp) + n_steiner)), dtype=np.int64)
    pts_full = np.vstack([allp, pts_out[n_new:]]) if n_steiner else allp.copy()
    tris = back[tris]; pts_out = pts_full
    chain_splits = {i: [(t, int(rep_old[int(new_of_old[j])])) for t, j in v] for i, v in chain_splits.items()}
    cons_ids = [int(rep_old[int(new_of_old[c])]) for c in cons_ids]
    chain_merge = {k: int(rep_old[int(new_of_old[k])]) for k in range(len(chain_ids))}
    if steiner_on_chains and len(pts_out) > len(allp) and len(seg_arr):
        # Steiner vertices inserted on original chain edges -> additional chain-edge splits
        extra = np.arange(len(allp), len(pts_out)); Pe = pts_out[extra]
        for i, (a, b) in enumerate(seg_arr):
            e = B[i] - A[i]; L2 = e @ e; w = Pe - A[i]
            t = (w @ e) / L2; dist = np.abs(w[:, 0] * e[1] - w[:, 1] * e[0]) / np.sqrt(L2)
            for k in np.where((dist <= 1e-9) & (t > 1e-9) & (t < 1 - 1e-9))[0]:
                chain_splits[i].append((float(t[k]), int(extra[k])))
        edge_splits_global = []
        for i, (a, b) in enumerate(seg_arr):
            inner = sorted({(t, j) for t, j in chain_splits.get(i, []) if j != int(a) and j != int(b)})
            if inner:
                edge_splits_global.append((int(a), int(b), [j for _, j in inner]))
    cen2 = pts_out[tris].mean(axis=1)
    if chains:
        # region classification by flood fill: chain edges separate material from void; the material side of a chain
        # edge is read from the exact level a small distance off the edge midpoint (both sides), then propagated across
        # all non-chain edges.  Components touching no chain edge fall back to the level at the centroid.
        # material side of a chain edge from the SHEET ORIENTATION: chains follow the surface orientation, and for an
        # outward-oriented sheet the cap triangle closing the solid along edge a->b lies on the side -n x t (t = b - a).
        # This is combinatorial (no level sampling), hence exact even where the sheet is tangent to the plane and the
        # chain vertices sit off the level set; the level is only sampled as a diagnostic.
        hand = 1.0 if (np.cross(u, v) @ n) > 0 else -1.0
        chain_edge_side: dict[tuple[int, int], int] = {}   # local 2D edge (min,max) -> material side as sign of (b-a) direction convention
        chain_edge_dir: dict[tuple[int, int], tuple[int, int]] = {}
        for ch, cl in zip(chains, closed):
            m = len(ch) if cl else len(ch) - 1
            for k in range(m):
                a2, b2 = lid[ch[k]], lid[ch[(k + 1) % len(ch)]]
                if a2 == b2:
                    continue
                chain_edge_dir[(min(a2, b2), max(a2, b2))] = (a2, b2)
        for i, (a2, b2) in enumerate(seg_arr):
            inner = sorted({(t, j) for t, j in chain_splits.get(i, []) if j != int(a2) and j != int(b2)})
            chain_nodes = [int(a2)] + [j for _, j in inner] + [int(b2)]
            for k in range(len(chain_nodes) - 1):
                if chain_nodes[k] != chain_nodes[k + 1]:
                    chain_edge_dir[(min(chain_nodes[k], chain_nodes[k + 1]), max(chain_nodes[k], chain_nodes[k + 1]))] = (chain_nodes[k], chain_nodes[k + 1])
        for key in chain_edge_dir:
            chain_edge_side[key] = 0
        e2t = defaultdict(list)
        for ti, tr3 in enumerate(tris):
            for a2, b2 in ((tr3[0], tr3[1]), (tr3[1], tr3[2]), (tr3[0], tr3[2])):
                e2t[(min(a2, b2), max(a2, b2))].append(ti)
        label = np.full(len(tris), -1, dtype=int); conf = np.zeros(len(tris)); level_disagreements = 0; level_checked = 0
        for key, (a2, b2) in chain_edge_dir.items():
            if key not in e2t:
                continue
            pa, pb = pts_out[a2], pts_out[b2]; mid = 0.5 * (pa + pb); t2 = pb - pa; L = np.linalg.norm(t2)
            if L < 1e-300:
                continue
            t2 = t2 / L
            s_mat = hand * np.array([t2[1], -t2[0]]) * orientation_sign     # 2D image of -n x t for the sheet's orientation
            # diagnostic: does the exact level agree (only where it is decisive)?
            off = max(0.3 * L, 1e-4)
            lev_p = float(sheet_level_2d(to3((mid + off * s_mat)[None]))[0]); lev_m = float(sheet_level_2d(to3((mid - off * s_mat)[None]))[0])
            if (lev_p < 0) != (lev_m < 0) and abs(lev_p - lev_m) > 1e-3:
                level_checked += 1
                if lev_p > lev_m:
                    level_disagreements += 1
            for ti in e2t[key]:
                c2 = cen2[ti] - mid; lab_ti = 1 if (c2 @ s_mat) > 0 else 0
                if conf[ti] < 1.0:
                    label[ti] = lab_ti; conf[ti] = 1.0
                elif label[ti] != lab_ti:
                    classification_conflicts[0] += 1
        # flood across non-chain edges, most confident seeds first; a conflict keeps the more confident label
        order = np.argsort(-conf); stack = [int(ti) for ti in order if label[ti] >= 0]
        while stack:
            ti = stack.pop(0); tr3 = tris[ti]
            for a2, b2 in ((tr3[0], tr3[1]), (tr3[1], tr3[2]), (tr3[0], tr3[2])):
                key = (min(a2, b2), max(a2, b2))
                if key in chain_edge_side:
                    continue
                for tj in e2t[key]:
                    if label[tj] < 0:
                        label[tj] = label[ti]; conf[tj] = conf[ti]; stack.append(tj)
                    elif label[tj] != label[ti]:
                        classification_conflicts[0] += 1
        rest = label < 0
        if rest.any():
            # a region no chain edge touches is bounded only by triangulation-hull edges (a port face: void, the
            # sheet has no boundary there) or by the cube polygon (the cut plane: material where the level says so)
            label[rest] = is_material(to3(cen2[rest])).astype(int) if unseeded_material else 0
        keep = label == 1
        classification_report.update({"chain_edges": int(len(chain_edge_dir)), "level_checked": int(level_checked), "level_disagreements": int(level_disagreements),
                                      "conflicts": int(classification_conflicts[0]), "unseeded_triangles": int(rest.sum())})
    else:
        keep = is_material(to3(cen2)) if unseeded_material else np.zeros(len(tris), bool); label = keep.astype(int); chain_edge_side = {}
    import os
    if os.environ.get("SHEET_DEBUG_DIR"):
        dd = Path(os.environ["SHEET_DEBUG_DIR"]); k = len(list(dd.glob("cls_*.npz")))
        np.savez(dd / f"cls_{k}.npz", vertices=pts_out, tris=tris, label=label, chain_edges=np.asarray(list(chain_edge_side), dtype=np.int64).reshape(-1, 2), n=n, d=d)
    tris = tris[keep]
    # needle repair before the flip pass: a Steiner apex a hair beside a chain edge (see _repair_cap_needles)
    if len(tris):
        shared = set(range(len(chain_ids))) | {int(j) for lst in chain_splits.values() for _, j in lst}
        removable = {int(cons_ids[k]) for k in (removable_points or ()) if k < len(cons_ids)} - shared
        tris, needle_rep = _repair_cap_needles(pts_out, tris, len(allp), removable=removable)
        classification_report.update(needle_rep)
    # simple cap quality pass: flip the longest edge of the worst triangles when neither a chain edge nor a PSLG
    # segment (carrier edge, cube polygon) is involved; orientation and quality guards live in _flip_edges
    if len(tris) and cap_flip_quality > 0.0:
        protected = {(min(int(back[a]), int(back[b])), max(int(back[a]), int(back[b]))) for a, b in segs_s}
        pts3 = to3(pts_out); flips_total = 0
        for _ in range(4):
            q_cap = tri_quality(pts3, tris); bad = np.where(q_cap < cap_flip_quality)[0]
            if not len(bad):
                break
            longest = []
            for i in bad[np.argsort(q_cap[bad])]:
                f = [int(v) for v in tris[i]]; e = [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]
                longest.append(max(e, key=lambda ab: np.linalg.norm(pts3[ab[0]] - pts3[ab[1]])))
            tris, nf = _flip_edges(pts3, tris, longest, protected=protected); flips_total += nf
            if nf == 0:
                break
        q_cap = tri_quality(pts3, tris)
        classification_report.update({"cap_flips": int(flips_total), "cap_min_quality": float(q_cap.min()), "cap_below_flip_quality": int((q_cap < cap_flip_quality).sum())})
    gid = np.full(len(pts_out), -1, dtype=np.int64)
    for k, g in enumerate(chain_ids):
        gid[k] = g
    merged_chain = {}
    for k, g in enumerate(chain_ids):
        r = chain_merge.get(k, k)
        if r != k and r < len(chain_ids):
            merged_chain[g] = chain_ids[r]  # global chain vertex g is replaced by chain vertex of r
    new_xyz = []; base = len(V)
    needed = set(np.unique(tris).tolist()) | set(cons_ids) | {j for _, _, js in edge_splits_global for j in js}
    for k in sorted(needed):
        if gid[k] < 0:
            gid[k] = base + len(new_xyz); new_xyz.append(to3(pts_out[k:k + 1])[0])
    tris_g = gid[tris]
    cons_global = [int(gid[c]) for c in cons_ids]
    splits_global = [(int(gid[a]), int(gid[b]), [int(gid[j]) for j in js]) for a, b, js in edge_splits_global]
    return np.asarray(new_xyz).reshape(-1, 3), tris_g, cons_global, splits_global, merged_chain


def split_edges_in_triangles(F: np.ndarray, labels: list[str], splits, V: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Fan-split every triangle containing edge (a,b) or (b,a) at the given interior points (ordered from a to b)."""
    if not splits:
        return F, labels
    by_edge = {}
    for a, b, js in splits:
        by_edge[(a, b)] = [a] + js + [b]; by_edge[(b, a)] = [b] + js[::-1] + [a]
    out = []; out_lab = []
    for f, lab in zip(F, labels):
        f = [int(x) for x in f]; done = False
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]; c = f[(k + 2) % 3]
            if (a, b) in by_edge:
                chain = by_edge[(a, b)]
                for m in range(len(chain) - 1):
                    out.append((chain[m], chain[m + 1], c)); out_lab.append(lab)
                done = True; break
        if not done:
            out.append(tuple(f)); out_lab.append(lab)
    return np.asarray(out, dtype=np.int64), out_lab


def merge_duplicate_vertices(V: np.ndarray, F: np.ndarray, decimals: int = 9):
    key = np.round(V, decimals); _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel(); remap = np.arange(len(V)); remap = first[inv]
    F2 = remap[F]; F2 = F2[(F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])]
    V2, F3 = compact(V, F2)
    # mapping old id -> new id
    used, inv2 = np.unique(F2.ravel(), return_inverse=True); new_of = {int(u): k for k, u in enumerate(used)}
    Fr = remap[F]; keep_mask = (Fr[:, 0] != Fr[:, 1]) & (Fr[:, 1] != Fr[:, 2]) & (Fr[:, 0] != Fr[:, 2])
    return V2, F3, lambda old: new_of.get(int(remap[old]), -1), keep_mask


def project_to_sheet_in_planes(geometry, V: np.ndarray, planes, iters: int = 4, fd: float = 1e-6, tol: float = 1e-9, inside_margin: float = 2e-4,
                               max_step: float | None = None, min_inplane_fraction: float = 0.25) -> np.ndarray:
    """Newton projection onto |phi| - tau = 0; vertices lying on clip planes move only within their plane(s).
    A plane vertex whose in-plane gradient is below min_inplane_fraction of the full gradient (sheet nearly tangent to
    the plane) is left where it is, and no vertex moves further than max_step per iteration."""
    V = np.asarray(V, dtype=np.float64).copy()
    normals = np.zeros((len(V), 3, 3)); count = np.zeros(len(V), dtype=int)
    for _, n, d in planes:
        on = np.abs(V @ n - d) <= tol
        for i in np.where(on)[0]:
            if count[i] < 3:
                normals[i, count[i]] = n; count[i] += 1
    for _ in range(iters):
        f = sheet_level(geometry, V); g = np.zeros_like(V)
        for k in range(3):
            e = np.zeros(3); e[k] = fd
            g[:, k] = (sheet_level(geometry, V + e) - sheet_level(geometry, V - e)) / (2 * fd)
        g_full2 = np.einsum("ij,ij->i", g, g)
        for i in np.where(count > 0)[0]:  # remove the plane-normal components (orthonormalized: planes may be oblique)
            basis = []
            for c in range(count[i]):
                nn = normals[i, c].copy()
                for b in basis:
                    nn -= (nn @ b) * b
                if np.linalg.norm(nn) > 1e-9:
                    basis.append(nn / np.linalg.norm(nn))
            for b in basis:
                g[i] -= (g[i] @ b) * b
        g2 = np.einsum("ij,ij->i", g, g); ok = (g2 > 1e-20) & (g2 >= min_inplane_fraction * g_full2)
        step = np.zeros_like(V); step[ok] = (f[ok] / g2[ok])[:, None] * g[ok]
        if max_step is not None:
            L = np.linalg.norm(step, axis=1); big = L > max_step
            step[big] *= (max_step / L[big])[:, None]
        # interior vertices must stay strictly inside the retained half-spaces (no overshoot through a clip plane)
        for _, n, d in planes:
            s_now = V @ n - d; s_new = (V - step) @ n - d
            cross = (count == 0) & (s_new > -inside_margin) & (s_now < s_new)
            if cross.any():
                allowed = (s_now[cross] + inside_margin) / np.maximum(s_now[cross] - s_new[cross], 1e-300)
                step[cross] *= np.clip(allowed, 0.0, 1.0)[:, None]
        V -= step
    return V


def _edge_counts(F: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in np.asarray(F, dtype=np.int64):
        for e in ((a, b), (b, c), (a, c)):
            counts[(min(e), max(e))] += 1
    return counts


def _collapse_edges(V: np.ndarray, F: np.ndarray, planes, candidates, tol: float = 1e-9, fold_cos: float = 0.2, q_floor: float = 0.05) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse the candidate edges (a, b) that are admissible, in the given order, one collapse per vertex per call.
    A vertex may only be merged into a neighbour lying on every clip plane the vertex lies on (the boundary never
    leaves its planes); vertices on two or more planes are never removed; the link condition (no third common
    neighbour beyond the apexes of the edge's triangles) and a fold-over check (surviving triangles keep their
    orientation) guard the topology."""
    V = np.asarray(V, dtype=np.float64); F = np.asarray(F, dtype=np.int64)
    onp = np.stack([np.abs(V @ n - d) <= tol for _, n, d in planes], axis=1) if planes else np.zeros((len(V), 0), bool)
    nplanes = onp.sum(axis=1)
    v2t: dict[int, list[int]] = defaultdict(list); nbrs: dict[int, set[int]] = defaultdict(set)
    for i, f in enumerate(F):
        for v in f:
            v2t[int(v)].append(i)
        nbrs[int(f[0])].update((int(f[1]), int(f[2]))); nbrs[int(f[1])].update((int(f[0]), int(f[2]))); nbrs[int(f[2])].update((int(f[0]), int(f[1])))
    N0 = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); Q0 = tri_quality(V, F)
    remap = np.arange(len(V)); touched: set[int] = set(); done = 0
    for a, b in candidates:
        a = int(a); b = int(b)
        if a == b or a in touched or b in touched:
            continue
        same = bool((onp[a] == onp[b]).all())
        can_a = (nplanes[a] <= 1 or same) and bool((onp[a] <= onp[b]).all())      # a may vanish into b
        can_b = (nplanes[b] <= 1 or same) and bool((onp[b] <= onp[a]).all())
        if not (can_a or can_b):
            continue
        if can_a and can_b:
            drop, keep = (a, b) if (nplanes[a] < nplanes[b] or (nplanes[a] == nplanes[b] and a > b)) else (b, a)
        elif can_a:
            drop, keep = a, b
        else:
            drop, keep = b, a
        shared = [i for i in v2t[a] if i in set(v2t[b])]
        apexes = {int(v) for i in shared for v in F[i] if int(v) not in (a, b)}
        if (nbrs[a] & nbrs[b]) - apexes:              # link condition
            continue
        ok = True
        for i in v2t[drop]:
            if i in shared:
                continue
            f = [keep if int(v) == drop else int(v) for v in F[i]]
            n1 = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
            if np.linalg.norm(N0[i]) < 1e-18:              # already degenerate (coincident vertices): any collapse is an improvement
                continue
            if n1 @ N0[i] <= fold_cos * np.linalg.norm(n1) * np.linalg.norm(N0[i]):
                ok = False; break
            q1 = float(tri_quality(V, np.asarray([f]))[0])
            if q1 < q_floor and q1 < Q0[i]:            # never turn a triangle into a needle
                ok = False; break
        if not ok:
            continue
        remap[drop] = keep; touched.update((a, b, *apexes)); done += 1
    if done == 0:
        return V, F, 0
    F = remap[F]; F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    V, F = compact(V, F)
    return V, F, done


def collapse_short_edges(V: np.ndarray, F: np.ndarray, planes, min_len: float, max_passes: int = 10, boundary_only: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse every edge shorter than min_len (shortest first) subject to the rules of _collapse_edges."""
    V = np.asarray(V, dtype=np.float64).copy(); F = np.asarray(F, dtype=np.int64).copy(); total = 0
    for _ in range(max_passes):
        if boundary_only:
            onp = np.zeros(len(V), dtype=int)
            for _, n, d in planes:
                onp += (np.abs(V @ n - d) <= 1e-9)
            E = np.asarray([e for e in _directed_boundary(F) if onp[e[0]] > 0 and onp[e[1]] > 0], dtype=np.int64).reshape(-1, 2)
        else:
            E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), axis=1), axis=0)
        if not len(E):
            break
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1); short = np.where(L < min_len)[0]
        if not len(short):
            break
        order = short[np.lexsort((E[short, 1], E[short, 0], L[short]))]
        V, F, done = _collapse_edges(V, F, planes, [tuple(E[i]) for i in order])
        if done == 0:
            break
        total += done
    return V, F, total


def collapse_edges_on_lines(V: np.ndarray, F: np.ndarray, planes, tol: float = 1e-9, max_passes: int = 5) -> tuple[np.ndarray, np.ndarray, int]:
    """A sheet edge whose endpoints lie on the same two planes runs along the line of those planes: the sheet crosses
    such a line at points, never along a segment, so the edge is a snapped sliver; both caps would attach to it and
    make it non-manifold.  Collapse it (the merge stays on the line)."""
    V = np.asarray(V, dtype=np.float64).copy(); F = np.asarray(F, dtype=np.int64).copy(); total = 0
    for _ in range(max_passes):
        onp = np.stack([np.abs(V @ n - d) <= tol for _, n, d in planes], axis=1) if planes else np.zeros((len(V), 0), bool)
        E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), axis=1), axis=0)
        on_line = (onp[E[:, 0]] & onp[E[:, 1]]).sum(axis=1) >= 2
        if not on_line.any():
            break
        L = np.linalg.norm(V[E[on_line, 0]] - V[E[on_line, 1]], axis=1); cand = E[on_line][np.argsort(L)]
        V, F, done = _collapse_edges(V, F, planes, [tuple(e) for e in cand])
        if done == 0:
            break
        total += done
    return V, F, total


def _tri_min_angle_2d(P2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Minimum interior angle (degrees) of planar triangles; 0 for a degenerate one."""
    if not len(T):
        return np.empty(0)
    p = P2[T]; a = np.linalg.norm(p[:, 1] - p[:, 2], axis=1); b = np.linalg.norm(p[:, 0] - p[:, 2], axis=1); c = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    def ang(x, y, z):
        return np.degrees(np.arccos(np.clip((y * y + z * z - x * x) / np.maximum(2 * y * z, 1e-300), -1.0, 1.0)))
    return np.minimum(np.minimum(ang(a, b, c), ang(b, a, c)), ang(c, a, b))


def _repair_cap_needles(P2: np.ndarray, tris: np.ndarray, n_input: int, *, removable: set | None = None, angle_min: float = 2.0, max_passes: int = 4) -> tuple[np.ndarray, dict]:
    """Remove needle triangles of the cap whose apex, or an endpoint of whose long edge, is a vertex the cap owns.

    Triangle's quality refinement ('q20') with no Steiner points allowed on the chains ('YY') cannot split a chain
    segment its circumcenter encroaches and places the point a hair beside the segment instead, and a carrier SUPPORT
    point forced into a shallow intrusion of the band into a carrier triangle lands a hair from the chain likewise:
    a triangle of angle 1e-3 degrees standing on that chain edge, protected from the flip pass, survives into the
    volume mesh, where every tetrahedron built on it is a sliver (all 41 sub-degree tetrahedra of three production
    meshes, 2026-09-05) and a stiffness bomb of up to 11x the converged operator scale.

    A vertex the cap owns (a Steiner point, or a support point listed in `removable` that is not shared with the
    sheet) is collapsed onto another vertex of the needle: the needle vanishes and its fan follows the target.  The
    2D link condition, orientation of every fan triangle and absence of duplicate triangles are required; the target
    leaving the best fan is taken, and a collapse is accepted only when it improves on the needle's angle.  When the
    apex owns no removable vertex it is moved off the long edge into its fan.  Chain vertices, carrier nodes and
    crossings never move.  P2 is modified in place when an apex is moved."""
    tris = np.asarray(tris, dtype=np.int64).copy()
    removed = moved = 0; skipped_input = 0; removable = set(int(k) for k in (removable or ()))

    def area2(T):
        p = P2[T]; return (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])

    def owned(v):
        return v >= n_input or v in removable

    for _ in range(max_passes):
        ang = _tri_min_angle_2d(P2, tris); bad = np.flatnonzero(ang < angle_min)
        if not len(bad):
            break
        sgn = np.sign(area2(tris)); rows = {tuple(sorted(int(v) for v in f)) for f in tris}
        changed: set[int] = set(); progress = 0
        for i in bad[np.argsort(ang[bad])]:
            if int(i) in changed:
                continue
            f = [int(v) for v in tris[i]]; p = P2[f]
            e = [np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[0] - p[1])]; m = int(np.argmax(e))
            apex, ends = f[m], [f[k] for k in range(3) if k != m]
            candidates = [c for c in [apex] + ends if owned(c)]
            if not candidates:
                skipped_input += 1; continue                      # chain vertices, carrier nodes, crossings only
            best = None
            for c in candidates:
                star = [int(s) for s in np.flatnonzero((tris == c).any(axis=1))]
                if changed.intersection(star):
                    continue
                link_c = {int(v) for s in star for v in tris[s] if int(v) != c}
                for t in [v for v in f if v != c]:
                    star_t = np.flatnonzero((tris == t).any(axis=1))
                    link_t = {int(v) for s in star_t for v in tris[s] if int(v) != t}
                    opposite = {int(v) for s in star if t in tris[s] for v in tris[s] if int(v) not in (c, t)}
                    if (link_c & link_t) != opposite:            # 2D link condition
                        continue
                    keep = [s for s in star if t not in tris[s]]
                    new = tris[keep].copy(); new[new == c] = t
                    a2 = area2(new) if len(new) else np.empty(0)
                    if len(new) and (np.any(np.sign(a2) != sgn[keep]) or np.any(np.abs(a2) < 1e-24)):
                        continue
                    if any(tuple(sorted(int(v) for v in r)) in rows for r in new):
                        continue
                    q = float(_tri_min_angle_2d(P2, new).min()) if len(new) else 90.0
                    if best is None or q > best[0]:
                        best = (q, c, t, star, keep, new)
            if best is not None and best[0] > ang[i]:
                q, c, t, star, keep, new = best
                for s in star:
                    rows.discard(tuple(sorted(int(v) for v in tris[s])))
                tris[keep] = new; rows.update(tuple(sorted(int(v) for v in r)) for r in new)
                for s in star:
                    if s not in keep:
                        tris[s] = -1
                changed.update(star); removed += 1; progress += 1
                continue
            if not owned(apex):
                continue
            # move the apex off the long edge, into its fan
            star = [int(s) for s in np.flatnonzero((tris == apex).any(axis=1))]
            if changed.intersection(star):
                continue
            A, Bv = P2[ends[0]], P2[ends[1]]; ab = Bv - A; L = float(np.linalg.norm(ab)); nrm = np.array([-ab[1], ab[0]]) / max(L, 1e-300)
            side = np.sign((P2[apex] - A) @ nrm)
            if side == 0:
                others = [int(v) for s in star for v in tris[s] if int(v) not in (apex, *ends)]
                side = np.sign(np.mean([(P2[o] - A) @ nrm for o in others])) if others else 1.0
            old = P2[apex].copy()
            for frac in (0.3, 0.15, 0.05):
                P2[apex] = old + frac * L * side * nrm
                a2 = area2(tris[star])
                if np.all(np.sign(a2) == sgn[star]) and float(_tri_min_angle_2d(P2, tris[star]).min()) > ang[i]:
                    moved += 1; progress += 1; changed.update(star); break
                P2[apex] = old
        tris = tris[(tris >= 0).all(axis=1)]
        if progress == 0:
            break
    ang = _tri_min_angle_2d(P2, tris) if len(tris) else np.asarray([90.0])
    return tris, {"cap_needle_apex_collapsed": int(removed), "cap_needle_apex_moved": int(moved), "cap_needle_input_apex_skipped": int(skipped_input),
                  "cap_needles_remaining": int((ang < angle_min).sum()), "cap_min_angle_degrees": float(ang.min())}


def _flip_edges(V: np.ndarray, F: np.ndarray, edges, fold_cos: float = 0.5, protected: set | None = None) -> tuple[np.ndarray, int]:
    """Flip interior edges (b, c) -> (a, d) when both new triangles keep the local orientation, improve the worse of the
    two qualities, and the new edge does not exist yet.  Edges in `protected` (chain / constraint segments) stay."""
    F = np.asarray(F, dtype=np.int64).copy()
    e2f: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, f in enumerate(F):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
            e2f[(min(a, b), max(a, b))].append(i)
    q = tri_quality(V, F); changed: set[int] = set(); flipped = 0
    for b, c in edges:
        key = (min(b, c), max(b, c)); tris = e2f.get(key, [])
        if len(tris) != 2 or tris[0] in changed or tris[1] in changed or (protected is not None and key in protected):
            continue
        t1, t2 = tris
        f1 = [int(v) for v in F[t1]]; k = next(j for j in range(3) if f1[j] not in key); a, b1, c1 = f1[k], f1[(k + 1) % 3], f1[(k + 2) % 3]
        d = next(int(v) for v in F[t2] if int(v) not in key)
        if (min(a, d), max(a, d)) in e2f:
            continue
        new = np.asarray([[a, b1, d], [a, d, c1]], dtype=np.int64)
        n_old = np.cross(V[b1] - V[a], V[c1] - V[a]); n_old2 = np.cross(V[F[t2, 1]] - V[F[t2, 0]], V[F[t2, 2]] - V[F[t2, 0]])
        # orient t2's normal like t1's (they share the edge in opposite directions when consistently oriented)
        if n_old2 @ n_old < 0:
            n_old2 = -n_old2
        ref = n_old + n_old2
        n_new = np.cross(V[new[:, 1]] - V[new[:, 0]], V[new[:, 2]] - V[new[:, 0]])
        if np.any(n_new @ ref <= fold_cos * np.linalg.norm(n_new, axis=1) * np.linalg.norm(ref)):
            continue
        if tri_quality(V, new).min() <= min(q[t1], q[t2]):
            continue
        F[t1] = new[0]; F[t2] = new[1]; changed.update((t1, t2)); flipped += 1
        e2f[(min(a, d), max(a, d))] = [t1, t2]
    return F, flipped


def remove_needle_triangles(V: np.ndarray, F: np.ndarray, planes, q_min: float = 0.05, max_passes: int = 3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Triangles below quality q_min (4 sqrt(3) A / sum l^2): collapse their shortest edge; those that cannot collapse
    (both endpoints pinned) have their longest edge flipped when that is admissible."""
    V = np.asarray(V, dtype=np.float64).copy(); F = np.asarray(F, dtype=np.int64).copy(); collapsed = 0; flipped = 0
    for _ in range(max_passes):
        q = tri_quality(V, F); bad = np.where(q < q_min)[0]
        if not len(bad):
            break
        cands = []
        for i in bad[np.argsort(q[bad])]:
            f = [int(v) for v in F[i]]; e = [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]
            cands.append(min(e, key=lambda ab: np.linalg.norm(V[ab[0]] - V[ab[1]])))
        V, F, nc = _collapse_edges(V, F, planes, cands); collapsed += nc
        q = tri_quality(V, F); bad = np.where(q < q_min)[0]
        if not len(bad):
            break
        longest = []
        for i in bad:
            f = [int(v) for v in F[i]]; e = [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])]
            longest.append(max(e, key=lambda ab: np.linalg.norm(V[ab[0]] - V[ab[1]])))
        F, nf = _flip_edges(V, F, longest); flipped += nf
        if nc == 0 and nf == 0:
            break
    q = tri_quality(V, F) if len(F) else np.asarray([1.0])
    return V, F, {"collapsed": int(collapsed), "flipped": int(flipped), "remaining_below": int((q < q_min).sum()), "min_quality": float(q.min())}


def self_intersecting_faces(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Indices of triangles pymeshlab flags as intersecting another non-adjacent triangle (broad phase; the coplanar
    branch of that test over-reports, so use self_intersections for the verified pairs)."""
    import pymeshlab as ml
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(V, dtype=np.float64), face_matrix=np.asarray(F, dtype=np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    return np.where(ms.current_mesh().face_selection_array())[0]


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _proper_overlap_2d(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> bool:
    def inside(p, T):
        sgn = [_cross2(T[(k + 1) % 3] - T[k], p - T[k]) for k in range(3)]
        return (min(sgn) > eps) or (max(sgn) < -eps)
    if any(inside(p, Q) for p in P) or any(inside(q, P) for q in Q):
        return True
    for i in range(3):
        a, b = P[i], P[(i + 1) % 3]
        for j in range(3):
            c, d = Q[j], Q[(j + 1) % 3]
            r = b - a; t_ = d - c; den = _cross2(r, t_)
            if abs(den) < 1e-18:
                continue
            t = _cross2(c - a, t_) / den; u = _cross2(c - a, r) / den
            if eps < t < 1 - eps and eps < u < 1 - eps:
                return True
    return False


def tri_tri_proper(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> bool:
    """True when two triangles intersect in a set of positive length/area (touching at a point or along an edge is not an
    intersection).  Moller's interval test with a coplanar 2D branch."""
    P = np.asarray(P, dtype=np.float64); Q = np.asarray(Q, dtype=np.float64)
    nQ = np.cross(Q[1] - Q[0], Q[2] - Q[0]); nP = np.cross(P[1] - P[0], P[2] - P[0])
    lQ = np.linalg.norm(nQ); lP = np.linalg.norm(nP)
    if lQ < 1e-300 or lP < 1e-300:
        return False
    nQ = nQ / lQ; nP = nP / lP
    dP = (P - Q[0]) @ nQ; dQ = (Q - P[0]) @ nP
    if (dP > eps).all() or (dP < -eps).all() or (dQ > eps).all() or (dQ < -eps).all():
        return False
    if (np.abs(dP) <= eps).all() and (np.abs(dQ) <= eps).all():
        ax = int(np.argmax(np.abs(nQ))); keep = [k for k in range(3) if k != ax]
        return _proper_overlap_2d(P[:, keep], Q[:, keep], eps)
    D = np.cross(nP, nQ)
    if np.linalg.norm(D) < 1e-14:
        return False

    def interval(T, dist):
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if dist[i] * dist[j] < 0:
                t = dist[i] / (dist[i] - dist[j]); pts.append(T[i] + t * (T[j] - T[i]))
            elif abs(dist[i]) <= eps:
                pts.append(T[i])
        if len(pts) < 2:
            return None
        proj = np.asarray(pts) @ D; return proj.min(), proj.max()
    I1 = interval(P, dP); I2 = interval(Q, dQ)
    if I1 is None or I2 is None:
        return False
    lo, hi = max(I1[0], I2[0]), min(I1[1], I2[1])
    if hi - lo <= 1e-10:
        return False
    # the common segment must run through the interior of at least one triangle: two triangles touching along a
    # shared edge (or an edge lying on the other's boundary) have a common segment but do not intersect
    D = D / np.linalg.norm(D); tmid = 0.5 * (lo + hi)

    def strictly_inside(T, nT, point, tol=1e-9):
        b = []
        for k in range(3):
            e = T[(k + 1) % 3] - T[k]; w = point - T[k]
            b.append(np.cross(e, w) @ nT)
        return min(b) > tol * max(abs(x) for x in b + [1e-300])
    # a point of the common line at parameter tmid: rebuild it from P's interval points (any point with proj == tmid)
    pts = []
    for T, dist in ((P, dP), (Q, dQ)):
        for i in range(3):
            j = (i + 1) % 3
            if dist[i] * dist[j] < 0:
                t = dist[i] / (dist[i] - dist[j]); pts.append(T[i] + t * (T[j] - T[i]))
            elif abs(dist[i]) <= eps:
                pts.append(T[i])
    pts = np.asarray(pts); proj = pts @ D; base = pts[int(np.argmin(np.abs(proj - tmid)))]
    point = base + (tmid - base @ D) * D
    return strictly_inside(P, nP, point) or strictly_inside(Q, nQ, point)


def self_intersections(V: np.ndarray, F: np.ndarray, fold_cos: float = -0.7) -> dict:
    """Verified self-intersections: pymeshlab candidates re-tested pairwise with tri_tri_proper (pairs sharing a vertex
    excluded), plus fold-overs (edge-adjacent triangles whose normals point against each other)."""
    V = np.asarray(V, dtype=np.float64); F = np.asarray(F, dtype=np.int64)
    if len(F) == 0:
        return {"pairs": [], "folds": [], "candidates": 0}
    cand = self_intersecting_faces(V, F); pairs = []
    if len(cand):
        lo = V[F[cand]].min(axis=1); hi = V[F[cand]].max(axis=1)
        for p in range(len(cand)):
            for q in range(p + 1, len(cand)):
                if (lo[p] <= hi[q] + 1e-12).all() and (lo[q] <= hi[p] + 1e-12).all():
                    if set(F[cand[p]].tolist()) & set(F[cand[q]].tolist()):
                        continue
                    if tri_tri_proper(V[F[cand[p]]], V[F[cand[q]]]):
                        pairs.append((int(cand[p]), int(cand[q])))
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); N /= np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
    e2f: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, f in enumerate(F):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
            e2f[(min(a, b), max(a, b))].append(i)
    folds = [(int(t[0]), int(t[1])) for t in e2f.values() if len(t) == 2 and N[t[0]] @ N[t[1]] < fold_cos]
    return {"pairs": pairs, "folds": folds, "candidates": int(len(cand))}


class EmptyCellError(ValueError):
    """The clipped sheet has no triangles: the cell carries no material."""


def remove_nonmanifold_and_flat(V: np.ndarray, F: np.ndarray, planes, max_passes: int = 5) -> tuple[np.ndarray, dict]:
    """After remeshing: for edges shared by 3+ triangles, drop the smallest-area offenders until every edge has at most
    two triangles (flat in-plane triangles are handled separately by lift_flat_triangles)."""
    F = np.asarray(F, dtype=np.int64); dropped_flat = 0; dropped_nm = 0
    for _ in range(max_passes):
        e2f: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, f in enumerate(F):
            for a, b in ((0, 1), (1, 2), (0, 2)):
                e2f[(min(f[a], f[b]), max(f[a], f[b]))].append(i)
        bad = [fs for fs in e2f.values() if len(fs) > 2]
        if not bad:
            break
        P = V[F]; area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        drop = set()
        for fs in bad:
            fs_sorted = sorted(fs, key=lambda i: area[i])
            drop.update(fs_sorted[:len(fs) - 2])
        dropped_nm += len(drop); F = F[[i for i in range(len(F)) if i not in drop]]
    return F, {"dropped_flat": dropped_flat, "dropped_nonmanifold": dropped_nm}


def remesh_sheet(V: np.ndarray, F: np.ndarray, *, size: float, iterations: int = 10, feature_degrees: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic explicit remeshing (pymeshlab) of the open sheet surface; boundary vertices stay on the boundary polylines."""
    import pymeshlab as ml
    ms = ml.MeshSet(); ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(V, dtype=np.float64), face_matrix=np.asarray(F, dtype=np.int32)))
    ms.meshing_isotropic_explicit_remeshing(iterations=int(iterations), targetlen=ml.PureValue(float(size)), featuredeg=float(feature_degrees),
                                            adaptive=False, selectedonly=False, splitflag=True, collapseflag=True, swapflag=True, smoothflag=True, reprojectflag=True)
    m = ms.current_mesh(); V2 = np.asarray(m.vertex_matrix(), dtype=np.float64); F2 = np.asarray(m.face_matrix(), dtype=np.int64)
    return compact(V2, F2)


def _face_band_samples(geometry, planes, box_max, src, carrier_n: int, samples_per_cell: int, level_tolerance: float):
    """Symmetric cell-centred lattice of one box face with the band test (|phi| - tau < level_tolerance, retained side).
    Returns (axis, inplane axes, n_u, n_v, U, V (2D coords), P (3D), level, near mask, material mask)."""
    axis, side = {"box_x_min": (0, 0), "box_x_max": (0, 1), "box_y_min": (1, 0), "box_y_max": (1, 1), "box_z_min": (2, 0), "box_z_max": (2, 1)}[src]
    inplane = [k for k in range(3) if k != axis]; spu = samples_per_cell * carrier_n
    n_u = int(round(box_max[inplane[0]] * carrier_n)); n_v = int(round(box_max[inplane[1]] * carrier_n))
    su = (np.arange(n_u * samples_per_cell) + 0.5) / spu; sv = (np.arange(n_v * samples_per_cell) + 0.5) / spu
    U, W = np.meshgrid(su, sv, indexing="ij"); P = np.zeros((U.size, 3)); P[:, axis] = box_max[axis] if side else 0.0; P[:, inplane[0]] = U.ravel(); P[:, inplane[1]] = W.ravel()
    lev = sheet_level(geometry, P); inside = np.ones(len(P), bool)
    for _, n, d in planes:
        inside &= (P @ n - d) <= 1e-9
    return axis, inplane, n_u, n_v, U.ravel(), W.ravel(), P, lev, inside & (lev < level_tolerance), inside & (lev < 0.0)


def carrier_active_set(geometry, layout, active_ports, *, carrier_n: int | None = None, samples: int = 48, interior_margin: float = 0.003, level_tolerance: float = 0.05,
                       support_min_depth: float = 0.05) -> tuple[np.ndarray, list, dict]:
    """Mesh-independent carrier TRIANGLES the material band reaches on each port face (band = |phi| <= tau within
    level_tolerance, retained side of the cut), by barycentric sampling of every carrier trace triangle: these are the
    cap constraints (PSLG) and the support points.  The cut face is a port like the box faces (skin coupling,
    decision of record 2026-09-04): its cap is constrained to the cut_0 carrier trace and its nodes are active by
    the same triangle rule (the trace is cube-symmetric by construction, see cut_carrier).  Returns (shell coords, per-face triangle arrays restricted to the
    touched triangles, {face: {triangle index: material support point well inside the triangle and the band}}).
    carrier_n is accepted for API symmetry with equivariant_active_nodes and unused here."""
    shell_coords, _, per_face = carrier_shell(layout, tuple(active_ports))       # box faces AND the cut face (a port)
    planes = clip_planes(geometry)
    bary = np.asarray([(i / samples, j / samples, 1 - (i + j) / samples) for i in range(samples + 1) for j in range(samples + 1 - i)])
    kept = []; support_points: dict[str, dict[int, np.ndarray]] = {}
    for src, tris in per_face:
        keep = np.zeros(len(tris), bool); pts_face: dict[int, np.ndarray] = {}
        for k, tri in enumerate(tris):
            P = bary @ shell_coords[tri]; lev = sheet_level(geometry, P); inside = np.ones(len(P), bool)
            for _, n, d in planes:
                inside &= (P @ n - d) <= 1e-9
            # active: the band reaches the triangle within `level_tolerance` (mesh independent AND a superset of any
            # mesher's support: a fine port node can only land on a triangle the band actually touches).
            strict = inside & (lev < 0.0) & (bary.min(axis=1) > interior_margin)
            near = inside & (lev < level_tolerance)
            if near.any():
                keep[k] = True
            if strict.any():
                # support point: material sample well inside the triangle and deep in the band (guarantees nonzero weights
                # on all three vertices, hence stiffness on every active carrier node).  Only where the band is at least
                # support_min_depth deep inside this triangle: a point forced into a shallow intrusion of the band (a
                # lens a hair from the chain) makes a needle cap triangle, and every tetrahedron on a needle has
                # shape-function gradients of 1/width, a stiffness bomb of up to 11x the operator scale (2026-09-05).
                tau = -lev[strict]; deep = tau >= support_min_depth
                if deep.any():
                    score = np.minimum(bary[strict].min(axis=1) / 0.33, tau / max(tau.max(), 1e-12)); score[~deep] = -np.inf
                    pts_face[k] = P[strict][int(np.argmax(score))]
        kept.append((src, tris[keep])); support_points[src] = {int(k): v for k, v in pts_face.items()}
    return shell_coords, kept, support_points


def equivariant_active_nodes(geometry, layout, active_ports, *, carrier_n: int, samples_per_cell: int = 32, level_tolerance: float = 0.05) -> set[int]:
    """Active carrier NODES by a rule that is exactly invariant under the 48 cube symmetries and contains every vertex
    of every carrier triangle carrier_active_set marks (same face lattice): a node is active iff one of the closed
    carrier squares containing its location is reached by the band.  Squares and the cell-centred lattice are
    symmetric, unlike the carrier triangulation (each square is split along one diagonal, faces crossed by the cut
    are clipped), and a touched triangle lies in a touched square whose closure holds its vertices.  Grid nodes see
    their four squares, nodes on a grid line (cut-line vertices) the two squares they separate.  Returns layout global
    scalar ids."""
    shell_coords, _, per_face = carrier_shell(layout, tuple(a for a in active_ports if a != "cut_0"))
    planes = clip_planes(geometry); box_max = np.asarray(getattr(geometry, "box_max", (1.0, 1.0, 1.0)), dtype=np.float64)
    active: set[int] = set()
    for src, tris in per_face:
        nodes = np.unique(tris)
        if not len(nodes):
            continue
        axis, inplane, n_u, n_v, U, W, P, lev, near, strict = _face_band_samples(geometry, planes, box_max, src, carrier_n, samples_per_cell, level_tolerance)
        touched = near.reshape(n_u, samples_per_cell, n_v, samples_per_cell).any(axis=(1, 3))
        if not touched.any():
            continue
        uv = shell_coords[nodes][:, inplane] * carrier_n
        for g, (u, v) in zip(nodes, uv):
            us = {int(np.floor(u - 1e-9)), int(np.floor(u + 1e-9))}; vs = {int(np.floor(v - 1e-9)), int(np.floor(v + 1e-9))}
            if any(touched[a, b] for a in us for b in vs if 0 <= a < n_u and 0 <= b < n_v):
                active.add(int(g))
    return active


def geometric_active_mask(geometry, layout, port, *, carrier_n: int) -> np.ndarray:
    """Boolean mask over port.active_global_carrier_ids (compact order): the mesh-independent active carrier set =
    symmetric square rule (equivariant_active_nodes) united with the vertices of the cap-constraint triangles."""
    keys = list(layout.global_scalar_node_keys); g2c = dict(port.global_to_compact)
    lay2c = np.array([g2c.get(str(k), -1) for k in keys]); active = np.zeros(len(port.active_global_carrier_ids), bool)
    ports = list(port.active_global_port_ids)
    nodes = equivariant_active_nodes(geometry, layout, ports, carrier_n=carrier_n)
    # plus the vertices of every barycentrically touched cap-constraint triangle: the mesh port nodes lie on those
    # triangles, so this keeps the geometric set a superset of any mesh support (the few hairline triangles the
    # lattice misses add zero-stiffness rows and are the only non-invariant part of the set)
    for _, tris in carrier_active_set(geometry, layout, ports, carrier_n=carrier_n)[1]:
        nodes.update(int(g) for g in np.unique(tris))
    for g in nodes:
        if lay2c[g] >= 0:
            active[lay2c[g]] = True
    return active


def verify_active_superset(port, active: np.ndarray) -> dict:
    """The geometric active set must contain every carrier node any fine port node interpolates from.  Nothing is
    modified: P keeps its exact partition of unity and affine/rigid reproduction; active nodes that no fine node touches
    simply get zero rows and columns in the Schur operator (no material reaches them)."""
    Sc = port.scalar_fine_to_carrier.tocsr()
    support = np.asarray((Sc != 0).sum(axis=0)).ravel() > 0
    leak = np.where(support & ~active)[0]
    rep = {"active": int(active.sum()), "support": int(support.sum()), "support_outside_active": int(len(leak)),
           "active_without_support": int((active & ~support).sum())}
    if len(leak):
        col = Sc[:, leak]
        rep["max_weight_outside_active"] = float(np.abs(col.data).max()) if col.nnz else 0.0
    return rep


def split_all_t_junctions(V: np.ndarray, F: np.ndarray, labels, tol: float = 1e-9, max_passes: int = 3):
    """Any vertex lying strictly inside an edge of a triangle it does not belong to (a T-junction, e.g. a cap vertex on a
    sheet edge whose split was lost) splits that edge in every incident triangle.  KD-tree search; deterministic."""
    from scipy.spatial import cKDTree
    F = np.asarray(F, dtype=np.int64); labels = list(labels); total = 0
    for _ in range(max_passes):
        E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), axis=1), axis=0)
        tree = cKDTree(V); splits = []
        for a, b in E:
            pa, pb = V[a], V[b]; e = pb - pa; L2 = float(e @ e)
            if L2 < 1e-24:
                continue
            inner = []
            for c in tree.query_ball_point(0.5 * (pa + pb), 0.5 * np.sqrt(L2) + tol):
                if c == a or c == b:
                    continue
                w = V[c] - pa; t = float(w @ e) / L2
                if 1e-9 < t < 1 - 1e-9 and np.linalg.norm(w - t * e) <= tol:
                    inner.append((t, int(c)))
            if inner:
                inner.sort(); splits.append((int(a), int(b), [c for _, c in inner]))
        if not splits:
            break
        F, labels = split_edges_in_triangles(F, labels, splits, V); total += len(splits)
    return F, labels, total


def drop_floating_components(V: np.ndarray, F: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Connected components (by shared edges) of the closed surface that contain no port-face triangle (box face or cut
    face) are loose fragments of the sheet cut off by the clip planes; they carry no load and are removed (they would
    add floating rigid modes).  A fragment attached to the cut face alone is kept: the cut face is a port."""
    comps = _components(F); keep = np.zeros(len(F), bool); dropped = 0; dropped_area = 0.0
    P = V[F]; area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    for c in comps:
        c = np.asarray(c)
        if np.any(labels[c] != "tpms_free"):
            keep[c] = True
        else:
            dropped += 1; dropped_area += float(area[c].sum())
    if dropped:
        V2, F2 = compact(V, F[keep]); labels2 = labels[keep]
        return V2, F2, {"components": len(comps), "dropped": dropped, "dropped_area": dropped_area, "labels": labels2}
    return V, F, {"components": len(comps), "dropped": 0, "dropped_area": 0.0, "labels": labels}


def orient_outward(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Make triangle orientation consistent across shared edges (BFS), then flip globally so the volume is positive."""
    F = F.copy(); nf = len(F)
    edge_tris: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (a, b, c) in enumerate(F):
        for e in ((a, b), (b, c), (c, a)):
            edge_tris[(min(e), max(e))].append(i)
    seen = np.zeros(nf, bool)
    for root in range(nf):
        if seen[root]:
            continue
        seen[root] = True; stack = [root]
        while stack:
            i = stack.pop(); a, b, c = F[i]
            for e in ((a, b), (b, c), (c, a)):
                for j in edge_tris[(min(e), max(e))]:
                    if seen[j]:
                        continue
                    seen[j] = True
                    # neighbour must traverse the shared edge in the opposite direction
                    fj = F[j]; dir_j = [(fj[0], fj[1]), (fj[1], fj[2]), (fj[2], fj[0])]
                    if (e[0], e[1]) in dir_j:
                        F[j] = fj[[0, 2, 1]]
                    stack.append(j)
    for comp in _components(F):          # every closed shell is flipped by its own volume sign (several pieces of material)
        comp = np.asarray(comp)
        if signed_volume(V, F[comp]) < 0:
            F[comp] = F[comp][:, [0, 2, 1]]
    return F


def _snap_crossings_and_flatten_patches(V: np.ndarray, F: np.ndarray, n: np.ndarray, d: float, eps: float, planes, unit_grad: np.ndarray) -> tuple[np.ndarray, dict]:
    """Near-plane treatment of the sheet before its exact clip by the plane n.x = d (retained side n.x - d <= 0),
    decided by the local topology of the surface instead of by distance alone.

    A vertex within eps of the plane is one of two things.  If its one-ring reaches clearly beyond the plane on both
    sides (a neighbour farther than eps on each side) the surface CROSSES the plane there: the vertex is moved onto
    the plane (jointly with any plane it already lies on, displacement capped), so the boundary that the clip
    produces passes through mesh vertices instead of through slivers.  Otherwise the vertex belongs to a patch of
    surface that runs alongside the plane without crossing it (a tangency).  Such patches are handled as a whole,
    by connected component: when the material lies between the patch and the plane (the band level rises away from
    the plane, unit gradient . n < 0) the patch bounds a slab thinner than eps that the volume mesher cannot carry,
    so every vertex of the patch is moved onto the plane and the flat-triangle drop turns the slab into cap; when
    the void lies between, nothing is moved (a thin gap outside the solid is harmless).

    This replaces the blanket "snap everything within eps" of the earlier pipeline, whose two failure modes the
    population census exposed: snapping a vertex whose neighbours stay off the plane pinches the sheet onto the cap
    along an interior edge (an edge with four triangles), and snapping a vertex that merely passes near a face with
    the void in between fabricates a contact with that face where the exact geometry has none (a hole in the cap
    beside it).  Returns (vertices, report)."""
    V = np.asarray(V, dtype=np.float64); s_ = V @ n - d; near = np.abs(s_) < eps
    rep = {"crossing_vertices": 0, "patches_flattened": 0, "patch_vertices_flattened": 0, "patches_left": 0}
    if not near.any():
        return V, rep
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); m = len(V)
    A = coo_matrix((np.ones(2 * len(E)), (np.concatenate([E[:, 0], E[:, 1]]), np.concatenate([E[:, 1], E[:, 0]]))), shape=(m, m)).tocsr()
    A.data[:] = 1.0
    has_out = (A @ (s_ > eps).astype(np.float64)) > 0
    has_in = (A @ (s_ < -eps).astype(np.float64)) > 0
    crossing = near & has_out & has_in
    rep["crossing_vertices"] = int(crossing.sum())
    tangent = near & ~crossing
    flatten = np.zeros(m, bool)
    if tangent.any():
        idx = np.where(tangent)[0]; sub = A[idx][:, idx]
        ncomp, lab = connected_components(sub, directed=False)
        for k in range(ncomp):
            members = idx[lab == k]
            material_between = float(np.mean(unit_grad[members] @ n)) < 0.0
            if material_between:
                flatten[members] = True; rep["patches_flattened"] += 1; rep["patch_vertices_flattened"] += int(len(members))
            else:
                rep["patches_left"] += 1
    allow = crossing | flatten
    # patch vertices may sit farther than eps in the joint direction of a second plane; the cap on the move still holds
    V = snap_near_plane_constrained(V, n, d, eps * (1.0 + 1e-9), planes, allow=allow, max_move=2.0)
    return V, rep


def _flatten_tangent_edges(geometry, V: np.ndarray, F: np.ndarray, n: np.ndarray, d: float, planes, eps: float, max_rounds: int = 8) -> tuple[np.ndarray, int]:
    """The invariant behind the near-plane treatment: no INTERIOR sheet edge lies in a cap plane with the material
    between the sheet and the plane.  Such an edge is where the crossing curve runs tangentially along the surface
    for one edge length (both endpoints are crossing vertices, both incident triangles stay on the retained side): the
    sheet would touch the cap along that edge from inside the material, an edge with four triangles.  With the
    material between, the two incident triangles bound a slab that is as thin as the surface curvature makes it, so
    their third vertices are moved onto the plane too and the flat-triangle drop turns them into cap.  The move is
    accepted only if no triangle around the moved vertex changes orientation (the guard is the fold itself, not a
    distance threshold: a threshold always has a case just beyond it), and a third vertex farther than two snap
    distances is left to the watertight gate.  With the void between, the touch is harmless and nothing moves.
    Iterated: flattening can expose the next edge.  Returns (vertices, number of triangles flattened)."""
    V = V.copy(); total = 0
    v2t: dict[int, list[int]] = defaultdict(list)
    for t, f in enumerate(F):
        for v in f:
            v2t[int(v)].append(t)
    normals = lambda X, tris: np.cross(X[F[tris, 1]] - X[F[tris, 0]], X[F[tris, 2]] - X[F[tris, 0]])
    for _ in range(max_rounds):
        s_ = V @ n - d; onp = np.abs(s_) <= 1e-9
        e2t: dict[tuple[int, int], list[int]] = defaultdict(list)
        for t, f in enumerate(F):
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
                e2t[(min(a, b), max(a, b))].append(t)
        moved = np.zeros(len(V), bool)
        for (a, b), tris in e2t.items():
            if len(tris) != 2 or not (onp[a] and onp[b]):
                continue
            third = [int(v) for t in tris for v in F[t] if v != a and v != b]
            if any(onp[v] for v in third):
                continue                              # one triangle is already flat: the drop removes it
            if any(abs(s_[v]) > 2.0 * eps for v in third):
                continue
            mid = 0.5 * (V[a] + V[b]); g = np.zeros(3)
            for k in range(3):
                e = np.zeros(3); e[k] = 1e-6
                g[k] = float(sheet_level(geometry, (mid + e)[None])[0] - sheet_level(geometry, (mid - e)[None])[0]) / 2e-6
            if float(g @ n) >= 0.0:
                continue                              # void between the sheet and the plane: a harmless touch
            moved[third] = True
        if not moved.any():
            break
        for _guard in range(4):                       # drop the moves that would fold a neighbouring triangle
            # the third vertices sit up to two snap distances off the plane: the near threshold of the snap is widened
            # to that, and the joint move stays capped at three snap distances
            V2 = snap_near_plane_constrained(V, n, d, 2.0 * eps * (1.0 + 1e-9), planes, allow=moved, max_move=1.5)
            bad = np.zeros(len(V), bool)
            for v in np.where(moved)[0]:
                tris = np.asarray(v2t[int(v)]); N0 = normals(V, tris); N1 = normals(V2, tris)
                stays = ~(np.abs(V2[F[tris]] @ n - d) <= 1e-9).all(axis=1)          # triangles that do not become flat
                if (np.einsum("ij,ij->i", N0[stays], N1[stays]) <= 0.0).any():
                    bad[v] = True
            if not bad.any():
                break
            moved &= ~bad
            if not moved.any():
                break
        if not moved.any():
            break
        flat_now = int(((np.abs(V2[F] @ n - d) <= 1e-9).all(axis=1) & ~(np.abs(V[F] @ n - d) <= 1e-9).all(axis=1)).sum())
        if flat_now == 0:
            break
        V = V2; total += flat_now
    return V, total


def _enforce_no_tangent_edges(geometry, V: np.ndarray, F: np.ndarray, planes, eps: float) -> tuple[np.ndarray, np.ndarray, int]:
    """_flatten_tangent_edges on every plane, then the flat-triangle drop; returns (V, F, triangles flattened)."""
    total = 0
    for _, n, d in planes:
        V, k = _flatten_tangent_edges(geometry, V, F, n, d, planes, eps); total += k
    if total:
        flat = np.zeros(len(F), bool)
        for _, n, d in planes:
            flat |= (np.abs(V[F] @ n - d) <= 1e-9).all(axis=1)
        F = F[~flat]; V, F = compact(V, F)
    return V, F, total


def _remesh_and_clip(geometry, V_mc: np.ndarray, F_mc: np.ndarray, planes, size: float, *, margin_factor: float = 1.5,
                     snap_fraction: float = 0.2, collapse_fraction: float = 0.3, needle_quality: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict]:
    """Isotropic remesh of the sheet clipped with a margin outside every plane, then EXACT clipping by the planes.
    The remesher's own boundary (which it does not keep on the planes) is discarded; the boundary of the result comes
    from the clipping, so every boundary vertex lies on its planes to round-off and box corners are exact.  Slivers
    produced by the clipping are removed by short-edge collapse and needle repair, then the vertices are re-projected
    onto the level set within their planes.  Returns the open sheet and a report with the self-intersection count."""
    rep: dict = {"size": float(size)}
    margin = margin_factor * size; V, F = V_mc, F_mc
    for _, n, d in planes:
        V, F = clip_by_plane(V, F, n, d + margin)
    if len(F) == 0:
        raise EmptyCellError("the sheet does not enter the retained region")
    V, F = remesh_sheet(V, F, size=size); rep["remeshed"] = {"vertices": int(len(V)), "triangles": int(len(F))}
    V, _ = project_to_sheet(geometry, V)
    for _, n, d in planes:
        # unit normal of the level set at every vertex (recomputed: clipping renumbers); it tells the near-plane
        # treatment on which side of a tangent patch the material lies (see _snap_crossings_and_flatten_patches)
        g = np.zeros_like(V)
        for k in range(3):
            e = np.zeros(3); e[k] = 1e-6
            g[:, k] = (sheet_level(geometry, V + e) - sheet_level(geometry, V - e)) / 2e-6
        unit_g = g / np.maximum(np.linalg.norm(g, axis=1), 1e-300)[:, None]
        V, rep_snap = _snap_crossings_and_flatten_patches(V, F, n, d, snap_fraction * size, planes, unit_g)
        for key_, val_ in rep_snap.items():
            rep[key_] = rep.get(key_, 0) + val_
        V, F = clip_by_plane(V, F, n, d)
        V, ntan = _flatten_tangent_edges(geometry, V, F, n, d, planes, snap_fraction * size); rep["tangent_edges_flattened"] = rep.get("tangent_edges_flattened", 0) + int(ntan)
    if len(F) == 0:
        raise EmptyCellError("the sheet does not enter the retained region")
    flat = np.zeros(len(F), bool)
    for _, n, d in planes:
        flat |= (np.abs(V[F] @ n - d) <= 1e-9).all(axis=1)
    F = F[~flat]; V, F = compact(V, F); rep["flat_dropped"] = int(flat.sum())
    V, F, ncol = collapse_short_edges(V, F, planes, collapse_fraction * size)
    V = project_to_sheet_in_planes(geometry, V, planes, max_step=0.5 * size)
    V, F, ncol2 = collapse_short_edges(V, F, planes, collapse_fraction * size)
    V = project_to_sheet_in_planes(geometry, V, planes, iters=2, max_step=0.5 * size)
    V, F, _, _ = merge_duplicate_vertices(V, F)       # projections can land two vertices on one point: merge, drop the degenerate triangles
    # the in-plane projections can carry a box-face vertex to within the snap distance of the cut plane (a knife edge
    # inside the face): snap it onto the cut plane too, and collapse every sheet edge that now runs along a plane line
    for name, n, d in planes:
        if name == "cut_0":
            g = np.zeros_like(V)
            for k in range(3):
                e = np.zeros(3); e[k] = 1e-6
                g[:, k] = (sheet_level(geometry, V + e) - sheet_level(geometry, V - e)) / 2e-6
            unit_g = g / np.maximum(np.linalg.norm(g, axis=1), 1e-300)[:, None]
            V, rep_snap = _snap_crossings_and_flatten_patches(V, F, n, d, snap_fraction * size, planes, unit_g)
            V, ntan = _flatten_tangent_edges(geometry, V, F, n, d, planes, snap_fraction * size); rep_snap["tangent_edges_flattened"] = int(ntan)
            rep["late_cut_snap"] = rep_snap
    V, F, nline = collapse_edges_on_lines(V, F, planes); rep["line_edges_collapsed"] = int(nline)
    V, F, ncol3 = collapse_short_edges(V, F, planes, collapse_fraction * size); ncol2 += ncol3
    V, F, needle = remove_needle_triangles(V, F, planes, q_min=needle_quality)   # collapses/flips only: no vertex moves
    F, clean = remove_nonmanifold_and_flat(V, F, planes); V, F = compact(V, F)
    # a collapse or flip among plane vertices can leave a triangle lying in a plane: it belongs to the cap, not the sheet
    flat = np.zeros(len(F), bool)
    for _, n, d in planes:
        flat |= (np.abs(V[F] @ n - d) <= 1e-9).all(axis=1)
    F = F[~flat]; V, F = compact(V, F); rep["flat_dropped_late"] = int(flat.sum())
    # the invariant of the near-plane treatment, enforced after every step that moves or merges vertices
    V, F, ntan = _enforce_no_tangent_edges(geometry, V, F, planes, snap_fraction * size); rep["tangent_edges_flattened"] = int(ntan)
    rep["short_edges_collapsed"] = int(ncol + ncol2); rep["needles"] = needle; rep["nonmanifold_dropped"] = int(clean["dropped_nonmanifold"])
    counts = _edge_counts(F); rep["edges_with_3_or_more_triangles"] = int(sum(1 for c in counts.values() if c > 2))
    offb = [v for v in {x for e in _directed_boundary(F) for x in e} if not any(abs(V[v] @ n - d) <= 1e-9 for _, n, d in planes)]
    rep["boundary_vertices_off_plane"] = int(len(offb))
    sx = self_intersections(V, F)
    rep["self_intersecting_triangles"] = int(len(sx["pairs"]) + len(sx["folds"])); rep["self_intersection_candidates"] = sx["candidates"]; rep["fold_pairs"] = int(len(sx["folds"]))
    q = tri_quality(V, F) if len(F) else np.asarray([1.0])
    rep["vertices"] = int(len(V)); rep["triangles"] = int(len(F)); rep["min_quality"] = float(q.min())
    rep["max_level_residual"] = float(np.abs(sheet_level(geometry, V)).max()) if len(V) else 0.0
    return V, F, rep


@dataclass
class SheetSurfaceResult:
    V: np.ndarray
    F: np.ndarray
    labels: np.ndarray            # per-triangle label: "tpms_free", "cut_0", or a box face name
    carrier_nodes: dict[str, dict[int, int]]  # face -> {layout carrier global id: surface vertex id}
    report: dict = field(default_factory=dict)


def build_sheet_solid_surface(geometry, layout, *, n_per_unit: int = 64, remesh_size: float = 0.02, carrier_n: int = 32, pad: int = 4, cap_max_area: float | None = None,
                              quality: str = "q20", port_constraints: dict | None = None, constrain_carrier_edges: bool = True,
                              remesh_retry_factors: tuple[float, ...] = (1.0, 0.9, 1.12, 0.8), needle_quality: float = 0.05, thickness_scaling: bool = True,
                              cap_feature_fraction: float = 0.2) -> SheetSurfaceResult:
    """Closed, carrier-conforming surface of the sheet solid.

    n_per_unit / remesh_size: marching-cubes grid and isotropic remesh target (both refined automatically for thin walls
    when thickness_scaling is on).  port_constraints: optional {face: (points xyz, segments (pairs into points), keys)}
    replacing the layout carrier trace (multi-cell blocks); keys identify the constraint points in the returned
    carrier_nodes map.  The sheet is remeshed with a margin outside the planes and then clipped exactly
    (_remesh_and_clip); a result with self-intersections or non-manifold edges is retried with remesh_size scaled by
    the next factor, and the build fails closed when every factor fails."""
    t0 = time.perf_counter(); report: dict = {}
    # resolution follows the thinnest wall: t_min = 2 tau_min / max|grad phi| = 2 tau_min / (2 pi sqrt 3) = 0.184 tau_min
    t_min = 2.0 * float(min(geometry.tau_corners)) / (2.0 * np.pi * np.sqrt(3.0))
    n_requested, h_requested = int(n_per_unit), remesh_size
    if thickness_scaling and t_min > 0:
        n_per_unit = max(int(n_per_unit), int(np.ceil(3.0 / t_min)))
        if remesh_size is not None:
            remesh_size = min(float(remesh_size), 0.6 * t_min)
    report["resolution_effective"] = {"n_per_unit": int(n_per_unit), "remesh_size": remesh_size, "requested": {"n_per_unit": n_requested, "remesh_size": h_requested}, "wall_thickness_min": float(t_min)}
    V, F = marching_sheet(geometry, n_per_unit, pad); h = 1.0 / n_per_unit
    report["marching_cubes"] = {"vertices": int(len(V)), "triangles": int(len(F)), "h": h}
    V, res0 = project_to_sheet(geometry, V); report["projection"] = {"max_level_residual_before": res0, "max_level_residual_after": float(np.abs(sheet_level(geometry, V)).max())}
    planes = clip_planes(geometry)
    if 1.5 * max(remesh_retry_factors) * remesh_size >= pad * h:
        raise ValueError(f"marching-cubes pad {pad * h:.4f} is smaller than the remesh margin {1.5 * max(remesh_retry_factors) * remesh_size:.4f}")
    attempts = []; chosen = None
    for factor in remesh_retry_factors:
        size = float(remesh_size) * float(factor)
        try:
            Vr, Fr, rep_r = _remesh_and_clip(geometry, V, F, planes, size, snap_fraction=0.2, needle_quality=needle_quality)
        except EmptyCellError:
            raise
        except Exception as exc:
            attempts.append({"size": size, "error": str(exc)[:200]}); continue
        attempts.append(rep_r)
        if rep_r["self_intersecting_triangles"] == 0 and rep_r["edges_with_3_or_more_triangles"] == 0 and rep_r["boundary_vertices_off_plane"] == 0 and len(Fr):
            chosen = (Vr, Fr); break
    report["remesh_attempts"] = attempts
    if chosen is None:
        raise ValueError(f"sheet remesh failed for every retry factor: {attempts}")
    V, F = chosen; report["sheet_remesh"] = attempts[-1]
    report["flat_sheet_triangles"] = {"dropped": attempts[-1]["flat_dropped"], "lifted": 0}
    V = project_to_sheet_in_planes(geometry, V, planes, iters=2)
    # the projection moves vertices: re-establish the near-plane invariant, then no sheet triangle may lie in a
    # clip plane when the chains are read
    V, F, ntan = _enforce_no_tangent_edges(geometry, V, F, planes, 0.2 * (remesh_size if remesh_size is not None else h))
    flat = np.zeros(len(F), bool)
    for _, n, d in planes:
        flat |= (np.abs(V[F] @ n - d) <= 1e-9).all(axis=1)
    if flat.any():
        F = F[~flat]; V, F = compact(V, F)
    report["flat_sheet_triangles_before_caps"] = int(flat.sum()); report["tangent_edges_flattened_before_caps"] = int(ntan)
    report["clipped"] = {"vertices": int(len(V)), "triangles": int(len(F))}
    # orient the open sheet outward (normals toward increasing level = void) before the caps read the material side
    # of every chain edge from the triangle orientation
    cen = V[F].mean(axis=1); grad = np.zeros_like(cen)
    for k in range(3):
        e = np.zeros(3); e[k] = 1e-6
        grad[:, k] = (sheet_level(geometry, cen + e) - sheet_level(geometry, cen - e)) / 2e-6
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vote = np.einsum("ij,ij->i", N, grad)
    report["sheet_orientation"] = {"outward_triangles": int((vote > 0).sum()), "inward_triangles": int((vote < 0).sum())}
    if (vote < 0).sum() > (vote > 0).sum():
        F = F[:, [0, 2, 1]]
    labels = ["tpms_free"] * len(F)

    def is_material(P3: np.ndarray) -> np.ndarray:
        ok = sheet_level(geometry, P3) < 0.0
        for _, n2, d2 in planes:
            ok &= (P3 @ n2 - d2) <= 1e-9
        return ok

    def clipped_level(P3: np.ndarray) -> np.ndarray:
        """Signed level that is positive outside any retained half-space (used for the material-side test of chain edges)."""
        lev = sheet_level(geometry, P3)
        outside = np.zeros(len(P3), bool)
        for _, n2, d2 in planes:
            outside |= (P3 @ n2 - d2) > 1e-9
        return np.where(outside, np.abs(lev) + 1.0, lev)

    active = list(geometry.active_outer_port_sources())
    if port_constraints is None:
        shell_coords, per_face, support_points = carrier_active_set(geometry, layout, active, carrier_n=carrier_n)
        port_constraints = {}; report["active_carrier_nodes"] = {}
        for src, tris_c in per_face:
            used = np.unique(tris_c); local = {int(g): k for k, g in enumerate(used)}
            edges = sorted({(min(local[int(a)], local[int(b)]), max(local[int(a)], local[int(b)])) for f in tris_c for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2]))})
            pts = [shell_coords[g] for g in used] + [support_points[src][k] for k in sorted(support_points[src])]
            keys = [int(g) for g in used] + [-1 - k for k in sorted(support_points[src])]  # negative keys: support points
            port_constraints[src] = (np.asarray(pts), np.asarray(edges, dtype=np.int64), keys)
            report["active_carrier_nodes"][src] = int(len(used))
    carrier_nodes: dict[str, dict[int, int]] = {}; caps = {}; classification: dict[str, dict] = {}
    feature_size = cap_feature_fraction * float(remesh_size if remesh_size is not None else h)
    report["cap_feature_size"] = feature_size
    for name, n, d in planes:
        chains, closed = boundary_chains_on_plane(V, F, n, d); cls_rep: dict = {}; classification[name] = cls_rep
        if name in port_constraints and len(port_constraints[name][0]):
            cpts, cedges, ckeys = port_constraints[name]
            # chain vertices that also lie on another clip plane may not move (they belong to that plane's cap too)
            chain_v = sorted({i for ch in chains for i in ch}); fixed = set(); fixed_t = set()
            cpts_arr = np.asarray(cpts).reshape(-1, 3)
            for other, n2, d2 in planes:
                if other != name:
                    if chain_v:
                        on = np.abs(V[chain_v] @ n2 - d2) <= 1e-9; fixed.update(int(chain_v[k]) for k in np.where(on)[0])
                    on_t = np.abs(cpts_arr @ n2 - d2) <= 1e-9; fixed_t.update(int(k) for k in np.where(on_t)[0])
            fixed_t.update(k for k, key in enumerate(ckeys) if isinstance(key, (int, np.integer)) and int(key) < 0)   # support points: never snap targets
            v2t: dict[int, list[int]] = defaultdict(list)
            if chain_v:
                chain_set = set(chain_v)
                for t_, f_ in enumerate(F):
                    for v_ in f_:
                        if int(v_) in chain_set:
                            v2t[int(v_)].append(t_)
            def can_move(g, xyz, _V=V, _F=F, _v2t=v2t):
                """No incident sheet triangle may fold (normal turning by more than ~78 degrees) or degenerate."""
                for t_ in _v2t.get(int(g), ()):
                    f_ = _F[t_]; P_ = _V[f_].copy(); n0 = np.cross(P_[1] - P_[0], P_[2] - P_[0])
                    P_[list(f_).index(int(g))] = xyz; n1 = np.cross(P_[1] - P_[0], P_[2] - P_[0])
                    if np.linalg.norm(n1) < 1e-14 or (n0 @ n1) <= 0.2 * np.linalg.norm(n0) * np.linalg.norm(n1):
                        return False
                return True
            new_xyz, tris_g, cids, splits, merged = triangulate_plane_region(V, chains, closed, n, d, is_material, level_fn=clipped_level, constraint_points=np.asarray(cpts).reshape(-1, 3),
                                                                       constraint_segments=(np.asarray(cedges).reshape(-1, 2) if (len(cedges) and constrain_carrier_edges) else None), max_area=cap_max_area, quality=quality,
                                                                       classification_report=cls_rep, unseeded_material=False, feature_size=feature_size, fixed_vertices=fixed, can_move=can_move, fixed_targets=fixed_t,
                                                                       removable_points=[k for k, key in enumerate(ckeys) if isinstance(key, (int, np.integer)) and int(key) < 0])
            carrier_nodes[name] = {g: int(c) for g, c in zip(ckeys, cids) if not (isinstance(g, (int, np.integer)) and int(g) < 0)}
        elif not chains:
            caps[name] = {"chains": 0, "open_chains": 0, "triangles": 0, "area": 0.0, "edge_splits": 0}; continue
        else:
            poly = _retained_facet_polygon(planes, name, tuple(getattr(geometry, "box_max", (1.0, 1.0, 1.0))))
            segs = np.asarray([(k, (k + 1) % len(poly)) for k in range(len(poly))]) if len(poly) else None
            new_xyz, tris_g, _, splits, merged = triangulate_plane_region(V, chains, closed, n, d, is_material, level_fn=clipped_level, constraint_points=poly if len(poly) else None,
                                                                    constraint_segments=segs, max_area=cap_max_area, quality=quality, classification_report=cls_rep)
        V = np.vstack([V, new_xyz]) if len(new_xyz) else V
        F, labels = split_edges_in_triangles(F, labels, splits, V)
        F = np.vstack([F, tris_g]); labels += [name] * len(tris_g)
        if merged:  # chain vertices merged inside the cap: merge them on the sheet too
            rm = np.arange(len(V))
            for g_old, g_new in merged.items():
                rm[g_old] = g_new
            F = rm[F]; ok_tri = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2]); F = F[ok_tri]; labels = [l for l, o in zip(labels, ok_tri) if o]
        P = V[tris_g]; area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum() if len(tris_g) else 0.0
        caps[name] = {"chains": len(chains), "open_chains": int(sum(1 for c in closed if not c)), "triangles": int(len(tris_g)), "area": float(area), "edge_splits": len(splits)}
    report["caps"] = caps; report["cap_classification"] = {k: v for k, v in classification.items() if v}
    report["cap_needles"] = {"remaining": int(sum(v.get("cap_needles_remaining", 0) for v in classification.values())),
                             "min_angle_degrees": float(min([v.get("cap_min_angle_degrees", 90.0) for v in classification.values()] or [90.0])),
                             "apex_collapsed": int(sum(v.get("cap_needle_apex_collapsed", 0) for v in classification.values())),
                             "apex_moved": int(sum(v.get("cap_needle_apex_moved", 0) for v in classification.values()))}
    report["cap_classification_conflicts"] = int(sum(v.get("conflicts", 0) for v in classification.values()))
    report["cap_level_disagreements"] = int(sum(v.get("level_disagreements", 0) for v in classification.values()))
    F, labels, nfix2 = split_all_t_junctions(V, F, labels); report["t_junction_splits"] = int(nfix2)
    labels_arr = np.asarray(labels)
    V, F, remap, keep_mask = merge_duplicate_vertices(V, F); labels_arr = labels_arr[keep_mask]; report["degenerate_triangles_dropped"] = int((~keep_mask).sum())
    carrier_nodes = {face: {g: remap(c) for g, c in m.items()} for face, m in carrier_nodes.items()}  # provisional (see below)
    V, F, comp = drop_floating_components(V, F, labels_arr); labels_arr = comp.pop("labels"); report["components"] = comp
    F = orient_outward(V, F)
    wt = watertight_report(F); report["watertight"] = wt
    sx_final = self_intersections(V, F)
    report["self_intersecting_triangles"] = int(len(sx_final["pairs"])); report["self_intersection_candidates"] = sx_final["candidates"]
    report["closed_surface_fold_pairs"] = int(len(sx_final["folds"]))     # knife edges (cap against sheet) legitimately look like folds
    if sx_final["pairs"]:
        report["self_intersection_examples"] = [{"pair": list(pq), "labels": [str(labels_arr[pq[0]]), str(labels_arr[pq[1]])], "xyz": V[F[pq[0]]].round(5).tolist()} for pq in sx_final["pairs"][:6]]
    if wt["non_two_manifold_edges"]:
        e2f = defaultdict(list)
        for i, f in enumerate(F):
            for a, b in ((0, 1), (1, 2), (0, 2)):
                e2f[(min(f[a], f[b]), max(f[a], f[b]))].append(i)
        bad = [(e, [labels_arr[i] for i in fs]) for e, fs in e2f.items() if len(fs) != 2]
        report["non_manifold_examples"] = [{"edge": [int(e[0]), int(e[1])], "count": len(l), "labels": l, "xyz": V[list(e)].round(6).tolist()} for e, l in bad[:12]]
    outside = np.zeros(len(V), bool)
    for _, n, d in planes:
        outside |= (V @ n - d) > 1e-12
    report["vertices_outside_box"] = int(outside.sum())
    from .conforming_port_mesh import tri_quality
    if len(F) == 0:
        raise ValueError("sheet solid surface is empty: the cell has no material at this resolution (use finer n_per_unit / remesh_size)")
    q = tri_quality(V, F); report["surface_quality"] = {"min": float(q.min()), "p01": float(np.quantile(q, 0.01)), "below_0.3": int((q < 0.3).sum()), "triangles": int(len(F))}
    report["surface_quality_by_label"] = {lab: {"min": float(q[labels_arr == lab].min()), "below_0.3": int((q[labels_arr == lab] < 0.3).sum()), "n": int((labels_arr == lab).sum())}
                                          for lab in sorted(set(labels_arr.tolist())) if (labels_arr == lab).any()}
    # carrier identity is geometric, not index based: every topology step above renumbers vertices, so the map from
    # carrier node to surface vertex is rebuilt by coordinate here and is the conformity certificate of the cap.
    lookup = {tuple(np.round(p, 9)): i for i, p in enumerate(V)}
    carrier_nodes = {}; missing = 0
    for src, (pts, _edges, keys) in port_constraints.items():
        found: dict[int, int] = {}
        for key, point in zip(keys, np.asarray(pts).reshape(-1, 3)):
            if isinstance(key, (int, np.integer)) and int(key) >= 0:
                idx = lookup.get(tuple(np.round(point, 9)))
                if idx is None:
                    missing += 1          # the band only grazes this carrier triangle: its node carries no material
                else:
                    found[int(key)] = int(idx)
        carrier_nodes[src] = found
    report["carrier_nodes_on_surface"] = {src: len(m) for src, m in carrier_nodes.items()}
    report["carrier_nodes_not_on_surface"] = int(missing)
    report["volume"] = float(signed_volume(V, F)); report["seconds"] = time.perf_counter() - t0
    report["active_ports"] = [a for a in active if caps.get(a, {}).get("triangles", 0) > 0]
    return SheetSurfaceResult(V=V, F=F, labels=labels_arr, carrier_nodes=carrier_nodes, report=report)


def write_off(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    with open(path, "w") as fh:
        fh.write(f"OFF\n{len(V)} {len(F)} 0\n"); np.savetxt(fh, V, fmt="%.17g"); np.savetxt(fh, np.column_stack([np.full(len(F), 3), F]), fmt="%d")


@dataclass
class SheetMeshResult:
    mesh_path: Path
    node_count: int
    tet_count: int
    material_volume: float
    min_dihedral_degrees: float
    tets_below_5deg: int
    tets_below_10deg: int
    seconds: float
    carrier_nodes_preserved: int
    algorithm3d: int = 0
    nonmanifold_facets: int = 0       # volume facets with more than two tetrahedra (a boundary-recovery failure); 0 in a valid mesh
    sliver_smooth: dict | None = None  # boundary-locked sliver perturbation (smooth_slivers), opt-in: it trades slivers on sound bases for flat tetrahedra on the port face, which are worse for the operator (A/B 2026-09-05)


MESH_GENERATION_THREADS = 1     # HXT is deterministic for a fixed thread count only: pinned, so the mesh is byte-identical everywhere



_TET_FACES = np.asarray([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])
_DIRS26 = np.asarray([d for d in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3) if np.any(d)], dtype=np.float64)
_DIRS26 /= np.linalg.norm(_DIRS26, axis=1)[:, None]


def _tet_min_dihedral(P: np.ndarray) -> np.ndarray:
    """P: (n, 4, 3).  Minimum dihedral angle in degrees; -1 for a non-positive volume."""
    vol = np.einsum("ij,ij->i", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])) / 6.0
    N = []
    for k in range(4):
        a, b, c = P[:, _TET_FACES[k, 0]], P[:, _TET_FACES[k, 1]], P[:, _TET_FACES[k, 2]]
        n = np.cross(b - a, c - a); n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        n[np.einsum("ij,ij->i", n, P[:, k] - a) > 0] *= -1.0
        N.append(n)
    out = np.full(len(P), 180.0)
    for i in range(4):
        for j in range(i + 1, 4):
            out = np.minimum(out, 180.0 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", N[i], N[j]), -1, 1))))
    out[vol <= 0] = -1.0
    return out


def smooth_slivers(V: np.ndarray, T: np.ndarray, *, theta: float = 5.0, sweeps: int = 4) -> tuple[np.ndarray, dict]:
    """Sliver perturbation with every surface vertex locked: the interior vertex of each tetrahedron below `theta`
    degrees is moved to the position maximising the minimum dihedral angle over its star (pattern search over the 26
    lattice directions with a halving step), connectivity unchanged.  Every sliver Gmsh leaves in a sheet-solid mesh
    has three vertices on a boundary cap triangle and one interior vertex almost in that plane (2026-09-05), so this is
    the whole remedy for the ones a needle-free cap still admits.  Returns the new coordinates and a report."""
    V = np.asarray(V, dtype=np.float64).copy(); T = np.asarray(T, dtype=np.int64)
    faces = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), axis=1)
    u, cnt = np.unique(faces, axis=0, return_counts=True)
    on_surface = np.zeros(len(V), bool); on_surface[np.unique(u[cnt == 1])] = True
    inc: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(T):
        for v in t:
            inc[int(v)].append(i)
    d = _tet_min_dihedral(V[T]); before = {"min_dihedral_degrees": float(d.min()), "tets_below_5deg": int((d < 5).sum()), "tets_below_1deg": int((d < 1).sum())}
    moved_total = 0
    for _ in range(sweeps):
        cand = sorted({int(v) for t in np.flatnonzero(d < theta) for v in T[t] if not on_surface[v]})
        if not cand:
            break
        moved = 0
        for v in cand:
            star = np.asarray(inc[v]); Tt = T[star]; local = (Tt == v); Pst = V[Tt]
            cur = float(_tet_min_dihedral(Pst).min())
            L = float(np.median(np.linalg.norm(Pst.reshape(-1, 3) - V[v], axis=1))); h = 0.5 * L
            best, best_x = cur, V[v].copy()
            for _it in range(24):
                X = best_x + h * _DIRS26
                Pc = np.repeat(Pst[None], len(X), axis=0); Pc[:, local] = X[:, None, :]
                vals = _tet_min_dihedral(Pc.reshape(-1, 4, 3)).reshape(len(X), -1).min(axis=1); k = int(np.argmax(vals))
                if vals[k] > best + 1e-9:
                    best, best_x = float(vals[k]), X[k].copy()
                else:
                    h *= 0.5
                if h < 1e-7 * max(L, 1e-12):
                    break
            if best > cur + 1e-9:
                V[v] = best_x; moved += 1
        moved_total += moved
        d = _tet_min_dihedral(V[T])
        if moved == 0:
            break
    after = {"min_dihedral_degrees": float(d.min()), "tets_below_5deg": int((d < 5).sum()), "tets_below_1deg": int((d < 1).sum())}
    return V, {"before": before, "after": after, "vertices_moved": int(moved_total), "inverted": int((d < 0).sum()), "theta": theta, "sweeps": sweeps}


def mesh_sheet_solid(surface: SheetSurfaceResult, output_path: Path, *, interior_size_max: float = 0.05, algorithm3d: int = 10,
                     netgen_optimize: bool = True, optimize_passes: int = 2, threads: int = 8, isolate: bool = True, sliver_smooth: bool = False) -> SheetMeshResult:
    """Gmsh volume mesh of the closed surface.  With isolate=True Gmsh runs in a child process, so a Gmsh crash is
    reported as a RuntimeError instead of killing the production worker.

    Algorithm of record (2026-09-05): HXT (algorithm3d = 10).  Gmsh's legacy Delaunay (1) overlaps tetrahedra on
    ordinary, well-shaped cap triangles (pop_cut_0245: a cut-cap triangle of quality 0.15 with four tetrahedra on it)
    and segfaults on two of 60 cut cells even after the carrier and cap feature-size invariants removed every small
    feature; HXT meshed every one of those surfaces with the same quality (min dihedral 0.149 vs 0.149, 128 vs 129
    sub-5 degree tetrahedra on pop_cut_0003).  HXT's result depends on the thread count (and only on it: two runs at
    the same count are byte-identical), so generation runs with MESH_GENERATION_THREADS regardless of `threads`.
    A nonmanifold or crashed mesh fails closed: there is no retry."""
    if isolate:
        import json, subprocess, sys, tempfile
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "surface.npz"; result_path = Path(td) / "result.json"
            np.savez(state, V=surface.V, F=surface.F, labels=surface.labels, carrier_json=json.dumps({k: {str(g): int(v) for g, v in m.items()} for k, m in surface.carrier_nodes.items()}))
            code = ("import json, sys, numpy as np\nfrom pathlib import Path\n"
                    f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
                    f"from {__name__} import SheetSurfaceResult, mesh_sheet_solid\n"
                    f"z = np.load({str(state)!r}, allow_pickle=False)\n"
                    "cn = {k: {int(g): v for g, v in m.items()} for k, m in json.loads(str(z['carrier_json'])).items()}\n"
                    "surf = SheetSurfaceResult(V=z['V'], F=z['F'], labels=z['labels'].astype(str), carrier_nodes=cn)\n"
                    f"m = mesh_sheet_solid(surf, Path({str(output_path)!r}), interior_size_max={float(interior_size_max)!r}, algorithm3d={int(algorithm3d)}, "
                    f"netgen_optimize={bool(netgen_optimize)}, optimize_passes={int(optimize_passes)}, threads={int(threads)}, isolate=False, sliver_smooth={bool(sliver_smooth)})\n"
                    f"Path({str(result_path)!r}).write_text(json.dumps({{k: (str(v) if isinstance(v, Path) else v) for k, v in m.__dict__.items()}}))\n")
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            if proc.returncode != 0 or not result_path.exists():
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
                raise RuntimeError(f"Gmsh worker failed (return code {proc.returncode}): " + " | ".join(tail))
            payload = json.loads(result_path.read_text()); payload["mesh_path"] = Path(payload["mesh_path"])
            if payload.get("nonmanifold_facets", 0):
                # Gmsh Delaunay's boundary recovery overlaps tetrahedra when the surface carries features much smaller
                # than the sheet triangles (a carrier merge radius of 1/10 h produced 0.003 cap edges next to 0.03 sheet
                # triangles: 11 of 55 cut cells, 2026-09-05).  The root fix is upstream (carrier merge radius 1/3 h and
                # the cap feature-size invariant); a nonmanifold mesh is a defect and fails closed here.
                raise RuntimeError(f"Gmsh (algorithm {algorithm3d}) produced {payload['nonmanifold_facets']} nonmanifold volume facets")
            return SheetMeshResult(**payload)
    import gmsh
    t0 = time.perf_counter(); V, F, labels = surface.V, surface.F, surface.labels
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.logger.start()
    try:
        gmsh.model.add("sheet_solid")
        node_tag = {}; tag_count = 0; tag = 0
        # one volume per connected shell: a cut can leave several separate pieces of material, each a closed surface
        comps = _components(F)
        for comp in comps:
            comp = np.sort(np.asarray(comp, dtype=np.int64)); surf_tags = []   # original triangle order: node insertion order (hence the Gmsh result) is unchanged for one shell
            for lab in sorted(set(labels[comp].tolist())):
                tris = F[comp[labels[comp] == lab]]; tag += 1; gmsh.model.addDiscreteEntity(2, tag)
                used = np.unique(tris); new_ids = []; new_xyz = []
                for v in used:
                    if int(v) not in node_tag:
                        tag_count += 1; node_tag[int(v)] = tag_count; new_ids.append(tag_count); new_xyz.extend(V[v].tolist())
                if new_ids:
                    gmsh.model.mesh.addNodes(2, tag, new_ids, new_xyz)
                gmsh.model.mesh.addElementsByType(tag, 2, [], [node_tag[int(v)] for f in tris for v in f]); surf_tags.append(tag)
            loop = gmsh.model.geo.addSurfaceLoop(surf_tags); gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm3d); gmsh.option.setNumber("Mesh.MeshSizeMax", interior_size_max)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1); gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3); gmsh.option.setNumber("General.NumThreads", MESH_GENERATION_THREADS)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if netgen_optimize else 0)
        gmsh.model.mesh.generate(3)
        for _ in range(optimize_passes):
            gmsh.model.mesh.optimize("", niter=1)
        if netgen_optimize:
            gmsh.model.mesh.optimize("Netgen", niter=2)
        ntags, nxyz, _ = gmsh.model.mesh.getNodes(); nxyz = np.asarray(nxyz).reshape(-1, 3)
        order = {int(t): i for i, t in enumerate(ntags)}
        _, _, enodes = gmsh.model.mesh.getElements(3); log = gmsh.logger.get(); gmsh.logger.stop()
        if not enodes or len(enodes[0]) == 0:
            raise RuntimeError("Gmsh produced no tetrahedra; log tail: " + " | ".join(log[-12:]))
        tets = np.vectorize(order.get)(np.asarray(enodes[0]).reshape(-1, 4))
        preserved = 0
        for face, m in surface.carrier_nodes.items():
            for g, vid in m.items():
                if int(vid) < 0 or int(vid) not in node_tag:
                    continue
                i = order[node_tag[int(vid)]]
                if np.allclose(nxyz[i], V[vid], atol=1e-12):
                    preserved += 1
                else:
                    raise RuntimeError(f"Gmsh moved carrier node {g} on {face}")
    finally:
        gmsh.finalize()
    P = nxyz[tets]; vol = np.einsum("ij,ij->i", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])) / 6.0
    neg = vol < 0; tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy(); vol = np.abs(vol)
    sliver = None
    if sliver_smooth and len(tets):
        nxyz, sliver = smooth_slivers(nxyz, tets)
        if sliver["inverted"]:
            raise RuntimeError(f"sliver perturbation inverted {sliver['inverted']} tetrahedra")
        P = nxyz[tets]; vol = np.abs(np.einsum("ij,ij->i", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])) / 6.0)
    dih = _min_dihedral(nxyz, tets)
    faces = np.sort(np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    nonmanifold = int((np.unique(faces, axis=0, return_counts=True)[1] > 2).sum())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("MeshVersionFormatted 2\nDimension 3\n")
        fh.write(f"Vertices\n{len(nxyz)}\n"); np.savetxt(fh, np.column_stack([nxyz, np.zeros(len(nxyz), int)]), fmt="%.17g %.17g %.17g %d")
        fh.write(f"Tetrahedra\n{len(tets)}\n"); np.savetxt(fh, np.column_stack([tets + 1, np.ones(len(tets), int)]), fmt="%d %d %d %d %d")
        fh.write("End\n")
    return SheetMeshResult(mesh_path=output_path, node_count=int(len(nxyz)), tet_count=int(len(tets)), material_volume=float(vol.sum()),
                           min_dihedral_degrees=float(dih.min()), tets_below_5deg=int((dih < 5).sum()), tets_below_10deg=int((dih < 10).sum()),
                           seconds=time.perf_counter() - t0, carrier_nodes_preserved=preserved, algorithm3d=int(algorithm3d), nonmanifold_facets=nonmanifold,
                           sliver_smooth=sliver)
