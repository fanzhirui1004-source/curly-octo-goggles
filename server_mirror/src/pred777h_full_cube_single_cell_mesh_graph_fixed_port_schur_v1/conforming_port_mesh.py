"""Boundary-conforming fixed-port volume meshing (step 2).

The physical port faces are triangulated *exactly* by the deterministic carrier
trace layout, so the fine port nodes are the carrier nodes and the fixed-port
map ``u_fine = P q`` is a pure selection.  The remaining boundary (TPMS sheets,
inner collar planes) is taken from the coherent polyhedral domain surface; the
interior is meshed by Gmsh with the boundary mesh preserved.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


_BOX_PLANES = (("box_x_min", 0, 0.0), ("box_x_max", 0, 1.0), ("box_y_min", 1, 0.0), ("box_y_max", 1, 1.0), ("box_z_min", 2, 0.0), ("box_z_max", 2, 1.0))


def read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    if rows[0][0] != "OFF":
        raise ValueError("not an OFF file")
    nv, nf = int(rows[1][0]), int(rows[1][1])
    vertices = np.asarray([[float(x) for x in r[:3]] for r in rows[2:2 + nv]], dtype=np.float64)
    faces = np.asarray([[int(x) for x in r[1:4]] for r in rows[2 + nv:2 + nv + nf]], dtype=np.int64)
    return vertices, faces


def _port_planes(geometry) -> list[tuple[str, np.ndarray, float]]:
    planes = [(name, np.eye(3)[axis], value) for name, axis, value in _BOX_PLANES]
    if geometry.cut_plane is not None:
        p = np.asarray(geometry.cut_plane, dtype=np.float64)
        planes.append(("cut_0", p[:3] / np.linalg.norm(p[:3]), float(p[3]) / np.linalg.norm(p[:3])))
    return planes


def carrier_shell(layout, active_ports: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]:
    """Outer shell: every active port face triangulated by its carrier trace, in global carrier node ids."""

    coords = np.asarray(layout.global_scalar_node_coordinates, dtype=np.float64)
    per_face: list[tuple[str, np.ndarray]] = []
    for trace in layout.local_traces:
        src = str(trace.source_id)
        if src not in active_ports:
            continue
        l2g = layout.local_to_global_scalar[src].tocsr()
        local_to_global = np.empty(l2g.shape[0], dtype=np.int64)
        for i in range(l2g.shape[0]):
            cols = l2g.indices[l2g.indptr[i]:l2g.indptr[i + 1]]
            if len(cols) != 1:
                raise ValueError("trace node maps to more than one global carrier node")
            local_to_global[i] = int(cols[0])
        if not np.allclose(coords[local_to_global], np.asarray(trace.node_coordinates), atol=1e-12):
            raise ValueError(f"trace {src} coordinates disagree with the global carrier")
        per_face.append((src, local_to_global[np.asarray(trace.triangles, dtype=np.int64)]))
    tris = np.vstack([t for _, t in per_face])
    return coords, tris, per_face


def refine_shell(coords: np.ndarray, per_face: list[tuple[str, np.ndarray]], levels: int) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Uniform nested midpoint refinement of the carrier shell (shared edges refined identically)."""

    coords = np.asarray(coords, dtype=np.float64).copy()
    for _ in range(int(levels)):
        mid: dict[tuple[int, int], int] = {}
        new_coords = [coords]
        extra: list[np.ndarray] = []
        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in mid:
                mid[key] = len(coords) + len(extra); extra.append(0.5 * (coords[a] + coords[b]))
            return mid[key]
        refined = []
        for src, tris in per_face:
            out = []
            for a, b, c in tris:
                ab, bc, ca = midpoint(int(a), int(b)), midpoint(int(b), int(c)), midpoint(int(c), int(a))
                out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
            refined.append((src, np.asarray(out, dtype=np.int64)))
        coords = np.vstack([coords, np.asarray(extra)]) if extra else coords
        per_face = refined
    return coords, per_face


def watertight_report(tris: np.ndarray) -> dict[str, int]:
    e2f: dict[tuple[int, int], int] = defaultdict(int)
    for f in tris:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            e2f[(min(f[a], f[b]), max(f[a], f[b]))] += 1
    bad = sum(1 for v in e2f.values() if v != 2)
    return {"edges": len(e2f), "non_two_manifold_edges": bad}


def _components(tris: np.ndarray) -> list[np.ndarray]:
    e2f: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, f in enumerate(tris):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            e2f[(min(f[a], f[b]), max(f[a], f[b]))].append(i)
    adj: dict[int, set[int]] = defaultdict(set)
    for owners in e2f.values():
        for a in owners:
            adj[a].update(owners)
    seen = np.zeros(len(tris), bool); out = []
    for s in range(len(tris)):
        if seen[s]:
            continue
        stack = [s]; seen[s] = True; comp = []
        while stack:
            x = stack.pop(); comp.append(x)
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True; stack.append(y)
        out.append(np.asarray(comp))
    return out


def signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    P = V[F]
    return float(np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6.0)


def remesh_cavity_surfaces(V: np.ndarray, inner: np.ndarray, *, size: float, sharp_angle_degrees: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Remesh the closed inner (cavity) surfaces with Gmsh on their own discrete parametrization.

    The new vertices lie on the original polyhedral triangles (deviation at round-off), so the
    domain geometry is unchanged; only the surface triangulation quality changes.
    """

    import gmsh

    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("cavities")
        gmsh.model.addDiscreteEntity(2, 1)
        used = np.unique(inner); remap = {int(v): i + 1 for i, v in enumerate(used)}
        gmsh.model.mesh.addNodes(2, 1, [remap[int(v)] for v in used], V[used].ravel().tolist())
        gmsh.model.mesh.addElementsByType(1, 2, [], [remap[int(v)] for f in inner for v in f])
        gmsh.model.mesh.classifySurfaces(np.radians(sharp_angle_degrees), True, False, np.radians(180.0))
        gmsh.model.mesh.createGeometry()
        for key, value in (("Mesh.MeshSizeMax", size), ("Mesh.MeshSizeMin", 0.4 * size), ("Mesh.MeshSizeExtendFromBoundary", 0),
                           ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Algorithm", 6), ("Mesh.Smoothing", 5)):
            gmsh.option.setNumber(key, value)
        gmsh.model.mesh.clear(); gmsh.model.mesh.generate(2)
        ntags, nxyz, _ = gmsh.model.mesh.getNodes(); nxyz = np.asarray(nxyz).reshape(-1, 3); order = {int(t): i for i, t in enumerate(ntags)}
        _, _, enodes = gmsh.model.mesh.getElements(2)
        tris = np.vstack([np.vectorize(order.get)(np.asarray(e).reshape(-1, 3)) for e in enodes])
    finally:
        gmsh.finalize()
    if watertight_report(tris)["non_two_manifold_edges"]:
        raise ValueError("remeshed cavity surface is not watertight")
    return nxyz, tris




# ---- cavity-surface triangulation improvement (geometry preserved: vertices slide on the original polyhedron) ----
from scipy.spatial import cKDTree  # noqa: E402

def closest_point_on_tris(p, A, B, C):
    """closest points of p (n,3) on triangles (n,3) each; returns points."""
    ab=B-A; ac=C-A; ap=p-A; d1=(ab*ap).sum(1); d2=(ac*ap).sum(1); bp=p-B; d3=(ab*bp).sum(1); d4=(ac*bp).sum(1); cp=p-C; d5=(ab*cp).sum(1); d6=(ac*cp).sum(1)
    vc=d1*d4-d3*d2; vb=d5*d2-d1*d6; va=d3*d6-d5*d4; out=np.empty_like(p); done=np.zeros(len(p),bool)
    m=(d1<=0)&(d2<=0); out[m]=A[m]; done|=m
    m=(d3>=0)&(d4<=d3)&~done; out[m]=B[m]; done|=m
    m=(d6>=0)&(d5<=d6)&~done; out[m]=C[m]; done|=m
    m=(vc<=0)&(d1>=0)&(d3<=0)&~done; v=d1/np.where(d1-d3==0,1,d1-d3); out[m]=(A+v[:,None]*ab)[m]; done|=m
    m=(vb<=0)&(d2>=0)&(d6<=0)&~done; w=d2/np.where(d2-d6==0,1,d2-d6); out[m]=(A+w[:,None]*ac)[m]; done|=m
    m=(va<=0)&((d4-d3)>=0)&((d5-d6)>=0)&~done; den=(d4-d3)+(d5-d6); w=(d4-d3)/np.where(den==0,1,den); out[m]=(B+w[:,None]*(C-B))[m]; done|=m
    r=~done; den=va+vb+vc; v=vb/np.where(den==0,1,den); w=vc/np.where(den==0,1,den); out[r]=(A+v[:,None]*ab+w[:,None]*ac)[r]; return out
def tri_quality(V,T):
    Pp=V[T]; a=np.linalg.norm(Pp[:,1]-Pp[:,0],axis=1); b=np.linalg.norm(Pp[:,2]-Pp[:,1],axis=1); c=np.linalg.norm(Pp[:,0]-Pp[:,2],axis=1); s=(a+b+c)/2
    area=np.sqrt(np.clip(s*(s-a)*(s-b)*(s-c),0,None)); return 4*np.sqrt(3)*area/np.maximum(a*a+b*b+c*c,1e-300)
def normals(V,T):
    Pp=V[T]; n=np.cross(Pp[:,1]-Pp[:,0],Pp[:,2]-Pp[:,0]); l=np.linalg.norm(n,axis=1); return n/np.maximum(l,1e-300)[:,None]

def smooth_on_polyhedron(V0: np.ndarray, T: np.ndarray, iters: int = 20, step: float = 0.6, q_thresh: float = 0.6, knn: int = 16, frozen: np.ndarray | None = None):
    V=V0.copy(); Pp=V0[T]; tree=cKDTree(Pp.mean(1)); A0,B0,C0=Pp[:,0],Pp[:,1],Pp[:,2]
    nbr=defaultdict(set); v2t=defaultdict(list)
    for ti,f in enumerate(T):
        for u in f: v2t[int(u)].append(ti)
        for u,w in ((0,1),(1,2),(0,2)): nbr[int(f[u])].add(int(f[w])); nbr[int(f[w])].add(int(f[u]))
    # crease vertices (dihedral between incident faces > 30deg) are frozen to preserve sharp features exactly
    N0=normals(V0,T); crease=np.zeros(len(V),bool)
    e2f=defaultdict(list)
    for ti,f in enumerate(T):
        for u,w in ((0,1),(1,2),(0,2)): e2f[(min(f[u],f[w]),max(f[u],f[w]))].append(ti)
    for (a,b),fs in e2f.items():
        if len(fs)==2 and float(N0[fs[0]]@N0[fs[1]])<np.cos(np.radians(30)): crease[a]=crease[b]=True
    def project(p):
        idx=tree.query(p[None],k=knn)[1][0]; cands=closest_point_on_tris(np.repeat(p[None],len(idx),0),A0[idx],B0[idx],C0[idx]); d=np.linalg.norm(cands-p,axis=1); return cands[np.argmin(d)]
    moved=0; N=N0.copy()
    for it in range(iters):
        q=tri_quality(V,T); cand=sorted(set(int(u) for ti in np.where(q<q_thresh)[0] for u in T[ti]))
        Vn=V.copy(); ch=0
        for i in cand:
            if crease[i] or not nbr[i] or (frozen is not None and frozen[i]): continue
            nb=list(nbr[i]); target=V[nb].mean(0); d=target-V[i]
            # tangential component w.r.t. vertex normal (area-weighted incident normals)
            inc=v2t[i]; n=N[inc].mean(0); n/=max(np.linalg.norm(n),1e-300); d=d-(d@n)*n
            p=project(V[i]+step*d)
            Vt=V.copy(); Vt[i]=p
            if tri_quality(Vt,T[inc]).min()<=tri_quality(V,T[inc]).min(): continue
            if np.any(np.einsum('ij,ij->i',N[inc],normals(Vt,T[inc]))<0.8): continue
            Vn[i]=p; ch+=1
        V=Vn; N=normals(V,T); moved+=ch
        if ch==0: break
    return V, moved, crease



@dataclass(frozen=True)
class ConformingMeshResult:
    mesh_path: Path
    node_count: int
    tet_count: int
    carrier_node_count: int
    inner_component_count: int
    material_volume: float
    outer_shell_volume: float
    min_dihedral_degrees: float
    tets_below_5deg: int
    tets_below_10deg: int
    seconds: float
    gmsh_options: dict


def _min_dihedral(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    P = nodes[tets]; faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)); N = []
    for f in faces:
        n = np.cross(P[:, f[1]] - P[:, f[0]], P[:, f[2]] - P[:, f[0]]); n /= np.linalg.norm(n, axis=1)[:, None]; N.append(n)
    out = np.full(len(tets), 180.0)
    for i in range(4):
        for j in range(i + 1, 4):
            out = np.minimum(out, 180.0 - np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", N[i], N[j]), -1, 1))))
    return out


def build_conforming_port_mesh(*, off_path: Path, layout, geometry, active_ports: tuple[str, ...], output_path: Path,
                               interior_size_max: float = 0.05, algorithm3d: int = 10, optimize_passes: int = 3,
                               plane_tolerance: float = 1e-9, seed: int = 1, port_refine_levels: int = 1, netgen_optimize: bool = True, cavity_remesh_size: float | None = None, cavity_smooth: bool = True) -> ConformingMeshResult:
    import gmsh

    started = time.perf_counter()
    V, F = read_off(off_path)
    planes = _port_planes(geometry)
    on_port = np.zeros(len(F), bool)
    for name, normal, offset in planes:
        if name not in active_ports:
            continue
        on_port |= np.all(np.abs(V[F] @ normal - offset) < plane_tolerance, axis=1)
    inner = F[~on_port]
    if cavity_remesh_size is not None and len(inner):
        V, inner = remesh_cavity_surfaces(V, inner, size=cavity_remesh_size)
    smooth_report = None
    if cavity_smooth and len(inner):
        q_before = tri_quality(V, inner)
        V, moved, crease = smooth_on_polyhedron(V, inner)
        q_after = tri_quality(V, inner)
        smooth_report = {"moved_vertices": int(moved), "crease_frozen": int(crease.sum()), "q_min_before": float(q_before.min()), "q_min_after": float(q_after.min()),
                         "q_p5_before": float(np.quantile(q_before, 0.05)), "q_p5_after": float(np.quantile(q_after, 0.05))}
    shell_coords, shell_tris, per_face = carrier_shell(layout, active_ports)
    carrier_count = len(np.unique(shell_tris))
    shell_coords, per_face = refine_shell(shell_coords, per_face, port_refine_levels)
    shell_tris = np.vstack([t for _, t in per_face])
    wt_outer = watertight_report(shell_tris); wt_inner = watertight_report(inner) if len(inner) else {"edges": 0, "non_two_manifold_edges": 0}
    if wt_outer["non_two_manifold_edges"] or wt_inner["non_two_manifold_edges"]:
        raise ValueError(f"surface is not watertight: outer {wt_outer}, inner {wt_inner}")
    inner_comps = _components(inner) if len(inner) else []
    outer_vol = signed_volume(shell_coords, shell_tris)
    if outer_vol < 0:
        shell_tris = shell_tris[:, [0, 2, 1]]; per_face = [(s, t[:, [0, 2, 1]]) for s, t in per_face]; outer_vol = -outer_vol
    # cavity components: material normals point into the void, i.e. signed volume of each cavity shell is negative
    inner_fixed = []
    for comp in inner_comps:
        tris = inner[comp]
        if signed_volume(V, tris) > 0:
            tris = tris[:, [0, 2, 1]]
        inner_fixed.append(tris)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.logger.start()
    try:
        gmsh.model.add("conforming_port_cell")
        node_tag = 0; owned: dict[tuple[str, int], int] = {}
        def add_surface(tag: int, coords: np.ndarray, tris: np.ndarray, key: str):
            nonlocal node_tag
            gmsh.model.addDiscreteEntity(2, tag)
            used = np.unique(tris); new_ids = []; new_xyz = []
            for v in used:
                k = (key, int(v))
                if k not in owned:
                    node_tag += 1; owned[k] = node_tag; new_ids.append(node_tag); new_xyz.extend(coords[v].tolist())
            if new_ids:
                gmsh.model.mesh.addNodes(2, tag, new_ids, new_xyz)
            elems = [owned[(key, int(v))] for f in tris for v in f]
            gmsh.model.mesh.addElementsByType(tag, 2, [], elems)
        surf_tags = []
        for i, (src, tris) in enumerate(per_face):
            add_surface(i + 1, shell_coords, tris, "carrier"); surf_tags.append(i + 1)
        for j, tris in enumerate(inner_fixed):
            t = len(per_face) + 1 + j; add_surface(t, V, tris, "off"); surf_tags.append(t)
        outer_loop = gmsh.model.geo.addSurfaceLoop([i + 1 for i in range(len(per_face))])
        cavity_loops = [gmsh.model.geo.addSurfaceLoop([len(per_face) + 1 + j]) for j in range(len(inner_fixed))]
        gmsh.model.geo.addVolume([outer_loop] + cavity_loops); gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm3d)
        gmsh.option.setNumber("Mesh.MeshSizeMax", interior_size_max)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)
        gmsh.option.setNumber("Mesh.RandomSeed", seed)
        gmsh.option.setNumber("General.NumThreads", 8)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if netgen_optimize else 0)
        gmsh.model.mesh.generate(3)
        for _ in range(max(0, optimize_passes - 1)):
            gmsh.model.mesh.optimize("", niter=1)
        if netgen_optimize:
            gmsh.model.mesh.optimize("Netgen", niter=2)
        ntags, nxyz, _ = gmsh.model.mesh.getNodes(); nxyz = np.asarray(nxyz).reshape(-1, 3)
        order = {int(t): i for i, t in enumerate(ntags)}
        etypes, etags, enodes = gmsh.model.mesh.getElements(3)
        log = gmsh.logger.get(); gmsh.logger.stop()
        if not enodes or len(enodes[0]) == 0:
            raise RuntimeError("Gmsh produced no tetrahedra; log tail: " + " | ".join(log[-12:]))
        tets = np.asarray(enodes[0]).reshape(-1, 4); tets = np.vectorize(order.get)(tets)
        # carrier nodes must survive with identical coordinates
        carrier_ids = np.arange(carrier_count); carrier_xyz = shell_coords[carrier_ids]
        gm = np.vectorize(order.get)(np.asarray([owned[("carrier", int(v))] for v in carrier_ids]))
        if not np.allclose(nxyz[gm], carrier_xyz, atol=1e-12):
            raise RuntimeError("Gmsh moved a carrier node")
    finally:
        gmsh.finalize()
    P = nxyz[tets]; vol = np.einsum("ij,ij->i", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])) / 6.0
    neg = vol < 0; tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy(); vol = np.abs(vol)
    dih = _min_dihedral(nxyz, tets)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("MeshVersionFormatted 2\nDimension 3\n")
        fh.write(f"Vertices\n{len(nxyz)}\n"); np.savetxt(fh, np.column_stack([nxyz, np.zeros(len(nxyz), int)]), fmt="%.17g %.17g %.17g %d")
        fh.write(f"Tetrahedra\n{len(tets)}\n"); np.savetxt(fh, np.column_stack([tets + 1, np.ones(len(tets), int)]), fmt="%d %d %d %d %d")
        fh.write("End\n")
    return ConformingMeshResult(mesh_path=output_path, node_count=int(len(nxyz)), tet_count=int(len(tets)), carrier_node_count=int(len(carrier_ids)),
                                inner_component_count=len(inner_fixed), material_volume=float(vol.sum()), outer_shell_volume=outer_vol,
                                min_dihedral_degrees=float(dih.min()), tets_below_5deg=int((dih < 5).sum()), tets_below_10deg=int((dih < 10).sum()), seconds=time.perf_counter() - started,
                                gmsh_options={"Algorithm3D": algorithm3d, "MeshSizeMax": interior_size_max, "seed": seed, "optimize_passes": optimize_passes, "port_refine_levels": port_refine_levels, "shell_triangles": int(len(shell_tris)), "netgen_optimize": netgen_optimize, "cavity_remesh_size": cavity_remesh_size, "inner_triangles": int(len(inner)), "cavity_smooth": smooth_report})


__all__ = ["ConformingMeshResult", "build_conforming_port_mesh", "carrier_shell", "read_off", "watertight_report"]
