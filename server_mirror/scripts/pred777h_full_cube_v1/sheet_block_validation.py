#!/usr/bin/env python3
"""N x N x N block of sheet cells (true geometry, no collars) under the parent thickness field: monolithic Tet10 (free
interfaces, free external faces) vs fixed-port assembly of the per-cell labels, plus the constrained-monolithic
control (all external port traces tied to the P1 carrier, interfaces free).  Loads on the x = N face, x = 0 fixed.
The cut face is a port (2026-09-04): the block's cut cap conforms to the union of the cells' cut_0 traces and the
constrained control ties its fine nodes to those traces, like the box faces.

--shape 2 1 1 reproduces the STEP3 block; --shape 2 2 2 is the eight-cell assembly validation.
"""
from __future__ import annotations
import argparse, itertools, json, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np, scipy.sparse as sp
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot"); sss = import_module(f"{P}.sheet_solid_surface")
cpm = import_module(f"{P}.conforming_port_mesh"); pipe = import_module(f"{P}.sheet_label_pipeline"); fcg = import_module(f"{P}.full_cube_geometry")
import_module(f"{P}.frozen_backend").activate_frozen_backend()
from cctpms.fem.tet10 import assemble_global_tet10_stiffness
from cctpms.port.tet10_nodal_schur import TET10_EDGE_LOCAL_NODES
from cctpms.port.global_cell_patch_trace_layout import build_global_cell_patch_p1_prolongation
import pypardiso

FACES = ("box_x_min", "box_x_max", "box_y_min", "box_y_max", "box_z_min", "box_z_max")
FACE_AXIS = {"box_x_min": (0, 0), "box_x_max": (0, 1), "box_y_min": (1, 0), "box_y_max": (1, 1), "box_z_min": (2, 0), "box_z_max": (2, 1)}


def build_layout(spec, charts, carrier_n: int):
    return pipe.build_carrier_layout(charts, spec, carrier_n=carrier_n)


def p1_field_evaluator(tf):
    """Vectorized trilinear evaluation of the parent GlobalP1ThicknessField at world points."""
    origin = np.array([float(v) for v in tf.origin]); size = np.array([float(v) for v in tf.cell_size]); shape = np.array(tf.cell_shape, dtype=int)
    vals = np.array([float(v) for v in tf.vertex_values]).reshape(shape[0] + 1, shape[1] + 1, shape[2] + 1)
    def value(xyz):
        c = (np.asarray(xyz, dtype=np.float64) - origin) / size; c = np.clip(c, 0, shape - 1e-12)
        i = np.floor(c).astype(int); f = c - i; out = np.zeros(c.shape[:-1])
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = (f[..., 0] if di else 1 - f[..., 0]) * (f[..., 1] if dj else 1 - f[..., 1]) * (f[..., 2] if dk else 1 - f[..., 2])
                    out += w * vals[i[..., 0] + di, i[..., 1] + dj, i[..., 2] + dk]
        return out
    return value


def local_cut_plane(spec, off=(0, 0, 0)):
    """The cell's world cut in the local frame of the cell at block offset `off` (exact, from the base cell's spec),
    as the float tuple the mesher geometry carries; None without a cut."""
    cut = getattr(spec, "cut_plane_local", None)
    if cut is None:
        return None
    a, b, c, d = (Fraction(cut.a), Fraction(cut.b), Fraction(cut.c), Fraction(cut.d))
    d_off = d - (a * Fraction(off[0]) + b * Fraction(off[1]) + c * Fraction(off[2]))
    return (float(a), float(b), float(c), float(d_off))


def world_cut_plane(spec):
    """The base cell's cut in WORLD coordinates (what make_cell_spec expects for the other cells of the block)."""
    cut = getattr(spec, "cut_plane_local", None)
    if cut is None:
        return None
    o = [Fraction(v) for v in spec.placement.translation]
    a, b, c, d = (Fraction(cut.a), Fraction(cut.b), Fraction(cut.c), Fraction(cut.d))
    return type(cut)(cut.plane_id, cut.semantic_id, a, b, c, d + a * o[0] + b * o[1] + c * o[2])


class BlockGeom:
    """Block [0,Nx]x[0,Ny]x[0,Nz] with the parent thickness field: |phi(world)| - tau(world); the world cut, if any,
    in the block's frame (the base cell's local frame)."""
    def __init__(self, spec, geom, shape):
        self.tau = p1_field_evaluator(spec.thickness_field); self.origin = np.array([float(v) for v in spec.placement.translation])
        self.cut_plane = local_cut_plane(spec); self.box_max = tuple(float(s) for s in shape); self.collar_thickness = geom.collar_thickness
        self.tau_corners = tuple(float(self.tau(self.origin + np.array(c, dtype=float) * np.array(shape))) for c in itertools.product((0, 1), repeat=3))
    def material_band_components(self, points):
        xyz = np.asarray(points, dtype=np.float64); world = xyz + self.origin
        phi = np.cos(2.0 * np.pi * world).sum(axis=-1); t = np.abs(phi) - self.tau(world)
        return t, np.full(t.shape, np.inf)
    def active_outer_port_sources(self): return FACES + (("cut_0",) if self.cut_plane is not None else ())


def cell_specs(spec, geom, shape):
    """(offset, spec, geom) of every cell of the block; tau corners in the same corner order as geom.tau_corners."""
    tf = spec.thickness_field; o = spec.placement.translation
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    tauA = [float(tf.value(spec.placement.world(np.array(c, dtype=float)))) for c in corners]
    order = None
    for perm in itertools.permutations(range(8)):
        if all(abs(tauA[perm[k]] - float(geom.tau_corners[k])) < 1e-12 for k in range(8)):
            order = perm; break
    if order is None:
        raise SystemExit("could not match the corner order of tau_corners")
    out = []; wcut = world_cut_plane(spec)
    for off in itertools.product(range(shape[0]), range(shape[1]), range(shape[2])):
        if off == (0, 0, 0):
            # the manifest loader carries the cut unshifted; the exact local plane of the spec is authoritative
            out.append((off, spec, type(geom)(tau_corners=geom.tau_corners, cut_plane=local_cut_plane(spec), collar_thickness=geom.collar_thickness))); continue
        sp_ = fcg.make_cell_spec(thickness_field=tf, cell_world_origin=(o[0] + Fraction(off[0]), o[1] + Fraction(off[1]), o[2] + Fraction(off[2])),
                                 cell_id=f"{spec.cell_id}_{off[0]}{off[1]}{off[2]}", parent_world_cut_id=spec.parent_world_cut_id, world_cut=wcut)
        tau = tuple(float(tf.value(sp_.placement.world(np.array(corners[order[k]], dtype=float)))) for k in range(8))
        out.append((off, sp_, type(geom)(tau_corners=tau, cut_plane=local_cut_plane(spec, off), collar_thickness=geom.collar_thickness)))
    return out


def cell_label(spec, geom, layout, out, a):
    """Sheet label of one cell through the production pipeline stages."""
    out.mkdir(parents=True, exist_ok=True)
    surf = sss.build_sheet_solid_surface(geom, layout, n_per_unit=a.n_per_unit, remesh_size=a.remesh_size, carrier_n=a.carrier_n)
    if surf.report["watertight"]["non_two_manifold_edges"] or surf.report["self_intersecting_triangles"]:
        raise SystemExit(f"cell surface rejected: {out}")
    m = sss.mesh_sheet_solid(surf, out / "mesh.mesh", interior_size_max=a.size_max, algorithm3d=a.algorithm3d, threads=a.threads)
    stage = pipe.compile_port(out / "mesh.mesh", geom, spec, layout, carrier_n=a.carrier_n); sch = pipe.schur_label(stage, workers=a.workers)
    keys = list(layout.global_scalar_node_keys); g2c = dict(stage.port.global_to_compact); lay2c = np.array([g2c.get(str(k), -1) for k in keys])
    print(f"cell {out.name}: mesh {m.node_count} nodes, active {int(stage.active.sum())}, schur {sch.timing['schur_total']:.0f}s, rigid {sch.rigid_residual:.1e}", flush=True)
    return stage.port, stage.active, sch.schur_active, lay2c


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--geometry-manifest", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--shape", type=int, nargs=3, default=(2, 1, 1)); ap.add_argument("--n-per-unit", type=int, default=48); ap.add_argument("--remesh-size", type=float, default=0.03)
    ap.add_argument("--size-max", type=float, default=0.05); ap.add_argument("--algorithm3d", type=int, default=1); ap.add_argument("--no-smooth", action="store_true", help="ignored")
    ap.add_argument("--carrier-n", type=int, default=32); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); out = a.output_dir; out.mkdir(parents=True, exist_ok=True); rep = {"shape": list(a.shape)}; shape = tuple(a.shape); N = np.array(shape)
    rep["cut_plane_block_frame"] = None
    _, spec, geom = snap.load_geometry_manifest(a.geometry_manifest)
    cells = cell_specs(spec, geom, shape); layouts = {}; labels = {}; empty = []
    for off, sp_, gm_ in cells:
        if pipe.cut_contains_a_cell_edge(sp_) is not None:
            raise SystemExit(f"cell {off}: the cut contains a cell edge (GEOMETRY_DEGENERATE); move the world cut by an exact epsilon")
        charts, *_ = fcb.compile_full_cube_geometry_inputs(sp_); layouts[off] = build_layout(sp_, charts, a.carrier_n)
        try:
            labels[off] = cell_label(sp_, gm_, layouts[off], out / f"cell_{off[0]}{off[1]}{off[2]}", a)
        except sss.EmptyCellError:
            empty.append(off); print(f"cell {off}: EMPTY (the cut leaves no band)", flush=True)
    cells = [c for c in cells if c[0] not in empty]; rep["empty_cells"] = [list(o) for o in empty]
    layout0 = layouts[(0, 0, 0)]; rep["carrier_n"] = a.carrier_n; rep["cells"] = len(cells); rep["cut_plane_block_frame"] = local_cut_plane(spec)
    # ---- block surface: port constraints on the six block faces from the cell carriers (shifted) ----
    # every cell's own carrier trace (a cut clips the traces differently in every cell, and can remove a face entirely)
    shells = {off: dict(zip(("coords", "tris", "per_face"), cpm.carrier_shell(layouts[off], tuple(str(t.source_id) for t in layouts[off].local_traces)))) for off, _, _ in cells}
    def face_constraint(src, shifts):
        pts = []; segs = []; keysl = []; idx = {}
        for sh in shifts:
            shv = np.array(sh, dtype=float); found = [t for s_, t in shells[sh]["per_face"] if s_ == src]
            if not found:
                continue                          # the cut removed this face of this cell
            coords_ = shells[sh]["coords"]
            for f in found[0]:
                for v in f:
                    k = (int(v), sh)
                    if k not in idx:
                        idx[k] = len(pts); pts.append(coords_[v] + shv); keysl.append((int(v), sh))
                for a_, b_ in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
                    segs.append((idx[(int(a_), sh)], idx[(int(b_), sh)]))
        return np.asarray(pts).reshape(-1, 3), np.asarray(sorted({(min(x), max(x)) for x in segs}), dtype=np.int64).reshape(-1, 2), keysl
    pc = {}
    for src, (axis, side) in FACE_AXIS.items():
        shifts = [off for off, _, _ in cells if off[axis] == (shape[axis] - 1 if side else 0)]
        pc[src] = face_constraint(src, shifts)
    if local_cut_plane(spec) is not None:          # the cut face is a port: its cap conforms to every cell's cut_0 trace
        pc["cut_0"] = face_constraint("cut_0", [off for off, _, _ in cells])
        rep["cut_face_constraint"] = {"points": int(len(pc["cut_0"][0])), "segments": int(len(pc["cut_0"][1]))}
    bg = BlockGeom(spec, geom, shape); t0 = time.perf_counter()
    surf = sss.build_sheet_solid_surface(bg, layout0, n_per_unit=a.n_per_unit, remesh_size=a.remesh_size, carrier_n=a.carrier_n, port_constraints=pc, pad=6)
    rep["surface"] = {k: v for k, v in surf.report.items() if k not in ("caps", "remesh_attempts", "non_manifold_examples", "surface_quality_by_label", "cap_classification")}
    print("block surface:", json.dumps({k: v for k, v in rep["surface"].items() if k not in ("marching_cubes",)}, default=float), flush=True)
    if surf.report["watertight"]["non_two_manifold_edges"] or surf.report["self_intersecting_triangles"]:
        raise SystemExit("block surface rejected")
    m = sss.mesh_sheet_solid(surf, out / "block.mesh", interior_size_max=a.size_max, algorithm3d=a.algorithm3d, threads=a.threads)
    rep["mesh"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in m.__dict__.items()}; print("block mesh:", json.dumps(rep["mesh"], default=float), flush=True)
    V, F, labs = surf.V, surf.F, surf.labels
    # ---- Tet10 monolithic ----
    L = open(out / "block.mesh").read().split("\n"); i = L.index("Vertices"); nv = int(L[i + 1]); nxyz = np.array([list(map(float, l.split()[:3])) for l in L[i + 2:i + 2 + nv]])
    j = L.index("Tetrahedra"); nt = int(L[j + 1]); tets = np.array([list(map(int, l.split()[:4])) for l in L[j + 2:j + 2 + nt]]) - 1
    edges = {}
    def eid(p_, q_):
        k = (p_, q_) if p_ < q_ else (q_, p_)
        if k not in edges: edges[k] = len(edges)
        return edges[k]
    el10 = np.empty((len(tets), 10), np.int64); el10[:, :4] = tets
    for e, t in enumerate(tets):
        for jj, (p_, q_) in enumerate(TET10_EDGE_LOCAL_NODES): el10[e, 4 + jj] = nv + eid(int(t[p_]), int(t[q_]))
    mids = np.zeros((len(edges), 3))
    for (p_, q_), k in edges.items(): mids[k] = 0.5 * (nxyz[p_] + nxyz[q_])
    nodes10 = np.vstack([nxyz, mids]); n10 = len(nodes10); t0 = time.perf_counter()
    K = assemble_global_tet10_stiffness(nodes10, el10, E=1.0, nu=0.3, workers=a.workers).tocsr(); print(f"Tet10 dof {3*n10}, assembled {time.perf_counter()-t0:.0f}s", flush=True)
    key = lambda p: tuple(np.round(p, 9)); fine_lookup = {key(p): k for k, p in enumerate(nxyz)}
    # ---- global carrier set of the block (active nodes of every cell; shared interface nodes merged by coordinate) ----
    gk = {}; gc = []
    def gid(p):
        k = key(p)
        if k not in gk: gk[k] = len(gc); gc.append(np.asarray(k))
        return gk[k]
    maps = {}; c2s = {}
    for off, port, active, Ss, lay2c in ((off, *labels[off]) for off, _, _ in cells):
        cc = np.asarray(port.carrier_coordinates)[active]; maps[off] = np.array([gid(p + np.array(off, dtype=float)) for p in cc])
        m_ = np.full(len(active), -1); m_[active] = np.arange(int(active.sum())); c2s[off] = m_
    # the assembled carrier operator is block sparse (one dense q x q block per cell): kept sparse, because the dense
    # matrix of a 3 x 3 x 2 block is 65 GB and its eigendecomposition took the machine down (2026-09-05)
    # the assembled carrier operator is block sparse (one dense q x q block per cell) and symmetric: assembled as its
    # upper triangle only, int32 indices (the full int64 triplets of a 4 x 4 x 2 block were 57 GB, 2026-09-05)
    nG = len(gc); gcoords = np.asarray(gc); rows_, cols_, vals_ = [], [], []
    for off, _, _ in cells:
        mm = maps[off]; Sm = np.asarray(labels[off][2]); idx = (3 * mm[:, None] + np.arange(3)).ravel().astype(np.int32)
        ii, jj = np.meshgrid(idx, idx, indexing="ij"); up = ii <= jj
        rows_.append(ii[up]); cols_.append(jj[up]); vals_.append(Sm[up]); del ii, jj, up
    KG = sp.coo_matrix((np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))), shape=(3 * nG, 3 * nG)).tocsr(); del rows_, cols_, vals_   # UPPER triangle
    # interface consistency: for every internal interface the two cells' active node sets on it must coincide
    mism = 0; pairs = 0
    for off, _, _ in cells:
        for axis in range(3):
            nb = list(off); nb[axis] += 1; nb = tuple(nb)
            if nb not in maps: continue
            pairs += 1
            ccA = np.asarray(labels[off][0].carrier_coordinates)[labels[off][1]]; ccB = np.asarray(labels[nb][0].carrier_coordinates)[labels[nb][1]]
            sA = set(maps[off][np.abs(ccA[:, axis] - 1.0) < 1e-9].tolist()); sB = set(maps[nb][np.abs(ccB[:, axis]) < 1e-9].tolist()); mism += len(sA ^ sB)
    rep["interface_pairs"] = pairs; rep["interface_active_mismatch"] = int(mism); print("internal interfaces:", pairs, "active-set mismatch", mism, flush=True)
    from scipy.spatial import cKDTree
    tet_tree = cKDTree(nxyz[tets].mean(axis=1))
    def interp_vertex_field(u_vertex, pts):
        out_ = np.full((len(pts), 3), np.nan); idx = tet_tree.query(pts, k=48)[1]
        for i_, p in enumerate(pts):
            for ti in idx[i_]:
                a_, b_, c_, d_ = nxyz[tets[ti]]; Mm = np.column_stack([b_ - a_, c_ - a_, d_ - a_])
                try: l = np.linalg.solve(Mm, p - a_)
                except np.linalg.LinAlgError: continue
                lam = np.array([1 - l.sum(), *l])
                if lam.min() > -1e-6: out_[i_] = lam @ u_vertex[tets[ti]]; break
        return out_
    # ---- face P (fine face vertex of one cell face -> layout global scalar of that cell) ----
    def face_P(src, off):
        axis, side = FACE_AXIS[src]; fv = np.unique(F[labs == src]); fxyz = V[fv]; offv = np.array(off, dtype=float)
        if src not in {str(t.source_id) for t in layouts[off].local_traces} or not len(fv):
            return np.zeros(0, dtype=np.int64), sp.csr_matrix((0, layouts[off].scalar_node_count))
        inside = np.ones(len(fv), bool)
        for ax in range(3):
            if ax != axis: inside &= (fxyz[:, ax] >= offv[ax] - 1e-9) & (fxyz[:, ax] <= offv[ax] + 1.0 + 1e-9)
        fv = fv[inside]; fxyz = fxyz[inside]; lay = layouts[off]; tr = {str(t.source_id): t for t in lay.local_traces}[src]
        Ploc = build_global_cell_patch_p1_prolongation(tr, fxyz - offv, geometric_tolerance=1e-9)
        Pg = (Ploc @ lay.local_to_global_scalar[src].tocsr()).tocsr()
        return np.array([fine_lookup[key(p)] for p in fxyz]), Pg
    def cut_P(off):
        """Fine cut-face vertices inside the cell at `off` -> that cell's cut_0 trace (the cut face is a port)."""
        fv = np.unique(F[labs == "cut_0"]); fxyz = V[fv]; offv = np.array(off, dtype=float)
        lay = layouts[off]; traces = {str(t.source_id): t for t in lay.local_traces}
        if "cut_0" not in traces or not len(fv):
            return np.zeros(0, dtype=np.int64), sp.csr_matrix((0, lay.scalar_node_count))
        inside = np.all((fxyz >= offv - 1e-9) & (fxyz <= offv + 1.0 + 1e-9), axis=1)
        fv = fv[inside]; fxyz = fxyz[inside]
        if not len(fv):
            return np.zeros(0, dtype=np.int64), sp.csr_matrix((0, lay.scalar_node_count))
        Ploc = build_global_cell_patch_p1_prolongation(traces["cut_0"], fxyz - offv, geometric_tolerance=1e-9)
        Pg = (Ploc @ lay.local_to_global_scalar["cut_0"].tocsr()).tocsr()
        return np.array([fine_lookup[key(p)] for p in fxyz]), Pg
    def clip_P(Pg, off):
        Pl = Pg.tolil(copy=True); c2s_ = c2s[off]; lay2c_ = labels[off][3]
        for r in range(Pl.shape[0]):
            keep = [(c, v) for c, v in zip(Pl.rows[r], Pl.data[r]) if c2s_[lay2c_[c]] >= 0]; tot = sum(v for _, v in keep)
            if tot <= 1e-12: keep = list(zip(Pl.rows[r], Pl.data[r])); tot = sum(v for _, v in keep)
            Pl.rows[r] = [c for c, _ in keep]; Pl.data[r] = [v / tot for _, v in keep]
        return Pl.tocsr()
    def global_cols(off, ncols):
        c2s_ = c2s[off]; lay2c_ = labels[off][3]; mm = maps[off]
        return np.array([mm[c2s_[lay2c_[jj]]] if c2s_[lay2c_[jj]] >= 0 else -1 for jj in range(ncols)])
    # ---- loads on x = Nx: lumped tractions on fine face vertices of every cell face there, f_c = P^T f ----
    load_cells = [off for off, _, _ in cells if off[0] == shape[0] - 1]
    face_data = []
    for off in load_cells:
        fB, PB = face_P("box_x_max", off)
        if not len(fB):
            continue
        PB = clip_P(PB, off); colB = global_cols(off, PB.shape[1]); face_data.append((off, fB, PB, colB))
    wts = {}
    for f in F[labs == "box_x_max"]:
        pp = V[f]; area = 0.5 * np.linalg.norm(np.cross(pp[1] - pp[0], pp[2] - pp[0]))
        for v in f:
            fid = fine_lookup[key(V[v])]; wts[fid] = wts.get(fid, 0.0) + area / 3
    def traction(kind, yz):
        y, z = yz[:, 0], yz[:, 1]; t = np.zeros((len(yz), 3)); Ly, Lz = shape[1], shape[2]
        if kind == "tension_x": t[:, 0] = 1
        elif kind == "shear_y": t[:, 1] = 1
        elif kind == "bending_z": t[:, 0] = z - 0.5 * Lz
        elif kind == "wave_16H": t[:, 0] = np.sin(2 * np.pi * y / Ly) * np.sin(2 * np.pi * z / Lz)
        elif kind == "wave_8H": t[:, 0] = np.sin(4 * np.pi * y) * np.sin(4 * np.pi * z)
        elif kind == "wave_4H": t[:, 0] = np.sin(8 * np.pi * y) * np.sin(8 * np.pi * z)
        return t
    loads = {}
    for kind in ("tension_x", "shear_y", "bending_z", "wave_16H", "wave_8H", "wave_4H"):
        fF = np.zeros(3 * n10); fc = np.zeros((nG, 3))
        for off, fB, PB, colB in face_data:
            tf_ = traction(kind, nxyz[fB][:, 1:]) * np.array([wts[int(v)] for v in fB])[:, None]
            fF[(3 * fB[:, None] + np.arange(3)).ravel()] += tf_.ravel(); fcA = PB.T @ tf_
            for jj in range(PB.shape[1]):
                if colB[jj] >= 0: fc[colB[jj]] += fcA[jj]
        loads[kind] = (fF, fc.ravel())
    # ---- (a) monolithic free ----
    fixed_fine = np.where(np.abs(nodes10[:, 0]) < 1e-9)[0]; free = np.ones(3 * n10, bool); free[(3 * fixed_fine[:, None] + np.arange(3)).ravel()] = False
    # symmetric Pardiso on the upper triangle, and each factorisation is released after its loads are solved: the three
    # nonsymmetric factorisations held together exceeded 128 GiB on the 4 x 4 x 2 block (1.75e6 Tet10 dof, 2026-09-05)
    Kff = sp.triu(K[free][:, free].tocsr(), format="csr"); sol = pypardiso.PyPardisoSolver(mtype=-2); sol.set_iparm(1, 1); sol.set_iparm(2, 3); sol.set_iparm(60, 1)
    t0 = time.perf_counter(); sol.factorize(Kff); print(f"monolithic factorized {time.perf_counter()-t0:.0f}s ({Kff.shape[0]} dof, symmetric)", flush=True)
    U_F = {}
    for name, (fF, fG) in loads.items():
        uF = np.zeros(3 * n10); uF[free] = sol.solve(Kff, fF[free]); U_F[name] = uF
    sol.free_memory(everything=True); del sol, Kff
    # ---- (b) assembled ----
    fixedG = np.zeros(3 * nG, bool); fixedG[(3 * np.where(np.abs(gcoords[:, 0]) < 1e-9)[0][:, None] + np.arange(3)).ravel()] = True; freeG = ~fixedG
    dg = KG.diagonal(); zero_nodes = np.where((dg.reshape(-1, 3) <= 1e-12 * dg.max()).all(axis=1))[0]; rep["zero_stiffness_carrier_nodes"] = int(len(zero_nodes))   # carrier nodes the band only grazes carry a negligible support: removed from the free system like the unsupported ones
    if len(zero_nodes):
        zmask = np.zeros(3 * nG, bool); zmask[(3 * zero_nodes[:, None] + np.arange(3)).ravel()] = True; freeG &= ~zmask
    # symmetric indefinite Pardiso on the upper triangle (half the storage of the nonsymmetric mode; the 3 x 3 x 2 block's
    # 1.3e9 dense-block nonzeros ran the nonsymmetric factorization out of memory), in-core if it fits, out-of-core otherwise
    KGf = KG[freeG][:, freeG].tocsr()   # KG is already the upper triangle; the restriction keeps it upper
    solG = pypardiso.PyPardisoSolver(mtype=-2); solG.set_iparm(1, 1); solG.set_iparm(2, 3); solG.set_iparm(60, 1)
    t0 = time.perf_counter(); solG.factorize(KGf); print(f"assembled system factorized {time.perf_counter()-t0:.0f}s ({KGf.shape[0]} dof, {KGf.nnz} upper nonzeros)", flush=True)
    try:   # smallest eigenvalues by shift-invert through the Pardiso factorization (a diagnostic: the free assembled system must be positive definite)
        from scipy.sparse.linalg import LinearOperator, eigsh
        dKf = KGf.diagonal(); Asym = LinearOperator(KGf.shape, matvec=lambda x: KGf @ x + KGf.T @ x - dKf * x, dtype=np.float64)
        OPinv = LinearOperator(KGf.shape, matvec=lambda x: solG.solve(KGf, np.asarray(x, dtype=np.float64).ravel()), dtype=np.float64)
        evs = np.sort(eigsh(Asym, k=4, sigma=0.0, which="LM", OPinv=OPinv, return_eigenvectors=False, tol=1e-8))
    except Exception as exc:
        evs = np.full(4, np.nan); rep["assembled_eigenvalue_error"] = str(exc)[:200]
    rep["assembled_smallest_eigenvalues"] = evs.tolist()
    U_G = {}
    for name, (fF, fG) in loads.items():
        uG = np.zeros(3 * nG); uG[freeG] = solG.solve(KGf, fG[freeG]); U_G[name] = uG
    solG.free_memory(everything=True); del solG, KGf
    print(f"assembled carrier system: {nG} nodes, zero-stiffness nodes {len(zero_nodes)}, smallest eigenvalues {evs}", flush=True)
    # ---- (c) constrained monolithic: every external port-face fine node tied to the carrier ----
    tied = {}
    tie_jobs = [(src, off) for src, (axis, side) in FACE_AXIS.items() if src != "box_x_min" for off, _, _ in cells if off[axis] == (shape[axis] - 1 if side else 0)]
    tie_jobs += [("cut_0", off) for off, _, _ in cells]          # the cut face is a port in every cut cell
    for src, off in tie_jobs:
        if True:
            fids, Pg = cut_P(off) if src == "cut_0" else face_P(src, off)
            if not len(fids):
                continue
            Pl = clip_P(Pg, off).tolil(); cols = global_cols(off, Pg.shape[1])
            for r, fid in enumerate(fids):
                w = {}
                for jj, val in zip(Pl.rows[r], Pl.data[r]):
                    if cols[jj] < 0: continue
                    w[cols[jj]] = w.get(cols[jj], 0.0) + float(val)
                if not w or abs(sum(w.values()) - 1.0) > 1e-9: continue
                tied[fid] = w
    mid_tied = {}
    for (p_, q_), k in edges.items():
        if p_ in tied and q_ in tied:
            w = {}
            for src_w in (tied[p_], tied[q_]):
                for g_, val in src_w.items(): w[g_] = w.get(g_, 0.0) + 0.5 * val
            mid_tied[nv + k] = w
    tied_all = dict(tied); tied_all.update(mid_tied)
    free_nodes = [i_ for i_ in range(n10) if i_ not in tied_all]; col_free = {nd: k for k, nd in enumerate(free_nodes)}; nfree = len(free_nodes)
    rows, cols_, vals = [], [], []
    for nd in free_nodes:
        for c in range(3): rows.append(3 * nd + c); cols_.append(3 * col_free[nd] + c); vals.append(1.0)
    for nd, w in tied_all.items():
        for g_, val in w.items():
            for c in range(3): rows.append(3 * nd + c); cols_.append(3 * nfree + 3 * g_ + c); vals.append(val)
    Tc = sp.coo_matrix((vals, (rows, cols_)), shape=(3 * n10, 3 * nfree + 3 * nG)).tocsr()
    Kc = (Tc.T @ K @ Tc).tocsr(); fixed_c = np.zeros(Tc.shape[1], bool)
    for nd in fixed_fine:
        if nd in col_free: fixed_c[3 * col_free[nd]:3 * col_free[nd] + 3] = True
    fixed_c[3 * nfree:][fixedG] = True
    freec = (~fixed_c) & (np.diff(Kc.indptr) > 0); Kcf = sp.triu(Kc[freec][:, freec].tocsr(), format="csr"); solc = pypardiso.PyPardisoSolver(mtype=-2); solc.set_iparm(1, 1); solc.set_iparm(2, 3); solc.set_iparm(60, 1); solc.factorize(Kcf)
    print(f"constrained monolithic: tied fine nodes {len(tied_all)}, dof {Kcf.shape[0]}", flush=True)
    interior = np.where((np.abs(gcoords[:, 0] - np.round(gcoords[:, 0])) < 1e-9) & (gcoords[:, 0] > 1e-9) & (gcoords[:, 0] < shape[0] - 1e-9))[0]   # nodes on internal x-interfaces
    results = {}
    for name, (fF, fG) in loads.items():
        uF = U_F[name]; uG = U_G[name]
        fc_ = Tc.T @ fF; zc = np.zeros(Tc.shape[1]); zc[freec] = solc.solve(Kcf, fc_[freec]); uC = Tc @ zc
        cm, ca, cc = float(fF @ uF), float(fG @ uG), float(fF @ uC)
        results[name] = {"compliance_monolithic": cm, "compliance_assembled": ca, "compliance_constrained": cc, "assembled_vs_free": ca / cm - 1, "constrained_vs_free": cc / cm - 1,
                         "assembled_vs_constrained": ca / cc - 1}
        if len(interior):
            ui_m = interp_vertex_field(uF[:3 * nv].reshape(-1, 3), gcoords[interior]).ravel(); ui_c = interp_vertex_field(uC[:3 * nv].reshape(-1, 3), gcoords[interior]).ravel(); ui_a = uG[(3 * interior[:, None] + np.arange(3)).ravel()]
            okm = ~np.isnan(ui_m); ui_m, ui_a, ui_c = ui_m[okm], ui_a[okm], ui_c[okm]
            results[name].update({"interface_disp_rel_l2_vs_free": float(np.linalg.norm(ui_a - ui_m) / np.linalg.norm(ui_m)), "interface_disp_rel_l2_vs_constrained": float(np.linalg.norm(ui_a - ui_c) / np.linalg.norm(ui_c))})
        print(name, {k: (round(v, 5) if abs(v) > 1e-3 else f"{v:.2e}") for k, v in results[name].items()}, flush=True)
    rep["results"] = results; rep["block_tet10_dof"] = int(3 * n10); rep["global_carrier_nodes"] = int(nG)
    (out / "SHEET_BLOCK_VALIDATION.json").write_text(json.dumps(rep, indent=1, default=float)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
