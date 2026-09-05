#!/usr/bin/env python3
"""End-to-end proof of the coupling contract: a block of sheet cells with a SKIN bonded to its cut face.

The contract (decision of record 2026-09-04) says the skin couples to the cells through the cut-face carrier: the skin
is meshed on the cells' cut_0 carrier traces, so the joint displacement is a carrier P1 field on both sides.  This
script measures what that restriction costs and whether the assembly of the per-cell labels reproduces it.

Three models of the same geometry (same block mesh, SAME skin mesh, same loads), differing only in the space the
block-to-skin joint traction lives in:

  (REF)  full-fidelity glue: every fine cut-cap node of the block follows the skin's own P1 field at its position.
  (CON)  the contract, solved monolithically: the block's fine cut-cap nodes follow the CARRIER P1 field, and the
         carrier nodes are the skin's own nodes at those positions (midpoint refinement keeps the carrier nodes as a
         subset of the skin's joint nodes, so the coupling is exact and conforming).  The skin keeps every one of its
         own degrees of freedom: only the joint coupling is restricted, neither body is coarsened.
  (ASM)  what production does: the per-cell Schur operators on the carrier plus the same skin, joint through the
         carrier as in CON.

  CON vs REF  = the cost of restricting the joint to the carrier space (the number the route is judged on).
  ASM vs CON  = the assembly error (should match the cell-to-cell figure, order 1e-4).

Tying the skin's own joint nodes to the carrier instead would coarsen the skin to 1/32 as well and measure the skin's
discretization, not the coupling; that variant is `--skin-on-carrier` and belongs to the skin resolution study.

The block's external box faces are tied to the carrier in all three, so the cut joint is the only difference.  Loads
are tractions on the skin's outer face: the physical path, skin into core.  The block is fixed at x = 0.

The skin is a slab of thickness `--skin-thickness` extruded from the cut plane on the discarded side, meshed as Tet10
(each joint triangle gives one prism, split into three tets by the global-id rule so the split is conforming), so it
uses the same verified assembler as the cells.  `--skin-refine` midpoint-refines the joint triangulation: the skin
must be finer than the carrier or REF and CON would share a joint space and the comparison would be empty.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts/pred777h_full_cube_v1"))
P = "pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
fcb = import_module(f"{P}.full_cube_backend"); snap = import_module(f"{P}.mesher_neutral_snapshot"); sss = import_module(f"{P}.sheet_solid_surface")
cpm = import_module(f"{P}.conforming_port_mesh"); pipe = import_module(f"{P}.sheet_label_pipeline")
import_module(f"{P}.frozen_backend").activate_frozen_backend()
from cctpms.fem.tet10 import assemble_global_tet10_stiffness
from cctpms.port.tet10_nodal_schur import TET10_EDGE_LOCAL_NODES
from cctpms.port.global_cell_patch_trace_layout import build_global_cell_patch_p1_prolongation
import pypardiso
import sheet_block_validation as sbv

FACE_AXIS = sbv.FACE_AXIS
KEY = lambda p: tuple(np.round(np.asarray(p, dtype=np.float64), 9))


def tet4_to_tet10(nodes: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Straight-sided promotion: edge midpoints appended after the corner nodes."""
    edges: dict[tuple[int, int], int] = {}
    def eid(p, q):
        k = (p, q) if p < q else (q, p)
        if k not in edges: edges[k] = len(edges)
        return edges[k]
    nv = len(nodes); el = np.empty((len(tets), 10), np.int64); el[:, :4] = tets
    for e, t in enumerate(tets):
        for j, (p, q) in enumerate(TET10_EDGE_LOCAL_NODES): el[e, 4 + j] = nv + eid(int(t[p]), int(t[q]))
    mid = np.zeros((len(edges), 3))
    for (p, q), k in edges.items(): mid[k] = 0.5 * (nodes[p] + nodes[q])
    return np.vstack([nodes, mid]), el, edges


def refine_with_prolongation(xyz: np.ndarray, tris: np.ndarray, levels: int):
    """Uniform midpoint refinement of a conforming triangulation, with the P1 prolongation from the original nodes."""
    Pmat = sp.identity(len(xyz), format="csr")
    for _ in range(int(levels)):
        mid: dict[tuple[int, int], int] = {}; extra: list[np.ndarray] = []; rows = []; cols = []; vals = []
        def midpoint(a, b):
            k = (a, b) if a < b else (b, a)
            if k not in mid:
                mid[k] = len(xyz) + len(extra); extra.append(0.5 * (xyz[a] + xyz[b]))
                r = len(xyz) + len(extra) - 1
                rows.extend([r, r]); cols.extend([a, b]); vals.extend([0.5, 0.5])
            return mid[k]
        out = []
        for a, b, c in tris:
            ab, bc, ca = midpoint(int(a), int(b)), midpoint(int(b), int(c)), midpoint(int(c), int(a))
            out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        n_old = len(xyz)
        xyz = np.vstack([xyz, np.asarray(extra)]) if extra else xyz
        step = sp.vstack([sp.identity(n_old, format="csr"), sp.coo_matrix((vals, (np.asarray(rows) - n_old, cols)), shape=(len(xyz) - n_old, n_old)).tocsr()]).tocsr()
        Pmat = (step @ Pmat).tocsr(); tris = np.asarray(out, dtype=np.int64)
    return xyz, tris, Pmat


def extrude_slab(joint_xyz: np.ndarray, joint_tris: np.ndarray, normal: np.ndarray, thickness: float):
    """One prism layer on the outside of the joint plane, split into tets by the global-id rule (conforming)."""
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    top = joint_xyz + thickness * n[None, :]
    nodes = np.vstack([joint_xyz, top]); nb = len(joint_xyz); tets = []
    for tri in joint_tris:
        v = sorted(int(x) for x in tri)                     # global-id sorting: the quad diagonals agree between prisms
        a, b, c = v; A, B, C = a + nb, b + nb, c + nb
        tets += [(a, b, c, C), (a, b, C, B), (a, B, C, A)]
    tets = np.asarray(tets, dtype=np.int64)
    P = nodes[tets]; vol = np.einsum("ij,ij->i", P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])) / 6.0
    neg = vol < 0; tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()
    return nodes, tets, float(np.abs(vol).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-manifest", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--shape", type=int, nargs=3, default=(2, 1, 1))
    ap.add_argument("--n-per-unit", type=int, default=64); ap.add_argument("--remesh-size", type=float, default=0.02)
    ap.add_argument("--size-max", type=float, default=0.04); ap.add_argument("--algorithm3d", type=int, default=10)
    ap.add_argument("--carrier-n", type=int, default=32); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skin-thickness", type=float, default=0.05); ap.add_argument("--skin-refine", type=int, default=1)
    ap.add_argument("--skin-modulus", type=float, default=1.0, help="E of the skin relative to the cells (both nu = 0.3)")
    ap.add_argument("--skin-on-carrier", action="store_true", help="also run the variant whose SKIN sits on the carrier (the skin resolution study, not the coupling test)")
    a = ap.parse_args(); out = a.output_dir; out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"shape": list(a.shape), "skin": {"thickness": a.skin_thickness, "refine": a.skin_refine, "modulus": a.skin_modulus},
                 "resolution": {"n_per_unit": a.n_per_unit, "remesh_size": a.remesh_size, "size_max": a.size_max, "carrier_n": a.carrier_n}}
    shape = tuple(a.shape)
    _, spec, geom = snap.load_geometry_manifest(a.geometry_manifest)
    if sbv.local_cut_plane(spec) is None:
        raise SystemExit("the skin validation needs a cut block")
    cells = sbv.cell_specs(spec, geom, shape); layouts = {}; labels = {}; empty = []
    for off, sp_, gm_ in cells:
        charts, *_ = fcb.compile_full_cube_geometry_inputs(sp_); layouts[off] = pipe.build_carrier_layout(charts, sp_, carrier_n=a.carrier_n)
        try:
            labels[off] = sbv.cell_label(sp_, gm_, layouts[off], out / f"cell_{off[0]}{off[1]}{off[2]}", a)
        except sss.EmptyCellError:
            empty.append(off); print(f"cell {off}: EMPTY", flush=True)
    cells = [c for c in cells if c[0] not in empty]; rep["cells"] = len(cells); rep["empty_cells"] = [list(o) for o in empty]

    # ---------------- block surface and Tet10 mesh (port constraints from every cell's own carrier trace) ----------------
    shells = {off: dict(zip(("coords", "tris", "per_face"), cpm.carrier_shell(layouts[off], tuple(str(t.source_id) for t in layouts[off].local_traces)))) for off, _, _ in cells}
    def face_constraint(src, shifts):
        pts = []; segs = []; keysl = []; idx = {}
        for sh in shifts:
            found = [t for s_, t in shells[sh]["per_face"] if s_ == src]
            if not found: continue
            co = shells[sh]["coords"]; shv = np.array(sh, dtype=float)
            for f in found[0]:
                for v in f:
                    k = (int(v), sh)
                    if k not in idx: idx[k] = len(pts); pts.append(co[v] + shv); keysl.append(k)
                for a_, b_ in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])): segs.append((idx[(int(a_), sh)], idx[(int(b_), sh)]))
        return np.asarray(pts).reshape(-1, 3), np.asarray(sorted({(min(x), max(x)) for x in segs}), dtype=np.int64).reshape(-1, 2), keysl
    pc = {src: face_constraint(src, [off for off, _, _ in cells if off[ax] == (shape[ax] - 1 if side else 0)]) for src, (ax, side) in FACE_AXIS.items()}
    pc["cut_0"] = face_constraint("cut_0", [off for off, _, _ in cells])
    bg = sbv.BlockGeom(spec, geom, shape)
    surf = sss.build_sheet_solid_surface(bg, layouts[(0, 0, 0)], n_per_unit=a.n_per_unit, remesh_size=a.remesh_size, carrier_n=a.carrier_n, port_constraints=pc, pad=6)
    if surf.report["watertight"]["non_two_manifold_edges"] or surf.report["self_intersecting_triangles"]:
        raise SystemExit("block surface rejected")
    m = sss.mesh_sheet_solid(surf, out / "block.mesh", interior_size_max=a.size_max, algorithm3d=a.algorithm3d, threads=a.threads)
    rep["block_mesh"] = {"nodes": m.node_count, "tets": m.tet_count, "min_dihedral": m.min_dihedral_degrees, "tets_below_5deg": m.tets_below_5deg}
    print("block mesh:", json.dumps(rep["block_mesh"]), flush=True)
    V, F, labs = surf.V, surf.F, surf.labels
    L = (out / "block.mesh").read_text().split("\n"); i = L.index("Vertices"); nv = int(L[i + 1])
    nxyz = np.array([list(map(float, l.split()[:3])) for l in L[i + 2:i + 2 + nv]])
    j = L.index("Tetrahedra"); nt = int(L[j + 1]); tets = np.array([list(map(int, l.split()[:4])) for l in L[j + 2:j + 2 + nt]]) - 1
    nodes10, el10, edges = tet4_to_tet10(nxyz, tets); n10 = len(nodes10)
    t0 = time.perf_counter(); K = assemble_global_tet10_stiffness(nodes10, el10, E=1.0, nu=0.3, workers=a.workers).tocsr()
    print(f"block Tet10 dof {3 * n10}, assembled {time.perf_counter() - t0:.0f}s", flush=True)
    fine_lookup = {KEY(p): k for k, p in enumerate(nxyz)}

    # ---------------- carrier of the block: active nodes (the labels' dof) then every remaining cut-trace node ----------------
    uk: dict = {}; uc: list = []
    def uid(p):
        k = KEY(p)
        if k not in uk: uk[k] = len(uc); uc.append(np.asarray(k))
        return uk[k]
    maps = {}; c2s = {}
    for off, _, _ in cells:
        port, active, S_, lay2c = labels[off]
        cc = np.asarray(port.carrier_coordinates)[active]; maps[off] = np.array([uid(p + np.array(off, float)) for p in cc])
        m_ = np.full(len(active), -1); m_[active] = np.arange(int(active.sum())); c2s[off] = m_
    nA = len(uc)                                          # active carrier nodes: the assembled operator's dof
    joint_tris = [];
    for off, _, _ in cells:
        found = [t for s_, t in shells[off]["per_face"] if s_ == "cut_0"]
        if not found: continue
        co = shells[off]["coords"]; shv = np.array(off, dtype=float)
        for f in found[0]: joint_tris.append([uid(co[v] + shv) for v in f])
    joint_tris = np.unique(np.asarray(sorted(tuple(sorted(t)) for t in joint_tris), dtype=np.int64), axis=0)
    nU = len(uc); ucoords = np.asarray(uc)
    joint_nodes = np.unique(joint_tris)
    rep["carrier"] = {"active_nodes": nA, "union_nodes": nU, "joint_nodes": int(len(joint_nodes)), "joint_triangles": int(len(joint_tris)),
                      "joint_nodes_inactive": int((joint_nodes >= nA).sum())}
    print("carrier:", json.dumps(rep["carrier"]), flush=True)
    rep["carrier"]["joint_edges"] = cpm.watertight_report(joint_tris)
    KU = np.zeros((3 * nU, 3 * nU))
    for off, _, _ in cells:
        mm = maps[off]; Sm = labels[off][2]; idx = (3 * mm[:, None] + np.arange(3)).ravel(); KU[np.ix_(idx, idx)] += Sm

    # ---------------- the skin ----------------
    n_cut = np.asarray(sbv.local_cut_plane(spec)[:3], float); n_cut /= np.linalg.norm(n_cut)   # points to the discarded side
    jxyz, jtris, Pj = refine_with_prolongation(ucoords[joint_nodes].copy(), np.searchsorted(joint_nodes, joint_tris), a.skin_refine)
    snodes4, stets4, svol = extrude_slab(jxyz, jtris, n_cut, a.skin_thickness)
    snodes, stets, sedges = tet4_to_tet10(snodes4, stets4); ns = len(snodes)
    t0 = time.perf_counter(); Ks = assemble_global_tet10_stiffness(snodes, stets, E=a.skin_modulus, nu=0.3, workers=a.workers).tocsr()
    rep["skin_mesh"] = {"joint_nodes": int(len(jxyz)), "joint_triangles": int(len(jtris)), "tet10_nodes": ns, "tets": int(len(stets)), "volume": svol,
                        "joint_edge_median": float(np.median(np.linalg.norm(jxyz[jtris[:, 0]] - jxyz[jtris[:, 1]], axis=1)))}
    print(f"skin: {json.dumps(rep['skin_mesh'])}, assembled {time.perf_counter() - t0:.0f}s", flush=True)
    nj = len(jxyz)                                        # skin nodes 0..nj-1 are the joint layer (bottom of the slab)
    # skin joint layer -> carrier union dof (contract): the refinement prolongation composed with the joint node map
    Pjc = Pj.tocoo(); Pj_to_U = sp.coo_matrix((Pjc.data, (Pjc.row, joint_nodes[Pjc.col])), shape=(len(jxyz), nU)).tocsr()

    # ---------------- ties of the block's external port faces to the carrier ----------------
    def face_P(src, off):
        ax, side = FACE_AXIS[src]; fv = np.unique(F[labs == src]); offv = np.array(off, float)
        lay = layouts[off]; traces = {str(t.source_id): t for t in lay.local_traces}
        if src not in traces or not len(fv): return np.zeros(0, np.int64), sp.csr_matrix((0, lay.scalar_node_count))
        fxyz = V[fv]; inside = np.ones(len(fv), bool)
        for k in range(3):
            if k != ax: inside &= (fxyz[:, k] >= offv[k] - 1e-9) & (fxyz[:, k] <= offv[k] + 1.0 + 1e-9)
        fv = fv[inside]; fxyz = fxyz[inside]
        if not len(fv): return np.zeros(0, np.int64), sp.csr_matrix((0, lay.scalar_node_count))
        Ploc = build_global_cell_patch_p1_prolongation(traces[src], fxyz - offv, geometric_tolerance=1e-9)
        return np.array([fine_lookup[KEY(p)] for p in fxyz]), (Ploc @ lay.local_to_global_scalar[src].tocsr()).tocsr()
    def cut_P(off):
        fv = np.unique(F[labs == "cut_0"]); offv = np.array(off, float); lay = layouts[off]
        traces = {str(t.source_id): t for t in lay.local_traces}
        if "cut_0" not in traces or not len(fv): return np.zeros(0, np.int64), sp.csr_matrix((0, lay.scalar_node_count))
        fxyz = V[fv]; inside = np.all((fxyz >= offv - 1e-9) & (fxyz <= offv + 1.0 + 1e-9), axis=1)
        fv = fv[inside]; fxyz = fxyz[inside]
        if not len(fv): return np.zeros(0, np.int64), sp.csr_matrix((0, lay.scalar_node_count))
        Ploc = build_global_cell_patch_p1_prolongation(traces["cut_0"], fxyz - offv, geometric_tolerance=1e-9)
        return np.array([fine_lookup[KEY(p)] for p in fxyz]), (Ploc @ lay.local_to_global_scalar["cut_0"].tocsr()).tocsr()
    def layout_to_U(off, ncols):
        """layout global scalar index -> carrier union index (active nodes and every cut-trace node), -1 elsewhere."""
        co = np.asarray(layouts[off].global_scalar_node_coordinates) + np.array(off, float)
        return np.array([uk.get(KEY(p), -1) for p in co[:ncols]])
    tied: dict[int, dict[int, float]] = {}
    jobs = [(src, off) for src, (ax, side) in FACE_AXIS.items() if src != "box_x_min" for off, _, _ in cells if off[ax] == (shape[ax] - 1 if side else 0)]
    jobs += [("cut_0", off) for off, _, _ in cells]
    for src, off in jobs:
        fids, Pg = cut_P(off) if src == "cut_0" else face_P(src, off)
        if not len(fids): continue
        cols = layout_to_U(off, Pg.shape[1]); Pl = Pg.tolil()
        for r, fid in enumerate(fids):
            w: dict[int, float] = {}
            for jj, val in zip(Pl.rows[r], Pl.data[r]):
                if cols[jj] < 0: continue
                w[int(cols[jj])] = w.get(int(cols[jj]), 0.0) + float(val)
            tot = sum(w.values())
            if not w or abs(tot - 1.0) > 1e-6: continue
            tied[int(fid)] = {g: v / tot for g, v in w.items()}
    cut_fine = {int(v) for v in np.unique(F[labs == "cut_0"])}
    cut_tied = {fine_lookup[KEY(V[v])] for v in cut_fine if KEY(V[v]) in fine_lookup}
    mid_tied = {}
    for (p_, q_), k in edges.items():
        if p_ in tied and q_ in tied:
            w = {}
            for src_w in (tied[p_], tied[q_]):
                for g_, val in src_w.items(): w[g_] = w.get(g_, 0.0) + 0.5 * val
            mid_tied[nv + k] = w
    tied_all = dict(tied); tied_all.update(mid_tied)
    rep["tied_fine_nodes"] = len(tied_all)

    # the block's cut-cap fine nodes (corner + midside), which REF slaves to the skin instead of the carrier
    cut_slave = sorted({f for f in cut_tied if f in tied_all} | {nv + k for (p_, q_), k in edges.items() if p_ in cut_tied and q_ in cut_tied and (nv + k) in tied_all})
    rep["cut_fine_nodes_slaved"] = len(cut_slave)
    print(f"ties: {len(tied_all)} fine nodes to the carrier, of them {len(cut_slave)} on the cut cap", flush=True)

    # REF: those nodes follow the skin's P1 field instead.  Locate each in a skin joint triangle (barycentric).
    from scipy.spatial import cKDTree
    cen = jxyz[jtris].mean(axis=1); tree = cKDTree(cen)
    e1 = jxyz[jtris[:, 1]] - jxyz[jtris[:, 0]]; e2 = jxyz[jtris[:, 2]] - jxyz[jtris[:, 0]]
    skin_tie: dict[int, dict[int, float]] = {}; missed = 0
    for fid in cut_slave:
        p = nodes10[fid]; best = None
        for ti in tree.query(p, k=32)[1]:
            A = np.column_stack([e1[ti], e2[ti], n_cut]);
            try: lam = np.linalg.solve(A, p - jxyz[jtris[ti, 0]])
            except np.linalg.LinAlgError: continue
            w = np.array([1 - lam[0] - lam[1], lam[0], lam[1]])
            if w.min() > -1e-7 and abs(lam[2]) < 1e-7:
                best = (ti, np.clip(w, 0, None)); break
        if best is None: missed += 1; continue
        ti, w = best; w = w / w.sum(); skin_tie[fid] = {int(jtris[ti, k]): float(w[k]) for k in range(3)}
    rep["cut_fine_nodes_not_on_skin"] = missed
    print(f"REF: {len(skin_tie)} cut-cap fine nodes slaved to the skin ({missed} not located)", flush=True)
    if missed: raise SystemExit("a cut-cap fine node is not inside the skin joint mesh")

    # ---------------- loads: nodal tractions on the skin's outer face (identical in all three arms) ----------------
    fixed_fine = np.where(np.abs(nodes10[:, 0]) < 1e-9)[0]
    area = np.zeros(nj)
    for tri in jtris:
        pa, pb, pc = jxyz[tri]; A_ = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa))
        for v in tri: area[v] += A_ / 3.0
    basis = np.linalg.svd(np.eye(3) - np.outer(n_cut, n_cut))[0][:, :2].T          # two in-plane directions
    u_, v_ = basis[0], basis[1]
    pu = jxyz @ u_; pv = jxyz @ v_                                                  # physical in-plane coordinates
    su = (pu - pu.min()) / max(float(np.ptp(pu)), 1e-12); sv = (pv - pv.min()) / max(float(np.ptp(pv)), 1e-12)
    cu, cv = 0.5 * (pu.min() + pu.max()), 0.5 * (pv.min() + pv.max())
    extent = min(float(np.ptp(pu)), float(np.ptp(pv))); R_patch = 0.25 * extent      # a fitting footprint
    rr = np.sqrt((pu - cu) ** 2 + (pv - cv) ** 2)
    def traction(kind):
        """Service load cases of a sandwich panel skin (the first five) and a sinusoidal sweep (the last three), which
        is the restriction's transfer function against in-plane wavelength, not a load anyone applies."""
        f = np.zeros((nj, 3))
        if kind == "flatwise_tension": f += n_cut                                   # ASTM C297: uniform pull-off
        elif kind == "core_shear": f += u_                                          # ASTM C273: uniform in-plane shear
        elif kind == "pressure_gradient": f += np.outer(2 * (sv - 0.5), n_cut)      # panel flexure: linear normal pressure
        elif kind == "membrane_bending": f += np.outer(2 * (sv - 0.5), u_)          # in-plane bending of the skin
        elif kind == "fitting_patch": f[rr <= R_patch] += n_cut                     # a bolted insert footprint
        elif kind == "pressure_step": f[sv > 0.5] += n_cut                          # pressure over half the panel
        elif kind == "sine_16H": f += np.outer(np.sin(2 * np.pi * su) * np.sin(2 * np.pi * sv), n_cut)
        elif kind == "sine_8H": f += np.outer(np.sin(4 * np.pi * su) * np.sin(4 * np.pi * sv), n_cut)
        elif kind == "sine_4H": f += np.outer(np.sin(8 * np.pi * su) * np.sin(8 * np.pi * sv), n_cut)
        return f * area[:, None]
    SERVICE = ("flatwise_tension", "core_shear", "pressure_gradient", "membrane_bending", "fitting_patch", "pressure_step")
    SWEEP = ("sine_16H", "sine_8H", "sine_4H")
    KINDS = SERVICE + SWEEP
    loads = {}
    for k in KINDS:
        fs = np.zeros(3 * ns); fs[(3 * (nj + np.arange(nj))[:, None] + np.arange(3)).ravel()] = traction(k).ravel(); loads[k] = fs
    rep["loads"] = {"service": list(SERVICE), "diagnostic_sweep": list(SWEEP), "in_plane_extent": [float(np.ptp(pu)), float(np.ptp(pv))],
                    "fitting_patch_radius": float(R_patch), "fitting_patch_radius_in_carrier_spacings": float(R_patch * a.carrier_n),
                    "sine_wavelengths_in_carrier_spacings": {k: float(np.ptp(pu) / n * a.carrier_n) for k, n in (("sine_16H", 2), ("sine_8H", 4), ("sine_4H", 8))}}
    print("loads:", json.dumps(rep["loads"]), flush=True)

    # ---------------- the three arms ----------------
    # carrier joint nodes are the first len(joint_nodes) nodes of the refined skin joint mesh (refinement appends),
    # so a carrier joint node and its skin node are the same unknown; the remaining carrier nodes (box faces) are
    # their own columns.
    j_of_u = np.full(nU, -1, dtype=np.int64); j_of_u[joint_nodes] = np.arange(len(joint_nodes))
    assert np.allclose(jxyz[:len(joint_nodes)], ucoords[joint_nodes], atol=1e-12), "refinement moved the carrier nodes"
    nonjoint = np.where(j_of_u < 0)[0]; c_of_u = np.full(nU, -1, dtype=np.int64); c_of_u[nonjoint] = np.arange(len(nonjoint))
    rep["carrier"]["nonjoint_nodes"] = int(len(nonjoint))

    def build(tag, *, use_block, joint_through, use_operator, skin_on_carrier=False):
        """Columns: [free block fine dof] [skin dof] [non-joint carrier dof]."""
        t0 = time.perf_counter()
        slaved = set(tied_all) if use_block else set()
        bfree = [i for i in range(n10) if i not in slaved] if use_block else []
        bcol = {nd: k for k, nd in enumerate(bfree)}; nb = len(bfree)
        sfree = list(range(nj, ns)) if skin_on_carrier else list(range(ns))
        scol = {nd: k for k, nd in enumerate(sfree)}; nsk = len(sfree)
        off_s = 3 * nb; off_c = off_s + 3 * nsk; ncarr = nU if skin_on_carrier else len(nonjoint); ncols = off_c + 3 * ncarr
        def ucol(g, c):
            """columns of carrier union node g: its skin node when it is a joint node (the joint is the carrier and the
            carrier nodes are skin nodes), else its own column.  With skin_on_carrier every carrier node owns a column."""
            g = int(g)
            if skin_on_carrier: return off_c + 3 * g + c
            if j_of_u[g] >= 0: return off_s + 3 * scol[int(j_of_u[g])] + c
            return off_c + 3 * int(c_of_u[g]) + c
        r_, c_, v_ = [], [], []
        if use_block:
            for nd in bfree:
                for c in range(3): r_.append(3 * nd + c); c_.append(3 * bcol[nd] + c); v_.append(1.0)
            for nd, w in tied_all.items():
                use_skin = joint_through == "skin" and nd in skin_tie
                src = skin_tie[nd] if use_skin else w
                for g_, val in src.items():
                    for c in range(3):
                        col = (off_s + 3 * scol[int(g_)] + c) if use_skin else ucol(g_, c)
                        if col is None: continue
                        r_.append(3 * nd + c); c_.append(col); v_.append(float(val))
        Tb = sp.coo_matrix((v_, (r_, c_)), shape=(3 * n10, ncols)).tocsr()
        r_, c_, v_ = [], [], []
        for nd in sfree:
            for c in range(3): r_.append(3 * nd + c); c_.append(off_s + 3 * scol[nd] + c); v_.append(1.0)
        if skin_on_carrier:                                   # the resolution variant: the skin itself sits on the carrier
            Pc = Pj_to_U.tocoo()
            for r0, c0, w0 in zip(Pc.row, Pc.col, Pc.data):
                for c in range(3):
                    col = ucol(c0, c)
                    if col is None: continue
                    r_.append(3 * int(r0) + c); c_.append(col); v_.append(float(w0))
        Ts = sp.coo_matrix((v_, (r_, c_)), shape=(3 * ns, ncols)).tocsr()
        Kt = (Ts.T @ Ks @ Ts).tocsr()
        if use_block: Kt = (Kt + (Tb.T @ K @ Tb)).tocsr()
        if use_operator:                                      # the cell operators, on carrier columns via ucol
            r_, c_, v_ = [], [], []
            for g in range(nU):
                for c in range(3):
                    col = ucol(g, c)
                    if col is None: continue
                    r_.append(3 * g + c); c_.append(col); v_.append(1.0)
            Tu = sp.coo_matrix((v_, (r_, c_)), shape=(3 * nU, ncols)).tocsr()
            Kt = (Kt + (Tu.T @ sp.csr_matrix(KU) @ Tu)).tocsr()
        fixed = np.zeros(ncols, bool)
        for nd in fixed_fine:
            if nd in bcol: fixed[3 * bcol[nd]:3 * bcol[nd] + 3] = True
        for g in np.where(np.abs(ucoords[:, 0]) < 1e-9)[0]:
            for c in range(3):
                col = ucol(g, c)
                if col is not None: fixed[col] = True
        free = (~fixed) & (np.diff(Kt.indptr) > 0)
        Kf = Kt[free][:, free].tocsr(); solver = pypardiso.PyPardisoSolver(); solver.set_iparm(1, 1); solver.set_iparm(2, 3); solver.factorize(Kf)
        print(f"  {tag}: dof {Kf.shape[0]} (block {3 * nb}, skin {3 * nsk}, carrier {3 * ncarr}), factorized {time.perf_counter() - t0:.0f}s", flush=True)
        return {"Ts": Ts, "free": free, "Kf": Kf, "solver": solver, "ncols": ncols, "tag": tag}

    print("arms:", flush=True)
    A_ref = build("REF (joint at full fidelity)", use_block=True, joint_through="skin", use_operator=False)
    A_con = build("CON (joint through the carrier)", use_block=True, joint_through="carrier", use_operator=False)
    A_asm = build("ASM (assembled cell operators)", use_block=False, joint_through="carrier", use_operator=True)
    A_res = build("RES (skin itself on the carrier)", use_block=True, joint_through="carrier", use_operator=False, skin_on_carrier=True) if a.skin_on_carrier else None

    top = nj + np.arange(nj)
    def run(arm, fs):
        f = arm["Ts"].T @ fs; z = np.zeros(arm["ncols"]); z[arm["free"]] = arm["solver"].solve(arm["Kf"], f[arm["free"]])
        u_skin = (arm["Ts"] @ z).reshape(-1, 3)
        return float(f @ z), u_skin[top]
    results = {}
    for kind in KINDS:
        fs = loads[kind]
        if kind == SWEEP[0]: print("  -- diagnostic sweep (transfer function, not service loads) --", flush=True)
        cR, uR = run(A_ref, fs); cC, uC = run(A_con, fs); cA, uA = run(A_asm, fs)
        nrm = max(float(np.linalg.norm(uR)), 1e-300)
        row = {"compliance_reference": cR, "compliance_contract": cC, "compliance_assembled": cA,
               "contract_vs_reference": cC / cR - 1.0, "assembled_vs_contract": cA / cC - 1.0,
               "skin_disp_rel_l2_contract_vs_reference": float(np.linalg.norm(uC - uR) / nrm),
               "skin_disp_rel_l2_assembled_vs_contract": float(np.linalg.norm(uA - uC) / max(float(np.linalg.norm(uC)), 1e-300))}
        if A_res is not None:
            cS, uS = run(A_res, fs); row["compliance_skin_on_carrier"] = cS; row["skin_on_carrier_vs_reference"] = cS / cR - 1.0
        results[kind] = row
        print(kind, json.dumps({k: (round(v, 6) if abs(v) > 1e-4 else f"{v:.3e}") for k, v in row.items()}), flush=True)
    rep["results"] = results
    (out / "SHEET_SKIN_VALIDATION.json").write_text(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
