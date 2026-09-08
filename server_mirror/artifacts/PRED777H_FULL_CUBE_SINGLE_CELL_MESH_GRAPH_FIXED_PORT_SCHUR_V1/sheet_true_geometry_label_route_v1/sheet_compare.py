"""Compare sheet-geometry Tet10 fixed-port operators (supported carrier nodes only) in the carrier low-frequency norm."""
import sys, json, numpy as np
from pathlib import Path
R = Path('/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1'); sys.path.insert(0, str(R / 'src'))
from importlib import import_module
P = 'pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1'
adapter = import_module(f'{P}.cgal_mesh3_adapter'); fpa = import_module(f'{P}.fixed_port_adapter'); fcb = import_module(f'{P}.full_cube_backend')
snap = import_module(f'{P}.mesher_neutral_snapshot'); t10 = import_module(f'{P}.tet10_label')
import os
carrier_n = int(os.environ.get('CARRIER_N', '32')); gm = Path(sys.argv[1]); dirs = [Path(d) for d in sys.argv[2:]]
_, spec, geom = snap.load_geometry_manifest(gm)
charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
from cctpms.port.global_cell_patch_trace_layout import build_global_cell_patch_p1_trace_layout
layout = build_global_cell_patch_p1_trace_layout(charts, background_lower=(0.0,0.0,0.0), background_upper=(1.0,1.0,1.0), background_shape=(carrier_n,)*3, world_identity_rotation=spec.placement.rotation, world_identity_translation=spec.placement.translation)
# carrier identity + support from the first directory (all share the layout and the exact band, hence the support)
mp = {"domain_contract": "SHEET_SOLID_NO_COLLAR", "optimization_enabled": False, "semantic_plane_tolerance": 1e-10}
from dataclasses import replace as dc_replace
art = adapter.mesh_artifact_from_cgal_medit(dirs[0] / 'mesh.mesh', geometry=dc_replace(geom, cut_plane=None), mesher_version='5.4', mesher_parameters=mp, plane_tolerance=1e-10, source_geometry_hash=spec.spec_hash)
port = fpa.compile_fixed_port(art, spec, layout)
support = np.load(dirs[0] / 'carrier_support.npy')
for d in dirs[1:]:
    s2 = np.load(d / 'carrier_support.npy')
    if not np.array_equal(s2, support):
        print('WARNING support differs', d, int((s2 != support).sum()))
compact = dict(port.global_to_compact); n = len(port.active_global_carrier_ids)
M = np.zeros((n, n)); Lp = np.zeros((n, n))
for trace in layout.local_traces:
    if str(trace.source_id) not in port.active_global_port_ids: continue
    l2g = layout.local_to_global_scalar[str(trace.source_id)].tocsr(); ids = []
    for local in range(l2g.shape[0]):
        cols = l2g.indices[l2g.indptr[local]:l2g.indptr[local + 1]]; ids.append(compact[str(layout.global_scalar_node_keys[int(cols[0])])])
    X = np.asarray(trace.node_coordinates, dtype=np.float64)
    for tri in np.asarray(trace.triangles, dtype=np.int64):
        g = [ids[t] for t in tri]
        if not all(support[k] for k in g): continue
        p = X[tri]; e1 = p[1] - p[0]; e2 = p[2] - p[0]; area = 0.5 * np.linalg.norm(np.cross(e1, e2))
        if area <= 0: continue
        for a in range(3):
            for b in range(3): M[g[a], g[b]] += area / 12.0 * (2.0 if a == b else 1.0)
        # P1 stiffness on the triangle
        nrm = np.cross(e1, e2) / (2 * area); grads = [np.cross(nrm, p[(k + 2) % 3] - p[(k + 1) % 3]) / (2 * area) for k in range(3)]
        for a in range(3):
            for b in range(3): Lp[g[a], g[b]] += area * float(grads[a] @ grads[b])
idx = np.where(support)[0]; Ms = M[np.ix_(idx, idx)]; Ls = Lp[np.ix_(idx, idx)]
w, Vm = np.linalg.eigh(Ms); assert w.min() > 0, w.min()
Mih = Vm @ np.diag(w ** -0.5) @ Vm.T; mu, U = np.linalg.eigh(Mih @ Ls @ Mih); keep = mu <= (2 * np.pi / 0.25) ** 2  # wavelength >= 4H with H = 1/16 (fixed physical cutoff)
Ulow = Mih @ U[:, keep]; I3 = np.eye(3); W = np.kron(Mih, I3); Vl = np.kron(Ulow, I3)
print(f'supported carrier nodes {len(idx)}, q_supported {3*len(idx)}, lowfreq vector modes {Vl.shape[1]}')
S = {d.name: np.load(d / 'S_TET10_supported.npy') for d in dirs}
names = list(S)
def dist(Sa, Sb):
    D = Sa - Sb; out = {'raw': np.linalg.norm(D) / np.linalg.norm(Sb), 'whitened': np.linalg.norm(W @ D @ W) / np.linalg.norm(W @ Sb @ W)}
    Vb = Vl.T @ Sb @ Vl; Vd = Vl.T @ D @ Vl; out['lowfreq'] = np.linalg.norm(Vd) / np.linalg.norm(Vb)
    lam, Q = np.linalg.eigh(Vb); k = lam > 1e-9 * lam[-1]; r = np.einsum('ij,ij->j', Q[:, k], (Vl.T @ Sa @ Vl) @ Q[:, k]) / lam[k] - 1
    out['p50'] = float(np.median(r)); out['p95abs'] = float(np.quantile(np.abs(r), 0.95)); return out
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        o = dist(S[names[i]], S[names[j]]); print(f'{names[i]} vs {names[j]}: lowfreq {o["lowfreq"]:.4f} whitened {o["whitened"]:.4f} raw {o["raw"]:.4f} p50 {o["p50"]:+.4f} p95 {o["p95abs"]:.4f}')
