"""Schur back-substitution benchmark on a real k=1 mesh: SuperLU (current) vs MKL Pardiso multi-RHS, various chunk sizes."""
import sys, time, numpy as np, scipy.sparse as sp
from pathlib import Path
R = Path('/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1'); sys.path.insert(0, str(R / 'src'))
from importlib import import_module
P = 'pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1'
adapter = import_module(f'{P}.cgal_mesh3_adapter'); fpa = import_module(f'{P}.fixed_port_adapter'); fcb = import_module(f'{P}.full_cube_backend')
snap = import_module(f'{P}.mesher_neutral_snapshot'); t10 = import_module(f'{P}.tet10_label'); tcc = import_module(f'{P}.teacher_contract_calibration')
from cctpms.fem.tet10 import assemble_global_tet10_stiffness
case, mesh_path = sys.argv[1], Path(sys.argv[2]); gm = Path(sys.argv[3]); workers = int(sys.argv[4])
_, spec, geom = snap.load_geometry_manifest(gm)
charts, *_ = fcb.compile_full_cube_geometry_inputs(spec); layout = fcb.build_full_cube_trace_layout(spec, charts=charts)
mp = {"domain_contract": "COHERENT_TRIANGULATED_CLOSED_VOLUME_BOUNDING_POLYHEDRAL_COMPLEX", "optimization_enabled": False, "semantic_plane_tolerance": 1e-10}
art = adapter.mesh_artifact_from_cgal_medit(mesh_path, geometry=geom, mesher_version='5.4', mesher_parameters=mp, plane_tolerance=1e-10, source_geometry_hash=spec.spec_hash)
port = fpa.compile_fixed_port(art, spec, layout)
t0 = time.perf_counter(); promo = t10.promote_mesh_artifact_to_tet10(art, port)
K = assemble_global_tet10_stiffness(promo.node_coordinates, promo.elements, E=1.0, nu=0.3, workers=workers).tocsr(); n10 = len(promo.node_coordinates)
vec = sp.kron(promo.scalar_prolongation, sp.eye(3, format='csr'), format='csr')
constraint, _ = tcc.build_fixed_carrier_constraint(node_count=n10, port_node_indices=promo.port_node_indices, vector_prolongation=vec)
Kc = (constraint.T @ K @ constraint).tocsr(); Kc = (0.5 * (Kc + Kc.T)).tocsr(); q = 3 * len(port.active_global_carrier_ids); ni = Kc.shape[0] - q
Kqq = Kc[:q, :q].toarray(); Kqi = Kc[:q, q:].tocsr(); Kii = Kc[q:, q:].tocsr(); B = Kqi.T.tocsc()
print(f'{case}: fine dof {3*n10}, internal {ni}, q {q}, assembly+constraint {time.perf_counter()-t0:.1f}s', flush=True)
res = {}
# SuperLU, chunk 256 (current production path)
t0 = time.perf_counter(); from scipy.sparse.linalg import splu; lu = splu(Kii.tocsc(), permc_spec='COLAMD'); tf = time.perf_counter() - t0
t0 = time.perf_counter(); S = Kqq.copy()
for c0 in range(0, q, 256): S[:, c0:c0 + 256] -= Kqi @ lu.solve(B[:, c0:c0 + 256].toarray())
ts = time.perf_counter() - t0; res['superlu_256'] = S; print(f'superlu: factor {tf:.1f}s, solve {ts:.1f}s ({ts/q*1e3:.1f} ms/rhs)', flush=True); del lu
import pypardiso
for chunk in (256, 1024, 4096):
    t0 = time.perf_counter(); solver = pypardiso.PyPardisoSolver(); solver.set_iparm(1, 1); solver.set_iparm(2, 3); solver.factorize(Kii); tf = time.perf_counter() - t0
    t0 = time.perf_counter(); S = Kqq.copy()
    for c0 in range(0, q, chunk): S[:, c0:c0 + chunk] -= Kqi @ solver.solve(Kii, B[:, c0:c0 + chunk].toarray())
    ts = time.perf_counter() - t0; res[f'pardiso_{chunk}'] = S; solver.free_memory(everything=True)
    print(f'pardiso chunk {chunk}: factor {tf:.1f}s, solve {ts:.1f}s ({ts/q*1e3:.1f} ms/rhs)', flush=True)
ref = res['superlu_256']
for k, v in res.items(): print(k, 'max|dS|/max|S| vs superlu = %.2e' % (np.abs(v - ref).max() / np.abs(ref).max()))
