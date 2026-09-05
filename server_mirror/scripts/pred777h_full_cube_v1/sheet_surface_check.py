#!/usr/bin/env python3
"""Surface (and optional Gmsh volume) robustness check of one sheet-TPMS cell manifest; prints one JSON line.

Used by the population census: material volume estimate, empty screen, remesh retries, watertightness, verified
self-intersections, cap classification conflicts, surface quality, and (with --mesh) the Gmsh tetrahedralization.
"""
import sys, json, time, argparse, numpy as np, tempfile
from pathlib import Path
R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R / 'src'))
from importlib import import_module
P = 'pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1'
fcb = import_module(f'{P}.full_cube_backend'); snap = import_module(f'{P}.mesher_neutral_snapshot'); sss = import_module(f'{P}.sheet_solid_surface'); pipe = import_module(f'{P}.sheet_label_pipeline')
ap = argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('n', type=int); ap.add_argument('h', type=float); ap.add_argument('--mesh', action='store_true'); ap.add_argument('--size-max', type=float, default=0.04); ap.add_argument('--carrier-n', type=int, default=32); ap.add_argument('--threads', type=int, default=2)
a = ap.parse_args(); gm = Path(a.manifest)
_, spec, geom = snap.load_geometry_manifest(gm)
_degenerate = pipe.cut_contains_a_cell_edge(spec)
if _degenerate is not None:
    print(json.dumps({'case': gm.parent.name, 'n': a.n, 'h': a.h, 'status': 'GEOMETRY_DEGENERATE', 'error': _degenerate['reason'],
                      'tau': [round(float(t), 4) for t in geom.tau_corners], 'cut': None if geom.cut_plane is None else [round(float(v), 4) for v in geom.cut_plane]}))
    sys.exit(0)
charts, *_ = fcb.compile_full_cube_geometry_inputs(spec)
layout = pipe.build_carrier_layout(charts, spec, carrier_n=a.carrier_n)
out = {'case': gm.parent.name, 'n': a.n, 'h': a.h, 'tau': [round(float(t), 4) for t in geom.tau_corners], 'cut': None if geom.cut_plane is None else [round(float(v), 4) for v in geom.cut_plane]}
n = 120; xs = (np.arange(n) + 0.5) / n; grid = np.stack(np.meshgrid(xs, xs, xs, indexing='ij'), axis=-1).reshape(-1, 3)
inside = sss.sheet_level(geom, grid) < 0.0
for _, nn, dd in sss.clip_planes(geom): inside &= (grid @ nn - dd) <= 0.0
out['volume_est'] = round(float(inside.mean()), 5)
if out['volume_est'] < 1e-6:
    out['status'] = 'EMPTY'; print(json.dumps(out)); sys.exit(0)
t = time.perf_counter()
try:
    r = sss.build_sheet_solid_surface(geom, layout, n_per_unit=a.n, remesh_size=a.h, carrier_n=a.carrier_n)
except Exception as e:
    out.update({'status': 'SURFACE_FAIL', 'error': str(e)[:300], 'seconds': round(time.perf_counter() - t, 1)}); print(json.dumps(out)); sys.exit(0)
rep = r.report; last = rep['remesh_attempts'][-1]
out['cap_conflicts'] = rep.get('cap_classification_conflicts'); out['t_junction_splits'] = rep.get('t_junction_splits'); out['resolution_effective'] = rep.get('resolution_effective')
out.update({'attempts': len(rep['remesh_attempts']), 'retry_errors': [x.get('error') for x in rep['remesh_attempts'] if x.get('error')],
            'watertight': rep['watertight']['non_two_manifold_edges'], 'selfx': rep['self_intersecting_triangles'], 'components': rep['components']['components'], 'dropped': rep['components']['dropped'],
            'q_min': round(rep['surface_quality']['min'], 5), 'q_sheet_min': round(rep['surface_quality_by_label'].get('tpms_free', {}).get('min', 1.0), 5),
            'stage_q_min': round(last.get('min_quality', 0), 4), 'collapsed': last.get('short_edges_collapsed'), 'needles': last.get('needles'), 'flat_dropped': last.get('flat_dropped'),
            'volume': round(rep['volume'], 5), 'triangles': int(len(r.F)), 'surface_seconds': round(time.perf_counter() - t, 1)})
out['status'] = 'OK' if (out['watertight'] == 0 and out['selfx'] == 0) else 'SURFACE_BAD'
if a.mesh and out['status'] == 'OK':
    t = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as td:
            m = sss.mesh_sheet_solid(r, Path(td) / 'mesh.mesh', interior_size_max=a.size_max, algorithm3d=1, threads=a.threads, isolate=True)
        out.update({'tets': m.tet_count, 'min_dihedral': round(m.min_dihedral_degrees, 3), 'tets_below_5deg': m.tets_below_5deg, 'mesh_volume': round(m.material_volume, 5), 'mesh_seconds': round(time.perf_counter() - t, 1)})
    except Exception as e:
        out.update({'status': 'MESH_FAIL', 'error': str(e)[:300], 'mesh_seconds': round(time.perf_counter() - t, 1)})
print(json.dumps(out, default=float))
