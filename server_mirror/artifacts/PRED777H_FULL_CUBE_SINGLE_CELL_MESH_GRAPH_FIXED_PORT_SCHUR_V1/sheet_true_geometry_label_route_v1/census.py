"""True-geometry census of one cell: material volume, active carrier nodes per face, fragments, watertightness."""
import sys, json, numpy as np
from pathlib import Path
R=Path('/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1'); sys.path.insert(0,str(R/'src'))
from importlib import import_module
P='pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1'
fcb=import_module(f'{P}.full_cube_backend'); snap=import_module(f'{P}.mesher_neutral_snapshot'); sss=import_module(f'{P}.sheet_solid_surface')
from cctpms.port.global_cell_patch_trace_layout import build_global_cell_patch_p1_trace_layout
S=R/'artifacts/PRED777H_FULL_CUBE_SINGLE_CELL_MESH_GRAPH_FIXED_PORT_SCHUR_V1/science_pilot16_v1/samples'
name=sys.argv[1]; carrier_n=int(sys.argv[2]) if len(sys.argv)>2 else 32
_,spec,geom=snap.load_geometry_manifest(S/name/'geometry_material_manifest.json')
charts,*_=fcb.compile_full_cube_geometry_inputs(spec)
layout=build_global_cell_patch_p1_trace_layout(charts, background_lower=(0.,0.,0.), background_upper=(1.,1.,1.), background_shape=(carrier_n,)*3,
                                               world_identity_rotation=spec.placement.rotation, world_identity_translation=spec.placement.translation)
n=220; xs=(np.arange(n)+0.5)/n; X,Y,Z=np.meshgrid(xs,xs,xs,indexing='ij'); Pn=np.stack([X,Y,Z],axis=-1).reshape(-1,3)
inside=sss.sheet_level(geom,Pn)<0.0
for _,nn,dd in sss.clip_planes(geom): inside &= (Pn@nn-dd)<=0.0
vol=float(inside.mean()); tau=float(np.mean([float(t) for t in geom.tau_corners]))
out={'case':name,'tau_mean':round(tau,4),'volume':round(vol,5),'cut':geom.cut_plane is not None,'carrier_n':carrier_n}
if vol<1e-6:
    out['status']='EMPTY'; print(json.dumps(out)); sys.exit(0)
active=list(geom.active_outer_port_sources())
_,per_face,_=sss.carrier_active_set(geom,layout,active)
out['active_nodes_per_face']={src:int(len(np.unique(t))) for src,t in per_face}
out['active_nodes_total']=int(sum(out['active_nodes_per_face'].values()))
out['faces_with_material']=int(sum(1 for v in out['active_nodes_per_face'].values() if v>0))
try:
    surf=sss.build_sheet_solid_surface(geom,layout,n_per_unit=48,remesh_size=0.03,smooth=False)
    r=surf.report; out['components']=r.get('components'); out['watertight']=r['watertight']['non_two_manifold_edges']
    out['surface_volume']=round(r['volume'],5); out['status']='OK'
except Exception as e:
    out['status']='SURFACE_FAIL'; out['error']=str(e)[:120]
print(json.dumps(out))
