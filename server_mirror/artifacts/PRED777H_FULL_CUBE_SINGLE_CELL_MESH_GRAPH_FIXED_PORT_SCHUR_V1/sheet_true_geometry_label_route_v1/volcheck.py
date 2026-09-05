"""Material volume of the sheet cell: exact Monte-Carlo/grid integration vs the mesh vs the paper fit rho = C/1.755."""
import sys, json, numpy as np
from pathlib import Path
R=Path('/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1'); sys.path.insert(0,str(R/'src'))
from importlib import import_module
P='pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1'
snap=import_module(f'{P}.mesher_neutral_snapshot'); sss=import_module(f'{P}.sheet_solid_surface')
S=R/'artifacts/PRED777H_FULL_CUBE_SINGLE_CELL_MESH_GRAPH_FIXED_PORT_SCHUR_V1/science_pilot16_v1/samples'
rows=[]
for name, meshdir in (('G0 uncut','science_00_BALANCED_00_00:/root/autodl-tmp/_claude_diag/sheet/G0_c32_alg1'),
                      ('G1 shallow cut','science_12_BALANCED_02_01:/root/autodl-tmp/_claude_diag/sheet/G1_c32_alg1'),
                      ('G5 thin+cut','science_10_BALANCED_03_07:/root/autodl-tmp/_claude_diag/sheet/G5_c32_alg1')):
    m, md = meshdir.split(':')
    _,spec,geom=snap.load_geometry_manifest(S/m/'geometry_material_manifest.json')
    n=400; xs=(np.arange(n)+0.5)/n
    X,Y,Z=np.meshgrid(xs,xs,xs,indexing='ij'); Pnts=np.stack([X,Y,Z],axis=-1).reshape(-1,3)
    inside=sss.sheet_level(geom,Pnts)<0.0
    for _,nn,dd in sss.clip_planes(geom): inside &= (Pnts@nn-dd)<=0.0
    vol_exact=float(inside.mean())
    tau=np.array([float(t) for t in geom.tau_corners]); tau_mean=float(tau.mean())
    rho_paper=tau_mean/1.755
    # uncut volume of the same tau field (for the cut cases the paper formula applies to the full cell)
    inside_full=sss.sheet_level(geom,Pnts)<0.0; vol_full=float(inside_full.mean())
    try:
        L=open(Path(md)/'mesh.mesh').read().split('\n'); i=L.index('Vertices'); nv=int(L[i+1])
        V=np.array([list(map(float,l.split()[:3])) for l in L[i+2:i+2+nv]])
        j=L.index('Tetrahedra'); nt=int(L[j+1]); T=np.array([list(map(int,l.split()[:4])) for l in L[j+2:j+2+nt]])-1
        Pp=V[T]; vol_mesh=float(np.abs(np.einsum('ij,ij->i',Pp[:,1]-Pp[:,0],np.cross(Pp[:,2]-Pp[:,0],Pp[:,3]-Pp[:,0]))).sum()/6)
    except Exception as e:
        vol_mesh=float('nan')
    rows.append((name,tau_mean,vol_full,vol_exact,vol_mesh,rho_paper))
print(f"{'case':16s} {'tau(mean C)':>11s} {'rho exact(full)':>15s} {'rho paper=C/1.755':>18s} {'dev':>7s} {'rho exact(clipped)':>18s} {'rho mesh':>9s} {'mesh-exact':>10s}")
for name,tau_mean,vf,ve,vm,rp in rows:
    print(f"{name:16s} {tau_mean:11.4f} {vf:15.4f} {rp:18.4f} {100*(vf/rp-1):6.2f}% {ve:18.4f} {vm:9.4f} {100*(vm/ve-1) if vm==vm else float('nan'):9.2f}%")
