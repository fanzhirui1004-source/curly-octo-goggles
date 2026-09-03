"""Physics validation: 2x1x1 block of two G0 cells. Monolithic Tet10 solve with a FREE interface vs fixed-port assembly
(S_A + S_B on the shared carrier).  Loads: nodal forces on carrier nodes of the x=2 face; x=0 carrier face fixed."""
import sys, time, json, numpy as np, scipy.sparse as sp
from pathlib import Path
from collections import defaultdict
ROOT="/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1"; sys.path.insert(0, ROOT+"/src"); D="/root/autodl-tmp/_claude_diag"
P="pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
cpm=import_module(P+".conforming_port_mesh"); fcb=import_module(P+".full_cube_backend"); snap=import_module(P+".mesher_neutral_snapshot"); fpa=import_module(P+".fixed_port_adapter"); adapter=import_module(P+".cgal_mesh3_adapter"); t10=import_module(P+".tet10_label")
import_module(P+".frozen_backend").activate_frozen_backend()
from cctpms.fem.tet10 import assemble_global_tet10_stiffness
from cctpms.port.tet10_nodal_schur import TET10_EDGE_LOCAL_NODES
import pypardiso, gmsh

off=Path(sys.argv[1]); gm=Path(sys.argv[2]); S_npy=Path(sys.argv[3]); k=int(sys.argv[4]); size=float(sys.argv[5]); out=Path(sys.argv[6]); out.mkdir(parents=True, exist_ok=True)
_, spec, geom = snap.load_geometry_manifest(gm); charts,*_=fcb.compile_full_cube_geometry_inputs(spec); layout=fcb.build_full_cube_trace_layout(spec, charts=charts); active=tuple(sorted(str(c.source_id) for c in charts.charts))
V,F=cpm.read_off(off); on=np.zeros(len(F),bool)
for name,n,d in cpm._port_planes(geom):
    if name in active: on|=np.all(np.abs(V[F]@n-d)<1e-9,axis=1)
inner=F[~on]
coords,tris,per_face=cpm.carrier_shell(layout,active); ncar=len(coords)
coordsR,per_faceR=cpm.refine_shell(coords,per_face,k)
# ---- monolithic 2x1x1 shell: cell A faces except x_max, cell B (translated +1 in x) faces except x_min ----
shift=np.array([1.0,0,0])
def add_surface(tag, C, T, key, owned, cnt):
    gmsh.model.addDiscreteEntity(2, tag); ids=[]; xyz=[]
    for v in np.unique(T):
        kk=(key,int(v))
        if kk not in owned: cnt[0]+=1; owned[kk]=cnt[0]; ids.append(cnt[0]); xyz.extend(C[v].tolist())
    if ids: gmsh.model.mesh.addNodes(2, tag, ids, xyz)
    gmsh.model.mesh.addElementsByType(tag, 2, [], [owned[(key,int(v))] for f in T for v in f])
t0=time.perf_counter(); gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0); gmsh.model.add("block")
owned={}; cnt=[0]; tag=0; outer_tags=[]
# shared-edge consistency between A and B faces: nodes on the plane x=1 belong to A's x_max face (excluded) -> the y/z faces of A and B
# meet along x=1 edges; those nodes must be identical: key nodes by rounded coordinates instead of (cell, id)
def add_surface_xyz(tag_, XYZ, T):
    gmsh.model.addDiscreteEntity(2, tag_); ids=[]; xyz=[]; loc=[]
    for v in np.unique(T):
        key=tuple(np.round(XYZ[v],12))
        if key not in owned: cnt[0]+=1; owned[key]=cnt[0]; ids.append(cnt[0]); xyz.extend(XYZ[v].tolist())
    if ids: gmsh.model.mesh.addNodes(2, tag_, ids, xyz)
    gmsh.model.mesh.addElementsByType(tag_, 2, [], [owned[tuple(np.round(XYZ[v],12))] for f in T for v in f])
shellA=[(s,t) for s,t in per_faceR if s!="box_x_max"]; shellB=[(s,t) for s,t in per_faceR if s!="box_x_min"]
if cpm.signed_volume(coordsR,np.vstack([t for _,t in per_faceR]))<0: shellA=[(s,t[:,[0,2,1]]) for s,t in shellA]; shellB=[(s,t[:,[0,2,1]]) for s,t in shellB]
for s,t in shellA: tag+=1; add_surface_xyz(tag, coordsR, t); outer_tags.append(tag)
for s,t in shellB: tag+=1; add_surface_xyz(tag, coordsR+shift, t); outer_tags.append(tag)
comps=cpm._components(inner); cav_tags=[]
for XYZ in (V, V+shift):
    for c in comps:
        t=inner[c]; t=t[:,[0,2,1]] if cpm.signed_volume(XYZ,t)>0 else t; tag+=1; add_surface_xyz(tag, XYZ, t); cav_tags.append(tag)
loops=[gmsh.model.geo.addSurfaceLoop(outer_tags)]+[gmsh.model.geo.addSurfaceLoop([c]) for c in cav_tags]
gmsh.model.geo.addVolume(loops); gmsh.model.geo.synchronize()
for kk,v in (("Mesh.Algorithm3D",10),("Mesh.MeshSizeMax",size),("Mesh.MeshSizeExtendFromBoundary",1),("Mesh.Optimize",1),("Mesh.OptimizeNetgen",1),("General.NumThreads",8)): gmsh.option.setNumber(kk,v)
gmsh.model.mesh.generate(3); gmsh.model.mesh.optimize("Netgen",niter=2)
ntags,nxyz,_=gmsh.model.mesh.getNodes(); nxyz=np.asarray(nxyz).reshape(-1,3); order={int(t):i for i,t in enumerate(ntags)}
_,_,en=gmsh.model.mesh.getElements(3); tets=np.vectorize(order.get)(np.asarray(en[0]).reshape(-1,4)); gmsh.finalize()
Pp=nxyz[tets]; vol=np.einsum('ij,ij->i',Pp[:,1]-Pp[:,0],np.cross(Pp[:,2]-Pp[:,0],Pp[:,3]-Pp[:,0]))/6; neg=vol<0; tets[neg,1],tets[neg,2]=tets[neg,2].copy(),tets[neg,1].copy()
print(f"monolithic mesh: {len(nxyz)} nodes, {len(tets)} tets, volume {abs(vol).sum():.5f} (2 cells = {2*0.386472:.5f}), {time.perf_counter()-t0:.0f}s", flush=True)
# ---- Tet10 monolithic stiffness ----
edges={}
def eid(a,b):
    kk=(a,b) if a<b else (b,a)
    if kk not in edges: edges[kk]=len(edges)
    return edges[kk]
el10=np.empty((len(tets),10),np.int64); el10[:,:4]=tets
for e,t in enumerate(tets):
    for j,(a,b) in enumerate(TET10_EDGE_LOCAL_NODES): el10[e,4+j]=len(nxyz)+eid(int(t[a]),int(t[b]))
mids=np.zeros((len(edges),3))
for (a,b),i in edges.items(): mids[i]=0.5*(nxyz[a]+nxyz[b])
nodes10=np.vstack([nxyz,mids]); t0=time.perf_counter()
K=assemble_global_tet10_stiffness(nodes10, el10, E=1.0, nu=0.3, workers=8).tocsr(); n10=len(nodes10); print(f"Tet10 dof {3*n10}, assembled {time.perf_counter()-t0:.0f}s", flush=True)
# ---- carrier maps: A's carrier nodes (coords) and B's (coords+shift); shared face x=1 identified by coordinate ----
def key(p): return tuple(np.round(p,10))
carA=coords[:ncar]; carB=coords[:ncar]+shift
gkeys={}; 
def gid(p):
    kk=key(p)
    if kk not in gkeys: gkeys[kk]=len(gkeys)
    return gkeys[kk]
mapA=np.array([gid(p) for p in carA]); mapB=np.array([gid(p) for p in carB]); nG=len(gkeys); gcoords=np.zeros((nG,3))
for kk,i in gkeys.items(): gcoords[i]=kk
# fine node index of each global carrier node in the monolithic mesh
fine_lookup={key(p):i for i,p in enumerate(nxyz)}
fine_of_g=np.array([fine_lookup.get(key(gcoords[i]),-1) for i in range(nG)])
from scipy.spatial import cKDTree
tet_cent=nxyz[tets].mean(1); tet_tree=cKDTree(tet_cent)
def interp_vertex_field(u_vertex, pts):
    """P1 interpolation of a nodal (vertex) field at points, via point location in the Tet4 skeleton."""
    out=np.zeros((len(pts),3)); idx=tet_tree.query(pts,k=40)[1]
    for i,p in enumerate(pts):
        found=False
        for ti in idx[i]:
            a,b,c,d=nxyz[tets[ti]]; Mm=np.column_stack([b-a,c-a,d-a]); 
            try: l=np.linalg.solve(Mm,p-a)
            except np.linalg.LinAlgError: continue
            lam=np.array([1-l.sum(),*l])
            if lam.min()>-1e-8: out[i]=lam@u_vertex[tets[ti]]; found=True; break
        if not found: out[i]=np.nan
    return out
# ---- work-consistent loads: traction t(y,z) lumped on the k-refined VERTEX nodes of the x=2 face (monolithic),
#      and f_c = P^T f_fine on the carrier (assembled), where P is the exact nested interpolation of the face ----
x2=np.where(np.abs(gcoords[:,0]-2.0)<1e-9)[0]; x0=np.where(np.abs(gcoords[:,0]-0.0)<1e-9)[0]
faceB=[(s_,t_) for s_,t_ in per_faceR if s_=="box_x_max"][0][1]                      # refined triangles of B's x_max face (coordsR + shift)
fv=np.unique(faceB); fxyz=coordsR[fv]+shift; fpos={int(v):i for i,v in enumerate(fv)}
fine_face=np.array([fine_lookup[key(pp)] for pp in fxyz])                          # fine (vertex) node ids in the monolithic mesh
# lumped area weights on the refined face triangulation
wts=np.zeros(len(fv))
for tri in faceB:
    pp=coordsR[tri]; area=0.5*np.linalg.norm(np.cross(pp[1]-pp[0],pp[2]-pp[0]))
    for v in tri: wts[fpos[int(v)]]+=area/3
# nested interpolation P (fine face vertex -> carrier nodes of the same face) via the layout trace
from cctpms.port.global_cell_patch_trace_layout import build_global_cell_patch_p1_prolongation
trB=[t for t in layout.local_traces if str(t.source_id)=="box_x_max"][0]
Ploc=build_global_cell_patch_p1_prolongation(trB, coordsR[fv], geometric_tolerance=1e-10)   # (nf x n_trace_local)
l2g=layout.local_to_global_scalar["box_x_max"].tocsr(); Pglob=(Ploc@l2g).tocsr()               # (nf x n_global_carrier_of_A)
gcol=np.array([mapB[j] for j in range(ncar)])                                                  # A-carrier global column -> B-shifted global id
def traction(kind, yz):
    y,z=yz[:,0],yz[:,1]; t=np.zeros((len(yz),3))
    if kind=="tension_x": t[:,0]=1
    elif kind=="shear_y": t[:,1]=1
    elif kind=="bending_z": t[:,0]=z-0.5
    elif kind=="wave_8H": t[:,0]=np.sin(2*np.pi*y)*np.sin(2*np.pi*z)          # wavelength 1 = 16H
    elif kind=="wave_4H": t[:,0]=np.sin(4*np.pi*y)*np.sin(4*np.pi*z)          # wavelength 1/2 = 8H
    elif kind=="wave_2H": t[:,0]=np.sin(8*np.pi*y)*np.sin(8*np.pi*z)          # wavelength 1/4 = 4H
    return t
loads={}
for kind in ("tension_x","shear_y","bending_z","wave_8H","wave_4H","wave_2H"):
    tf=traction(kind, fxyz[:,1:])*wts[:,None]                                  # lumped fine nodal forces (nf,3)
    fc=np.zeros((nG,3)); fcA=Pglob.T@tf                                        # (n_global_carrier_A,3) in A-carrier column order
    for j in range(Pglob.shape[1]): fc[gcol[j]]+=fcA[j]
    loads[kind]=(tf, fc)
# monolithic: fix all fine nodes on x=0 (incl. midpoints)
fixed_fine=np.where(np.abs(nodes10[:,0])<1e-9)[0]; free=np.ones(3*n10,bool); free[(3*fixed_fine[:,None]+np.arange(3)).ravel()]=False
Kff=K[free][:,free].tocsr(); solver=pypardiso.PyPardisoSolver(); solver.set_iparm(1,1); solver.set_iparm(2,3); t0=time.perf_counter(); solver.factorize(Kff); print(f"monolithic factorized {time.perf_counter()-t0:.0f}s", flush=True)
# assembled: S_A, S_B from the cell label (same geometry, B translated => same S with mapB)
S=np.load(S_npy); q=S.shape[0]; assert q==3*ncar
# assembled global stiffness on nG carrier nodes
KG=np.zeros((3*nG,3*nG))
for m in (mapA,mapB):
    idx=(3*m[:,None]+np.arange(3)).ravel(); KG[np.ix_(idx,idx)]+=S
fixedG=np.zeros(3*nG,bool); fixedG[(3*x0[:,None]+np.arange(3)).ravel()]=True; freeG=~fixedG
KGf=KG[np.ix_(freeG,freeG)]
# constrained monolithic: fine VERTEX nodes of the x=2 face tied to carrier via Pglob (midpoint nodes tied via their edge endpoints)
nfine=len(nxyz); ncA=Pglob.shape[1]
rowsT=[]; colsT=[]; valsT=[]
face_vertex=set(int(v) for v in fine_face); mid_of_edge={}
for (a,b),i in edges.items():
    if a in face_vertex and b in face_vertex: mid_of_edge[nfine+i]=(a,b)
tied=set(face_vertex)|set(mid_of_edge); free_nodes=[i for i in range(n10) if i not in tied]
col_of_free={nd:i for i,nd in enumerate(free_nodes)}; nfree=len(free_nodes)
for nd in free_nodes:
    for c in range(3): rowsT.append(3*nd+c); colsT.append(3*col_of_free[nd]+c); valsT.append(1.0)
Pc=Pglob.tocoo(); fine_row_of={int(fine_face[i]):i for i in range(len(fine_face))}
def add_row(nd, weights):  # weights: dict carrierA_col -> w
    for j,w in weights.items():
        for c in range(3): rowsT.append(3*nd+c); colsT.append(3*nfree+3*j+c); valsT.append(w)
Pl=Pglob.tolil()
for v in face_vertex: add_row(v, {int(j):float(w) for j,w in zip(Pl.rows[fine_row_of[v]], Pl.data[fine_row_of[v]])})
for m_,(a,b) in mid_of_edge.items():
    wa={int(j):0.5*float(w) for j,w in zip(Pl.rows[fine_row_of[a]], Pl.data[fine_row_of[a]])}; wb={int(j):0.5*float(w) for j,w in zip(Pl.rows[fine_row_of[b]], Pl.data[fine_row_of[b]])}
    for j,w in wb.items(): wa[j]=wa.get(j,0.0)+w
    add_row(m_, wa)
Tc=sp.coo_matrix((valsT,(rowsT,colsT)),shape=(3*n10,3*nfree+3*ncA)).tocsr()
Kc=(Tc.T@K@Tc).tocsr(); fixed_c=np.zeros(Tc.shape[1],bool)
for nd in fixed_fine:
    if nd in col_of_free: fixed_c[3*col_of_free[nd]:3*col_of_free[nd]+3]=True
freec=(~fixed_c) & (np.diff(Kc.indptr) > 0); print("constrained system: dropped", int((np.diff(Kc.indptr) == 0).sum()), "empty (unsupported carrier) rows", flush=True); Kcf=Kc[freec][:,freec].tocsr(); solver_c=pypardiso.PyPardisoSolver(); solver_c.set_iparm(1,1); solver_c.set_iparm(2,3); solver_c.factorize(Kcf); print("constrained-monolithic factorized", flush=True)
results={}
for name,(tf,Lg) in loads.items():
    fG=Lg.ravel(); fF=np.zeros(3*n10); fF[(3*fine_face[:,None]+np.arange(3)).ravel()]=tf.ravel()
    uF=np.zeros(3*n10); uF[free]=solver.solve(Kff, fF[free]); uG=np.zeros(3*nG); uG[freeG]=np.linalg.solve(KGf, fG[freeG])
    comp_mono=float(fF@uF); comp_fp=float(fG@uG)
    fc_=Tc.T@fF; zc=np.zeros(Tc.shape[1]); zc[freec]=solver_c.solve(Kcf, fc_[freec]); uC=Tc@zc; comp_cm=float(fF@uC)
    # interface displacement at shared carrier nodes (x=1)
    x1=np.where(np.abs(gcoords[:,0]-1.0)<1e-9)[0]; ui_mono=interp_vertex_field(uF[:3*len(nxyz)].reshape(-1,3), gcoords[x1]).ravel(); ui_fp=uG[(3*x1[:,None]+np.arange(3)).ravel()]
    ok=~np.isnan(ui_mono); ui_mono=ui_mono[ok]; ui_fp=ui_fp[ok]
    # loaded-face displacement
    uL_mono=uF[(3*fine_of_g[x2][:,None]+np.arange(3)).ravel()]; uL_fp=uG[(3*x2[:,None]+np.arange(3)).ravel()]
    # work check: monolithic f.u vs assembled f_c.q are the same functional only through nesting; report both
    results[name]={"compliance_monolithic":comp_mono,"compliance_fixed_port":comp_fp,"compliance_rel_diff":comp_fp/comp_mono-1,"compliance_monolithic_x2_constrained":comp_cm,"constrained_vs_free":comp_cm/comp_mono-1,"assembled_vs_constrained":comp_fp/comp_cm-1,
                   "interface_disp_rel_l2":float(np.linalg.norm(ui_fp-ui_mono)/np.linalg.norm(ui_mono)),"loaded_face_disp_rel_l2":float(np.linalg.norm(uL_fp-uL_mono)/np.linalg.norm(uL_mono))}
    print(name, {kk:(round(v,6) if abs(v)>1e-3 else f"{v:.3e}") for kk,v in results[name].items()}, flush=True)
json.dump({"k":k,"size":size,"S":str(S_npy),"mono_tet10_dof":int(3*n10),"carrier_nodes":int(nG),"results":results}, open(out/"ASSEMBLY_VALIDATION.json","w"), indent=1)
