"""Jump analysis: restrict each S(d) to the untouched x=0 port face block, compare neighbours in the face-restricted low-frequency norm."""
import json, glob, sys, numpy as np
from pathlib import Path
from fractions import Fraction
ROOT="/root/autodl-tmp/cut_control_tpms_v1_full_cube_single_cell_v1"; sys.path.insert(0, ROOT+"/src"); D=Path("/root/autodl-tmp/_claude_diag/sweep")
P="pred777h_full_cube_single_cell_mesh_graph_fixed_port_schur_v1"
from importlib import import_module
fcb=import_module(P+".full_cube_backend"); snap=import_module(P+".mesher_neutral_snapshot"); t10=import_module(P+".tet10_label")
rows=[]
for man in sorted(D.glob("*/geometry_material_manifest.json")):
    m=json.load(open(man)); c=m["cut_plane"]["raw_coefficients"]["d"]; d=Fraction(c["numerator"],c["denominator"]); lab=man.parent/"label"/"FIXED_PORT_SCHUR_TET10.npz"
    if not lab.exists(): continue
    z=np.load(lab, allow_pickle=True); ids=[str(x) for x in z["carrier_ids"]]; xyz=z["carrier_coordinates"]; S=z["schur"]
    _, spec, geom = snap.load_geometry_manifest(man); charts,*_=fcb.compile_full_cube_geometry_inputs(spec); layout=fcb.build_full_cube_trace_layout(spec, charts=charts)
    # x=0 face carrier nodes (identical set for every d): select by coordinate
    sel=np.where(np.abs(xyz[:,0])<1e-12)[0]; key=[tuple(np.round(xyz[i],10)) for i in sel]
    rec=json.load(open(man.parent/"label"/"LABEL_RECEIPT.json"))
    rows.append({"d":d,"S":S,"sel":sel,"key":key,"xyz":xyz,"q":S.shape[0],"vol":rec["mesh"]["material_volume"],"dof":rec["tet10"]["fine_dof"]})
rows.sort(key=lambda r:r["d"])
# common x0 ordering by coordinate key
ref=rows[0]; order0={k:i for i,k in enumerate(ref["key"])}
def block(r):
    idx=np.array([r["sel"][r["key"].index(k)] for k in ref["key"]]); dofs=(3*idx[:,None]+np.arange(3)).ravel(); return r["S"][np.ix_(dofs,dofs)]
# face-restricted mass/Laplacian on the x0 face: build from the layout of the reference (box_x_min trace)
_, spec, geom = snap.load_geometry_manifest(sorted(D.glob("*/geometry_material_manifest.json"))[0]); charts,*_=fcb.compile_full_cube_geometry_inputs(spec); layout=fcb.build_full_cube_trace_layout(spec, charts=charts)
tr=[t for t in layout.local_traces if str(t.source_id)=="box_x_min"][0]; X=np.asarray(tr.node_coordinates); T=np.asarray(tr.triangles)
n=len(X); M=np.zeros((n,n)); L=np.zeros((n,n))
for tri in T:
    p=X[tri]; e1=p[1]-p[0]; e2=p[2]-p[0]; nrm=np.cross(e1,e2); area=0.5*np.linalg.norm(nrm); nn=nrm/(2*area); g=[np.cross(nn,p[(k+2)%3]-p[(k+1)%3])/(2*area) for k in range(3)]
    for a in range(3):
        for b in range(3): M[tri[a],tri[b]]+=area/12*(2 if a==b else 1); L[tri[a],tri[b]]+=area*(g[a]@g[b])
# map trace nodes to ref key order
tkey=[tuple(np.round(x,10)) for x in X]; perm=np.array([tkey.index(k) for k in ref["key"]]); M=M[np.ix_(perm,perm)]; L=L[np.ix_(perm,perm)]
w,V=np.linalg.eigh(M); Mih=V@np.diag(w**-0.5)@V.T; mu,U=np.linalg.eigh(Mih@L@Mih); keep=mu<=(2*np.pi/0.25)**2; U=Mih@U[:,keep]; Vv=np.kron(U,np.eye(3))
def dist(A,B): return np.linalg.norm(Vv.T@(A-B)@Vv)/np.linalg.norm(Vv.T@B@Vv)
print(f"x0-face block: {len(ref['key'])} nodes, lowfreq modes {Vv.shape[1]}")
print(" d        q     dof     vol      |dS_x0|/|S_x0| vs previous d   (grid lines at multiples of 1/16 = 4/64)")
prev=None
for r in rows:
    B=block(r); s=f"{float(r['d']):.5f} ({r['d']})  q={r['q']:5d} dof={r['dof']:7d} vol={r['vol']:.5f}"
    if prev is not None: s+=f"   step {float(prev[0]-r['d']):.4f}: {dist(B,prev[1]):.5f}" + ("   <-- crosses grid line" if any(float(prev[0])>k/16>=float(r['d']) or float(prev[0])>=k/16>float(r['d']) for k in range(1,16)) else "")
    print(s); prev=(r["d"],B)
