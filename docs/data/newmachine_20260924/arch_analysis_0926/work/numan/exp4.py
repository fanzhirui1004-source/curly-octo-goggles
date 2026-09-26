import numpy as np, toy, scipy.linalg as sla
from exp1 import probes
from exp3 import coarse_interp
rng=np.random.default_rng(11)
def perturbed_geo(c,sig,smooth=False):
    if smooth:   # spatially smooth multiplicative stiffness error (misread wall thickness field)
        X=np.stack([(c.ae//c.n+0.5)*c.h,(c.ae%c.n+0.5)*c.h],1)
        f=np.zeros(len(c.ae))
        for j in range(3):
            for l in range(3):
                f+=rng.normal()*np.cos(np.pi*j*X[:,0]+rng.uniform(0,6))*np.cos(np.pi*l*X[:,1]+rng.uniform(0,6))
        f/=f.std()
    else: f=rng.normal(size=len(c.ae))
    Kp=np.zeros_like(c.K)
    for k_,e in enumerate(c.ae):
        d=c.edofs[k_]; Kp[np.ix_(d,d)]+=(c.chi[e]*np.exp(sig*f[k_])+3e-3)*c.K0
    return -np.linalg.solve(Kp[np.ix_(c.I,c.I)],Kp[np.ix_(c.I,c.P)])
def galerkin(c,V,EI0):
    # u <- u + V (V^T K V)^-1 V^T r, r = -K_IP - K_II EI0 (all port unit vectors)
    r=-c.KIP-c.KII@EI0
    Kc=V.T@c.KII@V
    return EI0+V@np.linalg.solve(Kc,V.T@r)
def geo_coarse(c,m=4,rbm=False):
    # hat functions of coarse grid (spacing m*h) restricted to interior DOFs, 2 comps (+ rotation-enriched: x*hat, y*hat)
    gi=np.rint((c.X-np.array([c.x0,0]))/c.h).astype(int)[c.int_nodes]
    nc=c.n//m+1; cols=[]
    for I in range(nc):
        for J in range(nc):
            w=np.clip(1-np.abs(gi[:,0]-I*m)/m,0,None)*np.clip(1-np.abs(gi[:,1]-J*m)/m,0,None)
            if w.sum()==0: continue
            for d in range(2):
                v=np.zeros(len(c.I)); v[d::2]=w; cols.append(v)
            if rbm:
                v=np.zeros(len(c.I)); xr=(gi[:,0]-I*m)*c.h; yr=(gi[:,1]-J*m)*c.h
                v[0::2]=-yr*w; v[1::2]=xr*w; cols.append(v)
    V=np.array(cols).T
    keep=np.linalg.norm(V,axis=0)>0
    return V[:,keep]
def metrics(c,EIh,q):
    uI=c.EI@q; u=c.full(q,uI); En=np.einsum('ij,ij->j',u,c.K@u)
    e=(EIh-c.EI)@q; return np.mean(np.einsum('ij,ij->j',e,c.KII@e)/En)
for name,kw in [('FULL',{}),('medium',{'xcut':0.55}),('heavy',{'xcut':0.25})]:
    c=toy.Cell(32,[0.5,0.3,0.6,0.4],**kw); q=probes(c)
    Dm=1/np.sqrt(c.D); lam,W=np.linalg.eigh(c.KII*Dm[:,None]*Dm[None,:]); Vex=W*Dm[:,None]   # K_II v = lam D v
    print(f'== {name} nI={len(c.I)}')
    models={'M4 local stiffness err 20%':perturbed_geo(c,0.2),'M4s smooth stiffness err 20%':perturbed_geo(c,0.2,True),
            'M1+M4 mix':c.EI+0.1*(coarse_interp(c)-c.EI)+(perturbed_geo(c,0.15)-c.EI)}
    Vg=geo_coarse(c,4); Vr=geo_coarse(c,4,True); Vg8=geo_coarse(c,8,True)
    for mn,EIh in models.items():
        e0=metrics(c,EIh,q); row=[f'eps0={e0:.2e}']
        for k in [8]:
            row.append(f'tail{k}={metrics(c,c.cheb(EIh,np.eye(len(c.P)),k),q)/e0:.3f}')
        for m in [10,40]:
            row.append(f'eig{m}+t8={metrics(c,c.cheb(galerkin(c,Vex[:,:m],EIh),np.eye(len(c.P)),8),q)/e0:.4f}')
        row.append(f'geoH4({Vg.shape[1]})+t8={metrics(c,c.cheb(galerkin(c,Vg,EIh),np.eye(len(c.P)),8),q)/e0:.4f}')
        row.append(f'geoH4rbm({Vr.shape[1]})+t8={metrics(c,c.cheb(galerkin(c,Vr,EIh),np.eye(len(c.P)),8),q)/e0:.4f}')
        row.append(f'geoH8rbm({Vg8.shape[1]})+t8={metrics(c,c.cheb(galerkin(c,Vg8,EIh),np.eye(len(c.P)),8),q)/e0:.4f}')
        # network-span: interior outputs of the 'network' on a random bank of m smooth port directions (q-independent)
        for m in [12,40]:
            Qb=probes(c,m=m,kind='grf'); Vn=np.linalg.qr(EIh@Qb)[0]
            row.append(f'netspan{m}+t8={metrics(c,c.cheb(galerkin(c,Vn,EIh),np.eye(len(c.P)),8),q)/e0:.4f}')
        print(f'  {mn:30s}',' '.join(row))
    # zero start for reference
    Z=np.zeros_like(c.EI); e0=metrics(c,Z,q)
    print(f'  {"zero start":30s} eps0={e0:.1f} tail8={metrics(c,c.cheb(Z,np.eye(len(c.P)),8),q):.3f} geoH4rbm+t8={metrics(c,c.cheb(galerkin(c,Vr,c.cheb(Z,np.eye(len(c.P)),8)),np.eye(len(c.P)),8),q):.4f} (abs)')
