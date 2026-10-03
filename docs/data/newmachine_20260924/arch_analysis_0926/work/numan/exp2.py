import toy,numpy as np, scipy.linalg as sla
from exp1 import probes
rng=np.random.default_rng(1)
ks=[0,1,2,4,8,16,32]
def smooth_field(X,L=4):
    v=np.zeros(len(X))
    for j in range(L):
        for l in range(L):
            v+=rng.normal()*np.cos(np.pi*j*X[:,0]+rng.uniform(0,6.3))*np.cos(np.pi*l*X[:,1]+rng.uniform(0,6.3))/(1+j*j+l*l)
    return v
def analyze(c,EIh,label,q):
    uI=c.EI@q; u=c.full(q,uI); En=np.einsum('ij,ij->j',u,c.K@u)
    Dm=1/np.sqrt(c.D); lam,V=np.linalg.eigh(c.KII*Dm[:,None]*Dm[None,:])
    res=[]
    for k in ks:
        # apply tail as an operator on all port unit vectors -> Ehat_k
        Ek=c.cheb(EIh,np.eye(len(c.P)),k)
        F=Ek-c.EI
        dS=F.T@c.KII@F                      # S_hat - S (Galerkin identity, ports exact)
        eps=np.mean(np.einsum('ij,ij->j',q,dS@q)/En)
        # max relative spectral error of pencil on complement of rigid modes
        R=toy.rigid(c.X[c.port_nodes]); Q,_=np.linalg.qr(R); Z=sla.null_space(Q.T)
        Sr=Z.T@c.S@Z; dSr=Z.T@dS@Z
        dmax=sla.eigh(dSr,Sr,eigvals_only=True)[-1]
        res.append((k,eps,dmax))
    F0=EIh-c.EI
    y=V.T@((F0@q)/Dm[:,None]); en=(lam[:,None]*y**2).sum(1); en/=en.sum()
    print(f"{label:28s} eps_k/eps_0:", ' '.join(f"{r[1]/res[0][1]:.2e}" for r in res), f"| eps0={res[0][1]:.3f}",
          "| dmax_k:", ' '.join(f"{r[2]:.3g}" for r in res[:6]), f"| err energy <a: {en[lam<lam[-1]/30].sum():.3f}")
    return res
if __name__=='__main__':
  print('2010 observed net-start ratio: 1 .359 .169 .102 .0059 5.1e-4 9.5e-5')
  for name,kw in [('FULL',{}),('heavy',{'xcut':0.25})]:
    c=toy.Cell(32,[0.5,0.3,0.6,0.4],**kw); q=probes(c)
    print('==',name)
    nP=len(c.P); nI=len(c.I)
    # M2 white per-DOF relative noise on E (random linear map)
    W=rng.normal(size=c.EI.shape)*np.abs(c.EI).mean()
    analyze(c,c.EI+0.01*W,'M2 white noise',q)
    # M3 smooth amplitude error: EI*(1+ smooth field)
    Xi=np.repeat(c.X[c.int_nodes],2,0); s=smooth_field(Xi)
    analyze(c,c.EI*(1+0.2*s[:,None]),'M3 smooth modulation',q)
    # M1 coarse-grid interpolation of exact E (factor 4), bilinear, zero where inactive
    n=c.n; m=4; Xg=c.X[c.int_nodes]
    # sample exact full field on coarse nodes via nearest used node, interpolate bilinear
    Efull=np.zeros((2*c.nn,nP)); Efull[c.P]=np.eye(nP); Efull[c.I]=c.EI
    Hc=1.0/(n/m); idx={}
    Xall=c.X-np.array([c.x0,0])
    gi=np.rint(Xall/c.h).astype(int)
    lut={(a,b):i for i,(a,b) in enumerate(gi)}
    Interp=np.zeros((len(c.I),2*c.nn))
    for r,node in enumerate(c.int_nodes):
        x,y=gi[node]; x0=(x//m)*m; y0=(y//m)*m
        ws=[];ids=[]
        for dx in (0,m):
            for dy in (0,m):
                key=(x0+dx,y0+dy)
                w=(1-abs(x-x0-dx)/m)*(1-abs(y-y0-dy)/m)
                if key in lut and w>0: ws.append(w); ids.append(lut[key])
        ws=np.array(ws); ws/=ws.sum() if len(ws) else 1
        for w,i in zip(ws,ids):
            Interp[2*r,2*i]+=w; Interp[2*r+1,2*i+1]+=w
    EIc=Interp@Efull
    analyze(c,EIc,'M1 coarse interp (H=4h)',q)
    # M4 wrong geometry: element stiffness perturbed 20% random per element
    for sig in [0.05,0.2]:
        chi_p=c.chi.copy(); chi_p[c.ae]*=np.exp(sig*rng.normal(size=len(c.ae)))
        Kp=np.zeros_like(c.K)
        for k_,e in enumerate(c.ae):
            d=c.edofs[k_]; Kp[np.ix_(d,d)]+=(chi_p[e]+3e-3)*c.K0
        EIp=-np.linalg.solve(Kp[np.ix_(c.I,c.I)],Kp[np.ix_(c.I,c.P)])
        analyze(c,EIp,f'M4 elem stiffness +-{int(sig*100)}%',q)
    # M5 wrong global thickness (smooth geometry error): tau shifted 5%
    c2=toy.Cell(32,np.array([0.5,0.3,0.6,0.4])*1.05,**kw)
    if c2.nn==c.nn:
        analyze(c,c2.EI,'M5 tau +5% (smooth geo err)',q)
