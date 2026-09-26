import toy,numpy as np
rng=np.random.default_rng(0)
ks=[0,1,2,4,8,16,32]
def probes(c,m=32,kind='force'):
    # force class: q = S^+ f, f smooth random self-equilibrated nodal port forces
    Xp=c.X[c.port_nodes]
    R=toy.rigid(Xp); Q,_=np.linalg.qr(R)
    F=[]
    for i in range(m):
        kx,ky=rng.normal(size=(2,4))
        f=np.zeros(len(c.P))
        ph=rng.uniform(0,2*np.pi,(2,4))
        for d in range(2):
            v=sum(np.cos((j+1)*np.pi*Xp[:,0]+ph[d,j])*kx[j]+np.cos((j+1)*np.pi*Xp[:,1]+ph[d,j])*ky[j] for j in range(4))
            f[d::2]=v
        f-=Q@(Q.T@f); F.append(f)
    F=np.array(F).T
    Sp=np.linalg.pinv(c.S,rcond=1e-12)
    q=Sp@F
    if kind=='grf': q=F
    q-=Q@(Q.T@q)
    return q
for name,kw in [('FULL',{}),('medium xcut=.55',{'xcut':0.55}),('heavy xcut=.25',{'xcut':0.25})]:
    c=toy.Cell(32,[0.5,0.3,0.6,0.4],**kw)
    Dm=1/np.sqrt(c.D); lam,V=np.linalg.eigh(c.KII*Dm[:,None]*Dm[None,:])
    q=probes(c)
    uI=c.EI@q; u=c.full(q,uI); En=np.einsum('ij,ij->j',u,c.K@u)
    out=[]
    for k in ks:
        x=c.cheb(np.zeros_like(uI),q,k); e=c.full(np.zeros_like(q),x-uI)
        out.append(np.mean(np.einsum('ij,ij->j',e,c.K@e)/En))
    # spectral energy of exact interior field lifting error (zero start) : e0 = -uI in D^-1K basis
    y=(V.T@((uI)/Dm[:,None]))  # coordinates in D^{1/2} scaled basis
    en=(lam[:,None]*y**2).sum(1); en/=en.sum()
    print(f"{name}: nI={len(c.I)} lmax={c.lmax:.3f} lmin={lam[0]:.2e} kappa={lam[-1]/lam[0]:.0f} frac_eigs<lmax/30={np.mean(lam<lam[-1]/30):.3f}")
    print('   zero-start energy excess by k', np.array(out))
    print('   exact-field energy share in lam<lmax/30: %.3f, lam<lmax/100: %.3f'%(en[lam<lam[-1]/30].sum(), en[lam<lam[-1]/100].sum()))
