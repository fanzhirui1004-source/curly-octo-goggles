import sys, numpy as np
sys.path.insert(0,'/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/pack/work/numan')
import toy
def make(n=20, xcut=0.3, tau=(0.5,0.3,0.6,0.4)):
    c=toy.Cell(n,list(tau),xcut=xcut)
    Xp=c.X[c.port_nodes]; R=toy.rigid(Xp); Qr,_=np.linalg.qr(R)
    # non-rigid port basis
    nP=len(c.P); Pn=np.eye(nP)-Qr@Qr.T
    w,V=np.linalg.eigh(Pn); B=V[:,w>0.5]           # orthonormal basis of rigid-complement
    Sr=B.T@c.S@B; lam,U=np.linalg.eigh(Sr)
    c.B=B; c.Qr=Qr; c.Slam=lam; c.SU=B@U           # S eigenpairs (Rayleigh), port-space vectors
    c.Sp=c.SU@np.diag(1/lam)@c.SU.T                # pseudo-inverse on complement
    c.Sm12=c.SU@np.diag(lam**-0.5)@c.SU.T
    return c
def probes(c,m,kind,rng):
    Xp=c.X[c.port_nodes]; F=[]
    for i in range(m):
        kx,ky=rng.normal(size=(2,4)); f=np.zeros(len(c.P)); ph=rng.uniform(0,2*np.pi,(2,4))
        for d in range(2):
            f[d::2]=sum(np.cos((j+1)*np.pi*Xp[:,0]+ph[d,j])*kx[j]+np.cos((j+1)*np.pi*Xp[:,1]+ph[d,j])*ky[j] for j in range(4))
        f-=c.Qr@(c.Qr.T@f); F.append(f)
    F=np.array(F).T
    q=c.Sp@F if kind=='force' else F
    q-=c.Qr@(c.Qr.T@q)
    return q/np.linalg.norm(q,axis=0)
def metrics(c,EIh,qf):
    Err=EIh-c.EI
    M=Err.T@c.KII@Err
    ex=np.einsum('ij,ij->j',qf,M@qf)/np.einsum('ij,ij->j',qf,c.S@qf)
    G=c.Sm12@M@c.Sm12; g=np.linalg.eigvalsh((G+G.T)/2)
    # per S-eigenmode relative excess
    per=np.einsum('ij,ij->j',c.SU,M@c.SU)/c.Slam
    return dict(force_mean=ex.mean(), force_max=ex.max(), mu1=g[-1], top4=g[-4:][::-1], per_mode=per)
if __name__=='__main__':
    c=make()
    print('nI',len(c.I),'nP',len(c.P),'S Rayleigh range',c.Slam[:3],c.Slam[-1], 'ratio', c.Slam[-1]/c.Slam[0])
    Dm=1/np.sqrt(c.D); l=np.linalg.eigvalsh(c.KII*Dm[:,None]*Dm[None,:]); print('kappa(D^-1K_II)',l[-1]/l[0], 'kappa(K_II)',np.linalg.cond(c.KII))
