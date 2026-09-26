import toy,numpy as np, scipy.linalg as sla
rng=np.random.default_rng(7)
n=32
tau_t=[0.45,0.35,0.55,0.4]      # test corners (0,0),(1,0),(1,1),(0,1)
tau_n=[0.5,0.45,0.4,0.6]        # neighbour at x in [-1,0]: corners (-1,0),(0,0),(0,1),(-1,1); shared (0,0)=t0,(0,1)=t3
tau_n[1]=tau_t[0]; tau_n[2]=tau_t[3]

def coarse_interp(c,m=4):
    nP=len(c.P); Efull=np.zeros((2*c.nn,nP)); Efull[c.P]=np.eye(nP); Efull[c.I]=c.EI
    gi=np.rint((c.X-np.array([c.x0,0]))/c.h).astype(int); lut={(a,b):i for i,(a,b) in enumerate(gi)}
    Interp=np.zeros((len(c.I),2*c.nn))
    for r,node in enumerate(c.int_nodes):
        x,y=gi[node]; x0=(x//m)*m; y0=(y//m)*m; ws=[];ids=[]
        for dx in (0,m):
            for dy in (0,m):
                key=(x0+dx,y0+dy); w=(1-abs(x-x0-dx)/m)*(1-abs(y-y0-dy)/m)
                if key in lut and w>0: ws.append(w); ids.append(lut[key])
        ws=np.array(ws); ws/=ws.sum()
        for w,i in zip(ws,ids): Interp[2*r,2*i]+=w; Interp[2*r+1,2*i+1]+=w
    return Interp@Efull
def perturbed(c,sig):
    chi_p=c.chi.copy(); chi_p[c.ae]*=np.exp(sig*rng.normal(size=len(c.ae)))
    Kp=np.zeros_like(c.K)
    for k_,e in enumerate(c.ae):
        d=c.edofs[k_]; Kp[np.ix_(d,d)]+=(chi_p[e]+3e-3)*c.K0
    return -np.linalg.solve(Kp[np.ix_(c.I,c.I)],Kp[np.ix_(c.I,c.P)])

def lattice(ct,cn,EIh_t):
    # global port nodes by coordinates
    keys={}
    def gid(X):
        out=[]
        for x in X:
            k=(round(x[0]*n),round(x[1]*n))
            if k not in keys: keys[k]=len(keys)
            out.append(keys[k])
        return np.array(out)
    gt=gid(ct.X[ct.port_nodes]); gn=gid(cn.X[cn.port_nodes])
    N=len(keys); Xg=np.zeros((N,2))
    for k,v in keys.items(): Xg[v]=np.array(k)/n
    def R(g):
        M=np.zeros((2*len(g),2*N)); M[0::2,2*g]=np.eye(len(g)); M[1::2,2*g+1]=np.eye(len(g)); return M
    Rt,Rn=R(gt),R(gn)
    St=ct.S; Sn=cn.S
    F_t=EIh_t-ct.EI; Sht=St+F_t.T@ct.KII@F_t
    A=Rt.T@St@Rt+Rn.T@Sn@Rn; Ah=Rt.T@Sht@Rt+Rn.T@Sn@Rn
    clamp=np.abs(Xg[:,0]+1)<1e-9
    free=np.flatnonzero(~np.repeat(clamp,2))
    loads=[];labels=[]
    for who,xr in [('test',(0,1)),('nbr',(-1,0))]:
        sel=np.flatnonzero((np.abs(Xg[:,1])<1e-9)&(Xg[:,0]>=xr[0]-1e-9)&(Xg[:,0]<=xr[1]+1e-9)&~clamp)
        for d in range(2):
            f=np.zeros(2*N); f[2*sel+d]=1.0/len(sel); loads.append(f); labels.append(f'{who}_{"xy"[d]}')
    Fm=np.array(loads).T
    U=np.zeros_like(Fm); Uh=np.zeros_like(Fm)
    U[free]=np.linalg.solve(A[np.ix_(free,free)],Fm[free]); Uh[free]=np.linalg.solve(Ah[np.ix_(free,free)],Fm[free])
    C=np.einsum('ij,ij->j',Fm,U); Ch=np.einsum('ij,ij->j',Fm,Uh)
    qt=Rt@U; qth=Rt@Uh
    w=np.einsum('ij,ij->j',qt,St@qt)/C
    eps=np.einsum('ij,ij->j',qt,Sht@qt)/np.einsum('ij,ij->j',qt,St@qt)-1
    comp=(C-Ch)/C
    dq=qth-qt; qerr=np.sqrt(np.einsum('ij,ij->j',dq,St@dq)/np.einsum('ij,ij->j',qt,St@qt))
    As=[ct.A(j) for j in range(4)]
    def sens(q,EI):
        u=ct.full(q,EI@q); return np.array([-np.einsum('ij,ij->j',u,Aj@u) for Aj in As])
    s=sens(qt,ct.EI); sf=sens(qth,EIh_t); s_fo=sens(qt,EIh_t); s_so=sens(qth,ct.EI)
    rel=lambda a: np.linalg.norm(a-s,axis=0)/np.linalg.norm(s,axis=0)
    return dict(labels=labels,w=w,eps=eps,comp=comp,bound=w*eps,qerr=qerr,sens=rel(sf),sens_field=rel(s_fo),sens_sol=rel(s_so))

if __name__=='__main__':
  for name,kw in [('FULL',{}),('medium',{'xcut':0.55}),('heavy',{'xcut':0.3})]:
    ct=toy.Cell(n,tau_t,**kw); cn=toy.Cell(n,tau_n,x0=-1.0)
    EIh=ct.EI+0.35*(coarse_interp(ct)-ct.EI)+ (perturbed(ct,0.3)-ct.EI)
    print('=====',name,'test nI',len(ct.I))
    for k in [0,2,4,8,16]:
        Ek=ct.cheb(EIh,np.eye(len(ct.P)),k)
        r=lattice(ct,cn,Ek)
        gap=r['bound']-r['comp']
        print(f' k={k:2d}  '+' | '.join(f"{L}: w={r['w'][i]:.3f} eps={r['eps'][i]:.4f} comp={r['comp'][i]:.4f} qerr={r['qerr'][i]:.4f}<= {np.sqrt(max(gap[i],0)/r['w'][i]):.4f} sens={r['sens'][i]:.4f} (fld {r['sens_field'][i]:.4f}, sol {r['sens_sol'][i]:.4f})" for i,L in enumerate(r['labels'])))
