"""Single-geometry learning toy: which error components does Adam + normalised energy loss leave?
Models: dense W (no architectural bias) and deep-linear (A@B, incremental-learning bias).
Variants: energy loss; +smoother-in-loop (k Chebyshev sweeps on the output, trained end-to-end);
+adversarial worst directions; L2 field loss (Hessian ~ D instead of K); Galerkin enrichment."""
import sys, json, numpy as np, torch, toy_setup as T
torch.set_num_threads(4)
dt=torch.float64
cfg=dict(n=24,xcut=0.5,tau=(0.5,0.3,0.6,0.4))
if len(sys.argv)>1 and sys.argv[1]=='full': cfg=dict(n=24,xcut=None,tau=(0.35,0.25,0.4,0.3))
c=T.make(**cfg)
rng=np.random.default_rng(0)
KII=torch.tensor(c.KII,dtype=dt); KIP=torch.tensor(c.KIP,dtype=dt); S=torch.tensor(c.S,dtype=dt)
EI=torch.tensor(c.EI,dtype=dt); D=torch.tensor(c.D,dtype=dt)
nI,nP=c.EI.shape
qtest=T.probes(c,256,'force',np.random.default_rng(99))
def cheb(x,q,k,alpha=30.0):
    if k==0: return x
    b_=c.lmax; a_=b_/alpha; th=(b_+a_)/2; de=(b_-a_)/2; sig=th/de
    f=-KIP@q; r=f-KII@x; z=r/D[:,None]; rho=1/sig; d=z/th
    for i in range(k):
        x=x+d; r=r-KII@d; z=r/D[:,None]; rn=1/(2*sig-rho); d=rn*rho*d+2*rn/de*z; rho=rn
    return x
def batch(bs,adv=None,frac=0.0):
    nb=bs//2
    q=np.concatenate([T.probes(c,nb,'force',rng),T.probes(c,bs-nb,'grf',rng)],1)
    if adv is not None and frac>0:
        na=int(bs*frac); q[:,:na]=adv[:,rng.integers(0,adv.shape[1],na)]*rng.choice([-1,1],na)
    return torch.tensor(q,dtype=dt)
def run(name,model='dense',k=0,loss='energy',adv_frac=0.0,steps=6000,lr=1e-3,bs=16,width=None,fp32=False,m_enrich=0):
    torch.manual_seed(0)
    tdt=torch.float32 if fp32 else dt
    if model=='dense':
        W=torch.zeros(nI,nP,dtype=tdt,requires_grad=True); params=[W]; getW=lambda:W
    else:
        r=width or nP
        A=(1e-3*torch.randn(nI,r,dtype=tdt)).requires_grad_(); B=(1e-3*torch.randn(r,nP,dtype=tdt)).requires_grad_()
        params=[A,B]; getW=lambda:A@B
    Venr=None
    if m_enrich:
        Venr=(1e-2*torch.randn(nI,m_enrich,dtype=tdt)).requires_grad_(); params.append(Venr)
    opt=torch.optim.Adam(params,lr=lr)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=steps,pct_start=0.05)
    adv=None; hist=[]
    def field(q):
        u=(getW().to(dt))@q
        u=cheb(u,q,k)
        if Venr is not None:   # Galerkin correction in span(Venr): exact energy minimisation, small m x m solve
            V=Venr.to(dt); r=-KIP@q-KII@u
            G=V.T@KII@V+1e-12*torch.eye(V.shape[1],dtype=dt)
            u=u+V@torch.linalg.solve(G,V.T@r)
        return u
    def evalW():
        with torch.no_grad():
            I=torch.eye(nP,dtype=dt); EIh=field(I).numpy()
        return EIh
    for s in range(steps):
        if adv_frac>0 and s%200==0 and s>0:
            m=T.metrics(c,evalW(),qtest[:,:4])
            Err=evalW()-c.EI; M=Err.T@c.KII@Err; G=c.Sm12@M@c.Sm12; g,U=np.linalg.eigh((G+G.T)/2)
            adv=c.Sm12@U[:,-8:]; adv/=np.linalg.norm(adv,axis=0)
        q=batch(bs,adv,adv_frac)
        u=field(q); e=u-EI@q; den=torch.einsum('ij,ij->j',q,S@q)
        if loss=='energy': L=(torch.einsum('ij,ij->j',e,KII@e)/den).mean()
        elif loss=='l2D':  L=(torch.einsum('ij,ij->j',e,D[:,None]*e)/den).mean()
        if fp32: L=L.float()
        opt.zero_grad(); L.backward(); opt.step(); sch.step()
        if s%1000==999 or s==steps-1:
            m=T.metrics(c,evalW(),qtest); hist.append((s+1,m['force_mean'],m['mu1']))
    m=T.metrics(c,evalW(),qtest)
    # error share by S-mode (softest 10% of modes)
    per=m['per_mode']; nm=len(per); soft=per[:max(1,nm//10)].mean(); stiff=per[-max(1,nm//10):].mean()
    # smoothing headroom of the final field
    EIh=evalW()
    with torch.no_grad():
        post=[T.metrics(c,cheb(torch.tensor(EIh,dtype=dt),torch.eye(nP,dtype=dt),kk).numpy(),qtest)['force_mean'] for kk in (1,4,8)]
    out=dict(name=name,force_mean=m['force_mean'],force_max=m['force_max'],mu1=m['mu1'],top4=list(m['top4']),soft10=soft,stiff10=stiff,post_k148=post,hist=hist)
    print(json.dumps({k:(round(v,5) if isinstance(v,float) else v) for k,v in out.items() if k!='hist'}),flush=True)
    print('   hist',[(h[0],round(h[1],4),round(h[2],3)) for h in hist],flush=True)
    return out
if __name__=='__main__':
    print('cfg',cfg,'nI',nI,'nP',nP,'Rayleigh ratio',c.Slam[-1]/c.Slam[0])
    res=[]
    steps=int(sys.argv[2]) if len(sys.argv)>2 else 6000
    for kw in [dict(name='dense_energy'),dict(name='dense_energy_fp32',fp32=True),dict(name='dense_energy_k4',k=4),
               dict(name='dense_energy_adv',adv_frac=0.25),dict(name='dense_l2D',loss='l2D'),
               dict(name='deeplin_energy',model='deep'),dict(name='deeplin_energy_k4',model='deep',k=4),
               dict(name='dense_energy_enrich8',m_enrich=8),dict(name='dense_energy_k4_adv',k=4,adv_frac=0.25)]:
        res.append(run(steps=steps,**kw))
    json.dump(res,open(f"toy_learn_{sys.argv[1] if len(sys.argv)>1 else 'cut'}.json",'w'),default=float)
