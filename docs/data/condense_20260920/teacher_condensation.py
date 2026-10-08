import json, os, numpy as np, time, sys
from scipy.linalg import cho_factor, cho_solve
L=json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
by={int(e['seat']):e for e in L}

def unpack_upper(path, q):
    v=np.load(path, mmap_mode='r')
    M=np.zeros((q,q), dtype=np.float64)
    M[np.triu_indices(q)]=np.asarray(v, dtype=np.float64)
    M+=M.T; M[np.diag_indices(q)]*=0.5
    return M

def spec_drop6(M):
    w,V=np.linalg.eigh(M)
    return w, V

print(f"{'seat':>7} {'q':>6} {'nbox':>6} {'ncut':>6} {'kap_S':>10} {'kap_cond':>10} {'ratio':>6} "
      f"{'null_S':>9} {'null_cond':>9} {'BB%':>6} {'BC%':>6} {'CC%':>6} {'exact':>9} {'dynS':>8} {'dynC':>8} {'sec':>5}")
for seat in [int(x) for x in sys.argv[1:]]:
    t0=time.time(); e=by[seat]; q=int(e['q'])
    smp=json.load(open(os.path.join(e['packet'],'SAMPLE.json')))
    ncut_coord=int(smp['additional_scalar_cut_coordinates'])
    nc=3*ncut_coord; nb=q-nc
    S=unpack_upper(os.path.join(e['packet'],'S_UPPER.npy'), q)
    B=slice(0,nb); C=slice(nb,q)
    fBB=float(np.linalg.norm(S[B,B])); fBC=float(np.linalg.norm(S[B,C])) if nc else 0.
    fCC=float(np.linalg.norm(S[C,C])) if nc else 0.
    tot=fBB**2+fCC**2+2*fBC**2
    w,V=spec_drop6(S)
    kS=float(w[-1]/w[6]); nullS=float(abs(w[5])/w[-1])
    dS=np.diag(S); dynS=float(dS.max()/dS.min())
    if nc:
        Scc=np.array(S[C,C], order='F'); Scb=np.array(S[C,B])
        cf=cho_factor(Scc, lower=True, check_finite=False)
        X=cho_solve(cf, Scb, check_finite=False)
        Sb=np.array(S[B,B]) - Scb.T@X; Sb=0.5*(Sb+Sb.T)
        wb,Vb=spec_drop6(Sb)
        kC=float(wb[-1]/wb[6]); nullC=float(abs(wb[5])/wb[-1])
        dB=np.diag(Sb); dynC=float(dB.max()/dB.min())
        rng=np.random.default_rng(seat); f=rng.standard_normal(nb)
        f-=Vb[:,:6]@(Vb[:,:6].T@f)                       # make f orthogonal to the condensed nullspace
        ub=Vb[:,6:]@((Vb[:,6:].T@f)/wb[6:]); c1=float(f@ub)
        F=np.zeros(q); F[:nb]=f
        F-=V[:,:6]@(V[:,:6].T@F)
        U=V[:,6:]@((V[:,6:].T@F)/w[6:]); c2=float(F@U)
        chk=abs(c1/c2-1)
    else:
        kC=kS; nullC=nullS; dynC=dynS; chk=0.0
    print(f"{seat:>7} {q:>6} {nb:>6} {nc:>6} {kS:>10.4g} {kC:>10.4g} {kS/kC:>6.2f} "
          f"{nullS:>9.1e} {nullC:>9.1e} {100*fBB**2/tot:>6.2f} {100*2*fBC**2/tot:>6.2f} {100*fCC**2/tot:>6.2f} "
          f"{chk:>9.1e} {dynS:>8.3g} {dynC:>8.3g} {time.time()-t0:>5.1f}", flush=True)
    del S
