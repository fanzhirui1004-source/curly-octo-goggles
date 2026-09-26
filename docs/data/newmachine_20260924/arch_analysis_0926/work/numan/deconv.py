import json,numpy as np
from scipy.optimize import linprog
d=json.load(open('../../smooth_v2L1.json'))
alpha=d['alpha']
def pk(lam,k,a,b):
    s=(b+a)/(b-a); x=(b+a-2*lam)/(b-a)
    def T(k,x):
        x=np.asarray(x,float); out=np.empty_like(x)
        m=np.abs(x)<=1; out[m]=np.cos(k*np.arccos(x[m]))
        mp=x>1; out[mp]=np.cosh(k*np.arccosh(x[mp]))
        mn=x<-1; out[mn]=(-1)**k*np.cosh(k*np.arccosh(-x[mn]))
        return out
    return T(k,x)/T(k,np.array([s]))[0]
b=1.0; a=b/alpha
lam=np.concatenate([np.logspace(-5,np.log10(a),150,endpoint=False), np.linspace(a,b,250)])
for case,cd in d['per_case'].items():
  for cls in ['force_c','force']:
    for start in ['net','zero']:
        rows=cd[cls][start]; ks=[r['k'] for r in rows]; E=np.array([r['energy_mean'] for r in rows])
        P=np.array([pk(lam,k,a,b)**2 for k in ks])  # (K,J)
        # relative-scaled constraints
        tol=0.03
        A_ub=np.vstack([P/E[:,None], -P/E[:,None]]); b_ub=np.concatenate([np.full(len(E),1+tol), np.full(len(E),-(1-tol))])
        res={}
        for thr_name,thr in [('<a/10',a/10),('<a/3',a/3),('<a',a),('<0.2',0.2),('<0.5',0.5)]:
            c=(lam<thr).astype(float)/E[0]
            lo=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=(0,None),method='highs')
            hi=linprog(-c,A_ub=A_ub,b_ub=b_ub,bounds=(0,None),method='highs')
            res[thr_name]=(lo.fun if lo.status==0 else None, -hi.fun if hi.status==0 else None)
        print(case,cls,start,{k:(None if v[0] is None else round(v[0],4), None if v[1] is None else round(v[1],4)) for k,v in res.items()})
