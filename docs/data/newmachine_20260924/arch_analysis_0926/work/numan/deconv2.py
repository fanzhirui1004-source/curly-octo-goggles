import numpy as np
from scipy.optimize import linprog
from deconv import pk
ks=[0,1,2,4,8,16,32]
curves={
 '2010 heavy force_c':[0.3495,0.12548,0.05904,0.03568,0.0020515,1.795e-4,3.331e-5],
 '2010 heavy force':[0.19507,0.074786,0.038303,0.022059,0.0022768,3.939e-4,7.419e-5],
 '2003 med force_c':[13.5,9.9,7.9,6.1,4.7,3.8,2.95],
 '2003 med force':[7.2,5.1,4.0,3.1,2.3,1.9,1.45],
 '2006 med force_c':[3.6,2.6,2.0,1.45,1.04,0.81,0.61],
 '2006 med force':[2.2,1.4,1.0,0.72,0.46,0.35,0.25],
}
a=1/30; b=1.0
lam=np.concatenate([np.logspace(-6,np.log10(a),200,endpoint=False), np.linspace(a,b,200)])
P=np.array([pk(lam,k,a,b)**2 for k in ks])
for name,E in curves.items():
    E=np.array(E,float)
    tol=0.04 if E[0]>1 else 0.03   # the medium-cell numbers are rounded to 2 digits
    A_ub=np.vstack([P/E[:,None], -P/E[:,None]]); b_ub=np.concatenate([np.full(len(E),1+tol), np.full(len(E),-(1-tol))])
    out=[]
    for tn,t in [('<a/100',a/100),('<a/10',a/10),('<a/3',a/3),('<a',a),('>=a (rough)',None)]:
        c=((lam<t) if t is not None else (lam>=a)).astype(float)/E[0]
        lo=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=(0,None),method='highs'); hi=linprog(-c,A_ub=A_ub,b_ub=b_ub,bounds=(0,None),method='highs')
        out.append(f"{tn}: [{lo.fun:.2f},{-hi.fun:.2f}]" if lo.status==0 else f"{tn}: infeasible")
    print(f"{name:20s} E32/E0={E[-1]/E[0]:.3f}  E8/E0={E[4]/E[0]:.3f} ", ' '.join(out))
