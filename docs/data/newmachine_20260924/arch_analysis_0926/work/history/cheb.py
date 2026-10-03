import numpy as np
lmax=4.0; a=lmax/30
def damp(lam,k):
    # Chebyshev polynomial residual p_k(lam) with p(0)=1 on [a,lmax]
    t=lambda x: np.cosh(k*np.arccosh(x)) if abs(x)>=1 else np.cos(k*np.arccos(x))
    z0=(lmax+a)/(lmax-a); z=(lmax+a-2*lam)/(lmax-a)
    return abs(t(z))/t(z0)
for lam in [5.7e-4,2e-3,0.011,0.03,0.06,0.13,0.5,2.0]:
    print(lam, ['%.3g'%(damp(lam,k)**2) for k in [1,2,4,8,16,32]])
# route-1: kappa from lambda_min 0.007 normalized lmax 1 : degree needed
for kap in [143,1/0.0080]:
    r=(np.sqrt(kap)-1)/(np.sqrt(kap)+1)
    for k in [16,24,32,64]: print('kappa',round(kap),'k',k,'2r^k=',2*r**k)
