import numpy as np
lmax=4.0; a=lmax/30
def d2(lam,k):
    z0=(lmax+a)/(lmax-a); z=(lmax+a-2*lam)/(lmax-a)
    T=lambda x: np.cosh(k*np.arccosh(x)) if abs(x)>=1 else np.cos(k*np.arccos(x))
    return (T(z)/T(z0))**2
def avg(lo,hi,k):
    ls=np.exp(np.linspace(np.log(lo),np.log(hi),400)); return np.mean([d2(l,k) for l in ls])
# 2003 force_c budget: 24.5% lowest-200 (5.7e-4..0.011), ~40% mid (0.011..0.13), rest ~35.5% high (>0.13)
for k in [1,2,4,8,16,32]:
    s=0.245*avg(5.7e-4,0.011,k); m=0.40*avg(0.011,0.13,k); h=0.355*avg(0.13,4.0,k)
    print(k, 'pred frac %.3f'%(s+m+h), 'pred eps %.2f%%'%(13.5*(s+m+h)), 'soft %.3f mid %.3f high %.4f'%(s,m,h))
print('measured 2003 force_c: 13.5 9.9 7.9 6.1 4.7 3.8 2.95')
