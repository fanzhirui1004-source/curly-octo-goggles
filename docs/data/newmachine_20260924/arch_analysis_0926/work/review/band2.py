import numpy as np
def d2(lam,k,lmax,alpha=30):
    a=lmax/alpha; z0=(lmax+a)/(lmax-a); z=(lmax+a-2*lam)/(lmax-a)
    if abs(z)>=1: T=np.cosh(k*np.arccosh(abs(z)))*(np.sign(z)**k)
    else: T=np.cos(k*np.arccos(z))
    return (T/np.cosh(k*np.arccosh(z0)))**2
def avg(lo,hi,k,lmax):
    ls=np.exp(np.linspace(np.log(lo),np.log(hi),600)); return np.mean([d2(l,k,lmax) for l in ls])
lmax=4.0
print('2003: residual fraction of soft-200 band (5.7e-4..0.011) under Chebyshev k:')
for k in [8,32,64,128,256]:
    print(k, 'soft band keeps %.3f  mid(0.011-0.13) keeps %.4f'%(avg(5.7e-4,0.011,k,lmax),avg(0.011,0.13,k,lmax)))
# degree needed for eps<=1.2% with pure tail
for k in [32,48,64,96,128]:
    s=0.245*avg(5.7e-4,0.011,k,lmax); m=0.40*avg(0.011,0.13,k,lmax); h=0.355*avg(0.13,4,k,lmax)
    print('k',k,'pred eps %.2f%%'%(13.5*(s+m+h)))
# alternative: mid band concentrated near 0.011 vs 0.13 -> sensitivity of prediction
for lo,hi in [(0.011,0.03),(0.05,0.13)]:
    r=[13.5*(0.245*avg(5.7e-4,0.011,k,lmax)+0.40*avg(lo,hi,k,lmax)+0.355*avg(0.13,4,k,lmax)) for k in [1,2,4,8,16,32]]
    print('mid band in',lo,hi,' pred', np.round(r,2))
print('measured 13.5 9.9 7.9 6.1 4.7 3.8 2.95 (k=0,1,2,4,8,16,32)')
# 2010: check the high-k tail: lam_min 0.021 lmax 4.13
L=4.13
for k in [8,16,32]:
    print('2010 k',k,'p^2 at lam=0.021: %.3f, 0.05: %.3f, 0.1: %.4f, in-band max %.2e'%(d2(0.021,k,L),d2(0.05,k,L),d2(0.1,k,L),(1/np.cosh(k*np.arccosh((L+L/30)/(L-L/30))))**2))
net=np.array([0.3495,0.1255,0.0590,0.0357,0.00205,0.000180,3.33e-5]); zero=np.array([9907,3019,1129,1122,33.6,0.29,0.0042])
print('2010 net ratio', np.round(net/net[0],6)); print('2010 zero ratio', np.round(zero/zero[0],9))
