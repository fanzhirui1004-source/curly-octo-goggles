# 1D and 2D Q2 Laplacian, Jacobi-scaled: which wavelength (in elements) sits at lam = lmax/30 ?
import numpy as np, scipy.sparse as sp, scipy.linalg as la
# 1D Q2 element stiffness on [0,1] (nodes 0, .5, 1)
Ke=np.array([[7,-8,1],[-8,16,-8],[1,-8,7]])/3.0
Me=np.array([[4,2,-1],[2,16,2],[-1,2,4]])/30.0
def asm(ne,E):
    n=2*ne+1; K=np.zeros((n,n))
    for e in range(ne):
        i=[2*e,2*e+1,2*e+2]; K[np.ix_(i,i)]+=E
    return K
ne=64; K=asm(ne,Ke)[1:-1,1:-1]  # Dirichlet
D=np.diag(K); lam,V=la.eigh(K,np.diag(D))
lmax=lam.max(); a=lmax/30
for thr in [a, lmax/10, lmax/100]:
    j=np.searchsorted(lam,thr)
    # dominant wavelength of mode j: count sign changes on vertex nodes
    v=V[:,j]; vv=v[1::2] if len(v)%2 else v
    sc=np.sum(np.diff(np.sign(v[1::2]))!=0)
    print('1D: lmax=%.2f thr=%.4f -> mode %d/%d, ~half-waves %d over %d elements -> wavelength ~%.1f elements'%(lmax,thr,j,len(lam),sc+1,ne,2*ne/(sc+1)))
# 2D Q2 scalar Laplacian via tensor products
def k2(ne):
    K1=asm(ne,Ke)[1:-1,1:-1]; M1=asm(ne,Me)[1:-1,1:-1]
    return np.kron(K1,M1)+np.kron(M1,K1)
ne=24; K=k2(ne); D=np.diag(K); lam=la.eigh(K,np.diag(D),eigvals_only=True)
lmax=lam.max(); a=lmax/30
# lowest mode with wavelength L elements in one direction ~ compare with continuous: estimate via 1D fit
frac=np.mean(lam<a)
print('2D Q2 Laplacian ne=%d: lmax=%.2f, fraction of modes below lmax/30: %.3f, below lmax/10: %.3f'%(ne,lmax,frac,np.mean(lam<lmax/10)))
