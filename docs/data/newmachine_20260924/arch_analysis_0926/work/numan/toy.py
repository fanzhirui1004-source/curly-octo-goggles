"""2D plane-stress Q1 toy of a CutFEM-like TPMS cell: fill-fraction-scaled element stiffness (+ small floor, GP-like),
ports = box-face nodes with material, interior = the rest. Exact E, S, Chebyshev tail, lattice of 2 cells."""
import numpy as np, scipy.linalg as sla
np.set_printoptions(precision=4, suppress=True)

def ke_q1(h, E=1.0, nu=0.3):
    C = E/(1-nu**2)*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
    g = [-1/np.sqrt(3), 1/np.sqrt(3)]
    xi_n = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    K = np.zeros((8,8))
    for a in g:
        for b in g:
            dN = np.array([[xi_n[i,0]*(1+b*xi_n[i,1])/4, xi_n[i,1]*(1+a*xi_n[i,0])/4] for i in range(4)])*2/h
            B = np.zeros((3,8))
            for i in range(4):
                B[0,2*i]=dN[i,0]; B[1,2*i+1]=dN[i,1]; B[2,2*i]=dN[i,1]; B[2,2*i+1]=dN[i,0]
            K += B.T@C@B*(h/2)**2
    return K

class Cell:
    def __init__(self, n, tau, xcut=None, x0=0.0, sub=8, wfac=0.35, beta=3e-3, seed=0):
        self.n=n; h=1.0/n; self.h=h
        self.tau=np.array(tau,float)          # corners (0,0),(1,0),(1,1),(0,1)
        # element sub-points
        s=(np.arange(sub)+0.5)/sub
        ex,ey=np.meshgrid(np.arange(n),np.arange(n),indexing='ij'); ex=ex.ravel(); ey=ey.ravel()
        px=(ex[:,None,None]+s[None,:,None])*h; py=(ey[:,None,None]+s[None,None,:])*h
        px=np.broadcast_to(px,(n*n,sub,sub)).reshape(n*n,-1); py=np.broadcast_to(py,(n*n,sub,sub)).reshape(n*n,-1)
        phi=np.abs(np.cos(2*np.pi*px)+np.cos(2*np.pi*py))
        Nc=np.stack([(1-px)*(1-py),px*(1-py),px*py,(1-px)*py],-1)
        t=Nc@self.tau
        w=wfac*h*2
        z=(t-phi)/w; sg=1/(1+np.exp(-z))
        if xcut is not None:
            sg=sg*(px<xcut)
        chi=sg.mean(1)
        dchi=((sg*(1-sg)/w)[:,:,None]*Nc).mean(1)      # n*n x 4
        if xcut is not None: pass
        act=chi>2e-3
        self.chi=chi; self.dchi=dchi; self.act=act
        self.K0=ke_q1(h)
        # node numbering on (n+1)^2 grid
        nid=lambda i,j: i*(n+1)+j
        conn=np.stack([nid(ex,ey),nid(ex+1,ey),nid(ex+1,ey+1),nid(ex,ey+1)],1)
        self.conn_all=conn
        ae=np.flatnonzero(act); self.ae=ae
        used=np.unique(conn[ae]); self.used=used
        self.gmap=-np.ones((n+1)**2,int); self.gmap[used]=np.arange(len(used))
        nn=len(used); self.nn=nn
        X=np.stack(np.meshgrid(np.arange(n+1),np.arange(n+1),indexing='ij'),-1).reshape(-1,2)*h
        self.X=X[used]+np.array([x0,0])
        self.x0=x0
        K=np.zeros((2*nn,2*nn))
        self.edofs=[]
        for e in ae:
            ln=self.gmap[conn[e]]; d=np.stack([2*ln,2*ln+1],1).ravel()
            self.edofs.append(d)
            K[np.ix_(d,d)]+=(chi[e]+beta)*self.K0
        self.K=K
        xl=X[used]
        onb=(xl[:,0]<1e-12)|(xl[:,0]>1-1e-12)|(xl[:,1]<1e-12)|(xl[:,1]>1-1e-12)
        self.port_nodes=np.flatnonzero(onb); self.int_nodes=np.flatnonzero(~onb)
        self.P=np.stack([2*self.port_nodes,2*self.port_nodes+1],1).ravel()
        self.I=np.stack([2*self.int_nodes,2*self.int_nodes+1],1).ravel()
        KII=K[np.ix_(self.I,self.I)]; KIP=K[np.ix_(self.I,self.P)]
        self.KII=KII; self.KIP=KIP
        self.EI=-np.linalg.solve(KII,KIP)            # interior part of E
        self.S=K[np.ix_(self.P,self.P)]+KIP.T@self.EI
        self.S=(self.S+self.S.T)/2
        self.D=np.diag(KII).copy()
        self.lmax=np.linalg.eigvalsh(KII/np.sqrt(np.outer(self.D,self.D))).max()
    def A(self,j):
        A=np.zeros_like(self.K)
        for k,e in enumerate(self.ae):
            d=self.edofs[k]; A[np.ix_(d,d)]+=self.dchi[e,j]*self.K0
        return A
    def full(self,q,uI):
        u=np.zeros((2*self.nn,)+q.shape[1:]); u[self.P]=q; u[self.I]=uI; return u
    def cheb(self,uI,q,k,alpha=30.0,lmax=None):
        """k Chebyshev steps on K_II uI = -K_IP q, Jacobi preconditioned, interval [lmax/alpha, lmax]."""
        if k==0: return uI.copy()
        lmax=self.lmax if lmax is None else lmax
        b_=lmax; a_=lmax/alpha; th=(b_+a_)/2; de=(b_-a_)/2; sig=th/de
        f=-self.KIP@q
        x=uI.copy(); r=f-self.KII@x; z=r/self.D[:,None] if r.ndim==2 else r/self.D
        rho=1/sig; d=z/th
        for i in range(k):
            x=x+d
            r=r-self.KII@d
            z=r/self.D[:,None] if r.ndim==2 else r/self.D
            rho_n=1/(2*sig-rho)
            d=rho_n*rho*d+2*rho_n/de*z
            rho=rho_n
        return x

def rigid(X):
    R=np.zeros((2*len(X),3)); R[0::2,0]=1; R[1::2,1]=1; R[0::2,2]=-X[:,1]; R[1::2,2]=X[:,0]; return R
