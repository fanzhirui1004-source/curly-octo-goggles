import sys, os
sys.path.append('/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/pack/work/mechanics/pylib')
import numpy as np, scipy.sparse as sp, time
import cell
import pypardiso
class Big:
    def __init__(self, case, gamma=1e-4):
        t = time.time()
        C = cell.RealCell(case, gamma=gamma); self.C = C
        K = C.K
        self.KII = K[C.I][:, C.I].tocsr(); self.KIP = K[C.I][:, C.P].tocsr(); self.KPP = K[C.P][:, C.P].tocsr()
        self.ps = pypardiso.PyPardisoSolver(mtype=2)   # real SPD
        A = sp.triu(self.KII, format='csr'); A.sort_indices(); self.A = A
        self.ps.factorize(A)
        print('factor', case, C.nb, len(C.I), time.time() - t, flush=True)
    def solve(self, b):
        return self.ps.solve(self.A, np.ascontiguousarray(b))
    def extend(self, q):
        return -self.solve(self.KIP @ q)
