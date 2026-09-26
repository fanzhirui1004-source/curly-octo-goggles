"""Assemble the real CutFEM K of a packet cell (NETDATA.npz) on CPU: K = sum_e M_e.Tm + gamma sum_f F^T F."""
import sys, json, numpy as np, scipy.sparse as sp
sys.path.insert(0, '/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/pack/work/mechanics/lib')
import element_moments as EM
FIX = '/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/fixdl/FIX/'

class RealCell:
    def __init__(self, case, gamma=1e-4, E=1.0, nu=0.3):
        z = np.load(FIX + case + '/NETDATA.npz')
        self.z = z; self.n = n = int(z['n'])
        self.grid = z['grid'].astype(np.int64); self.cells = z['elem_cells'].astype(np.int64)
        self.en = z['elem_nodes'].astype(np.int64); self.M = z['moments']
        xi = self.grid[self.en] - 2 * self.cells[:, None, :] - 1
        keys = np.unique(xi.reshape(len(xi), -1), axis=0); assert len(keys) == 1
        self.xi = keys[0].reshape(27, 3)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
        self.Tm = EM.pattern_operators(self.xi, lam, mu, n)[1]      # 125 x 81 x 81
        N = len(self.grid); self.N = N; nb = 3 * N; self.nb = nb
        dofs = (3 * self.en[:, :, None] + np.arange(3)).reshape(len(self.en), 81); self.dofs = dofs
        Ke = np.einsum('em,mij->eij', self.M, self.Tm); self.Ke = Ke
        Kb = sp.csr_matrix((nb, nb))
        for lo in range(0, len(dofs), 3000):
            d_ = dofs[lo:lo + 3000]
            r = np.repeat(d_, 81, 1).ravel(); c = np.tile(d_, (1, 81)).ravel()
            Kb = Kb + sp.csr_matrix((Ke[lo:lo + 3000].ravel(), (r, c)), shape=(nb, nb))
        tpl = np.load(FIX + 'GP_TEMPLATES_n32.npz'); canon = tpl['canonical']; offs = tpl['offsets']
        tk = np.transpose(canon, (0, 2, 1)) @ canon
        f = z['gp_faces'].astype(np.int64); self.faces = f
        g = 2 * self.cells[f[:, 0]][:, None, :] + offs[f[:, 2]]
        M1 = 2 * n + 1
        ids = (g[..., 0] * M1 + g[..., 1]) * M1 + g[..., 2]
        nid = z['node_ids'].astype(np.int64)
        loc = np.searchsorted(nid, ids); assert np.all(nid[loc] == ids)
        gd = (3 * loc[:, :, None] + np.arange(3)).reshape(len(f), -1)
        G = sp.csr_matrix((nb, nb))
        for lo in range(0, len(f), 1500):
            g_ = gd[lo:lo + 1500]
            r = np.repeat(g_, 135, 1).ravel(); c = np.tile(g_, (1, 135)).ravel()
            G = G + sp.csr_matrix((tk[f[lo:lo + 1500, 2]].ravel(), (r, c)), shape=(nb, nb))
        self.Kbody, self.G, self.gamma = Kb, G, gamma
        self.K = (Kb + gamma * G).tocsr()
        port = z['is_port']; pm = np.repeat(port, 3)
        self.P = np.flatnonzero(pm); self.I = np.flatnonzero(~pm)
        self.xyz = self.grid / (2.0 * n)
        self.vf = self.M[:, 0] * n ** 3
    def check_diag(self):
        d = self.K.diagonal().reshape(-1, 3)
        ref = np.einsum('nii->ni', self.z['diag3'])
        return float(np.abs(d - ref).max() / np.abs(ref).max())
