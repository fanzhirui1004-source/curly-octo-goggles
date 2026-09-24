"""Low-risk inference acceleration of a trained MGNO / MGNO2 (task 55). Same linear map, different evaluation order:
  1. the geometry path is evaluated once per geometry (frozen): slot weights, grid-transfer weights, gates;
  2. every hyperedge layer is pre-assembled as two sparse matrices, the alpha-weighted gather G (H*Eh x N) and the
     beta-weighted scatter S (N x H*Eh, with the 1/deg normalization folded in):
        X <- (X + S @ mix_W(G @ X)) * (1 - pm) + X0 * pm
     (no E x 27 x B x F gather intermediate);
  3. restriction and prolongation (with the skip scale folded in) are sparse matrices;
  4. the adjoint E_hat^T is written out (transposed sparse matrices, transposed channel mixes, conv_transpose3d,
     reversed layer order), so the variational reaction S_hat q = E_hat^T K E_hat q needs no autograd graph.
Everything stays float32 like the trained model (K products in float64 as in trainlib); results agree with the autograd
model to float32 rounding (checked in t_fast.py). Training keeps using models.py.
"""
import torch
import torch.nn.functional as Fn
import models as MD

dev, f32, f64 = MD.dev, MD.f32, torch.float64


def _csr(rows, cols, vals, shape):
    """CSR from (unsorted, duplicate-free) triplets."""
    key = rows * shape[1] + cols
    o = torch.argsort(key)
    r, c, v = rows[o], cols[o], vals[o]
    crow = torch.zeros(shape[0] + 1, dtype=torch.int64, device=dev)
    crow[1:] = torch.cumsum(torch.bincount(r, minlength=shape[0]), 0)
    return torch.sparse_csr_tensor(crow.to(torch.int32), c.to(torch.int32), v.contiguous(), size=shape)


class _Hyper:
    """One hyperedge layer: nodes hn (Eh x 27), slot weights a, b (Eh x 27 x H), channel mixes W (H x F x F)."""

    def __init__(self, hn, a, b, W, deg, N):
        Eh, _, H = a.shape
        self.H, self.Eh, self.N = H, Eh, N
        he = (torch.arange(H, device=dev)[:, None, None] * Eh + torch.arange(Eh, device=dev)[None, :, None]).expand(H, Eh, 27)
        nodes = hn[None].expand(H, Eh, 27)
        ga = a.permute(2, 0, 1)                                                         # H x Eh x 27
        sb = b.permute(2, 0, 1) / deg[nodes]
        he, nodes, ga, sb = he.reshape(-1), nodes.reshape(-1), ga.reshape(-1).to(f32), sb.reshape(-1).to(f32)
        self.G = _csr(he, nodes, ga, (H * Eh, N))
        self.S = _csr(nodes, he, sb, (N, H * Eh))
        self.Gt = _csr(nodes, he, ga, (N, H * Eh))
        self.St = _csr(he, nodes, sb, (H * Eh, N))
        self.W = W.detach().to(f32).contiguous()
        self.Wt = self.W.transpose(1, 2).contiguous()

    def fwd(self, X):
        N, B, F = X.shape
        Z = torch.sparse.mm(self.G, X.reshape(N, B * F))
        Z = torch.bmm(Z.reshape(self.H, self.Eh * B, F), self.W)
        return torch.sparse.mm(self.S, Z.reshape(self.H * self.Eh, B * F)).reshape(N, B, F)

    def adj(self, Y):
        N, B, F = Y.shape
        Z = torch.sparse.mm(self.St, Y.reshape(N, B * F))
        Z = torch.bmm(Z.reshape(self.H, self.Eh * B, F), self.Wt)
        return torch.sparse.mm(self.Gt, Z.reshape(self.H * self.Eh, B * F)).reshape(N, B, F)


class FastNet:
    """Frozen MGNO / MGNO2 on one geometry: ext(qd) / ext_T(y) (the network map and its adjoint) and field(q),
    field_T(y), s_hat(q) with the trainlib.Geo wrapper (exact rigid part, exact port values)."""

    @torch.no_grad()
    def __init__(self, model, geo):
        self.model, self.geo = model, geo
        m, c = model, model.caches[geo.case]
        self.c = c
        self.N = c.N
        gp = m.geometry(geo.case)
        self.two = isinstance(m, MD.MGNO2)
        self.pm = c.is_port.to(f32)[:, None, None]
        self.W_in, self.W_out = m.W_in.detach().to(f32), m.W_out.detach().to(f32)
        ab = gp['ab']
        L = m.L_pre + m.L_post
        self.elem = [_Hyper(c.en, ab[:, :, 0, l], ab[:, :, 1, l], m.W[l], c.deg, c.N) for l in range(L)]
        if self.two:
            fab, xab = gp['fab'], gp['xab']
            Lg = m.L_pre + m.L_post + m.n_fringe
            fe, ff = (c.el_fade, c.gp_fade) if getattr(m, 'fringe_soft', False) else (None, None)   # models.MGNO2._hyper fade
            self.face = []
            for g in range(Lg):
                mask = c.gp_fringe if g >= m.L_pre + m.L_post else None
                hn, a, b = c.fn, fab[:, :, 0, g], fab[:, :, 1, g]
                if mask is not None:
                    hn, a, b = hn[mask], a[mask], b[mask]
                    if ff is not None:
                        b = b * ff[:, None, None]
                self.face.append(_Hyper(hn, a, b, m.Wf[g], c.fdeg, c.N))
            if fe is None:
                self.fringe = [_Hyper(c.en[c.el_fringe], xab[c.el_fringe][:, :, 0, l], xab[c.el_fringe][:, :, 1, l], m.Wx[l], c.deg, c.N)
                               for l in range(m.n_fringe)]
            else:
                self.fringe = [_Hyper(c.en[c.el_fringe], xab[c.el_fringe][:, :, 0, l], xab[c.el_fringe][:, :, 1, l] * fe[:, None, None],
                                      m.Wx[l], c.deg, c.N) for l in range(m.n_fringe)]
        # grid transfers
        self.R, self.Rt, self.P, self.Pt = [], [], [], []
        for l, t in enumerate(c.trans):
            i, v, w = t['i'], t['v'], t['w']
            wr = t['w'] * gp['rw'][l][0][i]
            den = torch.zeros(t['n_dst'], device=dev).index_add_(0, v, wr)
            rv = wr / den[v]
            self.R.append(_csr(v, i, rv, (t['n_dst'], t['n_src']))); self.Rt.append(_csr(i, v, rv, (t['n_src'], t['n_dst'])))
            denp = torch.zeros(t['n_src'], device=dev).index_add_(0, i, w)
            pv = m.skip[l].detach() * gp['rw'][l][1][i] * w / denp[i]
            self.P.append(_csr(i, v, pv, (t['n_src'], t['n_dst']))); self.Pt.append(_csr(v, i, pv, (t['n_dst'], t['n_src'])))
        self.gates = [g_.detach() for g_ in gp['gates']]
        self.convs = [w_.detach() for w_ in m.convs]
        self.cpl, self.levels, self.L_pre, self.L_post = m.cpl, m.levels, m.L_pre, m.L_post
        self.n_fringe = m.n_fringe if self.two else 0
        # wrapper (trainlib.Geo.field): rigid split and port overwrite, float32
        self.RP, self.RA, self.RPpinv = geo.RP.to(f32), geo.RA.to(f32), geo.RPpinv.to(f32)
        self.Pidx = geo.P

    # ------------------------------------------------------------------ pieces
    def _clamp(self, X, X0):
        return X * (1 - self.pm) + X0 * self.pm

    def _conv(self, Xc, l, k, transpose=False):
        t = self.c.trans[l]
        m_, B, F = t['m'], Xc.shape[1], Xc.shape[2]
        dense = torch.zeros((m_ ** 3, B, F), device=dev)
        if transpose:
            dense.index_copy_(0, t['dense_idx'], Xc * self.gates[l][:, k][:, None, :])
            v = Fn.conv_transpose3d(dense.permute(1, 2, 0).reshape(B, F, m_, m_, m_), self.convs[l][k], padding=1)
            return Xc + v.reshape(B, F, m_ ** 3).permute(2, 0, 1)[t['dense_idx']]
        dense.index_copy_(0, t['dense_idx'], Xc)
        v = Fn.conv3d(dense.permute(1, 2, 0).reshape(B, F, m_, m_, m_), self.convs[l][k], padding=1)
        return Xc + v.reshape(B, F, m_ ** 3).permute(2, 0, 1)[t['dense_idx']] * self.gates[l][:, k][:, None, :]

    def _sp(self, A, X):
        n, B, F = X.shape
        return torch.sparse.mm(A, X.reshape(n, B * F)).reshape(A.shape[0], B, F)

    def _layers(self):
        """The fine-level layer sequence as (kind, index) in forward order, split at the U-Net."""
        pre, post = [], []
        for l in range(self.L_pre):
            pre.append(self.elem[l])
            if self.two:
                pre.append(self.face[l])
        for l in range(self.L_post):
            post.append(self.elem[self.L_pre + l])
            if self.two:
                post.append(self.face[self.L_pre + l])
        for l in range(self.n_fringe):
            post.append(self.fringe[l]); post.append(self.face[self.L_pre + self.L_post + l])
        return pre, post

    # ------------------------------------------------------------------ model map and its adjoint (float32)
    @torch.no_grad()
    def ext(self, qd):
        c = self.c
        B = qd.shape[1]
        X0 = torch.zeros((self.N, B, self.W_in.shape[1]), device=dev)
        X0[c.port_nodes] = qd.reshape(-1, 3, B).permute(0, 2, 1) @ self.W_in
        pre, post = self._layers()
        X = X0
        for h in pre:
            X = self._clamp(X + h.fwd(X), X0)
        skips, Xl = [], X
        for l in range(self.levels):
            skips.append(Xl)
            Xl = self._sp(self.R[l], Xl)
            for j in range(self.cpl):
                Xl = self._conv(Xl, l, j)
        for l in reversed(range(self.levels)):
            for j in range(self.cpl):
                Xl = self._conv(Xl, l, self.cpl + j)
            Xl = skips[l] + self._sp(self.P[l], Xl)
        X = self._clamp(Xl, X0)
        for h in post:
            X = self._clamp(X + h.fwd(X), X0)
        return (X @ self.W_out).permute(0, 2, 1).reshape(-1, B)

    @torch.no_grad()
    def ext_T(self, y):
        c = self.c
        B = y.shape[1]
        Xb = y.reshape(self.N, 3, B).permute(0, 2, 1) @ self.W_out.T
        X0b = torch.zeros_like(Xb)
        pre, post = self._layers()
        for h in reversed(post):
            X0b += Xb * self.pm
            A = Xb * (1 - self.pm)
            Xb = A + h.adj(A)
        X0b += Xb * self.pm
        G = Xb * (1 - self.pm)
        gskip = []
        for l in range(self.levels):
            gskip.append(G)
            G = self._sp(self.Pt[l], G)
            for j in reversed(range(self.cpl)):
                G = self._conv(G, l, self.cpl + j, transpose=True)
        for l in reversed(range(self.levels)):
            for j in reversed(range(self.cpl)):
                G = self._conv(G, l, j, transpose=True)
            G = self._sp(self.Rt[l], G) + gskip[l]
        Xb = G
        for h in reversed(pre):
            X0b += Xb * self.pm
            A = Xb * (1 - self.pm)
            Xb = A + h.adj(A)
        X0b += Xb
        return (X0b[c.port_nodes] @ self.W_in.T).permute(0, 2, 1).reshape(-1, B)

    # ------------------------------------------------------------------ Geo wrapper and the variational reaction
    @torch.no_grad()
    def field(self, q):
        q32 = q.to(f32)
        cc = self.RPpinv @ q32
        u = self.ext(q32 - self.RP @ cc) + self.RA @ cc
        return u.index_copy(0, self.Pidx, q32)

    @torch.no_grad()
    def field_T(self, y):
        y = y.to(f32)
        yP = y[self.Pidx]
        yI = y.index_fill(0, self.Pidx, 0.0)
        qd = self.ext_T(yI)
        return yP + qd + self.RPpinv.T @ (self.RA.T @ yI - self.RP.T @ qd)

    @torch.no_grad()
    def s_hat(self, q):
        """S_hat q = E_hat^T K E_hat q (the K product in float64, as trainlib.Geo.s_hat_apply)."""
        u = self.field(q).to(f64)
        return self.field_T((self.geo.C.K @ u).to(f32)).to(f64)
