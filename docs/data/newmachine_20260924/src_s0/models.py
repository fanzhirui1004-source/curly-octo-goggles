"""Model registry for step 1. Every model: forward(geo, qd) -> field (nb, B) float32, linear in qd.

'diffusion' : plumbing test and tiny baseline: T steps of damped graph diffusion on the active-node graph (element node
              pairs weighted by the element material volume), ports held; learnable step sizes only.
"""
import numpy as np
import torch
import torch.nn as nn

dev = torch.device('cuda:0')
f32 = torch.float32


class GeoCache:
    """Per-geometry tensors shared by models (built once, not parameters)."""

    def __init__(self, geo):
        nd = geo.nd
        self.N = len(nd['node_ids'])
        self.en = torch.as_tensor(nd['elem_nodes'].astype(np.int64), device=dev)            # E x 27
        self.M = torch.as_tensor(nd['moments'], device=dev)                                  # E x 125 (fp64)
        self.is_port = torch.as_tensor(nd['is_port'], device=dev)
        self.port_nodes = torch.nonzero(self.is_port).squeeze(1)
        self.int_nodes = torch.nonzero(~self.is_port).squeeze(1)
        self.grid = torch.as_tensor(nd['grid'].astype(np.int64), device=dev)
        self.weak = torch.as_tensor(nd['weak'], device=dev)

    def laplacian(self, weight='volume'):
        en = self.en
        w = self.M[:, 0].abs() if weight == 'volume' else torch.ones(len(en), dtype=torch.float64, device=dev)
        iu = torch.triu_indices(27, 27, 1, device=dev)
        a, b = en[:, iu[0]].reshape(-1), en[:, iu[1]].reshape(-1)
        ww = w[:, None].expand(-1, iu.shape[1]).reshape(-1)
        W = torch.sparse_coo_tensor(torch.stack([torch.cat([a, b]), torch.cat([b, a])]), torch.cat([ww, ww]),
                                    (self.N, self.N)).coalesce()
        deg = torch.sparse.sum(W, 1).to_dense()
        return W.to(f32).to_sparse_csr(), deg.to(f32)


def to_nodes(qd, cache):
    """port DOFs (np, B) node-major xyz -> node field (N, 3B) with zeros off the ports."""
    B = qd.shape[1]
    u = torch.zeros((cache.N, 3 * B), dtype=qd.dtype, device=qd.device)
    u[cache.port_nodes] = qd.reshape(-1, 3 * B)
    return u


def to_dofs(u, B):
    return u.reshape(-1, B)


class Diffusion(nn.Module):
    def __init__(self, geos, T=32, omega=0.8):
        super().__init__()
        self.T = T
        self.omega = nn.Parameter(torch.full((T,), float(omega)))
        self.cache = {}
        for g in geos:
            c = GeoCache(g)
            c.W, c.deg = c.laplacian()
            self.cache[g.case] = c

    def forward(self, geo, qd):
        c = self.cache[geo.case]
        B = qd.shape[1]
        u = to_nodes(qd, c)
        keep = c.is_port[:, None].to(f32)
        for k in range(self.T):
            avg = (c.W @ u) / c.deg[:, None]
            u = u + self.omega[k] * (1 - keep) * (avg - u)
        return to_dofs(u, B)


class _GLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qd, gl):
        ctx.gl = gl
        return gl.ext(qd.to(torch.float64)).to(qd.dtype)

    @staticmethod
    def backward(ctx, g):
        return ctx.gl.ext_T(g.to(torch.float64)).to(g.dtype), None


class GraphLiftModel(nn.Module):
    """Zero-parameter baseline (graph-harmonic extension), differentiable through its explicit adjoint."""

    def __init__(self, geos, weight='volume'):
        super().__init__()
        import ops as OP
        self.gl = {g.case: OP.GraphLift(g.C, weight=weight) for g in geos}

    def forward(self, geo, qd):
        return _GLFn.apply(qd, self.gl[geo.case])


# ---------------------------------------------------------------------------------------------------------------------
# MGNO v0: a fully learned multiscale linear graph neural operator (main line; no exact K or factorization inside).
#   q-path (linear in q, no biases, no activations): port embedding -> fine element-hyperedge message passing with
#   geometry-generated slot weights -> U-Net over the fixed vertex grids 33^3 / 17^3 / 9^3 (restriction with
#   geometry-generated node weights, 3x3x3 convolutions with geometry-generated per-voxel channel gates, prolongation)
#   -> fine post-smoothing -> linear read-out. Port states are clamped to the injected port embedding after every fine
#   layer (Dirichlet on the state), so information enters only from the ports.
#   geometry path (nonlinear, once per geometry): element features from the 125 moments (volume fraction and material
#   centroid/inertia moments in local coordinates), node flags (port/box/cut/weak) and the log stiffness diagonal,
#   two rounds of element<->node message passing, heads that emit the slot weights, restriction weights and gates.
# ---------------------------------------------------------------------------------------------------------------------
class MGCache:
    def __init__(self, geo, levels):
        c = GeoCache(geo)
        nd = geo.nd
        M = c.M
        vol = M[:, 0].clamp_min(1e-30)
        h3 = (1.0 / nd['n'].item()) ** 3 if np.ndim(nd['n']) == 0 else (1.0 / 32) ** 3
        vf = (vol / h3).clamp(0, 1)
        ef = torch.cat([vf[:, None], torch.log(vf[:, None] + 1e-6) / 10, M[:, 1:] / vol[:, None]], 1)
        c.efeat = ef.to(f32)                                                              # E x 126
        d3 = torch.as_tensor(nd['diag3'], device=dev).reshape(-1, 9).norm(dim=1)
        ld = torch.log(d3 / d3.median())
        flags = torch.stack([torch.as_tensor(nd[k], device=dev).to(f32) for k in ('is_port', 'is_box', 'is_cut', 'weak')], 1)
        g = c.grid.to(f32) / 64.0
        c.nfeat = torch.cat([flags, (ld / 5)[:, None].to(f32), torch.sin(2 * np.pi * g), torch.cos(2 * np.pi * g)], 1)   # N x 11
        c.deg = torch.bincount(c.en.reshape(-1), minlength=c.N).to(f32).clamp_min(1)
        # grid transfer: fine Q2 nodes (65^3 indices) -> 33^3 -> 17^3 -> 9^3, trilinear (full weighting) pairs
        c.trans = []
        src = c.grid
        m = 65
        for l in range(levels):
            mc = (m + 1) // 2
            pairs_i, pairs_v, pairs_w = [], [], []
            for off in np.ndindex(2, 2, 2):
                lo = torch.div(src, 2, rounding_mode='floor')
                odd = (src % 2 == 1)
                tgt = lo + torch.as_tensor(off, device=dev) * odd
                w = torch.where(odd, torch.full_like(src, 1, dtype=f32) * 0.5, torch.ones_like(src, dtype=f32))
                valid = torch.ones(len(src), dtype=torch.bool, device=dev)
                for a in range(3):
                    valid &= (off[a] == 0) | odd[:, a]
                ww = w.prod(1)
                idx = torch.nonzero(valid).squeeze(1)
                pairs_i.append(idx); pairs_v.append((tgt[idx, 0] * mc + tgt[idx, 1]) * mc + tgt[idx, 2]); pairs_w.append(ww[idx])
            pi, pv, pw = torch.cat(pairs_i), torch.cat(pairs_v), torch.cat(pairs_w)
            active = torch.unique(pv)
            vmap = torch.full((mc ** 3,), -1, dtype=torch.long, device=dev); vmap[active] = torch.arange(len(active), device=dev)
            c.trans.append(dict(i=pi, v=vmap[pv], w=pw, n_src=len(src), n_dst=len(active), m=mc, dense_idx=active))
            src = torch.stack(torch.unravel_index(active, (mc, mc, mc)), 1) if hasattr(torch, 'unravel_index') else \
                torch.stack([active // (mc * mc), (active // mc) % mc, active % mc], 1)
            m = mc
        self.c = c


def _mlp(i, h, o, n=2):
    layers = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n - 2):
        layers += [nn.Linear(h, h), nn.GELU()]
    layers += [nn.Linear(h, o)]
    return nn.Sequential(*layers)


class MGNO(nn.Module):
    def __init__(self, geos, F=32, H=4, L_pre=4, L_post=4, levels=3, Cg=64, conv_per_level=2, slot_dim=8, ckpt=False):
        super().__init__()
        self.F, self.H, self.L_pre, self.L_post, self.levels, self.cpl = F, H, L_pre, L_post, levels, conv_per_level
        self.ckpt = ckpt
        self.caches = {g.case: MGCache(g, levels).c for g in geos}
        self.elem_in = _mlp(126, Cg, Cg)
        self.node_in = _mlp(11 + Cg, Cg, Cg)
        self.mp_e = nn.ModuleList([_mlp(2 * Cg, Cg, Cg) for _ in range(2)])
        self.mp_n = nn.ModuleList([_mlp(2 * Cg, Cg, Cg) for _ in range(2)])
        self.slot = nn.Parameter(0.1 * torch.randn(27, slot_dim))
        L = L_pre + L_post
        self.slot_head = _mlp(2 * Cg + slot_dim, Cg, 2 * H * L)                        # alpha, beta per layer
        self.W = nn.Parameter(torch.randn(L, H, F, F) / np.sqrt(F) * 0.5)
        self.W_in = nn.Parameter(torch.randn(3, F) / np.sqrt(3))
        self.W_out = nn.Parameter(torch.randn(F, 3) / np.sqrt(F) * 0.1)
        self.restrict_head = nn.ModuleList([_mlp(Cg, Cg, 2) for _ in range(levels)])      # restriction / prolongation
        self.gate_head = nn.ModuleList([_mlp(Cg, Cg, F * conv_per_level * 2) for _ in range(levels)])
        self.convs = nn.ParameterList([nn.Parameter(torch.randn(conv_per_level * 2, F, F, 3, 3, 3) / np.sqrt(27 * F) * 0.5)
                                       for _ in range(levels)])
        self.skip = nn.Parameter(torch.ones(levels))
        self._geo_cache = {}

    # ---------------- geometry path
    def geometry(self, case):
        c = self.caches[case]
        ge = self.elem_in(c.efeat)                                                         # E x Cg
        agg = torch.zeros((c.N, ge.shape[1]), device=dev).index_add_(0, c.en.reshape(-1), ge.repeat_interleave(27, 0)) / c.deg[:, None]
        gn = self.node_in(torch.cat([c.nfeat, agg], 1))
        for fe, fn in zip(self.mp_e, self.mp_n):
            ge = ge + fe(torch.cat([ge, gn[c.en].mean(1)], 1))
            agg = torch.zeros_like(gn).index_add_(0, c.en.reshape(-1), ge.repeat_interleave(27, 0)) / c.deg[:, None]
            gn = gn + fn(torch.cat([gn, agg], 1))
        E = c.en.shape[0]
        z = torch.cat([ge[:, None, :].expand(E, 27, -1), gn[c.en], self.slot[None].expand(E, -1, -1)], 2)
        ab = self.slot_head(z).reshape(E, 27, 2, self.L_pre + self.L_post, self.H) * 0.2   # small start
        rw, gates, gl = [], [], gn
        for l, t in enumerate(c.trans):
            r = self.restrict_head[l](gl)
            rw.append((torch.nn.functional.softplus(r[:, 0]) + 1e-3, torch.nn.functional.softplus(r[:, 1]) + 1e-3))
            gsum = torch.zeros((t['n_dst'], gl.shape[1]), device=dev).index_add_(0, t['v'], gl[t['i']] * t['w'][:, None])
            wsum = torch.zeros(t['n_dst'], device=dev).index_add_(0, t['v'], t['w'])
            gl = gsum / wsum[:, None]
            gates.append(2 * torch.sigmoid(self.gate_head[l](gl)).reshape(-1, self.cpl * 2, self.F))
        return dict(ab=ab, rw=rw, gates=gates)

    # ---------------- q path
    def _fine(self, X, X0, pm, c, ab, layer):
        a, b = ab[:, :, 0, layer], ab[:, :, 1, layer]                                    # E x 27 x H
        Xg = X[c.en]                                                                        # E x 27 x B x F
        Z = torch.einsum('eah,eabf->ehbf', a, Xg)
        Z = torch.einsum('ehbf,hfg->ehbg', Z, self.W[layer])
        Y = torch.einsum('eah,ehbg->eabg', b, Z)
        dX = torch.zeros_like(X).index_add_(0, c.en.reshape(-1), Y.reshape(-1, *Y.shape[2:]))
        X = X + dX / c.deg[:, None, None]
        return X * (1 - pm) + X0 * pm

    def _restrict(self, X, t, w):
        num = torch.zeros((t['n_dst'],) + X.shape[1:], device=dev).index_add_(0, t['v'], X[t['i']] * (t['w'] * w[t['i']])[:, None, None])
        den = torch.zeros(t['n_dst'], device=dev).index_add_(0, t['v'], t['w'] * w[t['i']])
        return num / den[:, None, None]

    def _prolong(self, Xc, t, w):
        """fine_i = w_i * sum_v t_w(i, v) Xc_v / sum_v t_w(i, v)  (geometry weights only: linear in Xc)"""
        num = torch.zeros((t['n_src'],) + Xc.shape[1:], device=dev).index_add_(0, t['i'], Xc[t['v']] * t['w'][:, None, None])
        den = torch.zeros(t['n_src'], device=dev).index_add_(0, t['i'], t['w'])
        return num * (w / den)[:, None, None]

    def _convs(self, Xc, t, gate, k0, level):
        m, B, F = t['m'], Xc.shape[1], Xc.shape[2]
        for j in range(self.cpl):
            dense = torch.zeros((m ** 3, B, F), device=dev).index_copy(0, t['dense_idx'], Xc)
            v = dense.permute(1, 2, 0).reshape(B, F, m, m, m)
            v = torch.nn.functional.conv3d(v, self.convs[level][k0 + j], padding=1)
            v = v.reshape(B, F, m ** 3).permute(2, 0, 1)[t['dense_idx']]
            Xc = Xc + v * gate[:, k0 + j][:, None, :]
        return Xc

    def forward(self, geo, qd):
        c = self.caches[geo.case]
        gp = self.geometry(geo.case)
        B = qd.shape[1]
        q3 = qd.reshape(-1, 3, B).permute(0, 2, 1)                                       # port nodes x B x 3
        X0 = torch.zeros((c.N, B, self.F), device=dev)
        X0[c.port_nodes] = q3 @ self.W_in
        pm = c.is_port.to(f32)[:, None, None]
        X = X0
        ck = self.ckpt and torch.is_grad_enabled()
        fine = (lambda X_, l_: torch.utils.checkpoint.checkpoint(self._fine, X_, X0, pm, c, gp['ab'], l_, use_reentrant=False)) if ck \
            else (lambda X_, l_: self._fine(X_, X0, pm, c, gp['ab'], l_))
        for l in range(self.L_pre):
            X = fine(X, l)
        # U-Net over the vertex grids
        skips, Xl = [], X
        for l, t in enumerate(c.trans):
            skips.append(Xl)
            Xl = self._restrict(Xl, t, gp['rw'][l][0])
            Xl = self._convs(Xl, t, gp['gates'][l], 0, l)
        for l in reversed(range(self.levels)):
            t = c.trans[l]
            Xl = self._convs(Xl, t, gp['gates'][l], self.cpl, l)
            Xl = skips[l] + self.skip[l] * self._prolong(Xl, t, gp['rw'][l][1])
        X = Xl * (1 - pm) + X0 * pm
        for l in range(self.L_post):
            X = fine(X, self.L_pre + l)
        u = X @ self.W_out                                                                # N x B x 3
        return u.permute(0, 2, 1).reshape(-1, B)


# ---------------------------------------------------------------------------------------------------------------------
# MGNO v0.1: v0 plus ghost-face hyperedges and fringe layers (the block diagnosis of v0 put ~36% of the soft-direction
# error in the ghost-penalty energy and ~45% of the body error in elements with weak (fictitious-fringe) nodes).
#   - every fine layer is followed by a ghost-face layer: hyperedge = the 54 node slots of the two elements sharing the
#     face; slot weights generated from (face embedding, node embedding, slot embedding); same linear form as elements.
#   - n_fringe extra (element + ghost-face) layer pairs at the end, restricted to elements / faces touching weak nodes.
# Still fully learned and linear in q: no K, no factorization in the forward pass.
# ---------------------------------------------------------------------------------------------------------------------
class MGNO2(MGNO):
    def __init__(self, geos, F=32, H=4, L_pre=4, L_post=4, levels=3, Cg=64, conv_per_level=2, slot_dim=8, n_fringe=4):
        super().__init__(geos, F=F, H=H, L_pre=L_pre, L_post=L_post, levels=levels, Cg=Cg,
                         conv_per_level=conv_per_level, slot_dim=slot_dim)
        self.n_fringe = n_fringe
        self.face_in = _mlp(2 * Cg + 3, Cg, Cg)
        self.fslot = nn.Parameter(0.1 * torch.randn(27, slot_dim))
        Lg = L_pre + L_post + n_fringe                                                    # ghost layers
        Le_extra = n_fringe                                                               # extra element layers
        self.fslot_head = _mlp(2 * Cg + slot_dim, Cg, 2 * H * Lg)
        self.Wf = nn.Parameter(torch.randn(Lg, H, F, F) / np.sqrt(F) * 0.5)
        self.slot_head_x = _mlp(2 * Cg + slot_dim, Cg, 2 * H * max(Le_extra, 1))
        self.Wx = nn.Parameter(torch.randn(max(Le_extra, 1), H, F, F) / np.sqrt(F) * 0.5)
        for g in geos:
            c = self.caches[g.case]
            gpf = torch.as_tensor(g.nd['gp_faces'].astype(np.int64), device=dev)
            c.gp_owner, c.gp_nbr, c.gp_axis = gpf[:, 0], gpf[:, 1], gpf[:, 2]
            # 27-slot face stencil: the owner's two node layers on the face side (18) + the neighbour's middle layer (9),
            # ordered by (layer, in-plane coordinates) so that slots mean the same thing on every face
            o = c.grid[c.en] - 2 * torch.as_tensor(g.nd['elem_cells'].astype(np.int64), device=dev)[:, None, :]   # E x 27 x 3
            a = c.gp_axis
            oo, on = o[c.gp_owner], o[c.gp_nbr]                                              # F x 27 x 3
            def lay(t):
                return torch.gather(t, 2, a[:, None, None].expand(-1, 27, 1)).squeeze(2)
            b = (a + 1) % 3; d_ = (a + 2) % 3
            def key(t):
                return lay(t) * 9 + torch.gather(t, 2, b[:, None, None].expand(-1, 27, 1)).squeeze(2) * 3 + \
                    torch.gather(t, 2, d_[:, None, None].expand(-1, 27, 1)).squeeze(2)
            ko = key(oo) + (lay(oo) == 0) * 1000                                              # push layer 0 to the end
            kn = key(on) + (lay(on) != 1) * 1000
            io = torch.argsort(ko, 1)[:, :18]; inb = torch.argsort(kn, 1)[:, :9]
            c.fn = torch.cat([torch.gather(c.en[c.gp_owner], 1, io), torch.gather(c.en[c.gp_nbr], 1, inb)], 1)   # F x 27
            c.fdeg = torch.bincount(c.fn.reshape(-1), minlength=c.N).to(f32).clamp_min(1)
            wk = c.weak
            c.el_fringe = wk[c.en].any(1)
            c.gp_fringe = wk[c.fn].any(1)

    def geometry(self, case):
        out = super().geometry(case)
        c = self.caches[case]
        # recompute node/element embeddings (cheap) for the face heads
        ge = self.elem_in(c.efeat)
        agg = torch.zeros((c.N, ge.shape[1]), device=dev).index_add_(0, c.en.reshape(-1), ge.repeat_interleave(27, 0)) / c.deg[:, None]
        gn = self.node_in(torch.cat([c.nfeat, agg], 1))
        for fe, fn in zip(self.mp_e, self.mp_n):
            ge = ge + fe(torch.cat([ge, gn[c.en].mean(1)], 1))
            agg = torch.zeros_like(gn).index_add_(0, c.en.reshape(-1), ge.repeat_interleave(27, 0)) / c.deg[:, None]
            gn = gn + fn(torch.cat([gn, agg], 1))
        nf = len(c.gp_owner)
        ax = torch.nn.functional.one_hot(c.gp_axis, 3).to(f32)
        gf = self.face_in(torch.cat([ge[c.gp_owner], ge[c.gp_nbr], ax], 1))
        z = torch.cat([gf[:, None, :].expand(nf, 27, -1), gn[c.fn], self.fslot[None].expand(nf, -1, -1)], 2)
        Lg = self.L_pre + self.L_post + self.n_fringe
        out['fab'] = self.fslot_head(z).reshape(nf, 27, 2, Lg, self.H) * 0.2
        E = c.en.shape[0]
        zx = torch.cat([ge[:, None, :].expand(E, 27, -1), gn[c.en], self.slot[None].expand(E, -1, -1)], 2)
        out['xab'] = self.slot_head_x(zx).reshape(E, 27, 2, max(self.n_fringe, 1), self.H) * 0.2
        return out

    def _ck(self, fn, *args):
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _hyper(self, X, X0, pm, hn, deg, ab, W, layer, mask=None):
        a, b = ab[:, :, 0, layer], ab[:, :, 1, layer]
        if mask is not None:
            hn, a, b = hn[mask], a[mask], b[mask]
        Xg = X[hn]
        Z = torch.einsum('eah,eabf->ehbf', a, Xg)
        Z = torch.einsum('ehbf,hfg->ehbg', Z, W)
        Y = torch.einsum('eah,ehbg->eabg', b, Z)
        dX = torch.zeros_like(X).index_add_(0, hn.reshape(-1), Y.reshape(-1, *Y.shape[2:]))
        X = X + dX / deg[:, None, None]
        return X * (1 - pm) + X0 * pm

    def forward(self, geo, qd):
        c = self.caches[geo.case]
        gp = self.geometry(geo.case)
        B = qd.shape[1]
        q3 = qd.reshape(-1, 3, B).permute(0, 2, 1)
        X0 = torch.zeros((c.N, B, self.F), device=dev)
        X0[c.port_nodes] = q3 @ self.W_in
        pm = c.is_port.to(f32)[:, None, None]
        X = X0
        gl = 0
        for l in range(self.L_pre):
            X = self._ck(lambda X_, l_=l: self._fine(X_, X0, pm, c, gp['ab'], l_), X)
            X = self._ck(lambda X_, g_=gl: self._hyper(X_, X0, pm, c.fn, c.fdeg, gp['fab'], self.Wf[g_], g_), X); gl += 1
        skips, Xl = [], X
        for l, t in enumerate(c.trans):
            skips.append(Xl)
            Xl = self._restrict(Xl, t, gp['rw'][l][0])
            Xl = self._convs(Xl, t, gp['gates'][l], 0, l)
        for l in reversed(range(self.levels)):
            t = c.trans[l]
            Xl = self._convs(Xl, t, gp['gates'][l], self.cpl, l)
            Xl = skips[l] + self.skip[l] * self._prolong(Xl, t, gp['rw'][l][1])
        X = Xl * (1 - pm) + X0 * pm
        for l in range(self.L_post):
            X = self._ck(lambda X_, l_=l: self._fine(X_, X0, pm, c, gp['ab'], self.L_pre + l_), X)
            X = self._ck(lambda X_, g_=gl: self._hyper(X_, X0, pm, c.fn, c.fdeg, gp['fab'], self.Wf[g_], g_), X); gl += 1
        for l in range(self.n_fringe):
            X = self._ck(lambda X_, l_=l: self._hyper(X_, X0, pm, c.en, c.deg, gp['xab'], self.Wx[l_], l_, mask=c.el_fringe), X)
            X = self._ck(lambda X_, g_=gl: self._hyper(X_, X0, pm, c.fn, c.fdeg, gp['fab'], self.Wf[g_], g_, mask=c.gp_fringe), X); gl += 1
        u = X @ self.W_out
        return u.permute(0, 2, 1).reshape(-1, B)


REGISTRY = {'diffusion': Diffusion, 'graphlift': GraphLiftModel, 'mgno': MGNO, 'mgno2': MGNO2}


def build(name, geos, **kw):
    return REGISTRY[name](geos, **kw)
