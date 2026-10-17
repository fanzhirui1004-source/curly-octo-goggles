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
Opt-in (env FUSED_HYPER=1 at import, or fastnet.FUSED = True before building): the hyperedge layers run through the Triton
gather / scatter kernels of fused_hyper.py instead of the four CSR matrices (same map to float32 rounding, about a quarter
of the layer memory: slot weights instead of G, S, G^T, S^T). Default off.
B3 (model b3=True): the stiffness-share mixing is folded into the frozen scatter weights before the layers are built,
b <- b (1 + lambda (pi deg - 1)) (models.MGNO._b3), so both the CSR and the fused paths (and the adjoints) need nothing new.
"""
import os
import torch
import torch.nn.functional as Fn
import models as MD
import fastidx as FI

dev, f32, f64 = MD.dev, MD.f32, torch.float64
FUSED = os.environ.get('FUSED_HYPER', '0') == '1'


SLIM = os.environ.get('OPL_V3_SLIM', '1') == '1'
# src_v3, OPL_V3_SLOTS=recompute (opt-in): the hyperedge slot weights (a, b of every layer: ~270 MiB for a large cell,
# about half of its deployed state) are not stored; FastNet keeps the geometry embeddings (ge, gn: ~31 MiB) and
# recomputes the slot weights once per application (s_hat / field / field_T) with exactly the ops and shapes of
# models.MGNO2.geometry (bitwise the same values), dropping them afterwards. Default 'store' (src_v2 behaviour).
SLOTS = os.environ.get('OPL_V3_SLOTS', 'auto')                        # 'auto' (host budget) | 'store' | 'recompute'
_SHARED = {}


def _share(t):
    """Flag a tensor as shared by all cells (a model parameter or a quantity derived from parameters only): stream_ops
    leaves it on the device instead of copying it into every cell's streamed state (src_v3)."""
    try:
        t._opl_shared = True
    except (AttributeError, RuntimeError):
        pass
    return t


def _shared_t(W):
    """The transposed copy W^T (contiguous) of a parameter-derived W, one per parameter storage (src_v3)."""
    key = (W.data_ptr(), tuple(W.shape), tuple(W.stride()))
    t = _SHARED.get(key)
    if t is None:
        t = _SHARED[key] = _share(W.transpose(1, 2).contiguous())
    return t


def _csr(rows, cols, vals, shape):
    """CSR from (unsorted, duplicate-free) triplets."""
    key = rows * shape[1] + cols
    o = torch.argsort(key)
    r, c, v = rows[o], cols[o], vals[o]
    crow = torch.zeros(shape[0] + 1, dtype=torch.int64, device=dev)
    crow[1:] = torch.cumsum(torch.bincount(r, minlength=shape[0]), 0)
    return torch.sparse_csr_tensor(crow.to(torch.int32), c.to(torch.int32), v.contiguous(), size=shape)


SLOTFUSE = os.environ.get('OPL_V3_SLOTFUSE', '0') == '1'                         # src_v3 S19 (slotfused.clip_scale)


class _Hyper:
    """One hyperedge layer: nodes hn (Eh x 27), slot weights a, b (Eh x 27 x H), channel mixes W (H x F x F)."""

    def __init__(self, hn, a, b, W, deg, N):
        Eh, _, H = a.shape
        self.H, self.Eh, self.N = H, Eh, N
        self.fused = FUSED
        if self.fused:                                                                  # slot weights for fused_hyper
            self.hn, self.deg = hn.contiguous(), deg.to(f32).contiguous()
            self.a, self.b = a.detach().to(f32).contiguous(), b.detach().to(f32).contiguous()
            self.W = W.detach().to(f32).contiguous()
            if SLIM:                                                                    # src_v3: parameters stay shared
                _share(self.W); self.Wt = _shared_t(self.W)
            else:
                self.Wt = self.W.transpose(1, 2).contiguous()
            return
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

    def set_ab(self, a, b):
        self.a, self.b = a.detach().to(f32).contiguous(), b.detach().to(f32).contiguous()

    def drop_ab(self):
        self.a = self.b = None

    def fwd(self, X):
        N, B, F = X.shape
        if self.fused:
            import fused_hyper as FH
            Z = FH.gather(X.reshape(N, B * F).contiguous(), self.hn, self.a, self.deg, self.H, FH.BLK, False)
            Z = torch.bmm(Z.view(self.H, self.Eh * B, F), self.W).view(self.H, self.Eh, B * F)
            return FH.scatter(Z, self.hn, self.b, self.deg, N, FH.BLK, True).view(N, B, F)
        Z = torch.sparse.mm(self.G, X.reshape(N, B * F))
        Z = torch.bmm(Z.reshape(self.H, self.Eh * B, F), self.W)
        return torch.sparse.mm(self.S, Z.reshape(self.H * self.Eh, B * F)).reshape(N, B, F)

    def fwd_res(self, X):
        """X + fwd(X) with the residual folded into the scatter (fused path only)."""
        N, B, F = X.shape
        import fused_hyper as FH
        Z = FH.gather(X.reshape(N, B * F).contiguous(), self.hn, self.a, self.deg, self.H, FH.BLK, False)
        Z = torch.bmm(Z.view(self.H, self.Eh * B, F), self.W).view(self.H, self.Eh, B * F)
        return FH.scatter(Z, self.hn, self.b, self.deg, N, FH.BLK, True, init=X).view(N, B, F)

    def adj_res(self, Y):
        """Y + adj(Y) with the residual folded into the scatter (fused path only)."""
        N, B, F = Y.shape
        import fused_hyper as FH
        Z = FH.gather(Y.reshape(N, B * F).contiguous(), self.hn, self.b, self.deg, self.H, FH.BLK, True)
        Z = torch.bmm(Z.view(self.H, self.Eh * B, F), self.Wt).view(self.H, self.Eh, B * F)
        return FH.scatter(Z, self.hn, self.a, self.deg, N, FH.BLK, False, init=Y).view(N, B, F)

    def adj(self, Y):
        N, B, F = Y.shape
        if self.fused:
            import fused_hyper as FH
            Z = FH.gather(Y.reshape(N, B * F).contiguous(), self.hn, self.b, self.deg, self.H, FH.BLK, True)
            Z = torch.bmm(Z.view(self.H, self.Eh * B, F), self.Wt).view(self.H, self.Eh, B * F)
            return FH.scatter(Z, self.hn, self.a, self.deg, N, FH.BLK, False).view(N, B, F)
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
        self.pmask = c.is_port.bool()[:, None, None]                                     # where() forms (OPL_FASTIDX)
        self.W_in, self.W_out = m.W_in.detach().to(f32), m.W_out.detach().to(f32)
        ab = gp['ab']
        L = m.L_pre + m.L_post
        b3 = MD.MGNO._b3                                                                # B3 fold (identity for arg None)
        arg = (lambda kind, i: m._b3arg(c, kind, i) if getattr(m, 'b3', False) else None)
        self.elem = [_Hyper(c.en, ab[:, :, 0, l], b3(ab[:, :, 1, l], arg('le', l)), m.W[l], c.deg, c.N) for l in range(L)]
        if self.two:
            fab, xab = gp['fab'], gp['xab']
            Lg = m.L_pre + m.L_post + m.n_fringe
            fe, ff = (c.el_fade, c.gp_fade) if getattr(m, 'fringe_soft', False) else (None, None)   # models.MGNO2._hyper fade
            self.face = []
            for g in range(Lg):
                mask = c.gp_fringe if g >= m.L_pre + m.L_post else None
                hn, a, b = c.fn, fab[:, :, 0, g], b3(fab[:, :, 1, g], arg('lf', g))           # full-set fold, then mask
                if mask is not None:
                    hn, a, b = hn[mask], a[mask], b[mask]
                    if ff is not None:
                        b = b * ff[:, None, None]
                self.face.append(_Hyper(hn, a, b, m.Wf[g], c.fdeg, c.N))
            xb = [b3(xab[:, :, 1, l], arg('lx', l))[c.el_fringe] for l in range(m.n_fringe)]   # full-set fold, then mask
            if fe is None:
                self.fringe = [_Hyper(c.en[c.el_fringe], xab[c.el_fringe][:, :, 0, l], xb[l], m.Wx[l], c.deg, c.N)
                               for l in range(m.n_fringe)]
            else:
                self.fringe = [_Hyper(c.en[c.el_fringe], xab[c.el_fringe][:, :, 0, l], xb[l] * fe[:, None, None],
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
        if SLIM:                                                                        # src_v3: parameters stay shared
            for t_ in self.convs + [self.W_in, self.W_out]:
                _share(t_)
        self.cpl, self.levels, self.L_pre, self.L_post = m.cpl, m.levels, m.L_pre, m.L_post
        self.n_fringe = m.n_fringe if self.two else 0
        self._rc, self._rc_depth, self._rc_src = None, 0, None
        if SLOTS in ('recompute', 'auto') and FUSED and self.two:                     # src_v3: slot weights on demand
            self._rc_src = dict(ge=gp['ge'].detach(), gn=gp['gn'].detach())             # (auto: decided after the warm-up,
            if SLOTS == 'recompute':                                                    # recompute_slots / keep_slots)
                self.recompute_slots()
        # wrapper (trainlib.Geo.field): rigid split and port overwrite, float32
        self.RP, self.RA, self.RPpinv = geo.RP.to(f32), geo.RA.to(f32), geo.RPpinv.to(f32)
        self.Pidx = geo.P

    def slot_bytes(self):
        """Bytes of the stored slot weights (a, b of every hyperedge layer)."""
        return sum(t.numel() * t.element_size() for h in self._hypers() for t in (h.a, h.b) if t is not None)

    def recompute_slots(self):
        """src_v3: from now on recompute the slot weights in every application (bitwise the same values) instead of
        keeping them (OPL_V3_SLOTS=recompute, or auto when the host-memory budget is short). Before slim()."""
        if self._rc is None and self._rc_src is not None:
            self._rc = self._rc_src
            for h in self._hypers():
                h.drop_ab()
        return self._rc is not None

    def keep_slots(self):
        """src_v3 (auto): keep the stored slot weights; the recompute inputs are dropped."""
        if self._rc is None:
            self._rc_src = None

    def slim(self):
        """src_v3: after the freeze, keep only what ext / ext_T read from the per-geometry cache (port nodes, the
        U-Net levels' sizes and dense indices, the split-graph edges if any); the rest of models.MGCache (element and
        face node lists, float64 moments, input features, stiffness shares, transfer pairs) is no longer referenced
        from here (the caller drops the model's own reference)."""
        import types
        c = self.c
        keep = ('m', 'dense_idx', 'e_src', 'e_dst', 'e_k', 'e_ptr', 'n_dst', 'n_src')
        ns = types.SimpleNamespace(N=c.N, port_nodes=c.port_nodes, trans=[{k: t[k] for k in keep if k in t} for t in c.trans])
        if self._rc is not None:                                                        # what the slot recompute reads
            for k in ('en', 'fn', 'gp_owner', 'gp_nbr', 'gp_axis', 'b3_el', 'b3_gp', 'el_fringe', 'gp_fringe', 'el_fade', 'gp_fade'):
                if hasattr(c, k):
                    setattr(ns, k, getattr(c, k))
        self.c = ns
        return self

    # ------------------------------------------------------------------ slot weights on demand (OPL_V3_SLOTS=recompute)
    def _hypers(self):
        return list(self.elem) + (list(self.face) + list(self.fringe) if self.two else [])

    def _slot_tensors(self):
        """ab, fab, xab exactly as models.MGNO.geometry / MGNO2.geometry form them (same ops, same shapes)."""
        m, c, ge, gn = self.model, self.c, self._rc['ge'], self._rc['gn']
        E = c.en.shape[0]

        def clip(name, x, shape):
            st = m._bst[name]
            if st == 'off':
                return x
            return MD._SoftClip.apply(x, getattr(m, 'bnd_' + name).view(shape), st == 'mixed', m.bound_knee)
        def fused(name):                                                 # src_v3 S19: one Triton pass for * 0.2 + clip
            st = m._bst[name] if m.bounded else 'off'
            return SLOTFUSE and (st == 'off' or m.bound_knee > 0), st

        def head(mod, x, name, shape):
            on, st = fused(name)
            if on:
                import slotfused as SF
                raw = mod(x)
                A = getattr(m, 'bnd_' + name) if st != 'off' else None
                return SF.clip_scale(raw.reshape(-1, raw.shape[-1]), A, st, m.bound_knee, 0.2, m.H).reshape(shape)
            y = mod(x).reshape(shape) * 0.2
            return clip(name, y, (1, 1) + shape[2:4] + (1,)) if m.bounded else y
        L = m.L_pre + m.L_post
        z = torch.cat([ge[:, None, :].expand(E, 27, -1), gn[c.en], m.slot[None].expand(E, -1, -1)], 2)
        ab = head(m.slot_head, z, 'ab', (E, 27, 2, L, m.H))
        xab = head(m.slot_head_x, z, 'xab', (E, 27, 2, max(m.n_fringe, 1), m.H))
        del z
        nf = len(c.gp_owner)
        ax = torch.nn.functional.one_hot(c.gp_axis, 3).to(f32)
        gf = m.face_in(torch.cat([ge[c.gp_owner], ge[c.gp_nbr], ax], 1))
        zf = torch.cat([gf[:, None, :].expand(nf, 27, -1), gn[c.fn], m.fslot[None].expand(nf, -1, -1)], 2)
        Lg = L + m.n_fringe
        fab = head(m.fslot_head, zf, 'fab', (nf, 27, 2, Lg, m.H))
        del zf, gf
        return ab, fab, xab

    def _assign_slots(self, ab, fab, xab):
        """The _Hyper slot weights from ab / fab / xab, as __init__ derives them."""
        m, c = self.model, self.c
        L = m.L_pre + m.L_post
        b3 = MD.MGNO._b3
        arg = (lambda kind, i: m._b3arg(c, kind, i) if getattr(m, 'b3', False) else None)
        for l, h in enumerate(self.elem):
            h.set_ab(ab[:, :, 0, l], b3(ab[:, :, 1, l], arg('le', l)))
        fe, ff = (c.el_fade, c.gp_fade) if getattr(m, 'fringe_soft', False) else (None, None)
        for g, h in enumerate(self.face):
            a, b = fab[:, :, 0, g], b3(fab[:, :, 1, g], arg('lf', g))
            if g >= L:
                a, b = a[c.gp_fringe], b[c.gp_fringe]
                if ff is not None:
                    b = b * ff[:, None, None]
            h.set_ab(a, b)
        for l, h in enumerate(self.fringe):
            xb = b3(xab[:, :, 1, l], arg('lx', l))[c.el_fringe]
            h.set_ab(xab[c.el_fringe][:, :, 0, l], xb if fe is None else xb * fe[:, None, None])

    def slots(self):
        """Context: slot weights present inside (recomputed on entry, dropped on exit; no-op in 'store' mode)."""
        import contextlib
        me = self

        @contextlib.contextmanager
        def ctx():
            if me._rc is None:
                yield
                return
            if me._rc_depth == 0:
                with torch.no_grad():
                    me._assign_slots(*me._slot_tensors())
            me._rc_depth += 1
            try:
                yield
            finally:
                me._rc_depth -= 1
                if me._rc_depth == 0:
                    for h in me._hypers():
                        h.drop_ab()
        return ctx()

    # ------------------------------------------------------------------ pieces
    def _clamp(self, X, X0):
        if FI.ON:                                                                       # pm in {0, 1}: the same values
            return torch.where(self.pmask, X0, X)
        return X * (1 - self.pm) + X0 * self.pm

    def _conv(self, Xc, l, k, transpose=False):
        t = self.c.trans[l]
        m_, B, F = t['m'], Xc.shape[1], Xc.shape[2]
        if 'e_src' in t:                                                   # coarse_split: material-graph convolution
            import models as MD_
            if transpose:
                return Xc + MD_.graph_conv(Xc * self.gates[l][:, k][:, None, :], t, self.convs[l][k], transpose=True)
            return Xc + MD_.graph_conv(Xc, t, self.convs[l][k]) * self.gates[l][:, k][:, None, :]
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
    def ext(self, *args):
        with self.slots():
            return self._ext(*args)

    def _ext(self, qd):
        c = self.c
        B = qd.shape[1]
        X0 = torch.zeros((self.N, B, self.W_in.shape[1]), device=dev)
        X0[c.port_nodes] = qd.reshape(-1, 3, B).permute(0, 2, 1) @ self.W_in
        pre, post = self._layers()
        X = X0
        res = lambda h, X_: h.fwd_res(X_) if (FI.ON and h.fused) else X_ + h.fwd(X_)
        for h in pre:
            X = self._clamp(res(h, X), X0)
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
            X = self._clamp(res(h, X), X0)
        return (X @ self.W_out).permute(0, 2, 1).reshape(-1, B)

    @torch.no_grad()
    def ext_T(self, *args):
        with self.slots():
            return self._ext_T(*args)

    def _ext_T(self, y):
        c = self.c
        B = y.shape[1]
        Xb = y.reshape(self.N, 3, B).permute(0, 2, 1) @ self.W_out.T
        X0b = torch.zeros_like(Xb)
        pre, post = self._layers()
        zero = torch.zeros((), device=dev)
        pmk = (lambda X: torch.where(self.pmask, X, zero)) if FI.ON else (lambda X: X * self.pm)
        imk = (lambda X: torch.where(self.pmask, zero, X)) if FI.ON else (lambda X: X * (1 - self.pm))
        ares = lambda h, A_: h.adj_res(A_) if (FI.ON and h.fused) else A_ + h.adj(A_)
        for h in reversed(post):
            X0b += pmk(Xb)
            A = imk(Xb)
            Xb = ares(h, A)
        X0b += pmk(Xb)
        G = imk(Xb)
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
            X0b += pmk(Xb)
            A = imk(Xb)
            Xb = ares(h, A)
        X0b += Xb
        return (X0b[c.port_nodes] @ self.W_in.T).permute(0, 2, 1).reshape(-1, B)

    # ------------------------------------------------------------------ Geo wrapper and the variational reaction
    @torch.no_grad()
    def field(self, *args):
        with self.slots():
            return self._field(*args)

    def _field(self, q):
        q32 = q.to(f32)
        cc = self.RPpinv @ q32
        u = self.ext(q32 - self.RP @ cc) + self.RA @ cc
        u = u.index_copy(0, self.Pidx, q32)
        import trainlib as TL
        return TL.wrap(self.geo.C, u, self.model)                                       # physics wrapper (default off)

    def _tail_T(self, y):
        """Adjoint of the (linear) smoothing tail (explicit recurrence, no autograd graph)."""
        import trainlib as TL
        return TL.smooth_tail_T(self.geo.C, y, self.model.smooth_k, self.model.smooth_alpha)

    @torch.no_grad()
    def field_T(self, *args):
        with self.slots():
            return self._field_T(*args)

    def _field_T(self, y):
        if getattr(self.model, 'smooth_k', 0) or getattr(self.model, 'coarse_space', None):
            import trainlib as TL
            y = TL.wrap_T(self.geo.C, y, self.model)
        y = y.to(f32)
        yP = FI.rows(y, self.Pidx)
        yI = y.index_fill(0, self.Pidx, 0.0)
        qd = self.ext_T(yI)
        return yP + qd + self.RPpinv.T @ (self.RA.T @ yI - self.RP.T @ qd)

    @torch.no_grad()
    def s_hat(self, *args):
        with self.slots():
            return self._s_hat(*args)

    def _s_hat(self, q):
        """S_hat q = E_hat^T K E_hat q (the K product in float64, as trainlib.Geo.s_hat_apply)."""
        import trainlib as TL
        with TL._held(self.geo.C):                                                      # float32 K_e formed once
            u = self.field(q).to(f64)
            return self.field_T((self.geo.C.K @ u).to(f32)).to(f64)


def slim_deploy(op, geo, C, model=None, budget=None, cell_bytes=None):
    """src_v3 (OPL_V3_SLIM=1): after the warm-up application of a deployed cell (and any fingerprint that reads the model
    cache or geo.nd), drop what no later use reads: the model's per-geometry cache (FastNet keeps its own slim view),
    the float64 rigid modes of the geometry wrapper (FastNet holds float32 copies), the network input arrays (host).
    Also the post-warm-up decision on the two optional per-cell items (hostmem.Budget, OPL_V3_KE64 / OPL_V3_SLOTS = auto):
    the float64 partial-element matrices (teacher.Cell.store64) and the stored slot weights (else recomputed).
    cell_bytes: the cell's device footprint after the warm-up (the budget's estimate of a cell's state)."""
    import teacher as TE_
    fast = op.fast
    if budget is not None:
        kb = C.ke64_bytes() if hasattr(C, 'ke64_bytes') and TE_.KE64 == 'auto' else 0
        sb = fast.slot_bytes() if SLOTS == 'auto' and fast._rc_src is not None else 0
        other = None if cell_bytes is None else max(0, cell_bytes - sb)
        kp = getattr(C, '_kpp_cache', None)
        kpb = sum(t.numel() * t.element_size() for t in kp) if kp is not None and not torch.is_tensor(kp) else 0
        keep_slots, keep_ke = budget.decide(kb, sb, other, kpp_bytes=kpb)
        kept = 0
        if sb:
            if keep_slots:
                fast.keep_slots(); kept += sb
            else:
                fast.recompute_slots()
        if hasattr(C, 'store64') and C.store64(allow=keep_ke):
            kept += kb
        budget.account(kept)
    else:
        if hasattr(C, 'store64'):
            C.store64()                                                   # OPL_V3_KE64=store only
        if SLOTS == 'auto':
            fast.keep_slots()
    if not SLIM:
        return
    op.fast.slim()
    if model is not None:
        model.caches.pop(C.case, None)
    geo.RA = geo.RP = geo.RPpinv = None
    geo.nd = None
