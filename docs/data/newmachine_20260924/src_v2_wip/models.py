"""Model registry for step 1. Every model: forward(geo, qd) -> field (nb, B) float32, linear in qd.

'diffusion' : plumbing test and tiny baseline: T steps of damped graph diffusion on the active-node graph (element node
              pairs weighted by the element material volume), ports held; learnable step sizes only.

v2 options of MGNO / MGNO2 (model_args; all default off, and off is the v0 / v0.1 computation op for op):
  bounded=True    (B1) the q-path coefficients made by the geometry path are soft-clipped per group, a = A tanh(a_raw / A):
                  ab, fab, xab with group = layer x {alpha, beta} (buffers bnd_ab (2, L), bnd_fab (2, Lg), bnd_xab (2, Lx));
                  transfer weights w = w_max tanh(softplus(r) / w_max) + 1e-3 (bnd_rw (levels, 2), [level, restrict /
                  prolong]). A = inf is the identity (the clip is skipped). A from calib_b1.py -> set_bounds() or
                  model_args bounds=<calib file>; sat_stats(): fraction of |a_raw| > A per group in the last geometry().
  feat_v2=True    (B2) 5 more node features without any per-geometry normalisation (node_feats_v2), appended as the last
                  input columns of node_in; load_compat() zero-pads them in older checkpoints (same function at load).
  fringe_soft=True  (MGNO2, needs feat_v2) the fringe layers act on the fixed superset {hyperedges with max_i s_i > 0.01}
                  and every fringe hyperedge update is scaled by g = max_i s_i (soft weak score) instead of the hard
                  per-geometry-median weak flag.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

dev = torch.device(__import__('os').environ.get('OPL_DEV', 'cuda:0'))   # OPL_DEV=cpu for local unit tests
f32 = torch.float32
NF2 = 5                                                                   # feat_v2 node columns: rho, s, hop, plane, box
_DFULL = {}


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
def full_moments(n):
    """The 125 moments of a FULL element in the teacher's local coordinates (moments_ad.moments_rows, polyref_torch_fast):
    int_[-1,1]^3 xi^a eta^b zeta^c (a, b, c = 0..4, index 25 a + 5 b + c) times the physical measure (1 / 2n)^3."""
    I = [2.0 / (k + 1) if k % 2 == 0 else 0.0 for k in range(5)]
    return np.array([I[a] * I[b] * I[c] for a in range(5) for b in range(5) for c in range(5)]) / (2.0 * n) ** 3


def d_full_from_Tm(Tm, n):
    """D_full[s] = 3x3 diagonal block of slot s of the full-element stiffness K_e = sum_m M_full,m Tm_m -> (27, 3, 3)."""
    Tm = np.asarray(Tm.detach().cpu() if torch.is_tensor(Tm) else Tm, dtype=np.float64).reshape(125, 81, 81)
    Ke = np.einsum('m,mij->ij', full_moments(n), Tm)
    return np.stack([Ke[3 * s:3 * s + 3, 3 * s:3 * s + 3] for s in range(27)])


def d_full_table(path=None):
    """The shipped constant D_FULL.npz (next to this file): D (27, 3, 3), n, offsets (27, 3) = local node-grid offset of
    every slot (grid[elem_nodes[e, s]] - 2 elem_cells[e]); written by t_models_v2.py from the teacher's Tm."""
    p = Path(path) if path else Path(__file__).resolve().with_name('D_FULL.npz')
    if p not in _DFULL:
        z = np.load(p)
        _DFULL[p] = {k: z[k] for k in z.files}
    return _DFULL[p]


def node_feats_v2(nd, device=None):
    """B2 node features (fp64, one value per node), no per-geometry normalisation, invariant under the cube group for a
    correctly rotated nd (grid, elem_nodes slots, diag3 -> R D R^T, normal -> R n, offset -> offset + (sum n' - sum n) / 2):
      rho   = log(d3 / d_full), d3 = ||diag3_i||_F, d_full = || sum_{active e containing i} D_full[slot_e(i)] ||_F
      s     = sigmoid(-(log10(d3 / d_full) + 2) / 0.15): soft weak score, -> 1 below 1% of the full-solid diagonal
      hop   = element-graph hops to the nearest port node (BFS, cap 6)
      plane = signed distance to the cut plane in element sizes, material side (n.x <= offset, x in [0, 1]^3) positive,
              clamped to [-6, 6]; no plane (FULL cell): 6
      box   = distance to the nearest face of the cell box [0, 1]^3 in element sizes, capped at 6"""
    device = dev if device is None else device
    T = d_full_table()
    n = int(np.asarray(nd['n']))
    if int(T['n']) != n:
        raise ValueError(f'D_FULL is for n = {int(T["n"])}, geometry has n = {n}')
    en = torch.as_tensor(nd['elem_nodes'].astype(np.int64), device=device)
    grid = torch.as_tensor(nd['grid'].astype(np.int64), device=device)
    cells = torch.as_tensor(nd['elem_cells'].astype(np.int64), device=device)
    if not bool((grid[en] - 2 * cells[:, None, :] == torch.as_tensor(T['offsets'], device=device)[None]).all()):
        raise ValueError('SLOT_ORDER: elem_nodes slots do not follow the D_FULL local node order')
    N, E = grid.shape[0], en.shape[0]
    D = torch.as_tensor(T['D'], dtype=torch.float64, device=device)
    Dn = torch.zeros((N, 3, 3), dtype=torch.float64, device=device)
    for s_ in range(27):
        Dn.index_add_(0, en[:, s_], D[s_].expand(E, 3, 3))
    d3 = torch.as_tensor(nd['diag3'], dtype=torch.float64, device=device).reshape(-1, 9).norm(dim=1)
    ratio = d3.clamp_min(1e-300) / Dn.reshape(N, 9).norm(dim=1)
    port = torch.as_tensor(nd['is_port'], device=device)
    hop = torch.full((N,), 6.0, dtype=torch.float64, device=device)
    hop[port] = 0
    reached = port.clone()
    for k in range(1, 6):
        nb = torch.zeros(N, dtype=torch.bool, device=device)
        nb[en[reached[en].any(1)].reshape(-1)] = True
        nb &= ~reached
        hop[nb] = k
        reached |= nb
    nrm = torch.as_tensor(np.asarray(nd['normal'], dtype=np.float64), device=device)
    x = grid.to(torch.float64) / (2 * n)
    if float(nrm.norm()) == 0:
        plane = torch.full((N,), 6.0, dtype=torch.float64, device=device)
    else:
        plane = ((float(np.asarray(nd['offset'])) - x @ nrm) / nrm.norm() * n).clamp(-6, 6)
    g = grid.to(torch.float64)
    box = (torch.minimum(g, 2 * n - g).min(1).values / 2).clamp_max(6)
    return dict(rho=torch.log(ratio), s=torch.sigmoid(-(torch.log10(ratio) + 2) / 0.15), hop=hop, plane=plane, box=box)


class MGCache:
    def __init__(self, geo, levels, feat_v2=False):
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
        if feat_v2:
            fv = node_feats_v2(nd)
            c.s_weak = fv['s'].to(f32)
            c.nfeat2 = torch.stack([fv['rho'] / 5, fv['s'], fv['hop'] / 6, fv['plane'] / 6, fv['box'] / 6], 1).to(f32)   # N x NF2
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


class _SoftClip(torch.autograd.Function):
    """y = A tanh(x / A) where A is finite, y = x where A = inf (A broadcasts against x; mixed=False: A all finite).
    Saves only y (kept for the q path anyway): dy/dx = 1 - (y / A)^2."""

    @staticmethod
    def forward(ctx, x, A, mixed):
        if mixed:
            fin = torch.isfinite(A)
            As = torch.where(fin, A, torch.ones_like(A))
            y = torch.where(fin, (x / As).tanh_().mul_(As), x)
        else:
            fin, As = None, A
            y = (x / A).tanh_().mul_(A)
        ctx.save_for_backward(y, As)
        ctx.fin = fin
        return y

    @staticmethod
    def backward(ctx, g):
        y, As = ctx.saved_tensors
        d = torch.ops.aten.tanh_backward(g, (y / As).clamp_(-1, 1))                  # g (1 - (y / A)^2), one fused pass
        return (d if ctx.fin is None else torch.where(ctx.fin, d, g)), None, None


def _bounds_hook(module, incompatible):
    module._refresh_bounds()


def load_compat(model, sd):
    """Load a state dict into a (possibly v2) model: input layers widened by feat_v2 (model._pad_cols) get zero columns
    appended, so an older checkpoint gives the same function at load; B1 bound buffers missing from sd keep the model's
    values (inf = identity, or what set_bounds / bounds= put there). Strict otherwise: unexpected or missing keys and any
    other shape mismatch raise. Returns dict(padded=[...], kept=[...])."""
    own = model.state_dict()
    pad = getattr(model, '_pad_cols', {})
    extra = [k for k in sd if k not in own]
    if extra:
        raise KeyError(f'load_compat: unexpected keys {extra[:8]}')
    out, padded, kept = {}, [], []
    for k, v in own.items():
        if k not in sd:
            if k.startswith('bnd_'):
                out[k] = v; kept.append(k); continue
            raise KeyError(f'load_compat: missing key {k}')
        w = sd[k]
        if tuple(w.shape) == tuple(v.shape):
            out[k] = w
        elif k in pad and w.dim() == 2 and w.shape[0] == v.shape[0] and w.shape[1] + pad[k] == v.shape[1]:
            out[k] = torch.cat([w, torch.zeros((w.shape[0], pad[k]), dtype=w.dtype, device=w.device)], 1); padded.append(k)
        else:
            raise ValueError(f'load_compat: shape mismatch {k}: checkpoint {tuple(w.shape)}, model {tuple(v.shape)}')
    model.load_state_dict(out, strict=True)
    return dict(padded=padded, kept=kept)


class MGNO(nn.Module):
    def __init__(self, geos, F=32, H=4, L_pre=4, L_post=4, levels=3, Cg=64, conv_per_level=2, slot_dim=8, ckpt=False, sparse=False,
                 bounded=False, feat_v2=False, bounds=None):
        super().__init__()
        self.F, self.H, self.L_pre, self.L_post, self.levels, self.cpl = F, H, L_pre, L_post, levels, conv_per_level
        self.ckpt = ckpt and not sparse
        self.sparse = sparse                                                        # training-time sparse hyperedge layers
        self.bounded, self.feat_v2 = bounded, feat_v2
        self.caches = {g.case: MGCache(g, levels, feat_v2).c for g in geos}
        self.elem_in = _mlp(126, Cg, Cg)
        self.node_in = _mlp(11 + Cg + (NF2 if feat_v2 else 0), Cg, Cg)            # input [nfeat, agg (, nfeat2)]
        self._pad_cols = {'node_in.0.weight': NF2} if feat_v2 else {}
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
        self._rec, self._sat, self._bst = None, {}, {}                            # B1: raw capture (calib), saturation, state
        self.track_sat = True                                                      # B1: saturation counts in geometry()
        if bounded:
            self.register_buffer('bnd_ab', torch.full((2, L), float('inf')))
            self.register_buffer('bnd_rw', torch.full((levels, 2), float('inf')))
            self.register_load_state_dict_post_hook(_bounds_hook)
            self._refresh_bounds()
            if bounds is not None:
                self.set_bounds(bounds)

    def add_geo(self, geo):
        """Per-geometry cache for a geometry met after construction (step-2 pool training)."""
        if geo.case not in self.caches:
            self.caches[geo.case] = MGCache(geo, self.levels, self.feat_v2).c

    def drop_geo(self, case):
        self.caches.pop(case, None)

    # ---------------- B1 bounds
    def _refresh_bounds(self):
        """Python-side state of the bound buffers (no device sync in the forward): per tensor 'off' (all inf), 'all'
        (all finite) or 'mixed'; per (level, direction) transfer bound finite or not."""
        self._bst = {}
        for k in ('ab', 'fab', 'xab', 'rw'):
            b = getattr(self, 'bnd_' + k, None)
            if b is not None:
                fin = torch.isfinite(b)
                self._bst[k] = 'off' if not bool(fin.any()) else ('all' if bool(fin.all()) else 'mixed')
                if k == 'rw':
                    self._rw_fin = fin.tolist()

    def set_bounds(self, d):
        """B1 bounds {'ab': (2, L), 'fab': (2, Lg), 'xab': (2, Lx), 'rw': (levels, 2)} ([alpha/beta, layer] and [level,
        restrict/prolong]; inf = identity); a calib_b1.py output (path or its dict) works too. Absent names are kept."""
        if isinstance(d, (str, Path)):
            d = torch.load(d, map_location='cpu', weights_only=False)
        d = d.get('bounds', d)
        for k, v in d.items():
            b = getattr(self, 'bnd_' + k, None)
            if b is None:
                raise KeyError(f'set_bounds: no bound buffer for {k!r} (model built with bounded=True?)')
            with torch.no_grad():
                b.copy_(torch.as_tensor(v, dtype=b.dtype).reshape(b.shape))
        self._refresh_bounds()

    def sat_stats(self):
        """Fraction of |a_raw| > A per group in the last geometry() call ({name: nested list shaped like its bound, 'max':
        overall max}); groups with A = inf count 0 (also all groups while model.track_sat = False, which skips the count).
        A cheap out-of-distribution score for logging."""
        out = {}
        for k in ('ab', 'fab', 'xab', 'rw'):
            b = getattr(self, 'bnd_' + k, None)
            if b is not None:
                out[k] = self._sat[k].float().cpu().tolist() if k in self._sat else torch.zeros(b.shape).tolist()
        out['max'] = max([float(np.max(v)) for v in out.values()] or [0.0])
        return out

    def _clip(self, name, x, shape):
        """a = A tanh(a_raw / A) per group for one coefficient tensor (Eh x 27 x 2 x L x H); bound viewed as shape."""
        if self._rec is not None:
            self._rec[name] = x.detach()
        st = self._bst[name]
        if st == 'off':
            return x
        A = getattr(self, 'bnd_' + name).view(shape)
        if self.track_sat:
            with torch.no_grad():
                G, H = x.shape[2] * x.shape[3], x.shape[4]
                cnt = torch.count_nonzero((x.abs() > A).reshape(-1, G * H), dim=0).reshape(G, H).sum(1)
                self._sat[name] = (cnt / float(x.numel() // G)).reshape(x.shape[2], x.shape[3])
        return _SoftClip.apply(x, A, st == 'mixed')

    def _transfer(self, rw_raw, l):
        """Restriction / prolongation weights of level l: softplus(r) + 1e-3, or w_max tanh(softplus(r) / w_max) + 1e-3."""
        sp = [torch.nn.functional.softplus(rw_raw[:, j]) for j in range(2)]
        if self._rec is not None:
            self._rec.setdefault('rw', [None] * self.levels)[l] = tuple(s_.detach() for s_ in sp)
        out = []
        for j in range(2):
            if self._rw_fin[l][j]:
                A = self.bnd_rw[l, j]
                if self.track_sat:
                    with torch.no_grad():
                        if 'rw' not in self._sat:
                            self._sat['rw'] = torch.zeros_like(self.bnd_rw)
                        self._sat['rw'][l, j] = (sp[j] > A).float().mean()
                out.append(_SoftClip.apply(sp[j], A, False) + 1e-3)
            else:
                out.append(sp[j] + 1e-3)
        return tuple(out)

    # ---------------- geometry path
    def _node_in(self, c, agg):
        if not self.feat_v2:
            return self.node_in(torch.cat([c.nfeat, agg], 1))
        lin = self.node_in[0]                                                    # the old columns and the new ones as two
        k = lin.in_features - NF2                                                # products: zero new columns change nothing
        h = torch.nn.functional.linear(torch.cat([c.nfeat, agg], 1), lin.weight[:, :k].contiguous(), lin.bias) + \
            torch.nn.functional.linear(c.nfeat2, lin.weight[:, k:].contiguous())
        return self.node_in[1:](h)

    def geometry(self, case):
        c = self.caches[case]
        self._sat = {}
        ge = self.elem_in(c.efeat)                                                         # E x Cg
        agg = torch.zeros((c.N, ge.shape[1]), device=dev).index_add_(0, c.en.reshape(-1), ge.repeat_interleave(27, 0)) / c.deg[:, None]
        gn = self._node_in(c, agg)
        for fe, fn in zip(self.mp_e, self.mp_n):
            ge = ge + fe(torch.cat([ge, gn[c.en].mean(1)], 1))
            agg = torch.zeros_like(gn).index_add_(0, c.en.reshape(-1), ge.repeat_interleave(27, 0)) / c.deg[:, None]
            gn = gn + fn(torch.cat([gn, agg], 1))
        E = c.en.shape[0]
        z = torch.cat([ge[:, None, :].expand(E, 27, -1), gn[c.en], self.slot[None].expand(E, -1, -1)], 2)
        ab = self.slot_head(z).reshape(E, 27, 2, self.L_pre + self.L_post, self.H) * 0.2   # small start
        if self.bounded:
            ab = self._clip('ab', ab, (1, 1, 2, self.L_pre + self.L_post, 1))
        rw, gates, gl = [], [], gn
        for l, t in enumerate(c.trans):
            r = self.restrict_head[l](gl)
            if self.bounded:
                rw.append(self._transfer(r, l))
            else:
                rw.append((torch.nn.functional.softplus(r[:, 0]) + 1e-3, torch.nn.functional.softplus(r[:, 1]) + 1e-3))
            gsum = torch.zeros((t['n_dst'], gl.shape[1]), device=dev).index_add_(0, t['v'], gl[t['i']] * t['w'][:, None])
            wsum = torch.zeros(t['n_dst'], device=dev).index_add_(0, t['v'], t['w'])
            gl = gsum / wsum[:, None]
            gates.append(2 * torch.sigmoid(self.gate_head[l](gl)).reshape(-1, self.cpl * 2, self.F))
        return dict(ab=ab, rw=rw, gates=gates, ge=ge, gn=gn)

    # ---------------- q path
    def _pat(self, c, key, hn):
        """Sparse pattern of a hyperedge set, cached on the geometry cache (moves with it)."""
        import sparse_layers as SL
        if not hasattr(c, 'pats'):
            c.pats = {}
        if key not in c.pats:
            c.pats[key] = SL.pattern(hn, self.H, c.N)
        return c.pats[key]

    def _fine(self, X, X0, pm, c, ab, layer):
        a, b = ab[:, :, 0, layer], ab[:, :, 1, layer]                                    # E x 27 x H
        if self.sparse:
            import sparse_layers as SL
            dX = SL.hyper(X, a, b, self.W[layer], c.deg, self._pat(c, 'elem', c.en))
            return (X + dX) * (1 - pm) + X0 * pm
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
    def __init__(self, geos, F=32, H=4, L_pre=4, L_post=4, levels=3, Cg=64, conv_per_level=2, slot_dim=8, n_fringe=4, sparse=False,
                 bounded=False, feat_v2=False, fringe_soft=False, bounds=None):
        if fringe_soft and not feat_v2:
            raise ValueError('fringe_soft needs feat_v2 (the soft weak score s)')
        super().__init__(geos, F=F, H=H, L_pre=L_pre, L_post=L_post, levels=levels, Cg=Cg,
                         conv_per_level=conv_per_level, slot_dim=slot_dim, sparse=sparse, bounded=bounded, feat_v2=feat_v2)
        self.n_fringe = n_fringe
        self.fringe_soft = fringe_soft
        self.face_in = _mlp(2 * Cg + 3, Cg, Cg)
        self.fslot = nn.Parameter(0.1 * torch.randn(27, slot_dim))
        Lg = L_pre + L_post + n_fringe                                                    # ghost layers
        Le_extra = n_fringe                                                               # extra element layers
        self.fslot_head = _mlp(2 * Cg + slot_dim, Cg, 2 * H * Lg)
        self.Wf = nn.Parameter(torch.randn(Lg, H, F, F) / np.sqrt(F) * 0.5)
        self.slot_head_x = _mlp(2 * Cg + slot_dim, Cg, 2 * H * max(Le_extra, 1))
        self.Wx = nn.Parameter(torch.randn(max(Le_extra, 1), H, F, F) / np.sqrt(F) * 0.5)
        if bounded:
            self.register_buffer('bnd_fab', torch.full((2, Lg), float('inf')))
            self.register_buffer('bnd_xab', torch.full((2, max(Le_extra, 1)), float('inf')))
            self._refresh_bounds()
            if bounds is not None:
                self.set_bounds(bounds)
        for g in geos:
            self._face_setup(g)

    def _face_setup(self, g):
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
        if self.fringe_soft:                                  # fixed superset max_i s_i > 0.01, update scaled by g = max_i s_i
            ge_, gf_ = c.s_weak[c.en].max(1).values, c.s_weak[c.fn].max(1).values
            c.el_fringe, c.gp_fringe = ge_ > 0.01, gf_ > 0.01
            c.el_fade, c.gp_fade = ge_[c.el_fringe], gf_[c.gp_fringe]

    def add_geo(self, geo):
        if geo.case not in self.caches:
            super().add_geo(geo)
            self._face_setup(geo)

    def geometry(self, case):
        out = super().geometry(case)
        c = self.caches[case]
        ge, gn = out['ge'], out['gn']                                            # the base class's embeddings (same values)
        nf = len(c.gp_owner)
        ax = torch.nn.functional.one_hot(c.gp_axis, 3).to(f32)
        gf = self.face_in(torch.cat([ge[c.gp_owner], ge[c.gp_nbr], ax], 1))
        z = torch.cat([gf[:, None, :].expand(nf, 27, -1), gn[c.fn], self.fslot[None].expand(nf, -1, -1)], 2)
        Lg = self.L_pre + self.L_post + self.n_fringe
        out['fab'] = self.fslot_head(z).reshape(nf, 27, 2, Lg, self.H) * 0.2
        E = c.en.shape[0]
        zx = torch.cat([ge[:, None, :].expand(E, 27, -1), gn[c.en], self.slot[None].expand(E, -1, -1)], 2)
        out['xab'] = self.slot_head_x(zx).reshape(E, 27, 2, max(self.n_fringe, 1), self.H) * 0.2
        if self.bounded:
            out['fab'] = self._clip('fab', out['fab'], (1, 1, 2, Lg, 1))
            out['xab'] = self._clip('xab', out['xab'], (1, 1, 2, max(self.n_fringe, 1), 1))
        return out

    def _ck(self, fn, *args):
        if torch.is_grad_enabled() and not self.sparse:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _hyper(self, X, X0, pm, hn, deg, ab, W, layer, mask=None, fade=None):
        a, b = ab[:, :, 0, layer], ab[:, :, 1, layer]
        if mask is not None:
            hn, a, b = hn[mask], a[mask], b[mask]
        if fade is not None:                                                   # per-hyperedge scale of the update
            b = b * fade[:, None, None]
        if self.sparse:
            import sparse_layers as SL
            c = self._cur
            key = (id(deg) == id(c.deg), mask is not None)
            dX = SL.hyper(X, a, b, W, deg, self._pat(c, key, hn))
            return (X + dX) * (1 - pm) + X0 * pm
        Xg = X[hn]
        Z = torch.einsum('eah,eabf->ehbf', a, Xg)
        Z = torch.einsum('ehbf,hfg->ehbg', Z, W)
        Y = torch.einsum('eah,ehbg->eabg', b, Z)
        dX = torch.zeros_like(X).index_add_(0, hn.reshape(-1), Y.reshape(-1, *Y.shape[2:]))
        X = X + dX / deg[:, None, None]
        return X * (1 - pm) + X0 * pm

    def forward(self, geo, qd):
        c = self.caches[geo.case]
        self._cur = c
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
        fe, ff = (c.el_fade, c.gp_fade) if self.fringe_soft else (None, None)
        for l in range(self.n_fringe):
            X = self._ck(lambda X_, l_=l: self._hyper(X_, X0, pm, c.en, c.deg, gp['xab'], self.Wx[l_], l_, mask=c.el_fringe, fade=fe), X)
            X = self._ck(lambda X_, g_=gl: self._hyper(X_, X0, pm, c.fn, c.fdeg, gp['fab'], self.Wf[g_], g_, mask=c.gp_fringe, fade=ff), X); gl += 1
        u = X @ self.W_out
        return u.permute(0, 2, 1).reshape(-1, B)


REGISTRY = {'diffusion': Diffusion, 'graphlift': GraphLiftModel, 'mgno': MGNO, 'mgno2': MGNO2}


def build(name, geos, **kw):
    return REGISTRY[name](geos, **kw)
