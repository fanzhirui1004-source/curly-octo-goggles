"""B6: O_h (48-element cube group) data-side augmentation. A view of a trainlib.Geo shows the network the geometry
rotated by R (x' = .5 + R (x - .5); node grid g' = R (g - n) + n; make_rot.rotate / v_rot.node_map, cell_img) while
everything exact stays in the ORIGINAL frame: node order, port DOF order, q layout, K, banks, sensitivities (dM), rigid
split. Only what the network reads is transformed (rotate_nd), exactly (signed permutations, no rounding):
  grid, node_ids      g' = R (g - n) + n
  elem_cells          c' = (R (2 c + 1 - n) + n - 1) / 2                     (cell centres map to cell centres)
  elem_nodes          view slot of offset o' holds the original slot of offset o = R^T (o' - 1) + 1
  moments             local xi' = R xi (centred, [-1, 1]^3, index 25 a + 5 b + c):
                      M'[a'] = prod_i s_i^a'_i M[b], b_{p_i} = a'_i   with R[i, p_i] = s_i
  gp_faces            (owner, nbr, axis a) -> axis b with R e_a = s e_b; s < 0: owner and nbr swap (owner = lower cell)
  diag3               R D R^T;  normal R n;  offset + (sum n' - sum n) / 2;  taus: corners permuted (make_rot.rotate)
  flags, n            unchanged (node-wise, same node order)
Field of the view:  c = RP^+ q,  qd = q - RP c (original frame),  u = (I x R^T) model(view, (I x R) qd) + RA c,
u_P = q. Linear in q; since the rotated geometry has K' = Pi K Pi^T (Pi = node permutation x R, V01), the view's
energies equal the rotated geometry's network energies; labels (energies with the original K, sensitivities with the
original dM) need nothing new.
API: ELEMS (48 int 3x3, ELEMS[0] = I), index_of(R), rotate_nd(nd, R) -> dict, view(model, geo, k) -> view,
drop(model, view); helpers node_map, cell_img, rot_dofs (the maps a rotated packet uses).
A view is copy.copy(geo): it shares the base's tensors (C, banks, sens, adv, RP, RA, P, ...) as they are at creation; make
it while the base is resident on its device and drop it before the base is moved or dropped. Views are not pickled (the
bound field does not survive it and unpickling raises); rebuild them from the base."""
import copy, types, itertools
import numpy as np
import torch

f32 = torch.float32
ELEMS = []
for _p in itertools.permutations(range(3)):
    for _s in itertools.product((1, -1), repeat=3):
        _R = np.zeros((3, 3), dtype=np.int64); _R[np.arange(3), _p] = _s
        ELEMS.append(_R)
CORNERS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)     # tau corner k = 4 c0 + 2 c1 + c2
EXP = np.array(list(itertools.product(range(5), repeat=3)), dtype=np.int64)       # moment 25 a + 5 b + c -> (a, b, c)
KEYS = frozenset(('node_ids', 'grid', 'elem_cells', 'elem_nodes', 'moments', 'gp_faces', 'is_box', 'is_cut', 'is_port',
                  'weak', 'diag3', 'taus', 'normal', 'offset', 'n'))


def index_of(R):
    R = np.asarray(R)
    for k, E in enumerate(ELEMS):
        if R.shape == (3, 3) and np.array_equal(E, R):
            return k
    raise ValueError(f'index_of: not a signed 3x3 permutation matrix: {R.tolist()}')


def perm_sign(R):
    """(p, s) with R[i, p_i] = s_i, i.e. (R x)_i = s_i x_{p_i}."""
    R = ELEMS[index_of(R)]
    p = np.abs(R).argmax(1)
    return p, R[np.arange(3), p]


def node_map(ids, R, n=32):
    """Grid node ids -> ids of their images g' = R (g - n) + n (v_rot.node_map)."""
    g = np.stack(np.unravel_index(np.asarray(ids), (2 * n + 1,) * 3), 1).astype(np.int64)
    return np.ravel_multi_index(((g - n) @ np.asarray(R).T + n).T, (2 * n + 1,) * 3)


def cell_img(cells, R, n=32):
    """Cell indices (E, 3) -> image cells (v_rot.cell_img)."""
    return ((2 * np.asarray(cells, dtype=np.int64) + 1 - n) @ np.asarray(R).T + n - 1) // 2


def slot_perm(R, offs):
    """View slot t holds original slot slot_perm[t]: offs[t] = R (offs[slot_perm[t]] - 1) + 1."""
    img = (np.asarray(offs) - 1) @ np.asarray(R).T + 1
    return np.array([int(np.flatnonzero((img == offs[t]).all(1))[0]) for t in range(len(offs))])


def moment_map(R):
    """M'[m] = sgn[m] M[src[m]] for local coordinates xi' = R xi (exponents b_{p_i} = a'_i, sign prod_i s_i^a'_i)."""
    p, s = perm_sign(R)
    b = np.zeros_like(EXP); b[:, p] = EXP
    return b @ np.array([25, 5, 1]), np.prod(s[None, :] ** EXP, 1)


def corner_perm(R):
    """tau' [moved[k]] = tau[k] (make_rot.rotate)."""
    return ((((2 * CORNERS - 1) @ np.asarray(R).T) + 1) // 2) @ np.array([4, 2, 1])


def rotate_nd(nd, R):
    """NETDATA of the geometry rotated by R, in the ORIGINAL node and element order (see the module docstring).
    Unknown keys raise (a new network input needs its own rule here)."""
    extra = set(nd) - KEYS
    if extra:
        raise KeyError(f'rotate_nd: no rotation rule for {sorted(extra)}')
    R = ELEMS[index_of(R)]
    p, s = perm_sign(R)
    n = int(np.asarray(nd['n']))
    g = nd['grid'].astype(np.int64); cells = nd['elem_cells'].astype(np.int64); en = nd['elem_nodes']
    offs = g[en[0]] - 2 * cells[0]
    if not (g[en] - 2 * cells[:, None, :] == offs[None]).all():
        raise ValueError('SLOT_ORDER: elem_nodes slots are not one fixed local node order')
    out = dict(nd)
    g2 = (g - n) @ R.T + n
    out['grid'] = g2.astype(nd['grid'].dtype)
    out['node_ids'] = np.ravel_multi_index(g2.T, (2 * n + 1,) * 3).astype(np.asarray(nd['node_ids']).dtype)
    c2 = ((2 * cells + 1 - n) @ R.T + n - 1) // 2
    out['elem_cells'] = c2.astype(nd['elem_cells'].dtype)
    out['elem_nodes'] = np.ascontiguousarray(en[:, slot_perm(R, offs)])
    if not (g2[out['elem_nodes']] - 2 * c2[:, None, :] == offs[None]).all():
        raise AssertionError('SLOT_ORDER after rotation')
    src, sgn = moment_map(R)
    out['moments'] = nd['moments'][:, src] * sgn[None, :]
    f = np.asarray(nd['gp_faces']); b = np.argsort(p)[f[:, 2]]; flip = s[b] < 0
    f2 = f.copy(); f2[:, 2] = b
    f2[flip, 0], f2[flip, 1] = f[flip, 1], f[flip, 0]
    out['gp_faces'] = f2
    out['diag3'] = nd['diag3'][:, p][:, :, p] * (s[:, None] * s[None, :])[None]
    nrm = np.asarray(nd['normal'])
    n2 = nrm[p] * s
    out['normal'] = n2
    out['offset'] = np.asarray(float(np.asarray(nd['offset'])) + (float(n2.sum()) - float(nrm.sum())) / 2,
                               dtype=np.asarray(nd['offset']).dtype)
    t2 = np.empty_like(np.asarray(nd['taus'])); t2[corner_perm(R)] = np.asarray(nd['taus'])
    out['taus'] = t2
    return out


def rot_dofs(x, p, s):
    """(I x R) x for node-major xyz DOF blocks x (3 m, B): (R x)_i = s_i x_{p_i} per node (exact, differentiable)."""
    X = x.reshape(-1, 3, x.shape[1])
    return torch.stack([X[:, int(p[i])] * float(s[i]) for i in range(3)], 1).reshape(x.shape)


def _field(self, model, q):
    """trainlib.Geo.field through the rotated network: u = (I x R^T) model(view, (I x R) qd) + RA c, u_P = q."""
    q32 = q.to(f32)
    c = (self.RPpinv.to(f32) @ q32)
    qd = q32 - self.RP.to(f32) @ c
    u = rot_dofs(model(self, rot_dofs(qd, *self.oh_ps)), *self.oh_ps_t)
    u = u + self.RA.to(f32) @ c
    u = u.index_copy(0, self.P, q32)                                  # exact port values
    import trainlib as TL
    return TL.wrap(self.C, u, model)                                  # physics wrapper on the physical field (as Geo.field)


def view(model, geo, k):
    """View k (ELEMS[k]; 0 = identity) of a trainlib.Geo: case '<case>@oh<k>', nd rotated, field through the rotated
    network (so adversarial / evaluate / s_hat_apply use it too); its cache is registered with model.add_geo (models
    with per-geometry caches: MGNO / MGNO2; model=None registers nothing)."""
    if model is not None and not hasattr(model, 'add_geo'):
        raise TypeError(f'view: {type(model).__name__} has no add_geo (per-geometry caches); views need MGNO / MGNO2')
    if hasattr(geo, 'oh_R'):
        raise ValueError('view: build views from the base geometry, not from a view')
    k = int(k) if isinstance(k, (int, np.integer)) else index_of(k)
    R = ELEMS[k]
    p, s = perm_sign(R)
    pt = np.argsort(p)                                                # R^T: (R^T y)_j = s_{pt_j} y_{pt_j}
    v = copy.copy(geo)
    v.case, v.nd, v.oh_R, v.oh_k, v.oh_base = f'{geo.case}@oh{k}', rotate_nd(geo.nd, R), R, k, geo.case
    v.oh_ps, v.oh_ps_t = (tuple(p.tolist()), tuple(s.tolist())), (tuple(pt.tolist()), tuple(s[pt].tolist()))
    v.field = types.MethodType(_field, v)
    if model is not None:
        model.add_geo(v)
    return v


def drop(model, view):
    """Remove the view's model cache (the view object itself holds only shared references and its rotated nd)."""
    caches = getattr(model, 'caches', None)
    c = caches.pop(view.case, None) if caches is not None else None
    if c is not None and getattr(model, '_cur', None) is c:
        model._cur = None
