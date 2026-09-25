"""Step-2 data v2 (review C1): lattice-context q classes with traction-CONSISTENT loads, exact port reactions F, and the
fp64 fix of the NaN support banks. prep_geo.py is unchanged; this writes new files next to its output.

Classes (every direction rigid-free and at unit exact energy, exact 8-corner sensitivities, reactions F = S q):
  force_c    multiscale random smooth tractions on every box face with material (10% also on the cut surface), integrated
             consistently over the MATERIAL part of the face, f_i = sum_qp h^2 t(x_qp) N_i(x_qp) (lattice3 quadrature), so
             fictitious-fringe nodes get almost nothing (prep_data.to_ports puts the sampled field on every port node);
             Neumann solve (load projected onto the equilibrated subspace)
  face_c     one box face at a time (cycled): consistent traction on that face, rigid part removed on the face, other
             ports free; Neumann solve
  support_k  one box face on springs k_i = alpha diag(K)_ii on its port DOFs (alpha log-uniform in [0.3, 3], one per face),
             consistent tractions on the other box faces, cut band free; fp64 spring factor, non-finite solutions raise
  glued      the cell glued at full-DOF level (nodes at the same absolute position share DOFs) to its family FULL parent
             across one of its box faces that carries material; neighbour offsets +x, +y, +z, -z only (the acceptance gate
             uses -x and -y, lattice3 configs x / y: never those); the neighbour's far face clamped or on springs
             k = alpha diag (alternating over offsets); consistent tractions on the free box faces of both cells, 25% of the
             samples load ONLY the neighbour (the test cell is dragged almost rigidly); q = the test cell's port values
  --F-old        reactions F for the existing classes force, macro, grf, support, face
  --fix-support  recompute the old 'support' class (same definition as prep_geo) with fp64 spring factors where its bank
                 has non-finite values; the old files are kept as *.nan_bak.npy
Files per class: {split}_{cls}.npy (m x np fp32), {split}_{cls}_sens.npy (m x 8), {split}_{cls}_F.npy (m x np fp32);
DONE2.json (per-class stats, timings). Bank sizes from BANK_SPLITS like prep_geo (e.g. "512,64,64").
Usage: prep_geo2.py <body_dir> <data_dir> [--classes force_c,face_c,support_k,glued] [--F-old] [--fix-support] <case ...>
"""
import sys, json, time, gc, re, os
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import prep_data as PD
import prep_geo as PG

dev, dt = TE.dev, TE.dt
SPLITS = PD.SPLITS
NEW = ('force_c', 'face_c', 'support_k', 'glued')
OLD = ('force', 'macro', 'grf', 'support', 'face')
GLUE_OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1))                  # never (-1,0,0) / (0,-1,0): the gate's


# --------------------------------------------------------------------------------------------------------- quadrature
def _material(C, P):
    """Quadrature points inside the material: |phi| <= tau (trilinear corner thickness) and in the retained half-space."""
    from element_polyref import CUBE
    f = np.cos(2 * np.pi * P).sum(1)
    cube = np.asarray(CUBE, float)
    w8 = np.where(cube[:, None, :].astype(bool), P[None], 1 - P[None]).prod(-1)
    keep = np.abs(f) <= np.asarray(C.taus, float) @ w8
    if C.normal is not None and C.is_cut.any():
        keep &= (P @ np.asarray(C.normal, float)) <= C.offset
    return P[keep]


def _quad_nodes(C, P):
    """Q2 shape values N (m x 27) and node indices into C.nodes (m x 27) of the active element containing each point."""
    n = C.n
    c = np.minimum(np.floor(P * n).astype(np.int64), n - 1)
    key = (c[:, 0] * n + c[:, 1]) * n + c[:, 2]
    ok = np.isin(key, (C.cells[:, 0] * n + C.cells[:, 1]) * n + C.cells[:, 2])
    P, c = P[ok], c[ok]
    xi = P * n - c
    L = np.stack([2 * (xi - .5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - .5)], -1)       # m x 3(axis) x 3(o)
    M = 2 * n + 1
    idx, N = [], []
    for o in np.ndindex(3, 3, 3):
        g = 2 * c + np.asarray(o)
        ids = (g[:, 0] * M + g[:, 1]) * M + g[:, 2]
        loc = np.searchsorted(C.nodes, ids)
        if not (C.nodes[np.minimum(loc, len(C.nodes) - 1)] == ids).all():
            raise ValueError('QUAD_NODE_MISSING')
        idx.append(loc); N.append(L[:, 0, o[0]] * L[:, 1, o[1]] * L[:, 2, o[2]])
    return P, np.stack(idx, 1), np.stack(N, 1)


class Traction:
    """Consistent nodal forces of a traction field sampled at quadrature points: f (port DOFs, node-major xyz) = A t,
    A[(port node p), qp] = h^2 N_p(x_qp). Built for a box face (axis, value in {0, 1}) or for the cut surface."""

    def __init__(self, C, axis=None, value=None, cut=False, pts=6):
        n = C.n
        h = 1.0 / (n * pts)
        if cut:
            nrm = np.asarray(C.normal, float); nrm = nrm / np.linalg.norm(nrm)
            a = np.eye(3)[np.argmin(np.abs(nrm))]
            t1 = np.cross(nrm, a); t1 /= np.linalg.norm(t1); t2 = np.cross(nrm, t1)
            s = np.arange(-np.sqrt(3.0), np.sqrt(3.0) + h, h) + h / 2
            S1, S2 = np.meshgrid(s, s, indexing='ij')
            x0 = nrm * C.offset / np.linalg.norm(C.normal)                   # on the plane normal . x = offset (normal not unit)
            P = x0[None] + S1.reshape(-1, 1) * t1[None] + S2.reshape(-1, 1) * t2[None]
            P = P[(P >= 0).all(1) & (P <= 1).all(1)]
            P = _material_plane(C, P)
        else:
            s = np.arange(0, 1, h) + h / 2
            A, B = np.meshgrid(s, s, indexing='ij')
            P = np.zeros((A.size, 3)); o = [d for d in range(3) if d != axis]
            P[:, axis] = value; P[:, o[0]] = A.reshape(-1); P[:, o[1]] = B.reshape(-1)
            P = _material(C, P)
        P, idx, N = _quad_nodes(C, P)
        nz = N != 0
        q_i, k_i = np.nonzero(nz)
        pos = np.searchsorted(C.port_node_ids, C.nodes[idx[q_i, k_i]])
        pos_ok = C.port_node_ids[np.minimum(pos, len(C.port_node_ids) - 1)] == C.nodes[idx[q_i, k_i]]
        self.dropped = int((~pos_ok).sum())                                  # (face loads: none; cut surface: none expected)
        q_i, k_i, pos = q_i[pos_ok], k_i[pos_ok], pos[pos_ok]
        self.nodes = np.unique(pos)                                          # port-node positions that receive force
        self.m = len(P)
        self.X = torch.as_tensor(P, dtype=dt, device=dev)
        self.A = torch.sparse_coo_tensor(torch.as_tensor(np.stack([pos, q_i]), device=dev),
                                         torch.as_tensor(h * h * N[q_i, k_i], dtype=dt, device=dev),
                                         (len(C.port_node_ids), self.m)).coalesce().to_sparse_csr()
        self.area = h * h * self.m

    def forces(self, T):
        """T (m, 3, k) traction samples at the quadrature points -> (port DOFs, k)."""
        k = T.shape[2]
        return (self.A @ T.reshape(self.m, 3 * k)).reshape(-1, 3, k).reshape(-1, k)

    def random(self, k, gen):
        nw = k - k // 4
        return torch.cat([PD.plane_waves(self.X, nw, 0.5, 8.0, gen), PD.patches(self.X, k - nw, gen)], 2)


def _material_plane(C, P):
    from element_polyref import CUBE
    f = np.cos(2 * np.pi * P).sum(1)
    cube = np.asarray(CUBE, float)
    w8 = np.where(cube[:, None, :].astype(bool), P[None], 1 - P[None]).prod(-1)
    return P[np.abs(f) <= np.asarray(C.taus, float) @ w8]


def box_tractions(C, min_pts=64):
    """Traction operators of the six box faces that carry material: {(axis, side 0/2n): Traction}."""
    out = {}
    for a in range(3):
        for v in (0, 1):
            T = Traction(C, a, v)
            if T.m >= min_pts:
                out[(a, v * 2 * C.n)] = T
    return out


# --------------------------------------------------------------------------------------------------------- classes
def force_c(C, faces, total, gen, cut_frac=0.1, chunk=32):
    cut = Traction(C, cut=True) if (C.normal is not None and C.is_cut.any()) else None
    Qs = []
    for j in range(0, total, chunk):
        k = min(chunk, total - j)
        F = sum(T.forces(T.random(k, gen)) for T in faces.values())
        if cut is not None and cut.m:
            on = (torch.rand(k, device=dev, generator=gen) < cut_frac).to(dt)
            F = F + cut.forces(cut.random(k, gen)) * on[None, :]
        Qs.append(C.neumann(F))
    return torch.cat(Qs, 1), (0 if cut is None else cut.m)


def face_c(C, faces, total, gen, chunk=32):
    keys = list(faces)
    per = int(np.ceil(total / len(keys)))
    Qs = []
    for key in keys:
        T = faces[key]
        nodes = T.nodes
        dofs = torch.as_tensor((3 * nodes[:, None] + np.arange(3)[None]).reshape(-1), device=dev)
        Rf = TE.rigid_basis(C.port_node_ids[nodes], C.n)
        for j in range(0, per, chunk):
            k = min(chunk, per - j)
            F = T.forces(T.random(k, gen))
            ff = F[dofs]
            F = torch.zeros_like(F); F[dofs] = ff - Rf @ (Rf.T @ ff)
            Qs.append(C.neumann(F))
    q = torch.cat(Qs, 1)
    return q[:, torch.randperm(q.shape[1], generator=gen, device=dev)[:total]]


def _spd64(crow, col, vals, n):
    return TE.SPDSolver(crow, col, vals, n)                                  # fp64 (no fp32 attempt: see support_k)


def _finite(x, what):
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f'NONFINITE_{what}')
    return x


def support_k(C, faces, total, gen, chunk=32):
    g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
    keys = list(faces) if len(faces) > 1 else []                            # springs on one face, loads on another
    if not keys:
        return None
    per = int(np.ceil(total / len(keys)))
    ru, cu = C.ru.long(), C.cu.long()
    Qs = []
    for key in keys:
        a, side = key
        m = C.port_is_box & (g[:, a] == side)
        fdofs = C.P[torch.nonzero(torch.as_tensor(m, device=dev).repeat_interleave(3)).squeeze(1)]
        alpha = float(0.3 * 10 ** (torch.rand((), generator=gen, device=dev) * np.log10(10.0)))
        v = C.vals.clone(); v[C.diag[fdofs]] *= 1 + alpha                   # k_i = alpha diag(K)_ii
        s = 1 / torch.sqrt(v[C.diag])
        sol = _spd64(C.crow.int(), C.cu, (v * s[ru] * s[cu]).contiguous(), C.nb)
        others = [T for kk, T in faces.items() if kk != key]
        for j in range(0, per, chunk):
            k = min(chunk, per - j)
            f = sum(T.forces(T.random(k, gen)) for T in others)
            Fb = torch.zeros((C.nb, k), dtype=dt, device=dev); Fb[C.P] = f
            u = _finite(s[:, None] * sol.solve(s[:, None] * Fb), 'SUPPORT_K')
            Qs.append(u[C.P])
        sol.free(); del sol, v; gc.collect(); torch.cuda.empty_cache()
    q = torch.cat(Qs, 1)
    return q[:, torch.randperm(q.shape[1], generator=gen, device=dev)[:total]]


def family_full(case, packets):
    """The FULL parent of a case's family ('<fresh_split_id>_full'; rotated packets: the rotated FULL if it exists)."""
    m = re.match(r'^(fresh_[a-z]+_\d{4})_(.*)$', case)
    if m is None:
        return None
    fam, rest = m.groups()
    rot = re.search(r'(_rot[a-z0-9]+)$', rest)
    name = f'{fam}_full' + (rot.group(1) if rot else '')
    return name if (Path(packets) / name / 'FRESH_CONTEXT.json').exists() else None


def _mem(tag, **kw):
    """PREP_MEMLOG=1: device memory at a stage boundary (torch allocated / reserved, device free)."""
    if os.environ.get('PREP_MEMLOG'):
        free, tot = torch.cuda.mem_get_info()
        print(json.dumps(dict(event='MEM', tag=tag, alloc_GB=round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                              reserved_GB=round(torch.cuda.memory_reserved() / 2 ** 30, 2), free_GB=round(free / 2 ** 30, 2), **kw)), flush=True)


def _nbytes(v):
    if v.layout == torch.sparse_csr:
        return v.values().numel() * v.values().element_size() + v.col_indices().numel() * v.col_indices().element_size()
    return v.numel() * v.element_size()


def _offload(obj, min_bytes=64 << 20):
    """Move the large device tensors held directly by obj (a teacher.Cell) to the host; returns their names (_restore).
    Only where the data lives changes, never its value."""
    keys = [k for k, v in vars(obj).items() if torch.is_tensor(v) and v.is_cuda and _nbytes(v) >= min_bytes]
    for k in keys:
        setattr(obj, k, getattr(obj, k).to('cpu'))
    gc.collect(); torch.cuda.empty_cache()
    return keys


def _restore(obj, keys):
    for k in keys:
        setattr(obj, k, getattr(obj, k).to(dev))


def _spd_glued(crow, col, vals, n):
    """The two-cell factor: fp64; on a memory failure (two FULL cells: ~8e5 DOFs) free the cache and retry fp64, then fall back
    to an fp32 factor of the (diagonally scaled) system. Only the DIRECTIONS q come from this solve: their reactions,
    sensitivities and unit-energy normalisation are computed afterwards with the test cell's own exact factor."""
    for attempt in ('fp64', 'fp64_retry', 'fp32'):
        try:
            return TE.SPDSolver(crow, col, vals, n, **(dict(fdt=torch.float32) if attempt == 'fp32' else {})), attempt
        except Exception as e:
            if not PG._mem_error(e) or attempt == 'fp32':
                raise
            gc.collect(); torch.cuda.empty_cache()


def glued(Ct, body, total, gen, chunk=32, log=print):
    """Two-cell samples (see the module docstring). Returns (q (np_t, total), info) or (None, info)."""
    parent = family_full(Ct.case, TE.ROOT / 'packets')
    if parent is None:
        return None, dict(skipped='no family FULL parent')
    n = Ct.n
    gt = np.stack(np.unravel_index(Ct.nodes, (2 * n + 1,) * 3), 1)
    gp = np.stack(np.unravel_index(Ct.port_node_ids, (2 * n + 1,) * 3), 1)
    offs = [d for d in GLUE_OFFSETS if (Ct.port_is_box & (gp[:, np.flatnonzero(d)[0]] == (2 * n if sum(d) > 0 else 0))).sum() > 8]
    if not offs:
        return None, dict(skipped='no glue face with material', parent=parent)
    Cn = TE.Cell(parent, body, log=lambda s_: None); Cn.assemble()
    gn = np.stack(np.unravel_index(Cn.nodes, (2 * n + 1,) * 3), 1)
    ft, fn_ = box_tractions(Ct), box_tractions(Cn)
    off_t, off_n = _offload(Ct), _offload(Cn)                                # both K on the host during the two-cell factors
    try:
        return _glued_offsets(Ct, Cn, n, gn, ft, fn_, parent, offs, total, gen, chunk, log)
    finally:
        _restore(Ct, off_t)
        Cn._free(); del Cn; gc.collect(); torch.cuda.empty_cache()


def _glued_offsets(Ct, Cn, n, gn, ft, fn_, parent, offs, total, gen, chunk, log):
    gt = np.stack(np.unravel_index(Ct.nodes, (2 * n + 1,) * 3), 1)
    per = int(np.ceil(total / len(offs)))
    W = 6 * n + 1                                                           # absolute key over [-2n, 4n]^3
    kt = ((gt[:, 0] + 2 * n) * W + gt[:, 1] + 2 * n) * W + gt[:, 2] + 2 * n
    ordt = np.argsort(kt)
    Qs, info = [], dict(parent=parent, offsets=[], dofs=[])
    for oi, d in enumerate(offs):
        a = int(np.flatnonzero(d)[0]); sgn = int(sum(d))
        ga = gn + 2 * n * np.asarray(d)[None]
        kn = ((ga[:, 0] + 2 * n) * W + ga[:, 1] + 2 * n) * W + ga[:, 2] + 2 * n
        pos = np.searchsorted(kt[ordt], kn)
        shared = kt[ordt][np.minimum(pos, len(kt) - 1)] == kn
        tnode = ordt[np.minimum(pos, len(kt) - 1)]
        nnode = np.full(len(Cn.nodes), -1, np.int64)
        nnode[shared] = tnode[shared]
        own = np.flatnonzero(~shared)
        nnode[own] = len(Ct.nodes) + np.arange(len(own))
        Ntot = len(Ct.nodes) + len(own)
        dmap = torch.as_tensor((3 * nnode[:, None] + np.arange(3)[None]).reshape(-1), device=dev)
        # glued upper CSR: test entries as they are, neighbour entries renumbered (upper after renumbering), summed
        r = torch.cat([Ct.ru.to(dev).long(), dmap[Cn.ru.to(dev).long()]]); c = torch.cat([Ct.cu.to(dev).long(), dmap[Cn.cu.to(dev).long()]])
        v = torch.cat([Ct.vals.to(dev), Cn.vals.to(dev)])
        r, c = torch.minimum(r, c), torch.maximum(r, c)
        nb = 3 * Ntot
        key = r * nb + c
        key, inv = torch.unique(key, return_inverse=True)
        vals = torch.zeros(len(key), dtype=dt, device=dev).index_add_(0, inv, v)
        del r, c, v, inv
        ru, cu = key // nb, key % nb
        del key
        # far face of the neighbour: clamped (drop its DOFs) or springs k = alpha diag
        far_local = 2 * n if sgn > 0 else 0
        far_nodes = np.flatnonzero(gn[:, a] == far_local)
        far = torch.as_tensor(np.unique((3 * nnode[far_nodes][:, None] + np.arange(3)[None]).reshape(-1)), device=dev)
        clamp = oi % 2 == 0
        diag = torch.nonzero(ru == cu).squeeze(1)
        dvals = torch.zeros(nb, dtype=dt, device=dev); dvals[ru[diag]] = vals[diag]
        if clamp:
            keep = torch.ones(nb, dtype=torch.bool, device=dev); keep[far] = False
        else:
            keep = torch.ones(nb, dtype=torch.bool, device=dev)
            alpha = float(0.3 * 10 ** torch.rand((), generator=gen, device=dev))
            spring = torch.zeros(nb, dtype=dt, device=dev); spring[far] = alpha * dvals[far]
            vals = vals + torch.where(ru == cu, spring[ru], torch.zeros((), dtype=dt, device=dev))
            dvals = dvals + spring
        new = torch.full((nb,), -1, dtype=torch.long, device=dev); new[keep] = torch.arange(int(keep.sum()), device=dev)
        sel = keep[ru] & keep[cu]
        rA, cA, vA = new[ru[sel]], new[cu[sel]], vals[sel]
        del ru, cu, vals, sel
        nf = int(keep.sum())
        sA = 1 / torch.sqrt(dvals[keep])
        order = torch.argsort(rA * nf + cA)
        rA, cA, vA = rA[order], cA[order], vA[order]
        del order
        vA.mul_(sA[rA]).mul_(sA[cA])                                        # diagonal scaling in place
        crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(rA, minlength=nf), 0)]).int()
        cA = cA.int(); del rA
        gc.collect(); torch.cuda.empty_cache()
        _mem('glued_before_factor', offset=d, dofs=nf, nnz=int(vA.numel()))
        sol, prec = _spd_glued(crow, cA, vA.contiguous(), nf)
        _mem('glued_after_factor', offset=d, precision=prec)
        del cA, vA, crow
        # loads: free box faces of both cells (test: not the glued face; neighbour: neither glued nor far face)
        tface = (a, 2 * n if sgn > 0 else 0)
        nglue, nfar = (a, 0 if sgn > 0 else 2 * n), (a, far_local)
        tf = [T for kk, T in ft.items() if kk != tface]
        nf_ = [T for kk, T in fn_.items() if kk not in (nglue, nfar)]
        ptd = Ct.P                                                          # test port DOFs in the glued numbering
        pnd = dmap[Cn.P]                                                    # neighbour port DOFs in the glued numbering
        for j in range(0, per, chunk):
            k = min(chunk, per - j)
            only_n = (torch.rand(k, device=dev, generator=gen) < 0.25).to(dt)
            F = torch.zeros((nb, k), dtype=dt, device=dev)
            if tf:
                F.index_add_(0, ptd, sum(T.forces(T.random(k, gen)) for T in tf) * (1 - only_n)[None, :])
            if nf_:
                F.index_add_(0, pnd, sum(T.forces(T.random(k, gen)) for T in nf_))
            u = torch.zeros((nb, k), dtype=dt, device=dev)
            u[keep] = sA[:, None] * sol.solve(sA[:, None] * F[keep])
            Qs.append(_finite(u[ptd], 'GLUED'))
        sol.free(); del sol, F, u, dvals, sA, new, keep; gc.collect(); torch.cuda.empty_cache()
        info['offsets'].append(dict(offset=d, clamped=clamp, shared_nodes=int(shared.sum()), dofs=nf, precision=prec))
        log(json.dumps(dict(event='GLUED_OFFSET', case=Ct.case, parent=parent, offset=d, clamped=clamp, dofs=nf,
                            shared_nodes=int(shared.sum()), precision=prec)))
    q = torch.cat(Qs, 1)
    return q[:, torch.randperm(q.shape[1], generator=gen, device=dev)[:total]], info


# --------------------------------------------------------------------------------------------------------- output
def sens_F(C, q, chunk=128):
    S, F = [], []
    for j in range(0, q.shape[1], chunk):
        u = C.extend(q[:, j:j + chunk])
        S.append(C.sens2(u)); F.append((C.K @ u)[C.P])
    return torch.cat(S, 1), torch.cat(F, 1)


def save(d, cls, q, S, F):
    lo = 0
    for name, n_ in SPLITS:
        np.save(d / f'{name}_{cls}.npy', q[:, lo:lo + n_].T.contiguous().to(torch.float32).cpu().numpy())
        np.save(d / f'{name}_{cls}_sens.npy', S[:, lo:lo + n_].T.contiguous().cpu().numpy())
        np.save(d / f'{name}_{cls}_F.npy', F[:, lo:lo + n_].T.contiguous().to(torch.float32).cpu().numpy())
        lo += n_


def _nonfinite_bank(d, cls):
    return any(not np.isfinite(np.load(d / f'{s}_{cls}.npy', mmap_mode='r')).all() for s, _ in SPLITS if (d / f'{s}_{cls}.npy').exists())


def one(body, data, case, classes, F_old=False, fix_support=False, log=print):
    total = sum(n_ for _, n_ in SPLITS)
    d = data / case; d.mkdir(parents=True, exist_ok=True)
    rec = json.loads((d / 'DONE2.json').read_text()) if (d / 'DONE2.json').exists() else dict(case=case, banks={}, seconds={})
    t0 = time.perf_counter(); tt = rec['seconds']
    C = TE.Cell(case, body, log=lambda s_: None)
    try:
        if C.ni == 0:
            raise ValueError('NO_INTERIOR')
        _mem('start', case=case)
        C.assemble(); _mem('assembled'); C.dmoments(); _mem('dmoments')
        seed = (int.from_bytes(case.encode(), 'little') + 7919) % (2 ** 31)
        gen = torch.Generator(device=dev).manual_seed(seed)
        faces = box_tractions(C)
        raw, info = {}, {}
        t = time.perf_counter()
        if not faces and any(c_ in classes for c_ in ('force_c', 'face_c', 'support_k')):
            raise ValueError('NO_BOX_FACE_WITH_MATERIAL')
        if 'force_c' in classes or 'face_c' in classes:
            tt['neumann_precision'] = PG.factor_safe(C, neumann=True, interior=False, fp32_neumann=True)
            if 'force_c' in classes:
                raw['force_c'], info['force_c_cut_qp'] = force_c(C, faces, total, gen)
            if 'face_c' in classes:
                raw['face_c'] = face_c(C, faces, total, gen)
            C._free()
        tt['neumann_dirs2'] = time.perf_counter() - t; t = time.perf_counter()
        if 'support_k' in classes:
            q = support_k(C, faces, total, gen)
            if q is None:
                info['support_k'] = dict(skipped=f'{len(faces)} box face(s) with material')
            else:
                raw['support_k'] = q
        if fix_support and (d / 'train_support.npy').exists() and _nonfinite_bank(d, 'support'):
            orig = PG.spd_safe
            PG.spd_safe = _spd64
            try:
                g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
                X = torch.as_tensor(g / (2 * C.n), dtype=dt, device=dev)
                raw['support'] = _finite(PG.support_bank(C, X, g, total, gen), 'SUPPORT_FP64')
            finally:
                PG.spd_safe = orig
            for s_, _ in SPLITS:
                for suf in ('', '_sens'):
                    f = d / f'{s_}_support{suf}.npy'
                    if f.exists():
                        f.rename(d / f'{s_}_support{suf}.nan_bak.npy')
        tt['spring_dirs2'] = time.perf_counter() - t; t = time.perf_counter()
        _mem('before_glued', case=case)
        if 'glued' in classes:
            q, info['glued'] = glued(C, body, total, gen, log=log)
            if q is not None:
                raw['glued'] = q
        tt['glued_dirs'] = time.perf_counter() - t; t = time.perf_counter()
        tt['interior_precision'] = PG.factor_safe(C, neumann=False, fp32=True)
        for cls in list(raw):
            q = _finite(raw.pop(cls), cls.upper())
            qn = (q - C.Q @ (C.Q.T @ q)).norm(dim=0)
            if bool((qn <= 1e-12 * qn.max()).any()):
                raise ValueError(f'ZERO_DIRECTION_{cls.upper()}: {int((qn <= 1e-12 * qn.max()).sum())} columns')
            q = PG.normalize(C, q)
            rq = (q * q).sum(0)
            S, F = sens_F(C, q)
            _finite(S, cls.upper() + '_SENS'); _finite(F, cls.upper() + '_F')
            rec['banks'].setdefault(cls, {}).update(dict(rayleigh_quantiles=np.quantile((1 / rq).cpu().numpy(), [0, .1, .5, .9, 1]).tolist(),
                                     unit_energy_check=float(((F[:, :16] * q[:, :16]).sum(0) - 1).abs().max()), count=q.shape[1],
                                     **({k_: v_ for k_, v_ in info.get(cls, {}).items()} if isinstance(info.get(cls), dict) else {})))
            save(d, cls, q, S, F)
            del q, S, F
        if F_old:
            for cls in OLD:
                if not (d / f'train_{cls}.npy').exists():
                    continue
                for s_, _ in SPLITS:
                    qb = torch.as_tensor(np.load(d / f'{s_}_{cls}.npy'), device=dev).T.to(dt)
                    ok = torch.isfinite(qb).all(0)
                    F = torch.full_like(qb, float('nan'))
                    if ok.any():
                        F[:, ok] = torch.cat([C.apply(qb[:, ok][:, j:j + 128]) for j in range(0, int(ok.sum()), 128)], 1)
                    np.save(d / f'{s_}_{cls}_F.npy', F.T.contiguous().to(torch.float32).cpu().numpy())
                rec['banks'].setdefault(cls + '_F', {})['written'] = True
        for cls, v in info.items():
            if isinstance(v, dict) and 'skipped' in v:
                rec['banks'][cls] = v
        tt['normalize_sens_F2'] = time.perf_counter() - t
        rec.update(splits=SPLITS, classes=sorted(set(rec.get('classes', [])) | set(classes)), total_seconds2=time.perf_counter() - t0,
                   peak_GB2=torch.cuda.max_memory_allocated() / 2 ** 30, faces=[list(k) for k in faces])
        (d / 'DONE2.json').write_text(json.dumps(rec))
        log(json.dumps(dict(event='DONE2', case=case, seconds=rec['total_seconds2'], banks=rec['banks'])))
    finally:
        C._free()


def main(argv):
    import argparse, traceback
    ap = argparse.ArgumentParser()
    ap.add_argument('body'); ap.add_argument('data'); ap.add_argument('cases', nargs='+')
    ap.add_argument('--classes', default=','.join(NEW)); ap.add_argument('--F-old', action='store_true')
    ap.add_argument('--fix-support', action='store_true')
    a = ap.parse_args(argv)
    classes = [c for c in a.classes.split(',') if c]
    bad = [c for c in classes if c not in NEW]
    if bad:
        raise SystemExit(f'unknown classes {bad}')
    Q = Path(os.environ['PREP_LOCK']) if os.environ.get('PREP_LOCK') else None
    for ci, case in enumerate(a.cases):
        tag = f'prep2_{os.getpid()}_{ci}'
        if Q is not None:
            PG._acquire(Q, tag)
        torch.cuda.reset_peak_memory_stats()
        try:
            one(a.body, Path(a.data), case, classes, a.F_old, a.fix_support)
        except Exception as e:
            own = [f'{fr.name}:{fr.lineno}' for fr in traceback.extract_tb(e.__traceback__) if fr.filename.endswith(('prep_geo2.py', 'prep_geo.py'))]
            print(json.dumps(dict(event='FAILED2', case=case, error=repr(e)[:300], where=own, trace=traceback.format_exc()[-1500:])), flush=True)
        gc.collect(); torch.cuda.empty_cache()
        if Q is not None:
            PG._release(Q, tag)


if __name__ == '__main__':
    main(sys.argv[1:])
