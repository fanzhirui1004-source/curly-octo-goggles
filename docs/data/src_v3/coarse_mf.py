"""Matrix-free Q1 coarse prolongation of the equilibrium correction (src_v3, OPL_COARSE_PACK=1, deployment cells).

trainlib.coarse_setup forms V (interior DOFs x coarse columns): the Q1 hats of the ne^3 coarse elements (ne = 16 for
Q1_17) at the grid nodes, one column per (vertex, component) with interior support, scaled by 1 / sqrt(diag(A_c)).
src_v2 kept V and V^T as float32 COO tensors with int64 indices (~94 MB per large cell). Here only the column map
(vertex, component) -> column (int32), the column scale 1 / d (float64) and the grid id of every node (int32) are kept
(< 1 MB per cell); the entries are recomputed in the kernels exactly as before: the hat weight w is an exact dyadic
number (t = (g - R2 e) / R2), the entry is w * (1 / d) rounded once from float64 to float32 (the value src_v2 stored).
  prolong(C, c): V c   (interior x B; one program per block of interior DOFs, the 8 vertices summed in order)
  restrict(C, r): V^T r (coarse x B; float32 atomic adds into the coarse vector)
Both in float32 like the src_v2 sparse products (summation order differs: rounding only)."""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _hat_entry(gx, gy, gz, comp, a: tl.constexpr, b: tl.constexpr, c: tl.constexpr, colmap_ptr, invd_ptr, mi,
               R2: tl.constexpr, NE: tl.constexpr, NN: tl.constexpr):
    ex = tl.minimum(gx // R2, NE - 1)
    ey = tl.minimum(gy // R2, NE - 1)
    ez = tl.minimum(gz // R2, NE - 1)
    tx = (gx - ex * R2).to(tl.float64) / R2
    ty = (gy - ey * R2).to(tl.float64) / R2
    tz = (gz - ez * R2).to(tl.float64) / R2
    wx = tx if a == 1 else 1.0 - tx
    wy = ty if b == 1 else 1.0 - ty
    wz = tz if c == 1 else 1.0 - tz
    w = wx * wy * wz
    vid = ((ex + a) * NN + (ey + b)) * NN + (ez + c)
    col = tl.load(colmap_ptr + 3 * vid + comp, mask=mi, other=-1)
    ok = mi & (col >= 0) & (w != 0.0)
    dv = tl.load(invd_ptr + col, mask=ok, other=0.0)
    return col, ok, (w * dv).to(tl.float32)


@triton.jit
def _prolong(c_ptr, y_ptr, I_ptr, nodes_ptr, colmap_ptr, invd_ptr, ni, B, M: tl.constexpr, R2: tl.constexpr,
             NE: tl.constexpr, NN: tl.constexpr, BI: tl.constexpr, BB: tl.constexpr):
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    mi = i < ni
    dof = tl.load(I_ptr + i, mask=mi, other=0).to(tl.int64)
    node = dof // 3
    comp = (dof - node * 3).to(tl.int32)
    gid = tl.load(nodes_ptr + node, mask=mi, other=0)
    gx = gid // (M * M)
    gy = (gid // M) % M
    gz = gid % M
    cc = tl.arange(0, BB)
    mc = cc < B
    acc = tl.zeros((BI, BB), dtype=tl.float32)
    for a in tl.static_range(2):
        for b in tl.static_range(2):
            for c in tl.static_range(2):
                col, ok, v = _hat_entry(gx, gy, gz, comp, a, b, c, colmap_ptr, invd_ptr, mi, R2, NE, NN)
                x = tl.load(c_ptr + col[:, None].to(tl.int64) * B + cc[None, :], mask=ok[:, None] & mc[None, :], other=0.0)
                acc += v[:, None] * x
    tl.store(y_ptr + i[:, None].to(tl.int64) * B + cc[None, :], acc, mask=mi[:, None] & mc[None, :])


@triton.jit
def _restrict(r_ptr, z_ptr, I_ptr, nodes_ptr, colmap_ptr, invd_ptr, ni, B, M: tl.constexpr, R2: tl.constexpr,
              NE: tl.constexpr, NN: tl.constexpr, BI: tl.constexpr, BB: tl.constexpr):
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    mi = i < ni
    dof = tl.load(I_ptr + i, mask=mi, other=0).to(tl.int64)
    node = dof // 3
    comp = (dof - node * 3).to(tl.int32)
    gid = tl.load(nodes_ptr + node, mask=mi, other=0)
    gx = gid // (M * M)
    gy = (gid // M) % M
    gz = gid % M
    cc = tl.arange(0, BB)
    mc = cc < B
    r = tl.load(r_ptr + i[:, None].to(tl.int64) * B + cc[None, :], mask=mi[:, None] & mc[None, :], other=0.0)
    for a in tl.static_range(2):
        for b in tl.static_range(2):
            for c in tl.static_range(2):
                col, ok, v = _hat_entry(gx, gy, gz, comp, a, b, c, colmap_ptr, invd_ptr, mi, R2, NE, NN)
                tl.atomic_add(z_ptr + col[:, None].to(tl.int64) * B + cc[None, :], v[:, None] * r, mask=ok[:, None] & mc[None, :])


def meta(C, vc, slot, nc, invd, ne):
    """Kernel data of cell C: node grid ids (int32), column map (3 * (ne+1)^3, -1 where dropped), 1 / d (float64)."""
    dvc = C.dK.device
    nn = ne + 1
    colmap = np.full(3 * nn ** 3, -1, np.int32)
    vid = (vc[:, 0] * nn + vc[:, 1]) * nn + vc[:, 2]
    colmap[3 * vid + np.asarray(slot)] = np.arange(nc, dtype=np.int32)
    # every interior DOF's nonzero hats must hit kept columns (coarse_prolong drops only all-zero columns)
    n2 = 2 * C.n + 1
    nodes = np.asarray(C.nodes)
    I = C.I.cpu().numpy()
    g = np.stack(np.unravel_index(nodes[I // 3], (n2,) * 3), 1)
    r2 = (2 * C.n) // ne
    e = np.minimum(g // r2, ne - 1); t = g - r2 * e
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (t[:, 0] if a else r2 - t[:, 0]) * (t[:, 1] if b else r2 - t[:, 1]) * (t[:, 2] if c else r2 - t[:, 2])
                v = ((e[:, 0] + a) * nn + e[:, 1] + b) * nn + e[:, 2] + c
                bad = (w != 0) & (colmap[3 * v + I % 3] < 0)
                if bad.any():
                    raise ValueError('COARSE_MF_DROPPED_COLUMN_WITH_SUPPORT')
    return dict(nodes=torch.as_tensor(nodes.astype(np.int32), device=dvc), colmap=torch.as_tensor(colmap, device=dvc),
                invd=invd.to(torch.float64).contiguous(), nc=int(nc), M=int(n2), R2=int(r2), NE=int(ne), NN=int(nn))


def _launch(kern, C, x, out, B):
    m = C._c_mf
    ni = C.I.numel()
    BB = triton.next_power_of_2(max(B, 1))
    BI = max(16, 1024 // BB)
    kern[(triton.cdiv(ni, BI),)](x, out, C.I, m['nodes'], m['colmap'], m['invd'], ni, B, M=m['M'], R2=m['R2'], NE=m['NE'],
                                 NN=m['NN'], BI=BI, BB=BB, num_warps=4)


def prolong(C, c):
    """V c: c (nc, B) float32 -> (ni, B) float32."""
    c = c.contiguous()
    B = c.shape[1]
    y = torch.empty((C.I.numel(), B), dtype=torch.float32, device=c.device)
    for c0 in range(0, B, 16):
        nb = min(16, B - c0)
        if B <= 16:
            _launch(_prolong, C, c, y, B)
            break
        yc = torch.empty((C.I.numel(), nb), dtype=torch.float32, device=c.device)
        _launch(_prolong, C, c[:, c0:c0 + nb].contiguous(), yc, nb)
        y[:, c0:c0 + nb] = yc
    return y


def restrict(C, r):
    """V^T r: r (ni, B) float32 -> (nc, B) float32."""
    r = r.contiguous()
    B = r.shape[1]
    nc = C._c_mf['nc']
    z = torch.zeros((nc, B), dtype=torch.float32, device=r.device)
    for c0 in range(0, B, 16):
        nb = min(16, B - c0)
        if B <= 16:
            _launch(_restrict, C, r, z, B)
            break
        zc = torch.zeros((nc, nb), dtype=torch.float32, device=r.device)
        _launch(_restrict, C, r[:, c0:c0 + nb].contiguous(), zc, nb)
        z[:, c0:c0 + nb] = zc
    return z
