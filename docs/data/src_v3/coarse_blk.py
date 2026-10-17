"""Block-structured coarse inverse of the equilibrium correction (src_v3 S15, OPL_COARSE_BLK, deployment cells).

The coarse solve s = A_c^-1 z (A_c: the Jacobi-scaled Galerkin matrix of the cell's Q1 coarse space, nc ~ 9.3k on a full
plate cell) was applied with the explicit float32 inverse in packed symmetric tiles (S2): nc^2 / 2 floats, 168 MiB per
full cell, the largest per-cell item of a streamed analysis. A_c couples only coarse vertices at most w vertex spacings
apart (w read off the matrix), so the vertices with a coordinate in a central slab of width w on some axis (the
separator) split the others into 8 octants that do not couple. Block elimination with this separator (exact):
    y_o = A_oo^-1 z_o                                   (each octant)
    x_s = S^-1 (z_s - sum_o A_so y_o)                   S = A_ss - sum_o A_so A_oo^-1 A_os (Schur complement)
    x_o = A_oo^-1 (z_o - A_os x_s)
The octant inverses and S^-1 are formed in float64 (Cholesky) and stored in float32 as packed symmetric tiles (the
octants as one block-diagonal tile list), A_os as float32 COO triplets. On the full plate cells (nc = 9324, coupling width
2 through the ghost-penalty faces, separator 2664 columns): 29 MiB instead of 168, error against the float64 solve 1.7e-7
instead of 2.9e-7 (the blocks are formed in float64), coarse solve 0.26 ms instead of 0.39 (B = 1). The solve is
algebraically the same as the explicit inverse; only its float32 rounding differs. trainlib checks it against the
float64 solve like the packed inverse (OPL_COARSE_INV_TOL) and otherwise uses the packed inverse; small cells, where
the blocks do not save at least half of the tiles (OPL_COARSE_BLK_MAXFRAC), keep the packed inverse."""
import os
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import symtile as ST

f32 = torch.float32
MAX_FRAC = float(os.environ.get('OPL_COARSE_BLK_MAXFRAC', '0.5'))   # used only if its tiles are at most this fraction
                                                                     # of the packed inverse's


def coupling_width(A, vc):
    """Largest Chebyshev distance (in coarse vertex spacings) between the vertices of two columns with a nonzero entry
    of A (nc x nc, dense); vc (nc, 3) integer vertex coordinates."""
    v = torch.as_tensor(np.ascontiguousarray(vc), dtype=torch.int16, device=A.device)
    w = 0
    for r0 in range(0, A.shape[0], 1024):
        r1 = min(A.shape[0], r0 + 1024)
        nz = A[r0:r1] != 0
        if bool(nz.any()):
            d = (v[r0:r1, None, :] - v[None, :, :]).abs().amax(2)
            w = max(w, int(d[nz].max()))
            del d
        del nz
    return w


def partition(vc, w):
    """Octant (0..7) of every column, -1 for the separator: the vertices with a coordinate in the slab [m, m + w - 1]
    of some axis, m the middle of the cell's vertex range on that axis."""
    vc = np.asarray(vc)
    lab = np.zeros(len(vc), np.int64)
    sep = np.zeros(len(vc), bool)
    for k in range(3):
        c = vc[:, k]
        m = (int(c.min()) + int(c.max()) - (w - 1)) // 2
        sep |= (c >= m) & (c <= m + w - 1)
        lab = lab * 2 + (c > m + w - 1)
    lab[sep] = -1
    return lab


def _sym(A, r, c):
    """Block (r, c) of the symmetric matrix whose lower triangle is A's (the triangle a Cholesky factor of A reads)."""
    return torch.where(r[:, None] >= c[None, :], A[r[:, None], c], A[c[:, None], r].T)


def build(A, vc, shift=0.0):
    """Block data for the solve with A + shift I (A: nc x nc float64 dense on the device, symmetric up to rounding; its
    lower triangle is used, as by the Cholesky factor; not modified) or None when it does not pay (the separator is too
    large) or does not apply; returns (data, info)."""
    nc, dvc, T = A.shape[0], A.device, ST.T
    w = max(1, coupling_width(A, vc))
    lab = partition(vc, w)
    sep = np.flatnonzero(lab < 0)
    octs = [o for o in (np.flatnonzero(lab == k) for k in range(8)) if len(o)]
    nto = [triton.cdiv(len(o), T) for o in octs]
    tiles = sum(n * (n + 1) // 2 for n in nto) + ST.ntiles(len(sep))[1]
    info = dict(width=w, octants=[len(o) for o in octs], separator=int(len(sep)), tiles=int(tiles),
                tiles_packed=int(ST.ntiles(nc)[1]))
    if tiles > MAX_FRAC * ST.ntiles(nc)[1] or not len(sep) or not octs:
        return None, dict(info, reason='size')
    labt = torch.as_tensor(lab, device=dvc)
    for r0 in range(0, nc, 1024):                                        # the octants must not couple (exact zeros)
        r1 = min(nc, r0 + 1024)
        la, lb = labt[r0:r1, None], labt[None, :]
        if bool(((A[r0:r1] != 0) & (la >= 0) & (lb >= 0) & (la != lb)).any()):
            return None, dict(info, reason='coupled')
    si = torch.as_tensor(sep, device=dvc)
    S = A[si[:, None], si]
    if shift:
        S.diagonal().add_(shift)
    D, TI, TJ, rows, cols, vals, perm = [], [], [], [], [], [], []
    t0 = 0
    for o, nt in zip(octs, nto):
        oi = torch.as_tensor(o, device=dvc)
        Aoo = A[oi[:, None], oi]
        if shift:
            Aoo.diagonal().add_(shift)
        L, bad = torch.linalg.cholesky_ex(Aoo)
        if int(bad):
            return None, dict(info, reason='octant')
        Aos = _sym(A, oi, si)
        cz = torch.nonzero(Aos.abs().amax(0)).squeeze(1)                  # separator columns this octant couples to
        if len(cz):
            Ar = Aos[:, cz]
            S[cz[:, None], cz] = S[cz[:, None], cz] - Ar.T @ torch.cholesky_solve(Ar, L)
            nzr, nzc = torch.nonzero(Ar, as_tuple=True)
            rows.append(nzr + t0 * T); cols.append(cz[nzc]); vals.append(Ar[nzr, nzc].to(f32))
        D.append(ST.pack(torch.cholesky_inverse(L).to(f32)))
        ti, tj = ST._tile_index(nt, dvc)
        TI.append(ti + t0); TJ.append(tj + t0)
        p = torch.full((nt * T,), nc, dtype=torch.long, device=dvc)
        p[:len(o)] = oi
        perm.append(p)
        t0 += nt
        del Aoo, L, Aos
    Ls, bad = torch.linalg.cholesky_ex(S)
    if int(bad):
        return None, dict(info, reason='separator')
    Np = t0 * T
    perm = torch.cat(perm)
    if rows:
        idx, val = torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals)
    else:                                                                # no octant couples to the separator
        idx, val = torch.zeros((2, 0), dtype=torch.long, device=dvc), torch.zeros(0, dtype=f32, device=dvc)
    pmap = torch.where(perm < nc, perm, torch.full_like(perm, -1)).to(torch.int32)   # padded octant row -> column (-1)
    blk = dict(D=torch.cat(D), TI=torch.cat(TI).to(torch.int32), TJ=torch.cat(TJ).to(torch.int32),
               S=ST.pack(torch.cholesky_inverse(Ls).to(f32)), pmap=pmap, sep=si,
               dims=(int(nc), int(len(sep)), int(Np)))
    blk['Gc'], blk['Gj'], blk['Gv'] = _csr(idx[0], idx[1], val, Np)                  # A_os by octant rows
    blk['Tc'], blk['Tj'], blk['Tv'] = _csr(idx[1], idx[0], val, len(sep))             # A_so by separator rows
    lg = int((blk['Gc'][1:] - blk['Gc'][:-1]).max()) if Np else 0                    # longest rows (chunks of 32)
    lt = int((blk['Tc'][1:] - blk['Tc'][:-1]).max()) if len(sep) else 0
    blk['dims'] = blk['dims'] + (max(1, triton.cdiv(lg, 32)), max(1, triton.cdiv(lt, 32)))
    info['MiB'] = round(nbytes(blk) / 2 ** 20, 2)
    return blk, info


def _csr(r, c, v, n):
    """CSR (int32 row pointers, int32 columns sorted within a row, float32 values) of the triplets (r, c, v), n rows."""
    o = torch.argsort(r * (int(c.max()) + 1 if len(c) else 1) + c)
    crow = torch.zeros(n + 1, dtype=torch.int64, device=v.device)
    crow[1:] = torch.cumsum(torch.bincount(r, minlength=n), 0)
    return crow.to(torch.int32), c[o].to(torch.int32).contiguous(), v[o].contiguous()


def nbytes(blk):
    return sum(v.numel() * v.element_size() for v in blk.values() if torch.is_tensor(v))


@triton.jit
def _symm_map(P_ptr, X_ptr, Y_ptr, n, B, TI_ptr, TJ_ptr, map_ptr, TT: tl.constexpr, BB: tl.constexpr,
              XMAP: tl.constexpr, YMAP: tl.constexpr):
    """symtile._symm with the rows of X (XMAP) or of Y (YMAP) addressed through map (-1: none): the same tile products and
    atomic sums, so the octants' block-diagonal product reads the coarse vector in its own order or writes into it."""
    k = tl.program_id(0)
    I = tl.load(TI_ptr + k)
    J = tl.load(TJ_ptr + k)
    t = tl.arange(0, TT)
    c = tl.arange(0, BB)
    mc = c < B
    A = tl.load(P_ptr + k.to(tl.int64) * TT * TT + t[:, None] * TT + t[None, :])
    rI = I * TT + t
    rJ = J * TT + t
    mI = rI < n
    mJ = rJ < n
    if XMAP:
        xI = tl.load(map_ptr + rI, mask=mI, other=-1)
        xJ = tl.load(map_ptr + rJ, mask=mJ, other=-1)
    else:
        xI = tl.where(mI, rI, -1)
        xJ = tl.where(mJ, rJ, -1)
    if YMAP:
        yI = tl.load(map_ptr + rI, mask=mI, other=-1)
        yJ = tl.load(map_ptr + rJ, mask=mJ, other=-1)
    else:
        yI = tl.where(mI, rI, -1)
        yJ = tl.where(mJ, rJ, -1)
    XJ = tl.load(X_ptr + xJ[:, None].to(tl.int64) * B + c[None, :], mask=(xJ >= 0)[:, None] & mc[None, :], other=0.0)
    YI = tl.dot(A, XJ, input_precision="ieee")
    tl.atomic_add(Y_ptr + yI[:, None].to(tl.int64) * B + c[None, :], YI, mask=(yI >= 0)[:, None] & mc[None, :])
    if I != J:
        XI = tl.load(X_ptr + xI[:, None].to(tl.int64) * B + c[None, :], mask=(xI >= 0)[:, None] & mc[None, :], other=0.0)
        YJ = tl.dot(tl.trans(A), XI, input_precision="ieee")
        tl.atomic_add(Y_ptr + yJ[:, None].to(tl.int64) * B + c[None, :], YJ, mask=(yJ >= 0)[:, None] & mc[None, :])


@triton.jit
def _csr_sub(crow, col, val, x, base, bmap, out, B, BB: tl.constexpr, KB: tl.constexpr, NCH: tl.constexpr):
    """out[i] = base[bmap[i]] - sum_k val[k] x[col[k]] over row i's entries (bmap -1: base 0); one program per row, the
    entries in NCH chunks of KB (NCH * KB >= the longest row) summed in a fixed order (deterministic)."""
    i = tl.program_id(0)
    s0 = tl.load(crow + i)
    s1 = tl.load(crow + i + 1)
    c = tl.arange(0, BB)
    mc = c < B
    acc = tl.zeros((BB,), dtype=tl.float32)
    for ch in tl.static_range(NCH):
        k = s0 + ch * KB + tl.arange(0, KB)
        mk = k < s1
        j = tl.load(col + k, mask=mk, other=0)
        v = tl.load(val + k, mask=mk, other=0.0)
        xv = tl.load(x + j[:, None].to(tl.int64) * B + c[None, :], mask=mk[:, None] & mc[None, :], other=0.0)
        acc += tl.sum(v[:, None] * xv, axis=0)
    b = tl.load(bmap + i)
    bv = tl.load(base + b.to(tl.int64) * B + c, mask=mc & (b >= 0), other=0.0)
    tl.store(out + i.to(tl.int64) * B + c, bv - acc, mask=mc)


def _solve16(blk, z):
    nc, ns, Np, chG, chT = blk['dims']
    B = z.shape[1]
    dv = z.device
    ntD = len(blk['TI'])
    y = torch.zeros((Np, B), dtype=f32, device=dv)                      # y_o = A_oo^-1 z_o (padded octant order)
    _symm_map[(ntD,)](blk['D'], z, y, Np, B, blk['TI'], blk['TJ'], blk['pmap'], TT=ST.T, BB=16, XMAP=True, YMAP=False,
                      num_warps=4)
    t = torch.empty((ns, B), dtype=f32, device=dv)                      # z_s - A_so y
    _csr_sub[(ns,)](blk['Tc'], blk['Tj'], blk['Tv'], y, z, blk['sep'], t, B, BB=16, KB=32, NCH=chT, num_warps=1)
    xs = ST.matmul(blk['S'], ns, t)                                      # x_s = S^-1 (z_s - A_so y)
    r = torch.empty((Np, B), dtype=f32, device=dv)                      # z_o - A_os x_s
    _csr_sub[(Np,)](blk['Gc'], blk['Gj'], blk['Gv'], xs, z, blk['pmap'], r, B, BB=16, KB=32, NCH=chG, num_warps=1)
    s = torch.zeros((nc, B), dtype=f32, device=dv)                      # x_o = A_oo^-1 r, written in the coarse order
    _symm_map[(ntD,)](blk['D'], r, s, Np, B, blk['TI'], blk['TJ'], blk['pmap'], TT=ST.T, BB=16, XMAP=False, YMAP=True,
                      num_warps=4)
    s.index_copy_(0, blk['sep'], xs)
    return s


def solve(blk, z):
    """A^-1 z: z (nc, B) float32 in the coarse column order -> (nc, B) float32 (S15b): nine kernels per block of 16
    columns. The octant rows are read and written through the column map inside the tile products, the couplings A_os
    / A_so are CSR row sums (deterministic), so neither the permuted copies nor sparse-matrix objects (cuSPARSE made
    the two-stream PCG matvec of the 51-cell plate 2-10 % slower and irregular) are needed; the CPU issues nine kernels
    instead of about twenty (the two-stream PCG matvec is bound by the CPU's kernel launches)."""
    z = z.contiguous()
    B = z.shape[1]
    if B <= 16:
        return _solve16(blk, z)
    return torch.cat([_solve16(blk, z[:, c0:c0 + 16].contiguous()) for c0 in range(0, B, 16)], 1)


def solve_ref(blk, z):
    """The same solve by torch operations (tests): permuted copies, index_add couplings."""
    nc, ns, Np = blk['dims'][:3]
    pm, sep = blk['pmap'].long(), blk['sep'].long()
    zp = torch.where((pm >= 0)[:, None], z[pm.clamp_min(0)], torch.zeros((), dtype=z.dtype, device=z.device))
    y = ST.matmul_tiles(blk['D'], blk['TI'], blk['TJ'], Np, zp)
    gr = torch.repeat_interleave(torch.arange(Np, device=z.device), (blk['Gc'][1:] - blk['Gc'][:-1]).long())
    gj, gv = blk['Gj'].long(), blk['Gv'][:, None]
    xs = ST.matmul(blk['S'], ns, z[sep].index_add(0, gj, gv * y[gr], alpha=-1))
    xo = ST.matmul_tiles(blk['D'], blk['TI'], blk['TJ'], Np, zp.index_add(0, gr, gv * xs[gj], alpha=-1))
    s = torch.zeros_like(z)
    ok = pm >= 0
    s[pm[ok]] = xo[ok]
    s[sep] = xs
    return s
