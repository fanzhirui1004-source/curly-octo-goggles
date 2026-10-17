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
octants as one block-diagonal tile list), A_os as a float32 CSR matrix: ~20 MB per full cell instead of 168. The solve is
algebraically the same as the explicit inverse; only its float32 rounding differs. trainlib checks it against the
float64 solve like the packed inverse (OPL_COARSE_INV_TOL) and otherwise uses the packed inverse."""
import os
import numpy as np
import torch
import torch.nn.functional as F
import triton
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
    if tiles > MAX_FRAC * ST.ntiles(nc)[1] or not len(sep):
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
    G = torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals), (Np, len(sep)),
                                dtype=f32, device=dvc).coalesce()
    inv = torch.empty(nc, dtype=torch.long, device=dvc)                  # coarse column -> row of cat([x_o, x_s])
    pv = torch.nonzero(perm < nc).squeeze(1)
    inv[perm[pv]] = pv
    inv[si] = Np + torch.arange(len(sep), device=dvc)
    blk = dict(D=torch.cat(D), TI=torch.cat(TI).to(torch.int32), TJ=torch.cat(TJ).to(torch.int32),
               S=ST.pack(torch.cholesky_inverse(Ls).to(f32)), G=G.to_sparse_csr(), Gt=G.t().coalesce().to_sparse_csr(),
               perm=perm, sep=si, inv=inv, dims=(int(nc), int(len(sep)), int(Np)))
    info['MiB'] = round(nbytes(blk) / 2 ** 20, 2)
    return blk, info


def nbytes(blk):
    n = 0
    for v in blk.values():
        if torch.is_tensor(v):
            parts = [v.crow_indices(), v.col_indices(), v.values()] if v.layout == torch.sparse_csr else [v]
            n += sum(p.numel() * p.element_size() for p in parts)
    return n


def solve(blk, z):
    """A^-1 z: z (nc, B) float32 in the coarse column order -> (nc, B) float32."""
    nc, ns, Np = blk['dims']
    zp = F.pad(z, (0, 0, 0, 1))[blk['perm']]                              # octant rows (zero rows: tile padding)
    y = ST.matmul_tiles(blk['D'], blk['TI'], blk['TJ'], Np, zp)
    xs = ST.matmul(blk['S'], ns, z[blk['sep']] - torch.sparse.mm(blk['Gt'], y))
    xo = ST.matmul_tiles(blk['D'], blk['TI'], blk['TJ'], Np, zp - torch.sparse.mm(blk['G'], xs))
    return torch.cat([xo, xs])[blk['inv']]
