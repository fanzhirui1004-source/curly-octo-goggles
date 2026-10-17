"""Fused training-time hyperedge layer (task 66, tier 2): the same map and gradients as sparse_layers.HyperFn,
    Z  = G(a) X,   Z' = Z W_h,   dX = S(b) Z'     (G[(h,e), hn[e,s]] = a[e,s,h],  S[hn[e,s], (h,e)] = b[e,s,h] / deg)
with the node-row gathers and scatters in Triton kernels that serve ALL heads at once: every node row of X (and of dY in
the backward) is read once per hyperedge instead of once per head (cuSPARSE sees H * Eh rows), and the heads are summed
before the scatter (one float atomic row-add per (e, s) instead of H row-reads per incidence). The channel mixes and the
W gradient stay in cuBLAS (bmm, sparse_layers._gram). Backward: dZ' = gather(dY; b / deg), dW = sum Z^T dZ', dZ = dZ' W^T,
then one kernel scatters dX += sum_h a dZ_h and forms da[e,s,h] = <X[n], dZ_h>, db[e,s,h] = <dY[n] / deg, Z'_h>.
Summation order differs from cuSPARSE (float atomics), so results agree to float32 rounding, like index_add_.
Switch: env FUSED_HYPER=1 makes sparse_layers.hyper use it (default off: results unchanged)."""
import os
import torch
import triton
import triton.language as tl
import sparse_layers as SL
import hyperk as HK


@triton.jit
def _gather(src, hn, wgt, deg, out, Eh, BF, H: tl.constexpr, S: tl.constexpr, BLK: tl.constexpr, DEG: tl.constexpr):
    """out[h, e, c] = sum_s w[e, s, h] src[hn[e, s], c]   (w = wgt, or wgt / deg[hn] when DEG)."""
    e = tl.program_id(0)
    cols = tl.program_id(1) * BLK + tl.arange(0, BLK)
    cm = cols < BF
    hh = tl.arange(0, H)
    acc = tl.zeros((H, BLK), dtype=tl.float32)
    for s in tl.static_range(S):
        n = tl.load(hn + e * S + s).to(tl.int64)
        w = tl.load(wgt + (e * S + s) * H + hh)
        if DEG:
            w = w / tl.load(deg + n)
        x = tl.load(src + n * BF + cols, mask=cm, other=0.0)
        acc += w[:, None] * x[None, :]
    tl.store(out + (hh[:, None].to(tl.int64) * Eh + e) * BF + cols[None, :], acc, mask=cm[None, :])


@triton.jit
def _scatter(src, hn, wgt, deg, out, Eh, BF, rmask, H: tl.constexpr, S: tl.constexpr, BLK: tl.constexpr,
             DEG: tl.constexpr, RMASK: tl.constexpr = False):
    """out[hn[e, s], c] += sum_h w[e, s, h] src[h, e, c]   (RMASK, src_v3 S24: rows with rmask != 0 are skipped)."""
    e = tl.program_id(0)
    cols = tl.program_id(1) * BLK + tl.arange(0, BLK)
    cm = cols < BF
    hh = tl.arange(0, H)
    z = tl.load(src + (hh[:, None].to(tl.int64) * Eh + e) * BF + cols[None, :], mask=cm[None, :], other=0.0)
    for s in tl.static_range(S):
        n = tl.load(hn + e * S + s).to(tl.int64)
        w = tl.load(wgt + (e * S + s) * H + hh)
        if DEG:
            w = w / tl.load(deg + n)
        ok = cm
        if RMASK:
            ok = cm & (tl.load(rmask + n) == 0)
        tl.atomic_add(out + n * BF + cols, tl.sum(w[:, None] * z, axis=0), mask=ok)


@triton.jit
def _backward(X, dY, dZ, Zp, hn, A, deg, dX, dA, dB, Eh, BF, H: tl.constexpr, S: tl.constexpr, BLK: tl.constexpr):
    """dX[hn[e, s]] += sum_h a[e, s, h] dZ[h, e];  dA[e, s, h] += <X[hn], dZ[h, e]>;  dB[e, s, h] += <dY[hn] / deg, Z'[h, e]>
    (partial sums over this column block: dA, dB are zero-initialised and accumulated atomically)."""
    e = tl.program_id(0)
    cols = tl.program_id(1) * BLK + tl.arange(0, BLK)
    cm = cols < BF
    hh = tl.arange(0, H)
    base = (hh[:, None].to(tl.int64) * Eh + e) * BF + cols[None, :]
    dz = tl.load(dZ + base, mask=cm[None, :], other=0.0)
    zp = tl.load(Zp + base, mask=cm[None, :], other=0.0)
    for s in tl.static_range(S):
        n = tl.load(hn + e * S + s).to(tl.int64)
        a = tl.load(A + (e * S + s) * H + hh)
        g = tl.load(deg + n)
        x = tl.load(X + n * BF + cols, mask=cm, other=0.0)
        dy = tl.load(dY + n * BF + cols, mask=cm, other=0.0)
        tl.atomic_add(dX + n * BF + cols, tl.sum(a[:, None] * dz, axis=0), mask=cm)
        tl.atomic_add(dA + (e * S + s) * H + hh, tl.sum(x[None, :] * dz, axis=1))
        tl.atomic_add(dB + (e * S + s) * H + hh, tl.sum((dy / g)[None, :] * zp, axis=1))


@triton.jit
def _gather_t(src, hn, wgt, deg, out, Eh, BF, H: tl.constexpr, S: tl.constexpr, EB: tl.constexpr, BC: tl.constexpr,
              DEG: tl.constexpr):
    """_gather on a tile of EB hyperedges x BC columns (src_v3): the same sum per output element, in the same order."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < Eh
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    cm = cols < BF
    hh = tl.arange(0, H)
    acc = tl.zeros((EB, H, BC), dtype=tl.float32)
    for s in tl.static_range(S):
        n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
        w = tl.load(wgt + (e[:, None] * S + s) * H + hh[None, :], mask=me[:, None], other=0.0)
        if DEG:
            w = w / tl.load(deg + n, mask=me, other=1.0)[:, None]
        x = tl.load(src + n[:, None] * BF + cols[None, :], mask=me[:, None] & cm[None, :], other=0.0)
        acc += w[:, :, None] * x[:, None, :]
    ptr = out + (hh[None, :, None].to(tl.int64) * Eh + e[:, None, None]) * BF + cols[None, None, :]
    tl.store(ptr, acc, mask=me[:, None, None] & cm[None, None, :])


@triton.jit
def _scatter_t(src, hn, wgt, deg, out, Eh, BF, rmask, H: tl.constexpr, S: tl.constexpr, EB: tl.constexpr,
               BC: tl.constexpr, DEG: tl.constexpr, RMASK: tl.constexpr = False):
    """_scatter on a tile of EB hyperedges x BC columns (src_v3): the same products and atomic row adds (the head sum of
    a row may round differently: the atomics already make the order data dependent); RMASK as in _scatter."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < Eh
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    cm = cols < BF
    hh = tl.arange(0, H)
    z = tl.load(src + (hh[None, :, None].to(tl.int64) * Eh + e[:, None, None]) * BF + cols[None, None, :],
                mask=me[:, None, None] & cm[None, None, :], other=0.0)
    for s in tl.static_range(S):
        n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
        w = tl.load(wgt + (e[:, None] * S + s) * H + hh[None, :], mask=me[:, None], other=0.0)
        if DEG:
            w = w / tl.load(deg + n, mask=me, other=1.0)[:, None]
        ok = me
        if RMASK:
            ok = me & (tl.load(rmask + n, mask=me, other=1) == 0)
        tl.atomic_add(out + n[:, None] * BF + cols[None, :], tl.sum(w[:, :, None] * z, axis=1), mask=ok[:, None] & cm[None, :])


TILED = os.environ.get('OPL_V3_HTILE', '1') == '1'
# (EB hyperedges, BC columns, warps) per program, measured on the RTX 5090 (scripts/t_hyper.py, plate cell, F = 32):
# one program per hyperedge with 512-wide column blocks leaves 15/16 of the lanes idle at B = 1 (B * F = 32)
_TILE_G = {32: (8, 32, 4), 128: (4, 128, 4)}
_TILE_S = {32: (8, 32, 2), 128: (4, 128, 4)}


def _tile(table, BF):
    for lim in sorted(table):
        if BF <= lim:
            return table[lim]
    return None


def _grid(Eh, BF, blk):
    return (Eh, triton.cdiv(BF, blk))


def gather(src, hn, w, deg, H, blk, use_deg):
    """src (N, BF) -> (H, Eh, BF)."""
    Eh, S = hn.shape
    BF = src.shape[1]
    out = torch.empty((H, Eh, BF), dtype=torch.float32, device=src.device)
    if Eh == 0 or BF == 0:                                                  # empty hyperedge set (e.g. no fringe)
        return out
    if HK.usable(H, S, BF, w, hn) and src.is_contiguous():                 # src_v3 P16: CUDA kernel
        return HK.gather(src, hn, w, deg, out, use_deg)
    t = _tile(_TILE_G, BF) if TILED else None
    if t is not None:
        EB, BC, nw = t
        _gather_t[(triton.cdiv(Eh, EB), triton.cdiv(BF, BC))](src, hn, w, deg, out, Eh, BF, H=H, S=S, EB=EB, BC=BC,
                                                              DEG=use_deg, num_warps=nw)
        return out
    _gather[_grid(Eh, BF, blk)](src, hn, w, deg, out, Eh, BF, H=H, S=S, BLK=blk, DEG=use_deg)
    return out


def scatter(src, hn, w, deg, N, blk, use_deg, init=None, rmask=None, inplace=False):
    """src (H, Eh, BF) -> (N, BF); init (N, BF): the scatter adds onto a copy of init instead of zeros (a fused residual
    init + S Z, the same sum in another order). src_v3 S24: inplace (adds onto init itself, a fresh tensor of the caller),
    rmask (uint8 per row: rows != 0 receive nothing)."""
    H, Eh, BF = src.shape
    if init is None:
        out = torch.zeros((N, BF), dtype=torch.float32, device=src.device)
    else:
        out = init.reshape(N, BF) if inplace else init.reshape(N, BF).clone()
    if Eh == 0 or BF == 0:
        return out
    if HK.usable(H, hn.shape[1], BF, w, hn) and src.is_contiguous():       # src_v3 P16: CUDA kernel
        return HK.scatter(src, hn, w, deg, out, rmask, use_deg)
    rm = rmask if rmask is not None else out
    t = _tile(_TILE_S, BF) if TILED else None
    if t is not None:
        EB, BC, nw = t
        _scatter_t[(triton.cdiv(Eh, EB), triton.cdiv(BF, BC))](src, hn, w, deg, out, Eh, BF, rm, H=H, S=hn.shape[1], EB=EB,
                                                               BC=BC, DEG=use_deg, RMASK=rmask is not None, num_warps=nw)
        return out
    _scatter[_grid(Eh, BF, blk)](src, hn, w, deg, out, Eh, BF, rm, H=H, S=hn.shape[1], BLK=blk, DEG=use_deg,
                                 RMASK=rmask is not None)
    return out


class FusedHyperFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, a, b, W, deg, P, blk):
        N, B, F = X.shape
        H, Eh = P['H'], P['Eh']
        hn = P['hn'].contiguous()
        Xf = X.reshape(N, B * F).contiguous()
        ac, bc = a.detach().contiguous(), b.detach().contiguous()
        Z = gather(Xf, hn, ac, deg, H, blk, False)                                 # H x Eh x BF
        Zp = torch.bmm(Z.view(H, Eh * B, F), W).view(H, Eh, B * F)
        dX = scatter(Zp, hn, bc, deg, N, blk, True).view(N, B, F)
        ctx.save_for_backward(Xf, ac, bc, W, deg, Z, Zp)
        ctx.hn, ctx.shape, ctx.blk = hn, (N, B, F, H, Eh), blk
        return dX

    @staticmethod
    def backward(ctx, dY):
        Xf, a, b, W, deg, Z, Zp = ctx.saved_tensors
        N, B, F, H, Eh = ctx.shape
        hn, blk = ctx.hn, ctx.blk
        dYf = dY.reshape(N, B * F).contiguous()
        dZp = gather(dYf, hn, b, deg, H, blk, True)                                # H x Eh x BF
        dW = SL._gram(Z.view(H, Eh * B, F), dZp.view(H, Eh * B, F))
        dZ = torch.bmm(dZp.view(H, Eh * B, F), W.transpose(1, 2)).view(H, Eh, B * F).contiguous()
        dX = torch.zeros((N, B * F), dtype=torch.float32, device=dY.device)
        dA = torch.zeros_like(a); dB = torch.zeros_like(b)
        if Eh and B * F:
            _backward[_grid(Eh, B * F, blk)](Xf, dYf, dZ, Zp, hn, a, deg, dX, dA, dB, Eh, B * F, H=H, S=hn.shape[1], BLK=blk)
        return dX.view(N, B, F), dA, dB, dW, None, None, None


BLK = int(os.environ.get('FUSED_BLK', '512'))   # columns (of B*F) per program; 512 measured fastest (t_fused, RTX 5090)


def hyper(X, a, b, W, deg, P, blk=None):
    return FusedHyperFn.apply(X, a, b, W, deg, P, blk or BLK)
