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
def _scatter(src, hn, wgt, deg, out, Eh, BF, H: tl.constexpr, S: tl.constexpr, BLK: tl.constexpr, DEG: tl.constexpr):
    """out[hn[e, s], c] += sum_h w[e, s, h] src[h, e, c]."""
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
        tl.atomic_add(out + n * BF + cols, tl.sum(w[:, None] * z, axis=0), mask=cm)


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


def _grid(Eh, BF, blk):
    return (Eh, triton.cdiv(BF, blk))


def gather(src, hn, w, deg, H, blk, use_deg):
    """src (N, BF) -> (H, Eh, BF)."""
    Eh, S = hn.shape
    BF = src.shape[1]
    out = torch.empty((H, Eh, BF), dtype=torch.float32, device=src.device)
    _gather[_grid(Eh, BF, blk)](src, hn, w, deg, out, Eh, BF, H=H, S=S, BLK=blk, DEG=use_deg)
    return out


def scatter(src, hn, w, deg, N, blk, use_deg):
    """src (H, Eh, BF) -> (N, BF)."""
    H, Eh, BF = src.shape
    out = torch.zeros((N, BF), dtype=torch.float32, device=src.device)
    _scatter[_grid(Eh, BF, blk)](src, hn, w, deg, out, Eh, BF, H=H, S=hn.shape[1], BLK=blk, DEG=use_deg)
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
        _backward[_grid(Eh, B * F, blk)](Xf, dYf, dZ, Zp, hn, a, deg, dX, dA, dB, Eh, B * F, H=H, S=hn.shape[1], BLK=blk)
        return dX.view(N, B, F), dA, dB, dW, None, None, None


BLK = int(os.environ.get('FUSED_BLK', '128'))                              # columns (of B*F) per program


def hyper(X, a, b, W, deg, P, blk=None):
    return FusedHyperFn.apply(X, a, b, W, deg, P, blk or BLK)
