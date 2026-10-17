"""One hyperedge layer in one kernel (src_v3 P8, OPL_V3_HFUSE): fused_hyper's gather, the per-head channel mix and the
scatter, for a tile of EB hyperedges and one column block (one of the B right-hand sides, its F channels):

    Z[e, h, f]  = sum_s ga[e, s, h] X[hn[e, s], b, f]            (ga / deg[hn] with GDEG)        -- as fused_hyper.gather
    Z'[e, h, g] = sum_f Z[e, h, f] W[h, f, g]                                                    -- torch.bmm before
    out[hn[e, s], b, g] += sum_h sb[e, s, h] Z'[e, h, g]          (sb / deg[hn] with SDEG; rows with rmask != 0 skipped)

Z and Z' stay in registers: the (H, Eh, B F) intermediates are neither written nor read again, and one launch replaces
three. The gather sums in fused_hyper's order (the same Z); the channel mix sums over f in FC-wide chunks (cuBLAS's
order differs: rounding of the float32 sums); the scatter is fused_hyper's (head sum, then one float atomic add per
(e, s) row, data-dependent order as before)."""
import os
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_V3_HFUSE', '0') == '1'


@triton.jit
def _layer(X, hn, ga, sb, deg, W, out, rmask, Eh, BF, F: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
           EB: tl.constexpr, FC: tl.constexpr, GDEG: tl.constexpr, SDEG: tl.constexpr, RMASK: tl.constexpr):
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < Eh
    c0 = tl.program_id(1) * F                                               # column block: right-hand side b
    hh = tl.arange(0, H)
    g = tl.arange(0, F)
    mixed = tl.zeros((EB, H, F), dtype=tl.float32)
    for fc in tl.static_range(F // FC):
        f = fc * FC + tl.arange(0, FC)
        acc = tl.zeros((EB, H, FC), dtype=tl.float32)
        for s in range(S):                                                  # runtime loop (compile time)
            n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
            w = tl.load(ga + (e[:, None] * S + s) * H + hh[None, :], mask=me[:, None], other=0.0)
            if GDEG:
                w = w / tl.load(deg + n, mask=me, other=1.0)[:, None]
            x = tl.load(X + n[:, None] * BF + c0 + f[None, :], mask=me[:, None], other=0.0)
            acc += w[:, :, None] * x[:, None, :]
        Wc = tl.load(W + (hh[:, None, None] * F + f[None, :, None]) * F + g[None, None, :])          # [H, FC, F]
        mixed += tl.sum(acc[:, :, :, None] * Wc[None, :, :, :], axis=2)
    for s in range(S):
        n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
        w = tl.load(sb + (e[:, None] * S + s) * H + hh[None, :], mask=me[:, None], other=0.0)
        if SDEG:
            w = w / tl.load(deg + n, mask=me, other=1.0)[:, None]
        ok = me
        if RMASK:
            ok = me & (tl.load(rmask + n, mask=me, other=1) == 0)
        tl.atomic_add(out + n[:, None] * BF + c0 + g[None, :], tl.sum(w[:, :, None] * mixed, axis=1), mask=ok[:, None])


EB = int(os.environ.get('OPL_V3_HFUSE_EB', '4'))
FC = int(os.environ.get('OPL_V3_HFUSE_FC', '8'))
WARPS = int(os.environ.get('OPL_V3_HFUSE_WARPS', '4'))


def layer(X, hn, ga, sb, deg, W, out, gdeg, sdeg, rmask=None, eb=None, fc=None, warps=None):
    """out (N, B F) += the layer applied to X (N, B F); hn (Eh, S) node ids, ga / sb (Eh, S, H) gather / scatter
    weights, deg (N,), W (H, F, F); gdeg / sdeg: divide the gather / scatter weights by deg[node]; rmask (N,) uint8."""
    N, BF = X.shape
    H, F = W.shape[0], W.shape[1]
    Eh, S = hn.shape
    if Eh == 0 or BF == 0:
        return out
    eb, fc, warps = eb or EB, fc or FC, warps or WARPS
    _layer[(triton.cdiv(Eh, eb), BF // F)](X, hn, ga, sb, deg, W, out, rmask if rmask is not None else out, Eh, BF,
                                           F=F, H=H, S=S, EB=eb, FC=fc, GDEG=gdeg, SDEG=sdeg, RMASK=rmask is not None,
                                           num_warps=warps)
    return out
