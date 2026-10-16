"""One kernel per hyperedge-layer application in deployment (src_v3, OPL_V3_HFUSE=1; H = 4 heads).

fused_hyper runs a layer as three steps: a gather kernel Z[h, e] = sum_s wg[e, s, h] X[hn[e, s]], the channel mix
Z'_h = Z_h W_h (cuBLAS bmm) and a scatter kernel out[hn[e, s]] += sum_h ws[e, s, h] Z'_h[e] (float atomics), with Z and
Z' written to and read back from device memory. Here one program takes EB hyperedges through all three steps in
registers (the gather in the same order as fused_hyper, the channel mix as an IEEE float32 dot, the head sum before the
atomic add as before), so Z and Z' never reach memory and two of the three launches and the bmm disappear.
Forward with the residual folded in (fwd_res: out = X + S(b / deg) (G(a) X) W) and the adjoint (adj_res:
out = Y + G(a)^T (S(b / deg)^T Y) W^T). The channel mix is summed in another order than cuBLAS: results agree to float32
rounding (the scatter's atomics already make the order data dependent)."""
import os
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_V3_HFUSE', '0') == '1'
EB = int(os.environ.get('OPL_V3_HFUSE_EB', '16'))
NW = int(os.environ.get('OPL_V3_HFUSE_NW', '4'))


@triton.jit
def _layer4(X, hn, WG, WS, deg, W, out, Eh, NBF, S: tl.constexpr, F: tl.constexpr, NB: tl.constexpr, EB: tl.constexpr,
            DEG_G: tl.constexpr, DEG_S: tl.constexpr):
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < Eh
    f = tl.arange(0, F)
    W0 = tl.load(W + 0 * F * F + f[:, None] * F + f[None, :])
    W1 = tl.load(W + 1 * F * F + f[:, None] * F + f[None, :])
    W2 = tl.load(W + 2 * F * F + f[:, None] * F + f[None, :])
    W3 = tl.load(W + 3 * F * F + f[:, None] * F + f[None, :])
    for bb in tl.static_range(NB):
        a0 = tl.zeros((EB, F), tl.float32)
        a1 = tl.zeros((EB, F), tl.float32)
        a2 = tl.zeros((EB, F), tl.float32)
        a3 = tl.zeros((EB, F), tl.float32)
        for s in tl.static_range(S):
            n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
            x = tl.load(X + n[:, None] * NBF + bb * F + f[None, :], mask=me[:, None], other=0.0)
            base = (e * S + s) * 4
            w0 = tl.load(WG + base, mask=me, other=0.0)
            w1 = tl.load(WG + base + 1, mask=me, other=0.0)
            w2 = tl.load(WG + base + 2, mask=me, other=0.0)
            w3 = tl.load(WG + base + 3, mask=me, other=0.0)
            if DEG_G:
                dg = tl.load(deg + n, mask=me, other=1.0)
                w0 = w0 / dg
                w1 = w1 / dg
                w2 = w2 / dg
                w3 = w3 / dg
            a0 += w0[:, None] * x
            a1 += w1[:, None] * x
            a2 += w2[:, None] * x
            a3 += w3[:, None] * x
        z0 = tl.dot(a0, W0, input_precision="ieee")
        z1 = tl.dot(a1, W1, input_precision="ieee")
        z2 = tl.dot(a2, W2, input_precision="ieee")
        z3 = tl.dot(a3, W3, input_precision="ieee")
        for s in tl.static_range(S):
            n = tl.load(hn + e * S + s, mask=me, other=0).to(tl.int64)
            base = (e * S + s) * 4
            w0 = tl.load(WS + base, mask=me, other=0.0)
            w1 = tl.load(WS + base + 1, mask=me, other=0.0)
            w2 = tl.load(WS + base + 2, mask=me, other=0.0)
            w3 = tl.load(WS + base + 3, mask=me, other=0.0)
            if DEG_S:
                dg = tl.load(deg + n, mask=me, other=1.0)
                w0 = w0 / dg
                w1 = w1 / dg
                w2 = w2 / dg
                w3 = w3 / dg
            v = w0[:, None] * z0 + w1[:, None] * z1 + w2[:, None] * z2 + w3[:, None] * z3
            tl.atomic_add(out + n[:, None] * NBF + bb * F + f[None, :], v, mask=me[:, None])


def usable(h, X):
    return ON and h.H == 4 and X.dim() == 3 and X.shape[1] <= 4 and X.shape[2] in (16, 32, 64) and h.a is not None


def layer_res(h, X, adjoint):
    """X (N, B, F) float32 -> X + layer(X) (adjoint=False, as _Hyper.fwd_res) or X + layer^T(X) (as _Hyper.adj_res)."""
    N, B, F = X.shape
    Xf = X.reshape(N, B * F).contiguous()
    out = Xf.clone()
    if h.Eh:
        WG, WS, Wm = (h.b, h.a, h.Wt) if adjoint else (h.a, h.b, h.W)
        _layer4[(triton.cdiv(h.Eh, EB),)](Xf, h.hn, WG, WS, h.deg, Wm, out, h.Eh, B * F, S=h.hn.shape[1], F=F, NB=B, EB=EB,
                                          DEG_G=adjoint, DEG_S=not adjoint, num_warps=NW)
    return out.view(N, B, F)
