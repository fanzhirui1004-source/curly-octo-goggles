"""Fewer kernel launches in the network map and its adjoint (src_v3 S24, OPL_V3_NETFUSE; fastnet.FastNet).

The two-stream PCG matvec is bound by the CPU's kernel launches (~1400 per S_hat; removing ~150 with the concatenated
ghost-face product made the 51-cell plate's solve 7 % faster). Three groups of the network's launches are merged here,
each with the same float32 operations on the same values (no fused multiply-add):
  - forward layer + port clamp: where(port, X0, X + S Z) = where(port, X0, X) + S Z on the other rows only; the scatter
    adds onto the clamped copy and skips the port rows (fused_hyper.scatter rmask): 2 launches instead of 3;
  - adjoint layer masks: X0b += where(port, Xb, 0) and A = where(port, 0, Xb) in one kernel (mask_split), the layer's
    scatter adds onto A in place: 1 + 3 launches instead of 3 + 4;
  - U-Net convolutions: the dense grid input written in one kernel (zeros + index_copy_, and the gate product of the
    transposed convolution, before), the output Xc + v[dense_idx] (* gate) in one kernel (index, mul, add before);
    the cuDNN call is unchanged (same input layout and values): 2 + conv launches instead of 5 + conv.
Default on (OPL_V3_NETFUSE=0: the original paths). Checked on a full plate cell (scripts/t_netfuse.py, results/tests15/
netfuse2.log): with torch.backends.cudnn.deterministic every convolution (48) is bitwise equal to the original path (by
default the transposed convolutions are not reproducible run to run in either path: cuDNN's algorithm); the layers
with atomics differ from the original as the original from itself; S_hat 20.6 -> 19.7 ms (B = 1), 40.9 -> 37.1 ms
(B = 3); 51-cell plate with S21, S22 (v3p): 270.8 -> 260.3 s, C within 8e-9."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mask_split(xb, x0b, a, pm, total, C: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    r = k // C
    x = tl.load(xb + k, mask=m, other=0.0)
    p = tl.load(pm + r, mask=m, other=0) != 0
    y0 = tl.load(x0b + k, mask=m & p, other=0.0)
    tl.store(x0b + k, y0 + x, mask=m & p)
    tl.store(a + k, tl.where(p, 0.0, x), mask=m)


def mask_split(Xb, X0b, pm8):
    """X0b[port rows] += Xb[port rows] (in place); returns A = Xb with the port rows zeroed (a new tensor). Xb, X0b
    (N, B, F) float32 contiguous, pm8 (N,) uint8."""
    N = Xb.shape[0]
    C = Xb.numel() // max(N, 1)
    A = torch.empty_like(Xb)
    total = Xb.numel()
    if total:
        _mask_split[(triton.cdiv(total, 1024),)](Xb, X0b, A, pm8, total, C=C, BLOCK=1024, num_warps=4)
    return A


@triton.jit
def _mask_split_ip(xb, x0b, pidx, total, C: tl.constexpr, BLOCK: tl.constexpr):
    """src_v3 P12: over the port rows only: x0b[r] += xb[r]; xb[r] = 0 (xb is then imk(xb), in place)."""
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    q = k // C
    r = tl.load(pidx + q, mask=m, other=0).to(tl.int64)
    o = r * C + (k - q * C)
    x = tl.load(xb + o, mask=m, other=0.0)
    y0 = tl.load(x0b + o, mask=m, other=0.0)
    tl.store(x0b + o, y0 + x, mask=m)
    tl.store(xb + o, tl.zeros_like(x), mask=m)


def mask_split_inplace(Xb, X0b, pidx):
    """mask_split for an Xb the caller does not read again: X0b += pmk(Xb) and Xb's port rows zeroed in place (the same
    values as the copy mask_split returns), touching the port rows only (pidx: int32 port-row indices)."""
    N = Xb.shape[0]
    C = Xb.numel() // N
    total = len(pidx) * C
    if total:
        _mask_split_ip[(triton.cdiv(total, 1024),)](Xb, X0b, pidx, total, C=C, BLOCK=1024, num_warps=4)
    return Xb


@triton.jit
def _conv_in(xc, inv, g, dense, total, sg, B: tl.constexpr, F: tl.constexpr, GATE: tl.constexpr, BLOCK: tl.constexpr):
    """dense[j, b, f] = xc[inv[j], b, f] (* g[inv[j], f] with GATE; g rows sg apart), 0 where inv[j] < 0."""
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    j = k // (B * F)
    bf = k - j * (B * F)
    f = bf % F
    i = tl.load(inv + j, mask=m, other=-1).to(tl.int64)
    ok = m & (i >= 0)
    v = tl.load(xc + i * (B * F) + bf, mask=ok, other=0.0)
    if GATE:
        v = v * tl.load(g + i * sg + f, mask=ok, other=0.0)
    tl.store(dense + k, tl.where(ok, v, 0.0), mask=m)


@triton.jit
def _conv_out(xc, v, idx, g, out, total, sb, sf, sj, sg, B: tl.constexpr, F: tl.constexpr, GATE: tl.constexpr,
              BLOCK: tl.constexpr):
    """out[i, b, f] = xc[i, b, f] + v[b, f, idx[i]] (* g[i, f] with GATE; g rows sg apart): v by its strides."""
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    i = k // (B * F)
    bf = k - i * (B * F)
    b = bf // F
    f = bf - b * F
    j = tl.load(idx + i, mask=m, other=0).to(tl.int64)
    val = tl.load(v + b.to(tl.int64) * sb + f.to(tl.int64) * sf + j * sj, mask=m, other=0.0)
    if GATE:
        val = val * tl.load(g + i.to(tl.int64) * sg + f, mask=m, other=0.0)
    tl.store(out + k, tl.load(xc + k, mask=m, other=0.0) + val, mask=m)


def conv_in(Xc, inv, m3, g=None):
    """The dense (m3, B, F) grid of fastnet._conv: rows dense_idx get Xc (times the gate column g (N, F), unit column
    stride, if given)."""
    N, B, F = Xc.shape
    dense = torch.empty((m3, B, F), dtype=Xc.dtype, device=Xc.device)
    total = dense.numel()
    if total:
        assert g is None or g.stride(1) == 1
        _conv_in[(triton.cdiv(total, 1024),)](Xc, inv, g if g is not None else Xc, dense, total,
                                               g.stride(0) if g is not None else 0, B=B, F=F,
                                               GATE=g is not None, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return dense


def conv_out(Xc, v3, idx, g=None):
    """Xc + v3.permute(2, 0, 1)[idx] (* g[:, None, :]) for v3 = conv output reshaped (B, F, m3), by its strides."""
    N, B, F = Xc.shape
    out = torch.empty_like(Xc)
    total = out.numel()
    if total:
        assert g is None or g.stride(1) == 1
        sb, sf, sj = v3.stride()
        _conv_out[(triton.cdiv(total, 1024),)](Xc, v3, idx, g if g is not None else Xc, out, total, sb, sf, sj,
                                                g.stride(0) if g is not None else 0, B=B, F=F,
                                                GATE=g is not None, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return out
