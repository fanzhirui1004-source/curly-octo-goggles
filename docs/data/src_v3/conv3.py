"""The U-Net's 3x3x3 convolution (padding 1, F -> F channels) as one Triton implicit-GEMM kernel on the dense grid layout
FastNet._conv builds (src_v3 P10, OPL_V3_CONV3): input and output (m^3, B, F) (a voxel's F channels contiguous), weight
(F_out, F_in, 3, 3, 3) as torch.nn.functional.conv3d (cross-correlation). Per program: BM voxels of one right-hand
side, all F output channels; the 27 taps one after another, each an [BM, F] x [F, F] product in float32 ('ieee': no
TF32) accumulated in registers; taps outside the grid read zeros. cuDNN picks implicit_convolveNd_sgemm here (~89 us
per convolution at B = 1); the sums are the same, in another order (rounding of the float32 sums)."""
import os
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_V3_CONV3', '0') == '1'
BM = int(os.environ.get('OPL_V3_CONV3_BM', '32'))
WARPS = int(os.environ.get('OPL_V3_CONV3_WARPS', '8'))


@triton.jit
def _conv3(x_ptr, w_ptr, y_ptr, m, B, F: tl.constexpr, BM: tl.constexpr):
    v = tl.program_id(0) * BM + tl.arange(0, BM)
    b = tl.program_id(1)
    m3 = m * m * m
    mv = v < m3
    z = v // (m * m)
    yy = (v // m) % m
    xx = v % m
    f = tl.arange(0, F)
    acc = tl.zeros((BM, F), dtype=tl.float32)
    for k in tl.static_range(27):
        z2 = z + (k // 9 - 1)
        y2 = yy + ((k // 3) % 3 - 1)
        x2 = xx + (k % 3 - 1)
        ok = mv & (z2 >= 0) & (z2 < m) & (y2 >= 0) & (y2 < m) & (x2 >= 0) & (x2 < m)
        nv = ((z2 * m + y2) * m + x2).to(tl.int64)
        X = tl.load(x_ptr + (nv[:, None] * B + b) * F + f[None, :], mask=ok[:, None], other=0.0)          # [BM, F_in]
        Wk = tl.load(w_ptr + (k * F + f[:, None]) * F + f[None, :])                                        # [F_in, F_out]
        acc = tl.dot(X, Wk, acc, input_precision='ieee')
    tl.store(y_ptr + (v[:, None].to(tl.int64) * B + b) * F + f[None, :], acc, mask=mv[:, None])


_WT = {}


def _taps(W):
    """(F_out, F_in, 3, 3, 3) -> (27, F_in, F_out) contiguous, cached per weight tensor (shared by all cells)."""
    key = (W.data_ptr(), tuple(W.shape))
    t = _WT.get(key)
    if t is None:
        t = _WT[key] = W.permute(2, 3, 4, 1, 0).reshape(27, W.shape[1], W.shape[0]).contiguous()
    return t


def ready(W):
    """The weight's tap layout is cached (made here outside a CUDA-graph capture; during a capture an uncached weight
    falls back to cuDNN, so no cache tensor is allocated in a graph's pool)."""
    if (W.data_ptr(), tuple(W.shape)) in _WT:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    _taps(W)
    return True


def conv3d(dense, W, m, bm=None, warps=None):
    """dense (m^3, B, F) float32 contiguous -> conv3d(., W, padding=1) in the same (m^3, B, F) layout."""
    m3, B, F = dense.shape
    out = torch.empty_like(dense)
    _conv3[(triton.cdiv(m3, bm or BM), B)](dense, _taps(W), out, m, B, F=F, BM=bm or BM, num_warps=warps or WARPS)
    return out


@triton.jit
def _conv3s(xc_ptr, inv_ptr, didx_ptr, w_ptr, g_ptr, out_ptr, n, m, B, sg, F: tl.constexpr, BM: tl.constexpr,
            GIN: tl.constexpr, GOUT: tl.constexpr):
    """Sparse fused form (P10b): for active voxels a only (the outputs FastNet keeps),
        out[a, b, :] = xc[a, b, :] + (sum_taps X_tap W_tap) (* g[a, :] with GOUT),
    X_tap = xc[inv[neighbour of didx[a]], b, :] (* g[that row, :] with GIN), zero outside the grid or at inactive voxels.
    The dense grid is never formed (netfused.conv_in / conv_out and the inactive voxels' outputs are not computed)."""
    a = tl.program_id(0) * BM + tl.arange(0, BM)
    ma = a < n
    b = tl.program_id(1)
    v = tl.load(didx_ptr + a, mask=ma, other=0).to(tl.int32)
    z = v // (m * m)
    yy = (v // m) % m
    xx = v % m
    f = tl.arange(0, F)
    acc = tl.zeros((BM, F), dtype=tl.float32)
    for k in tl.static_range(27):
        z2 = z + (k // 9 - 1)
        y2 = yy + ((k // 3) % 3 - 1)
        x2 = xx + (k % 3 - 1)
        ok = ma & (z2 >= 0) & (z2 < m) & (y2 >= 0) & (y2 < m) & (x2 >= 0) & (x2 < m)
        an = tl.load(inv_ptr + (z2 * m + y2) * m + x2, mask=ok, other=-1)
        ok = ok & (an >= 0)
        an64 = an.to(tl.int64)
        X = tl.load(xc_ptr + (an64[:, None] * B + b) * F + f[None, :], mask=ok[:, None], other=0.0)
        if GIN:
            X = X * tl.load(g_ptr + an64[:, None] * sg + f[None, :], mask=ok[:, None], other=0.0)
        Wk = tl.load(w_ptr + (k * F + f[:, None]) * F + f[None, :])
        acc = tl.dot(X, Wk, acc, input_precision='ieee')
    a64 = a.to(tl.int64)
    if GOUT:
        acc = acc * tl.load(g_ptr + a64[:, None] * sg + f[None, :], mask=ma[:, None], other=0.0)
    base = (a64[:, None] * B + b) * F + f[None, :]
    tl.store(out_ptr + base, tl.load(xc_ptr + base, mask=ma[:, None], other=0.0) + acc, mask=ma[:, None])


BMS = int(os.environ.get('OPL_V3_CONV3S_BM', '32'))
WARPS_S = int(os.environ.get('OPL_V3_CONV3S_WARPS', '8'))


def conv_sparse(Xc, inv, didx, W, m, g=None, gate_in=False, bm=None, warps=None):
    """FastNet._conv in one kernel: Xc (n, B, F) active-voxel features, inv (m^3,) int32 voxel -> active index or -1,
    didx (n,) active -> voxel, W (F, F, 3, 3, 3) (the flipped weight for the transposed convolution), g (n, F) gate
    (unit column stride): gate_in multiplies the inputs (transposed form), else the convolution output."""
    n, B, F = Xc.shape
    out = torch.empty_like(Xc)
    if n == 0:
        return out
    assert Xc.is_contiguous() and (g is None or g.stride(1) == 1)
    _conv3s[(triton.cdiv(n, bm or BMS), B)](Xc, inv, didx, _taps(W), g if g is not None else Xc, out, n, m, B,
                                           g.stride(0) if g is not None else 0, F=F, BM=bm or BMS,
                                           GIN=g is not None and gate_in, GOUT=g is not None and not gate_in,
                                           num_warps=warps or WARPS_S, enable_fp_fusion=False)
    return out
