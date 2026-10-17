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
BM = int(os.environ.get('OPL_V3_CONV3_BM', '64'))
WARPS = int(os.environ.get('OPL_V3_CONV3_WARPS', '4'))


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
