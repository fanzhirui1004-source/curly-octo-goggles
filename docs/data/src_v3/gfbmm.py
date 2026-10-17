"""Masked row gather / scatter-add of the padded ghost-face DOF array (src_v3 S20, OPL_V3_GFBMM; teacher.GhostFaces).

With OPL_V3_GFBMM=1 the face DOF lists of the three face templates are one (3, 135, Fmax) int32 array padded with -1,
so a ghost-face product is four kernels instead of twelve (per template: a gather, two GEMMs, an index_add): the masked
gather below, two batched GEMMs (torch.bmm, every template at once; padded columns are zero and stay zero) and the
masked atomic scatter-add below. The two-stream PCG matvec is bound by the CPU's kernel launches, and the ghost-face
product is about a third of them (38 products per S_hat). The GEMMs are the same per template; the scatter's atomic
order differs (as index_add_'s already did)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(x, idx, out, total, B: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)                  # flat output element (row of idx, column)
    m = k < total
    r = k // B
    src = tl.load(idx + r, mask=m, other=-1).to(tl.int64)
    v = tl.load(x + src * B + (k - r * B), mask=m & (src >= 0), other=0.0)
    tl.store(out + k, v, mask=m)


@triton.jit
def _scatter(y, idx, src, total, B: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < total
    r = k // B
    dst = tl.load(idx + r, mask=m, other=-1).to(tl.int64)
    v = tl.load(src + k, mask=m, other=0.0)
    tl.atomic_add(y + dst * B + (k - r * B), v, mask=m & (dst >= 0))


def gather(x, idx):
    """x[idx] row by row (rows of x: x.shape[1] values; idx any shape, -1 -> zeros) -> (idx.numel(), b)."""
    b = x.shape[1]
    total = idx.numel() * b
    out = torch.empty((idx.numel(), b), dtype=x.dtype, device=x.device)
    if total:
        _gather[(triton.cdiv(total, 1024),)](x, idx, out, total, B=b, BLOCK=1024, num_warps=4)
    return out


def scatter_add(y, idx, src):
    """y[idx[i]] += src[i] (rows; idx -1 skipped; atomic float adds)."""
    b = y.shape[1]
    total = idx.numel() * b
    if total:
        _scatter[(triton.cdiv(total, 1024),)](y, idx, src.reshape(-1), total, B=b, BLOCK=1024, num_warps=4)
    return y
