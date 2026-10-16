"""Row gathers x[idx] for 2-D contiguous tensors (OPL_FASTIDX=1 / fastidx.ON; default off). Same values (a copy).

src_v2 route (OPL_FI_TRITON=0): element gathers through torch.take with a flat int64 element index cached on the index
tensor (per column count): the row-vectorised index kernel of this torch build runs at ~30 GB/s whenever a row is a
multiple of 16 bytes (float32 with 4, 8, 12, ... columns, float64 with >= 2 columns); element gathers do not. The caches
cost device memory (8 bytes per gathered element and column count, kept as long as the index tensor lives: e.g. ~360 MB
per large resident cell with B = 3) and are rebuilt after every load of a streamed cell.
src_v3 default (OPL_FI_TRITON=1): one Triton kernel gathers whole rows (int32 or int64 row index, any column count);
no cached index, no extra memory."""
import os
import torch

ON = os.environ.get('OPL_FASTIDX') == '1'
CACHE_MAX_COLS = 8
TRITON = os.environ.get('OPL_FI_TRITON', '1') == '1'

_k = None


def _kernel():
    global _k
    if _k is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _gather_flat(x_ptr, idx_ptr, out_ptr, total, B: tl.constexpr, BLOCK: tl.constexpr):
            k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)                    # flat output element (int32:
            m = k < total                                                         # total < 2^31, checked by the caller)
            r = k // B                                                            # constexpr B: no integer division
            src = tl.load(idx_ptr + r, mask=m, other=0).to(tl.int64)
            v = tl.load(x_ptr + src * B + (k - r * B), mask=m)
            tl.store(out_ptr + k, v, mask=m)
        _k = _gather_flat
    return _k


def _rows_triton(x, idx):
    import triton
    b = x.shape[1]
    flat = idx.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    total = flat.numel() * b
    if total >= 2 ** 31 - 4096:
        return x[idx]
    out = torch.empty((flat.numel(), b), dtype=x.dtype, device=x.device)
    if total:
        _kernel()[(triton.cdiv(total, 1024),)](x, flat, out, total, B=b, BLOCK=1024, num_warps=4)
    return out.view(*idx.shape, b)


def rows(x, idx):
    if not ON or x.dim() != 2 or x.shape[1] == 1 or not x.is_contiguous():
        return x[idx]
    if TRITON and x.is_cuda:
        return _rows_triton(x, idx)
    b = x.shape[1]
    if b > CACHE_MAX_COLS:                                              # wide blocks: no cached index (memory)
        return torch.take(x, (idx.reshape(-1, 1).long() * b + torch.arange(b, device=idx.device)).reshape(-1)).view(*idx.shape, b)
    cache = getattr(idx, '_fi_flat', None)
    flat = None if cache is None else cache.get(b)
    if flat is None:
        flat = (idx.reshape(-1, 1).long() * b + torch.arange(b, device=idx.device)).reshape(-1)
        try:
            if cache is None:
                cache = {}; idx._fi_flat = cache
            cache[b] = flat
        except (AttributeError, RuntimeError):
            pass
    return torch.take(x, flat).view(*idx.shape, b)
