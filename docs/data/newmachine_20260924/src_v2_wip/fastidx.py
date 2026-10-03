"""Row gathers x[idx] for 2-D contiguous tensors through element gathers (torch.take) when OPL_FASTIDX=1 (default off).
The row-vectorised index kernel of this torch build runs at ~30 GB/s whenever a row is a multiple of 16 bytes (float32
with 4, 8, 12, ... columns, float64 with >= 2 columns); element gathers do not. Same values (a copy).
The flat element index of an index tensor is cached on that tensor object (per column count and device), so it lives as
long as the index tensor itself (a streamed cell's index tensors are new objects after every load)."""
import os
import torch

ON = os.environ.get('OPL_FASTIDX') == '1'
CACHE_MAX_COLS = 8


def rows(x, idx):
    if not ON or x.dim() != 2 or x.shape[1] == 1 or not x.is_contiguous():
        return x[idx]
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
