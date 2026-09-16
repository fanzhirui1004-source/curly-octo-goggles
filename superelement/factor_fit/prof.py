#!/usr/bin/env python3
"""Where do the 11.6 s per E1-C step go?  Times the lifting apply at real scale, no teacher needed.

Hypothesis: torch.sparse.mm's backward w.r.t. the SPARSE operand's values is implemented as a
dense grad @ dense.T followed by a mask, which is d^3 = 2.1e12 flops in float64 -- while the
mathematically required cost is nnz * d = 2.8e9.  If so the backward, not the forward, is the bill.
"""
import torch, time
DEV = 'cuda'
d, nnz = 12792, 220_000
torch.manual_seed(0)
perm = torch.randperm(d, device=DEV)
Aset, Bset = perm[:d//2], perm[d//2:]
rows = Aset[torch.randint(0, d//2, (nnz,), device=DEV)]
cols = Bset[torch.randint(0, d//2, (nnz,), device=DEV)]
x = torch.randn(d, d, dtype=torch.float64, device=DEV)


def timeit(fn, n=3, warmup=1):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n


def sparse_fwd_bwd(backward):
    k = torch.zeros(nnz, dtype=torch.float64, device=DEV, requires_grad=True)
    diag = torch.arange(d, device=DEV)
    idx = torch.stack([torch.cat([diag, rows]), torch.cat([diag, cols])])
    val = torch.cat([torch.ones(d, device=DEV, dtype=torch.float64), k])
    S = torch.sparse_coo_tensor(idx, val, (d, d)).coalesce()
    y = torch.sparse.mm(S, x)
    if backward:
        y.sum().backward()
    return y


def chunked_fwd_bwd(backward, chunk=512):
    k = torch.zeros(nnz, dtype=torch.float64, device=DEV, requires_grad=True)
    y = x.clone()
    for j in range(0, d, chunk):
        s = slice(j, min(j + chunk, d))
        y[:, s] = y[:, s].index_add(0, rows, k.unsqueeze(1) * x[cols, s])
    if backward:
        y.sum().backward()
    return y


print('d = %d, nnz = %d, float64\n' % (d, nnz), flush=True)
print('sparse.mm  forward only        %8.3f s' % timeit(lambda: sparse_fwd_bwd(False)), flush=True)
print('sparse.mm  forward + backward  %8.3f s' % timeit(lambda: sparse_fwd_bwd(True)), flush=True)
print('chunked    forward only        %8.3f s' % timeit(lambda: chunked_fwd_bwd(False)), flush=True)
print('chunked    forward + backward  %8.3f s' % timeit(lambda: chunked_fwd_bwd(True)), flush=True)
# correctness: the two paths must agree
k = torch.full((nnz,), 0.01, dtype=torch.float64, device=DEV)
diag = torch.arange(d, device=DEV)
idx = torch.stack([torch.cat([diag, rows]), torch.cat([diag, cols])])
S = torch.sparse_coo_tensor(idx, torch.cat([torch.ones(d, device=DEV, dtype=torch.float64), k]), (d, d)).coalesce()
y1 = torch.sparse.mm(S, x)
y2 = x.clone()
for j in range(0, d, 512):
    s = slice(j, min(j + 512, d))
    y2[:, s] = y2[:, s].index_add(0, rows, k.unsqueeze(1) * x[cols, s])
print('\nagreement between paths: max abs diff %.3e' % float((y1 - y2).abs().max()), flush=True)
print('  (duplicate (row,col) pairs from random sampling are summed by BOTH paths, so this is a real check)')
