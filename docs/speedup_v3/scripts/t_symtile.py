"""Tests of src_v3 kernels: symtile (packed symmetric tiles) and fastidx (Triton row gather)."""
import os, sys, time
sys.path.insert(0, os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'))
import torch
import symtile as ST
import fastidx as FI

dev = torch.device('cuda')
torch.manual_seed(0)
for n in (5, 64, 65, 1000, 9276):
    G = torch.randn((n, n), dtype=torch.float64, device=dev)
    A64 = G @ G.T / n + torch.eye(n, dtype=torch.float64, device=dev)
    A = A64.to(torch.float32)
    P = ST.pack(A)
    As = torch.tril(A) + torch.tril(A, -1).T
    assert torch.equal(ST.unpack(P, n), As), n
    for B in (1, 2, 3, 6, 16, 17, 32):
        X = torch.randn((n, B), dtype=torch.float32, device=dev)
        Y = ST.matmul(P, n, X)
        ref = (As.double() @ X.double())
        err = float((Y.double() - ref).norm() / ref.norm())
        dense = As @ X
        errd = float((dense.double() - ref).norm() / ref.norm())
        assert err < 1e-6, (n, B, err)
    print('symtile n=%d ok: err %.2e (dense fp32 %.2e) packed MiB %.1f dense MiB %.1f' % (n, err, errd, P.numel() * 4 / 2**20, n * n * 4 / 2**20))
n = 9276
A = torch.randn((n, n), device=dev); A = A + A.T; P = ST.pack(A)
for B in (1, 3, 6, 16):
    X = torch.randn((n, B), device=dev)
    for f, name in ((lambda: ST.matmul(P, n, X), 'packed'), (lambda: A @ X, 'dense')):
        f(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            f()
        torch.cuda.synchronize(); print('B=%d %s %.3f ms' % (B, name, (time.perf_counter() - t) / 20 * 1e3))
FI.ON = True
for dtp in (torch.float32, torch.float64):
    for b in (2, 3, 6, 8, 16, 33):
        x = torch.randn((100000, b), dtype=dtp, device=dev)
        for idx in (torch.randint(0, 100000, (5000, 81), device=dev), torch.randint(0, 100000, (7, 135), device=dev, dtype=torch.int32)):
            FI.TRITON = True; y1 = FI.rows(x, idx)
            FI.TRITON = False; y0 = FI.rows(x, idx)
            assert torch.equal(y1, y0) and torch.equal(y1, x[idx.long()]), (dtp, b)
x = torch.randn((400000, 3), dtype=torch.float64, device=dev); idx = torch.randint(0, 400000, (11000, 81), device=dev)
for tri in (True, False):
    FI.TRITON = tri
    FI.rows(x, idx); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50):
        FI.rows(x, idx)
    torch.cuda.synchronize(); print('fastidx triton=%s %.3f ms' % (tri, (time.perf_counter() - t) / 50 * 1e3))
print('ALL_OK')
