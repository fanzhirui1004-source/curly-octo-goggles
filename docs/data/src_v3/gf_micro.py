import torch, time, json
dev = torch.device('cuda'); f32 = torch.float32
def timed(fn, rep=20):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return round(1000 * (time.perf_counter() - t) / rep, 4)
F, nb = 7550, 323424
B54 = torch.randn(54, 135, dtype=f32, device=dev)
idx = torch.randint(0, nb, (F, 135), device=dev)
out = {}
for b in (4, 6, 8, 12):
    x = torch.randn(nb, b, dtype=f32, device=dev); y = torch.zeros_like(x)
    X = x[idx].permute(1, 0, 2).reshape(135, -1)
    Z = B54 @ X
    Y = B54.t() @ Z
    Yr = Y.reshape(135, F, b).permute(1, 0, 2).reshape(-1, b)
    out[b] = dict(gather=timed(lambda: x[idx]), permcopy=timed(lambda: x[idx].permute(1, 0, 2).reshape(135, -1)),
                  gemm1=timed(lambda: B54 @ X), gemm2=timed(lambda: B54.t() @ Z),
                  back=timed(lambda: Y.reshape(135, F, b).permute(1, 0, 2).reshape(-1, b)),
                  index_add=timed(lambda: y.index_add_(0, idx.reshape(-1), Yr)),
                  gemm_alt=timed(lambda: torch.matmul(x[idx].reshape(F, 135 * 0 + 135, b).transpose(1, 2), B54.t())))
print(json.dumps(out))
