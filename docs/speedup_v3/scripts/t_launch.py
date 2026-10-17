"""CPU cost per kernel launch on this stack (the two-stream PCG matvec is bound by the CPU's launches, ~1400 per S_hat):
a Triton kernel through JITFunction (kernel[grid](...)), the same kernel through its cached CompiledKernel, a torch
elementwise op, torch indexing x[idx], index_add_, a small mm, a small bmm. Each is issued N times without
synchronising (a large tensor-free queue: the GPU work per launch is tiny), timed on the CPU.
usage: python t_launch.py [--n 2000]"""
import time, json, argparse
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x + i, tl.load(x + i, mask=i < n, other=0.0) + 1.0, mask=i < n)


ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=2000)
a = ap.parse_args()
dev = 'cuda'
x = torch.zeros(256, device=dev)
idx = torch.randint(0, 256, (135, 40), device=dev)
B = torch.randn(54, 135, device=dev); X = torch.randn(135, 40, device=dev)
Bs = torch.randn(3, 54, 135, device=dev); Xs = torch.randn(3, 135, 40, device=dev)
y = torch.zeros(256, 1, device=dev); src = torch.randn(135 * 40, 1, device=dev)


def cpu_us(fn, n=a.n):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t
    torch.cuda.synchronize()
    return round(1e6 * dt / n, 2)


r = {}
r['triton_jit'] = cpu_us(lambda: _k[(1,)](x, 256, BLOCK=256))
try:
    ck = _k.warmup(x, 256, BLOCK=256, grid=(1,))
    r['triton_compiled'] = cpu_us(lambda: ck[(1, 1, 1)](x, 256))
except Exception as e:                                                    # noqa: BLE001 (API differences)
    r['triton_compiled'] = repr(e)[:200]
r['torch_add_'] = cpu_us(lambda: x.add_(1.0))
r['torch_index'] = cpu_us(lambda: x[idx])
r['torch_index_add_'] = cpu_us(lambda: y.index_add_(0, idx.reshape(-1), src))
r['torch_mm'] = cpu_us(lambda: B @ X)
r['torch_bmm'] = cpu_us(lambda: torch.bmm(Bs, Xs))
r['torch_zeros'] = cpu_us(lambda: torch.zeros(256, device=dev))
print(json.dumps(r), flush=True)
