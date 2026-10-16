"""Fused Chebyshev sweeps of the equilibrium correction (src_v3, OPL_V3_CHEB=1, deployment cells, no autograd).

trainlib.smooth_tail and trainlib._cheb run k Jacobi-preconditioned Chebyshev sweeps on the interior equilibrium; each
sweep is one float32 K product (teacher._Kfp32) plus about ten small float64 PyTorch kernels (dtype conversions of the
whole vector around the product, the interior gather, the recurrence update, an out-of-place scatter-add, a zero fill).
Here every sweep is the float32 K product on a float32 copy of the iterate that is kept up to date, and ONE Triton
kernel for everything else. The float64 recurrence is evaluated with the same operations in the same order as the
PyTorch expressions it replaces and without fused multiply-add (enable_fp_fusion=False), so every iterate is bitwise the
same as in src_v2:
  smooth_tail   d0 = (dinv r) * (1 / theta),              d <- (a1 d) + ((c2 dinv) r),       x_I <- x_I + d
  _cheb         d0 = (b - dinv y_I) * (1 / theta),        d <- (a1 d) + (c2 (b - dinv y_I)), z <- z + d
with r = -y_I, y = K x (float32 product of the float32-rounded iterate, as _Kfp32 does), a1 = rho_n rho and
c2 = 2 rho_n / delta (Python floats, as before). A CUDA tensor divided by a Python scalar is computed by PyTorch as the
product with the scalar's float64 reciprocal (BinaryDivTrueKernel: 'a * reciprocal(b)'), hence the 1 / theta."""
import torch
import triton
import triton.language as tl


@triton.jit
def _tail_step(y_ptr, I_ptr, dinv_ptr, d_ptr, x_ptr, x32_ptr, co_ptr, ni, B,
               INIT: tl.constexpr, BI: tl.constexpr, BB: tl.constexpr):
    # co_ptr: float64 (1 / theta) for INIT, (a1, c2) otherwise (Triton would pass Python floats as float32)
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    mi = i < ni
    cc = tl.arange(0, BB)
    m = mi[:, None] & (cc < B)[None, :]
    rows = tl.load(I_ptr + i, mask=mi, other=0).to(tl.int64)
    g = rows[:, None] * B + cc[None, :]
    li = i[:, None].to(tl.int64) * B + cc[None, :]
    r = -(tl.load(y_ptr + g, mask=m, other=0.0).to(tl.float64))
    dv = tl.load(dinv_ptr + i, mask=mi, other=1.0)[:, None]
    if INIT:
        inv = tl.load(co_ptr)
        dn = (dv * r) * inv
    else:
        a1 = tl.load(co_ptr)
        c2 = tl.load(co_ptr + 1)
        d = tl.load(d_ptr + li, mask=m, other=0.0)
        dn = (a1 * d) + ((c2 * dv) * r)
    tl.store(d_ptr + li, dn, mask=m)
    xn = tl.load(x_ptr + g, mask=m, other=0.0) + dn
    tl.store(x_ptr + g, xn, mask=m)
    tl.store(x32_ptr + g, xn.to(tl.float32), mask=m)


@triton.jit
def _cheb_step(y_ptr, I_ptr, dinv_ptr, d_ptr, z_ptr, b_ptr, f32_ptr, co_ptr, ni, B,
               INIT: tl.constexpr, BI: tl.constexpr, BB: tl.constexpr):
    i = tl.program_id(0) * BI + tl.arange(0, BI)
    mi = i < ni
    cc = tl.arange(0, BB)
    m = mi[:, None] & (cc < B)[None, :]
    rows = tl.load(I_ptr + i, mask=mi, other=0).to(tl.int64)
    g = rows[:, None] * B + cc[None, :]
    li = i[:, None].to(tl.int64) * B + cc[None, :]
    dv = tl.load(dinv_ptr + i, mask=mi, other=1.0)[:, None]
    Az = dv * tl.load(y_ptr + g, mask=m, other=0.0).to(tl.float64)
    t = tl.load(b_ptr + li, mask=m, other=0.0) - Az
    if INIT:
        inv = tl.load(co_ptr)
        dn = t * inv
    else:
        a1 = tl.load(co_ptr)
        c2 = tl.load(co_ptr + 1)
        d = tl.load(d_ptr + li, mask=m, other=0.0)
        dn = (a1 * d) + (c2 * t)
    tl.store(d_ptr + li, dn, mask=m)
    zn = tl.load(z_ptr + li, mask=m, other=0.0) + dn
    tl.store(z_ptr + li, zn, mask=m)
    tl.store(f32_ptr + g, zn.to(tl.float32), mask=m)


def _launch(kern, n, B, *args, init):
    BB = triton.next_power_of_2(max(B, 1))
    BI = max(16, 1024 // BB)
    kern[(triton.cdiv(n, BI),)](*args, INIT=init, BI=BI, BB=BB, num_warps=4, enable_fp_fusion=False)


def _coeffs(theta, delta, k, dt, device):
    """Row 0: (1 / theta, 0); row i >= 1: (a1, c2) of update i, the Python-float sequence of trainlib (float64)."""
    sigma = theta / delta; rho = 1 / sigma
    out = [(1.0 / theta, 0.0)]
    for i in range(k - 1):
        rho_n = 1 / (2 * sigma - rho)
        out.append((rho_n * rho, 2 * rho_n / delta))
        rho = rho_n
    return torch.tensor(out, dtype=dt, device=device)


def smooth_tail(C, Kc, u, k, lmin, lmax, dt):
    """trainlib.smooth_tail (deployment, no autograd), bitwise: returns x (float64, nb x B)."""
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2
    dinv = (1 / C.dK[C.I]).contiguous()
    x = u.to(dt, copy=True).contiguous()                                  # the float64 iterate (never aliases u)
    x32 = x.to(torch.float32)                                             # float32 copy, as _Kfp32 rounds it
    ni, B = C.I.numel(), x.shape[1]
    d = torch.empty((ni, B), dtype=dt, device=x.device)
    co = _coeffs(theta, delta, k, dt, x.device)
    for i in range(k):
        y = Kc.mm32(x32)
        _launch(_tail_step, ni, B, y, C.I, dinv, d, x, x32, co[i], ni, B, init=(i == 0))
    return x


def cheb(C, Kc, z0, b, k, lmin, lmax, dt):
    """trainlib._cheb (deployment, no autograd), bitwise: returns z (float64, ni x B)."""
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2
    dinv = (1 / C.dK[C.I]).contiguous()
    z = z0.to(dt, copy=True).contiguous()
    b = b.to(dt).contiguous()
    ni, B = C.I.numel(), z.shape[1]
    full32 = torch.zeros((C.nb, B), dtype=torch.float32, device=z.device)
    full32[C.I] = z.to(torch.float32)
    d = torch.empty((ni, B), dtype=dt, device=z.device)
    co = _coeffs(theta, delta, k, dt, z.device)
    for i in range(k):
        y = Kc.mm32(full32)
        _launch(_cheb_step, ni, B, y, C.I, dinv, d, z, b, full32, co[i], ni, B, init=(i == 0))
    return z
