"""Matrix-free float32 element product (src_v3 P13, OPL_V3_KMF): y += sum_e P_e^T K_e P_e x for the float32 correction
product (teacher._Kfp32) without forming the element matrices.

A deployment element is a Q2 hexahedron (27 nodes, 81 DOFs) whose stiffness is the moment-fitted sum
    K_e = sum_m M_e[m] T_m,   M_e[m] = int_(cut part of e) xi^a eta^b zeta^c dV   (m = (a, b, c), a, b, c <= 4)
(element_moments.pattern_operators: T_m holds the monomial coefficients of grad N_i . C . grad N_j). The integrands are
polynomials of degree <= 4 in each local coordinate, which the 5-point Gauss rule per axis integrates exactly, so with
the weights w_e = V^-1 M_e (V[m, q] = xi_q^a eta_q^b zeta_q^c at the 125 tensor Gauss points; float64, then rounded)
    K_e = sum_q w_e[q] B_q^T C B_q      (the same matrix in exact arithmetic; float64 check: 8e-15 relative)
and K_e x_e = sum_q w_e[q] sum_d dN/dx_d(q) sigma_.d(q), sigma = lam tr(eps) I + 2 mu eps, eps from the gradient
of x_e at q. The product then reads 125 weights and 27 node indices per element (~0.6 kB) instead of the 3321 packed
float32 entries (13.3 kB) of the formed matrix, and needs no formation of the matrices (the E x 125 x 3321 GEMM of
_Kfp32.hold). Arithmetic: per tile of EB elements and PB quadrature points, the nine gradient components as float32
dots (input_precision 'ieee': no TF32) of the shared derivative table (3 x 128 x 32, zero padded) with the gathered
element displacements, the stress, the weighted back projection as nine more dots; atomic adds into y as before.
Rounding differs from the formed-matrix product (float32 error against float64 of the same order, 3e-7 median on
random cut boxes, see scripts/t_kmf.py); the flop count grows with the number of columns while the formed-matrix
product reads the matrices once for all columns, so it is used for narrow products only (B <= MAXB)."""
import os
import numpy as np
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_V3_KMF', '0') == '1'
MAXB = int(os.environ.get('OPL_V3_KMF_MAXB', '2'))
EB = int(os.environ.get('OPL_V3_KMF_EB', '32'))
PB = int(os.environ.get('OPL_V3_KMF_PB', '16'))
WARPS = int(os.environ.get('OPL_V3_KMF_WARPS', '4'))
_TAB = {}


def _lag(k, x):
    return {-1: x * (x - 1) / 2, 0: 1 - x * x, 1: x * (x + 1) / 2}[int(k)]


def _dlag(k, x):
    return {-1: x - 0.5, 0: -2 * x, 1: x + 0.5}[int(k)]


def tables(xi_nodes, n, device):
    """(DN, Vinv3T) for one node ordering xi_nodes (27 x 3 local coordinates in {-1, 0, 1}, the ordering the templates
    use) and grid parameter n: DN (3, 128, 32) float32, DN[d, q, i] = dN_i/dx_d at Gauss point q (physical: 2n d/dxi);
    Vinv3T (125, 125) float64 with w = M @ Vinv3T."""
    key = (np.asarray(xi_nodes, np.int64).tobytes(), int(n), str(device))
    t = _TAB.get(key)
    if t is None:
        g, _ = np.polynomial.legendre.leggauss(5)
        V1i = np.linalg.inv(np.vander(g, 5, increasing=True).T)               # w1 = V1i m1 (V1[a, p] = g_p^a)
        Vinv3 = np.kron(np.kron(V1i, V1i), V1i)                               # [(p, q, r), (a, b, c)], row-major
        xi = np.asarray(xi_nodes).reshape(27, 3)
        DN = np.zeros((3, 128, 32))
        for qi, (p, q, r) in enumerate(np.ndindex(5, 5, 5)):
            x = (g[p], g[q], g[r])
            for i in range(27):
                v = [_lag(xi[i, d], x[d]) for d in range(3)]
                dv = [_dlag(xi[i, d], x[d]) for d in range(3)]
                DN[0, qi, i] = dv[0] * v[1] * v[2] * 2 * n
                DN[1, qi, i] = v[0] * dv[1] * v[2] * 2 * n
                DN[2, qi, i] = v[0] * v[1] * dv[2] * 2 * n
        t = _TAB[key] = (torch.tensor(DN, dtype=torch.float32, device=device).contiguous(),
                         torch.tensor(Vinv3.T, dtype=torch.float64, device=device).contiguous())
    return t


def weights(M, Vinv3T, chunk=8192):
    """Quadrature weights of the elements (float64 moments M, E x 125): float32 (128, E), rows 125.. zero."""
    E = M.shape[0]
    WT = torch.zeros((128, E), dtype=torch.float32, device=M.device)
    for lo in range(0, E, chunk):
        WT[:125, lo:lo + chunk] = (M[lo:lo + chunk] @ Vinv3T).t().to(torch.float32)
    return WT


def nodes(dofs):
    """Element node indices (E x 27 int32) of node-major element DOFs (dofs[e, 3 i + c] = 3 node + c)."""
    return (dofs[:, 0::3] // 3).to(torch.int32).contiguous()


@triton.jit
def _kmf(x_ptr, y_ptr, nd_ptr, w_ptr, dn_ptr, E, LD, lam, mu, EB: tl.constexpr, PB: tl.constexpr):
    """Elements pid * EB ..; one column c = program_id(1) of x (row stride LD)."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < E
    col = tl.program_id(1)
    i = tl.arange(0, 32)
    mi = i < 27
    m2 = mi[:, None] & me[None, :]
    nd = tl.load(nd_ptr + e[None, :].to(tl.int64) * 27 + i[:, None], mask=m2, other=0).to(tl.int64)     # [32, EB]
    r0 = (3 * nd) * LD + col
    u0 = tl.load(x_ptr + r0, mask=m2, other=0.0)
    u1 = tl.load(x_ptr + r0 + LD, mask=m2, other=0.0)
    u2 = tl.load(x_ptr + r0 + 2 * LD, mask=m2, other=0.0)
    y0 = tl.zeros((32, EB), dtype=tl.float32)
    y1 = tl.zeros((32, EB), dtype=tl.float32)
    y2 = tl.zeros((32, EB), dtype=tl.float32)
    for pb in tl.static_range(0, 128, PB):
        q = pb + tl.arange(0, PB)
        a0 = tl.load(dn_ptr + (0 * 128 + q[:, None]) * 32 + i[None, :])                                  # [PB, 32]
        a1 = tl.load(dn_ptr + (1 * 128 + q[:, None]) * 32 + i[None, :])
        a2 = tl.load(dn_ptr + (2 * 128 + q[:, None]) * 32 + i[None, :])
        g00 = tl.dot(a0, u0, input_precision='ieee')                        # g_dc = d u_c / d x_d at the points [PB, EB]
        g01 = tl.dot(a0, u1, input_precision='ieee')
        g02 = tl.dot(a0, u2, input_precision='ieee')
        g10 = tl.dot(a1, u0, input_precision='ieee')
        g11 = tl.dot(a1, u1, input_precision='ieee')
        g12 = tl.dot(a1, u2, input_precision='ieee')
        g20 = tl.dot(a2, u0, input_precision='ieee')
        g21 = tl.dot(a2, u1, input_precision='ieee')
        g22 = tl.dot(a2, u2, input_precision='ieee')
        w = tl.load(w_ptr + q[:, None].to(tl.int64) * E + e[None, :], mask=me[None, :], other=0.0)       # [PB, EB]
        ld = lam * (g00 + g11 + g22)
        s00 = w * (ld + 2 * mu * g00)
        s11 = w * (ld + 2 * mu * g11)
        s22 = w * (ld + 2 * mu * g22)
        s01 = w * (mu * (g01 + g10))
        s02 = w * (mu * (g02 + g20))
        s12 = w * (mu * (g12 + g21))
        b0 = tl.trans(a0)                                                   # [32, PB]
        b1 = tl.trans(a1)
        b2 = tl.trans(a2)
        y0 = tl.dot(b0, s00, y0, input_precision='ieee')                    # y_p[i] += sum_d dN_i/dx_d sigma_pd
        y0 = tl.dot(b1, s01, y0, input_precision='ieee')
        y0 = tl.dot(b2, s02, y0, input_precision='ieee')
        y1 = tl.dot(b0, s01, y1, input_precision='ieee')
        y1 = tl.dot(b1, s11, y1, input_precision='ieee')
        y1 = tl.dot(b2, s12, y1, input_precision='ieee')
        y2 = tl.dot(b0, s02, y2, input_precision='ieee')
        y2 = tl.dot(b1, s12, y2, input_precision='ieee')
        y2 = tl.dot(b2, s22, y2, input_precision='ieee')
    tl.atomic_add(y_ptr + r0, y0, mask=m2)
    tl.atomic_add(y_ptr + r0 + LD, y1, mask=m2)
    tl.atomic_add(y_ptr + r0 + 2 * LD, y2, mask=m2)


def matvec_(y, nd, WT, DN, x, lam, mu, eb=None, pb=None, warps=None):
    """y += K x (float32, in place): nd (E x 27 int32 element nodes), WT (128 x E float32 weights), DN (3 x 128 x 32),
    x, y (nb x B) float32 contiguous; lam, mu the Lame constants of the templates."""
    E = nd.shape[0]
    if E == 0:
        return y
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32
    B = x.shape[1]
    eb, pb, warps = eb or EB, pb or PB, warps or WARPS
    _kmf[(triton.cdiv(E, eb), B)](x, y, nd, WT, DN, E, B, float(lam), float(mu), EB=eb, PB=pb, num_warps=warps)
    return y
