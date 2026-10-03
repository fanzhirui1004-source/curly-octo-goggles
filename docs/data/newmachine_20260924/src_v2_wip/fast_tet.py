"""Fused per-tetrahedron moments (Triton, float64) for the polyhedral moment integrator (OPL_TET_TRITON=1, default off).

  out[t, 25 a + 5 b + c] = |det E_t| * sum_q w_q x_q^a y_q^b z_q^c,   P_q = v0 + sum_k r_qk (v_k - v0)

the same quadrature as polyref_torch_fast / moments_ad._quad (collapsed Gauss rule tref, tw), evaluated per tetrahedron
in registers instead of a batched (5 x q) @ (q x 25) product over materialised power tables; summation order differs
(rounding level). The backward (reverse mode w.r.t. the vertices, for the design sensitivities) is written out:
with p(P) = sum_m G_m phi_m(P) and s = J sum_q w_q p(P_q),
  ds/dv_k = J sum_q w_q r_qk grad p(P_q) + (sum_q w_q p(P_q)) dJ/de_k        (k = 1..3)
  ds/dv0  = J sum_q w_q (1 - sum_k r_qk) grad p(P_q) - sum_k (sum_q w_q p(P_q)) dJ/de_k
  dJ/de_1 = sgn(det) e_2 x e_3, dJ/de_2 = sgn(det) e_3 x e_1, dJ/de_3 = sgn(det) e_1 x e_2."""
import os
import torch
import triton
import triton.language as tl

BLOCK = int(os.environ.get('OPL_TET_BLOCK', '8'))
WARPS = int(os.environ.get('OPL_TET_WARPS', '4'))


def enabled():
    return os.environ.get('OPL_TET_TRITON') == '1'


@triton.jit
def _geom(tets_ptr, offs, mask):
    b = offs * 12
    v0x = tl.load(tets_ptr + b + 0, mask=mask, other=0.0)
    v0y = tl.load(tets_ptr + b + 1, mask=mask, other=0.0)
    v0z = tl.load(tets_ptr + b + 2, mask=mask, other=0.0)
    e1x = tl.load(tets_ptr + b + 3, mask=mask, other=0.0) - v0x
    e1y = tl.load(tets_ptr + b + 4, mask=mask, other=0.0) - v0y
    e1z = tl.load(tets_ptr + b + 5, mask=mask, other=0.0) - v0z
    e2x = tl.load(tets_ptr + b + 6, mask=mask, other=0.0) - v0x
    e2y = tl.load(tets_ptr + b + 7, mask=mask, other=0.0) - v0y
    e2z = tl.load(tets_ptr + b + 8, mask=mask, other=0.0) - v0z
    e3x = tl.load(tets_ptr + b + 9, mask=mask, other=0.0) - v0x
    e3y = tl.load(tets_ptr + b + 10, mask=mask, other=0.0) - v0y
    e3z = tl.load(tets_ptr + b + 11, mask=mask, other=0.0) - v0z
    return v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z


@triton.jit
def _powers(v, k):
    """[B, 8] table v ** k, k = 0..4 (columns 5..7 zero)."""
    v2 = v * v
    V = v[:, None]
    V2 = v2[:, None]
    return tl.where(k == 0, 1.0, tl.where(k == 1, V, tl.where(k == 2, V2, tl.where(k == 3, V2 * V,
                    tl.where(k == 4, V2 * V2, 0.0)))))


@triton.jit
def _dpowers(v, k):
    """[B, 8] table d(v ** k)/dv = k v ** (k - 1), k = 0..4."""
    v2 = v * v
    V = v[:, None]
    V2 = v2[:, None]
    return tl.where(k == 1, 1.0, tl.where(k == 2, 2.0 * V, tl.where(k == 3, 3.0 * V2, tl.where(k == 4, 4.0 * V2 * V, 0.0))))


@triton.jit
def _fwd(tets_ptr, tref_ptr, tw_ptr, out_ptr, T, Q: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offs < T
    v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z = _geom(tets_ptr, offs, mask)
    det = e1x * (e2y * e3z - e2z * e3y) - e1y * (e2x * e3z - e2z * e3x) + e1z * (e2x * e3y - e2y * e3x)
    J = tl.abs(det)
    k = tl.arange(0, 8)[None, :]
    acc = tl.zeros([BLOCK_T, 8, 8, 8], dtype=tl.float64)
    for q in range(Q):
        r1 = tl.load(tref_ptr + 3 * q + 0); r2 = tl.load(tref_ptr + 3 * q + 1); r3 = tl.load(tref_ptr + 3 * q + 2)
        w = tl.load(tw_ptr + q)
        x = v0x + r1 * e1x + r2 * e2x + r3 * e3x
        y = v0y + r1 * e1y + r2 * e2y + r3 * e3y
        z = v0z + r1 * e1z + r2 * e2z + r3 * e3z
        xp = _powers(x, k) * w
        yz = _powers(y, k)[:, :, None] * _powers(z, k)[:, None, :]                  # B x 8 x 8
        acc += xp[:, :, None, None] * yz[:, None, :, :]
    acc = acc * J[:, None, None, None]
    ia = tl.arange(0, 8)[None, :, None, None]; ib = tl.arange(0, 8)[None, None, :, None]; ic = tl.arange(0, 8)[None, None, None, :]
    col = 25 * ia + 5 * ib + ic
    ok = (ia < 5) & (ib < 5) & (ic < 5) & mask[:, None, None, None]
    tl.store(out_ptr + offs[:, None, None, None] * 125 + col, acc, mask=ok)


@triton.jit
def _bwd(tets_ptr, tref_ptr, tw_ptr, g_ptr, gt_ptr, T, Q: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = offs < T
    v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z = _geom(tets_ptr, offs, mask)
    det = e1x * (e2y * e3z - e2z * e3y) - e1y * (e2x * e3z - e2z * e3x) + e1z * (e2x * e3y - e2y * e3x)
    J = tl.abs(det)
    sg = tl.where(det >= 0, 1.0, -1.0)
    k = tl.arange(0, 8)[None, :]
    ia = tl.arange(0, 8)[None, :, None, None]; ib = tl.arange(0, 8)[None, None, :, None]; ic = tl.arange(0, 8)[None, None, None, :]
    ok = (ia < 5) & (ib < 5) & (ic < 5) & mask[:, None, None, None]
    G = tl.load(g_ptr + offs[:, None, None, None] * 125 + 25 * ia + 5 * ib + ic, mask=ok, other=0.0)   # B x 8 x 8 x 8
    S = tl.zeros([BLOCK_T], dtype=tl.float64)
    g0x = tl.zeros([BLOCK_T], dtype=tl.float64); g0y = tl.zeros([BLOCK_T], dtype=tl.float64); g0z = tl.zeros([BLOCK_T], dtype=tl.float64)
    g1x = tl.zeros([BLOCK_T], dtype=tl.float64); g1y = tl.zeros([BLOCK_T], dtype=tl.float64); g1z = tl.zeros([BLOCK_T], dtype=tl.float64)
    g2x = tl.zeros([BLOCK_T], dtype=tl.float64); g2y = tl.zeros([BLOCK_T], dtype=tl.float64); g2z = tl.zeros([BLOCK_T], dtype=tl.float64)
    g3x = tl.zeros([BLOCK_T], dtype=tl.float64); g3y = tl.zeros([BLOCK_T], dtype=tl.float64); g3z = tl.zeros([BLOCK_T], dtype=tl.float64)
    for q in range(Q):
        r1 = tl.load(tref_ptr + 3 * q + 0); r2 = tl.load(tref_ptr + 3 * q + 1); r3 = tl.load(tref_ptr + 3 * q + 2)
        w = tl.load(tw_ptr + q)
        x = v0x + r1 * e1x + r2 * e2x + r3 * e3x
        y = v0y + r1 * e1y + r2 * e2y + r3 * e3y
        z = v0z + r1 * e1z + r2 * e2z + r3 * e3z
        zp, dz = _powers(z, k), _dpowers(z, k)                                        # B x 8
        yp, dy = _powers(y, k), _dpowers(y, k)
        xp, dx = _powers(x, k), _dpowers(x, k)
        Gz = tl.sum(G * zp[:, None, None, :], axis=3)                                 # B x 8(a) x 8(b)
        Gdz = tl.sum(G * dz[:, None, None, :], axis=3)
        Gyz = tl.sum(Gz * yp[:, None, :], axis=2)                                     # B x 8(a)
        Gdyz = tl.sum(Gz * dy[:, None, :], axis=2)
        Gydz = tl.sum(Gdz * yp[:, None, :], axis=2)
        p = tl.sum(Gyz * xp, axis=1)
        px = tl.sum(Gyz * dx, axis=1)
        py = tl.sum(Gdyz * xp, axis=1)
        pz = tl.sum(Gydz * xp, axis=1)
        S += w * p
        c0 = w * (1.0 - r1 - r2 - r3)
        g0x += c0 * px; g0y += c0 * py; g0z += c0 * pz
        g1x += w * r1 * px; g1y += w * r1 * py; g1z += w * r1 * pz
        g2x += w * r2 * px; g2y += w * r2 * py; g2z += w * r2 * pz
        g3x += w * r3 * px; g3y += w * r3 * py; g3z += w * r3 * pz
    d1x = sg * (e2y * e3z - e2z * e3y); d1y = sg * (e2z * e3x - e2x * e3z); d1z = sg * (e2x * e3y - e2y * e3x)
    d2x = sg * (e3y * e1z - e3z * e1y); d2y = sg * (e3z * e1x - e3x * e1z); d2z = sg * (e3x * e1y - e3y * e1x)
    d3x = sg * (e1y * e2z - e1z * e2y); d3y = sg * (e1z * e2x - e1x * e2z); d3z = sg * (e1x * e2y - e1y * e2x)
    o = offs * 12
    tl.store(gt_ptr + o + 3, J * g1x + S * d1x, mask=mask); tl.store(gt_ptr + o + 4, J * g1y + S * d1y, mask=mask)
    tl.store(gt_ptr + o + 5, J * g1z + S * d1z, mask=mask)
    tl.store(gt_ptr + o + 6, J * g2x + S * d2x, mask=mask); tl.store(gt_ptr + o + 7, J * g2y + S * d2y, mask=mask)
    tl.store(gt_ptr + o + 8, J * g2z + S * d2z, mask=mask)
    tl.store(gt_ptr + o + 9, J * g3x + S * d3x, mask=mask); tl.store(gt_ptr + o + 10, J * g3y + S * d3y, mask=mask)
    tl.store(gt_ptr + o + 11, J * g3z + S * d3z, mask=mask)
    tl.store(gt_ptr + o + 0, J * g0x - S * (d1x + d2x + d3x), mask=mask)
    tl.store(gt_ptr + o + 1, J * g0y - S * (d1y + d2y + d3y), mask=mask)
    tl.store(gt_ptr + o + 2, J * g0z - S * (d1z + d2z + d3z), mask=mask)


def tet_moments(tets, tref, tw):
    """(T, 4, 3) float64 -> (T, 125) float64 (J folded in)."""
    tets = tets.contiguous()
    T = tets.shape[0]
    out = torch.empty((T, 125), dtype=torch.float64, device=tets.device)
    if T:
        _fwd[(triton.cdiv(T, BLOCK),)](tets, tref.contiguous(), tw.contiguous(), out, T, Q=tref.shape[0], BLOCK_T=BLOCK, num_warps=WARPS)
    return out


def tet_moments_vjp(tets, tref, tw, G):
    tets = tets.contiguous()
    T = tets.shape[0]
    gt = torch.empty_like(tets)
    if T:
        _bwd[(triton.cdiv(T, BLOCK),)](tets, tref.contiguous(), tw.contiguous(), G.contiguous(), gt, T, Q=tref.shape[0],
                                       BLOCK_T=BLOCK, num_warps=WARPS)
    return gt


class TetMoments(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tets, tref, tw):
        ctx.save_for_backward(tets, tref, tw)
        return tet_moments(tets, tref, tw)

    @staticmethod
    def backward(ctx, G):
        tets, tref, tw = ctx.saved_tensors
        return tet_moments_vjp(tets, tref, tw, G), None, None
