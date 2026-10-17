"""Fused elementwise tail of the slot-weight heads (src_v3 S19, OPL_V3_SLOTFUSE): y = SoftClip(raw * 0.2) in one pass.

fastnet._slot_tensors (the slot weights recomputed in every application when they are not stored, S5 / S14) applies,
after each head's last Linear, the scale 0.2 and models._SoftClip (knee > 0: identity below T = knee A, beyond it
sign(x) (T + (A - T) tanh((|x| - T) / (A - T)))). PyTorch runs that as ~11 elementwise kernels over the heads' outputs
(~77 M floats on a full plate cell, ~3.5 ms of the ~9.7 ms recompute). Here one Triton program per row block does the
same float32 operations in the same order: x = raw * 0.2f, T = A * knee, |x|, |x| - T, A - T, an IEEE division
(div_rn), tanh (libdevice, as PyTorch's float tanh), (A - T) * tanh, T + ..., the sign, and the selections; no fused
multiply-add (enable_fp_fusion=False), so every intermediate is rounded as in PyTorch. Checked on a full plate cell
(scripts/t_slotfuse.py, results/tests12/slotfuse.log): ab, fab, xab bitwise equal (torch.equal, ~100 M values);
_slot_tensors 7.8 -> 3.2 ms, with _assign_slots 9.4 -> 4.8 ms per recompute."""
import os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ON = os.environ.get('OPL_V3_SLOTFUSE', '1') == '1'


@triton.jit
def _clip_scale(raw, A, out, n_rows, scale, knee, C: tl.constexpr, H: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                MODE: tl.constexpr):
    """MODE 0: no clip (x = raw * scale); 1: clip, every bound finite; 2: clip, infinite bounds pass x through."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BC)
    m = (r < n_rows)[:, None] & (c < C)[None, :]
    off = r[:, None].to(tl.int64) * C + c[None, :]
    x = tl.load(raw + off, mask=m, other=0.0) * scale
    if MODE == 0:
        y = x
    else:
        a = tl.load(A + c // H, mask=c < C, other=1.0)[None, :]
        if MODE == 2:
            fin = (a < float('inf')) & (a > float('-inf'))                     # torch.isfinite (NaN: False)
            a = tl.where(fin, a, 1.0)
        t = a * knee
        ax = tl.abs(x)
        den = a - t
        th = libdevice.tanh(libdevice.div_rn(ax - t, den))
        v = t + den * th
        w = tl.where(x > 0, v, tl.where(x < 0, -v, 0.0 * v))
        y = tl.where(ax > t, w, x)
        if MODE == 2:
            y = tl.where(fin, y, x)
    tl.store(out + off, y, mask=m)


def clip_scale(raw, A, state, knee, scale=0.2, H=4):
    """SoftClip(raw * scale) as models._SoftClip.apply(raw * scale, A.view(1, 1, G, H-broadcast), state == 'mixed', knee)
    for a head output raw (rows, G * H) float32 contiguous, A (G,) the group bounds (state 'off': no clip)."""
    raw = raw.contiguous()
    rows, C = raw.shape
    out = torch.empty_like(raw)
    if rows == 0:
        return out
    mode = 0 if state == 'off' else (2 if state == 'mixed' else 1)
    if mode and knee <= 0:
        raise NotImplementedError('slotfused: knee = 0 (plain tanh clip) is not fused')
    BC = triton.next_power_of_2(C)
    BR = max(1, 4096 // BC)
    Af = A.reshape(-1).to(torch.float32).contiguous() if mode else raw
    _clip_scale[(triton.cdiv(rows, BR),)](raw, Af, out, rows, float(scale), float(knee), C=C, H=H, BR=BR, BC=BC, MODE=mode,
                                          num_warps=4, enable_fp_fusion=False)
    return out


ASSIGN = os.environ.get('OPL_V3_SLOTASSIGN', '0') == '1'


@triton.jit
def _assign(ab, fac, lam, A, Bo, R, LT: tl.constexpr, L: tl.constexpr, H: tl.constexpr, B3: tl.constexpr,
            BLOCK: tl.constexpr):
    """A[l, r, h] = ab[r, 0, l, h], Bo[l, r, h] = ab[r, 1, l, h] (* (1 + lam[l] fac[r]) with B3) for l < L, ab rows r of
    (2, LT, H) values: the per-layer slot weights of fastnet._assign_slots, the B3 fold in the same float32 operations."""
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = k < L * R * H
    l = k // (R * H)
    rem = k - l * (R * H)
    r = (rem // H).to(tl.int64)
    h = rem - (rem // H) * H
    a = tl.load(ab + (r * 2 * LT + l) * H + h, mask=m, other=0.0)
    b = tl.load(ab + (r * 2 * LT + LT + l) * H + h, mask=m, other=0.0)
    if B3:
        u = 1.0 + tl.load(lam + l, mask=m, other=0.0) * tl.load(fac + r, mask=m, other=0.0)
        b = b * u
    tl.store(A + k, a, mask=m)
    tl.store(Bo + k, b, mask=m)


def assign(ab, L, lam=None, fac=None):
    """ab (rows, 27, 2, LT, H) contiguous -> (A, B), each (L, rows, 27, H): layer l's a = A[l], b = B[l] (contiguous)."""
    E, S, two, LT, H = ab.shape
    R = E * S
    A = torch.empty((L, E, S, H), dtype=ab.dtype, device=ab.device)
    Bo = torch.empty_like(A)
    n = L * R * H
    if n:
        b3 = lam is not None
        if b3:                                                     # torch's b * (1 + lam * fac) runs in float32
            fac, lam = fac.reshape(-1).to(ab.dtype).contiguous(), lam.to(ab.dtype).contiguous()
            assert fac.numel() == R and lam.numel() >= L
        _assign[(triton.cdiv(n, 1024),)](ab, fac if b3 else ab, lam if b3 else ab, A, Bo, R, LT=LT, L=L, H=H,
                                         B3=b3, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return A, Bo
