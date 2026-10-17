"""Fused elementwise tail of the slot-weight heads (src_v3 S19, OPL_V3_SLOTFUSE): y = SoftClip(raw * 0.2) in one pass.

fastnet._slot_tensors (the slot weights recomputed in every application when they are not stored, S5 / S14) applies,
after each head's last Linear, the scale 0.2 and models._SoftClip (knee > 0: identity below T = knee A, beyond it
sign(x) (T + (A - T) tanh((|x| - T) / (A - T)))). PyTorch runs that as ~11 elementwise kernels over the heads' outputs
(~77 M floats on a full plate cell, ~3.5 ms of the ~9.7 ms recompute). Here one Triton program per row block does the
same float32 operations in the same order: x = raw * 0.2f, T = A * knee, |x|, |x| - T, A - T, an IEEE division
(div_rn), tanh (libdevice, as PyTorch's float tanh), (A - T) * tanh, T + ..., the sign, and the selections; no fused
multiply-add (enable_fp_fusion=False), so every intermediate is rounded as in PyTorch. The result is checked bitwise
against the PyTorch path (scripts/t_slotfuse.py) before the switch is used."""
import os
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

ON = os.environ.get('OPL_V3_SLOTFUSE', '0') == '1'


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
