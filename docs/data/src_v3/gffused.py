"""Ghost-penalty face product in one kernel per face template (src_v3 P9, OPL_V3_GFFUSE): y += sum_f P_f^T B^T B P_f x
for the float32 correction product (teacher.GhostFaces.matvec, GF_CAT layout: face DOFs as a 135 x faces int32 array,
faces grouped by template). Per tile of FB faces: the faces' 135 DOF values gathered from x, t = B x_f (B: 54 x 135),
y_f = B^T t, atomically added to y at the faces' DOFs; the same factored form B^T (B X) as before (two GEMMs per
template and an index_add_ per product, with the gathered X and Y in memory), in another summation order (fp32 dot in
'ieee' precision: no TF32). One program per tile of faces and column of x."""
import os
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_V3_GFFUSE', '0') == '1'


@triton.jit
def _gf(x_ptr, y_ptr, iT_ptr, B_ptr, nf, ldT, c0, LD, FB: tl.constexpr, KC: tl.constexpr):
    """Faces c0 .. c0 + nf - 1 of the 135 x ldT array iT (columns), template matrix B (54 x 135, row-major); one column
    (right-hand side) per program row: c = program_id(1)."""
    pid = tl.program_id(0)
    c = tl.program_id(1)
    f = pid * FB + tl.arange(0, FB)
    mf = f < nf
    i = tl.arange(0, 64)                                                    # rows of B (54 used)
    mi = i < 54
    t = tl.zeros((FB, 64), dtype=tl.float32)
    for jc in tl.static_range(0, 136, KC):                                  # t = X B^T over 135 DOFs in KC chunks
        j = jc + tl.arange(0, KC)
        mj = j < 135
        d = tl.load(iT_ptr + j[None, :].to(tl.int64) * ldT + c0 + f[:, None], mask=mf[:, None] & mj[None, :],
                    other=0).to(tl.int64)
        X = tl.load(x_ptr + d * LD + c, mask=mf[:, None] & mj[None, :], other=0.0)                        # [FB, KC]
        Bt = tl.load(B_ptr + i[None, :] * 135 + j[:, None], mask=mj[:, None] & mi[None, :], other=0.0)      # [KC, 64]
        t = tl.dot(X, Bt, t, input_precision='ieee')
    for jc in tl.static_range(0, 136, KC):                                  # y_f = t B, scattered in KC chunks
        j = jc + tl.arange(0, KC)
        mj = j < 135
        Bc = tl.load(B_ptr + i[:, None] * 135 + j[None, :], mask=mi[:, None] & mj[None, :], other=0.0)      # [64, KC]
        Y = tl.dot(t, Bc, input_precision='ieee')                                                         # [FB, KC]
        d = tl.load(iT_ptr + j[None, :].to(tl.int64) * ldT + c0 + f[:, None], mask=mf[:, None] & mj[None, :],
                    other=0).to(tl.int64)
        tl.atomic_add(y_ptr + d * LD + c, Y, mask=mf[:, None] & mj[None, :])


FB = int(os.environ.get('OPL_V3_GFFUSE_FB', '32'))
KC = int(os.environ.get('OPL_V3_GFFUSE_KC', '32'))
WARPS = int(os.environ.get('OPL_V3_GFFUSE_WARPS', '4'))


def matvec(gf, x, y, fb=None, kc=None, warps=None):
    """y += G x through the GhostFaces object gf (GF_CAT layout: gf.iTc (135 x faces int32), gf.cuts, gf.B[k] 54 x 135);
    x, y (nb x b) float32 contiguous."""
    iT = gf.iTc
    nc = x.shape[1]
    fb, kc, warps = fb or FB, kc or KC, warps or WARPS
    ldT = iT.shape[1]
    for k, Bm in enumerate(gf.B):
        lo, hi = gf.cuts[k], gf.cuts[k + 1]
        if hi <= lo:
            continue
        _gf[(triton.cdiv(hi - lo, fb), nc)](x, y, iT, Bm.contiguous(), hi - lo, ldT, lo, nc, FB=fb, KC=kc, num_warps=warps)
    return y
