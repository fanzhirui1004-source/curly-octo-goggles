"""Ghost-penalty product G x = sum_f P_f^T B_d^T B_d P_f x in one kernel per face template (src_v3, OPL_V3_GFFUSE=1,
float32, the equilibrium correction's K product).

teacher.GhostFaces.matvec runs, per template d (54 x 135), a gather of the face vectors X (135 x faces), two cuBLAS
GEMMs Z = B_d X, Y = B_d^T Z and an index_add_ of Y: 4 launches per template, 12 per float32 K product plus the zero fill
and the gamma scaling (about 500 of the ~1770 launches of one S_hat application). Here one program takes FB faces of
one template through all of it: gathered face vectors, the two products as IEEE float32 dots (row chunks of 64),
float32 atomic adds into y. Same products; the sums run in another order (the index_add_ was already atomic)."""
import os
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_V3_GFFUSE', '0') == '1'
FB = int(os.environ.get('OPL_V3_GFFUSE_FB', '32'))
NW = int(os.environ.get('OPL_V3_GFFUSE_NW', '4'))


@triton.jit
def _gf_tpl(x, y, idxT, Bm, nf, NBF, R: tl.constexpr, Q: tl.constexpr, FB: tl.constexpr, RC: tl.constexpr,
            QP: tl.constexpr, NB: tl.constexpr):
    f = tl.program_id(0) * FB + tl.arange(0, FB)
    mf = f < nf
    q = tl.arange(0, QP)
    mq = q < Q
    NRC: tl.constexpr = (R + RC - 1) // RC
    for bb in tl.static_range(NB):
        Z = tl.zeros((FB, QP), tl.float32)
        for c in tl.static_range(NRC):
            r = c * RC + tl.arange(0, RC)
            mr = r < R
            m2 = mf[:, None] & mr[None, :]
            idx = tl.load(idxT + r[None, :].to(tl.int64) * nf + f[:, None], mask=m2, other=0).to(tl.int64)
            X = tl.load(x + idx * NBF + bb, mask=m2, other=0.0)                                       # FB x RC
            Bt = tl.load(Bm + q[None, :] * R + r[:, None], mask=mr[:, None] & mq[None, :], other=0.0)  # RC x QP (B^T)
            Z += tl.dot(X, Bt, input_precision="ieee")                                                # FB x QP
        for c in tl.static_range(NRC):
            r = c * RC + tl.arange(0, RC)
            mr = r < R
            m2 = mf[:, None] & mr[None, :]
            idx = tl.load(idxT + r[None, :].to(tl.int64) * nf + f[:, None], mask=m2, other=0).to(tl.int64)
            Bc = tl.load(Bm + q[:, None] * R + r[None, :], mask=mq[:, None] & mr[None, :], other=0.0)  # QP x RC
            Yc = tl.dot(Z, Bc, input_precision="ieee")                                                # FB x RC
            tl.atomic_add(y + idx * NBF + bb, Yc, mask=m2)


def usable(gf, x):
    return ON and x.dtype == torch.float32 and x.dim() == 2 and x.shape[1] <= 4 and x.is_contiguous() \
        and all(B.dtype == torch.float32 for B in gf.B)


def matvec(gf, x, y):
    """y += G x (float32, x / y: nb x b contiguous, b <= 4), as GhostFaces.matvec."""
    b = x.shape[1]
    for iT, B in zip(gf.idxT, gf.B):
        R, nf = iT.shape
        if nf == 0:
            continue
        Q = B.shape[0]
        _gf_tpl[(triton.cdiv(nf, FB),)](x, y, iT, B, nf, b, R=R, Q=Q, FB=FB, RC=64, QP=triton.next_power_of_2(Q), NB=b,
                                         num_warps=NW)
    return y
