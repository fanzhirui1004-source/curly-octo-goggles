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

MODE = int(os.environ.get('OPL_V3_GFFUSE', '0') or 0)                 # 1: one kernel per template (atomics);
ON = MODE == 1                                                          # 2: two-phase, no atomics (deterministic)


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


FB = int(os.environ.get('OPL_V3_GFFUSE_FB', '16'))
KC = int(os.environ.get('OPL_V3_GFFUSE_KC', '16'))
WARPS = int(os.environ.get('OPL_V3_GFFUSE_WARPS', '2'))


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


# ---------------------------------------------------------------------- two-phase form (P9b, OPL_V3_GFFUSE=2)
@triton.jit
def _gf_face(x_ptr, Y_ptr, iT_ptr, B_ptr, nf, ldT, c0, LD, nc, c, FB: tl.constexpr, KC: tl.constexpr):
    """Phase 1: Y[c0 + f, :135] = B^T (B x_f) for faces c0 .. c0 + nf - 1 of one template, column c of x (Y rows 136
    wide, one column of x at a time)."""
    pid = tl.program_id(0)
    f = pid * FB + tl.arange(0, FB)
    mf = f < nf
    i = tl.arange(0, 64)
    mi = i < 54
    t = tl.zeros((FB, 64), dtype=tl.float32)
    for jc in tl.static_range(0, 136, KC):
        j = jc + tl.arange(0, KC)
        mj = j < 135
        d = tl.load(iT_ptr + j[None, :].to(tl.int64) * ldT + c0 + f[:, None], mask=mf[:, None] & mj[None, :],
                    other=0).to(tl.int64)
        X = tl.load(x_ptr + d * LD + c, mask=mf[:, None] & mj[None, :], other=0.0)
        Bt = tl.load(B_ptr + i[None, :] * 135 + j[:, None], mask=mj[:, None] & mi[None, :], other=0.0)
        t = tl.dot(X, Bt, t, input_precision='ieee')
    for jc in tl.static_range(0, 136, KC):
        j = jc + tl.arange(0, KC)
        mj = j < 135
        Bc = tl.load(B_ptr + i[:, None] * 135 + j[None, :], mask=mi[:, None] & mj[None, :], other=0.0)
        Y = tl.dot(t, Bc, input_precision='ieee')
        tl.store(Y_ptr + (c0 + f[:, None]).to(tl.int64) * 136 + j[None, :], Y, mask=mf[:, None] & mj[None, :])


@triton.jit
def _gf_reduce(Y_ptr, y_ptr, ud_ptr, ptr_ptr, slot_ptr, nu, LD, c, DB: tl.constexpr, MAXD: tl.constexpr):
    """Phase 2: y[ud[u], c] += sum of the face rows' entries at slot[ptr[u] : ptr[u+1]] (in that order); every DOF
    once, no atomics (deterministic)."""
    u = tl.program_id(0) * DB + tl.arange(0, DB)
    mu = u < nu
    p0 = tl.load(ptr_ptr + u, mask=mu, other=0)
    p1 = tl.load(ptr_ptr + u + 1, mask=mu, other=0)
    acc = tl.zeros((DB,), dtype=tl.float32)
    for k in range(MAXD):
        ok = mu & (p0 + k < p1)
        sl = tl.load(slot_ptr + p0 + k, mask=ok, other=0).to(tl.int64)
        acc += tl.load(Y_ptr + sl, mask=ok, other=0.0)
    d = tl.load(ud_ptr + u, mask=mu, other=0).to(tl.int64)
    tl.store(y_ptr + d * LD + c, tl.load(y_ptr + d * LD + c, mask=mu, other=0.0) + acc, mask=mu)


def incidence(gf):
    """Per GhostFaces object (cached on it): the DOFs the faces touch (ud, sorted), and for each the face-row slots
    (face column * 136 + local DOF) of its incidences in ascending order (CSR: ptr, slot)."""
    inc = getattr(gf, '_gf_inc', None)
    if inc is None:
        iT = gf.iTc
        n135, nf = iT.shape
        dof = iT.t().reshape(-1).to(torch.int64)                            # face-major: slot = f * 136 + j
        f_ = torch.arange(nf, device=iT.device).repeat_interleave(n135)
        j_ = torch.arange(n135, device=iT.device).repeat(nf)
        slot = f_ * 136 + j_
        o = torch.argsort(dof * (nf * 136) + slot)                          # by DOF, then slot
        dof, slot = dof[o], slot[o]
        ud, cnt = torch.unique_consecutive(dof, return_counts=True)
        ptr = torch.zeros(len(ud) + 1, dtype=torch.int64, device=iT.device)
        ptr[1:] = torch.cumsum(cnt, 0)
        inc = gf._gf_inc = dict(ud=ud.to(torch.int32).contiguous(), ptr=ptr.to(torch.int32).contiguous(),
                                slot=slot.to(torch.int32).contiguous(), maxd=int(cnt.max()), nf=nf)
    return inc


def matvec2(gf, x, y, fb=None, kc=None, db=128, warps=None):
    """Two-phase y += G x (deterministic): face products into a (faces x 136) buffer, then one sum per touched DOF."""
    iT = gf.iTc
    inc = incidence(gf)
    nc = x.shape[1]
    fb, kc, warps = fb or FB, kc or KC, warps or WARPS
    ldT = iT.shape[1]
    Yb = torch.empty((inc['nf'], 136), dtype=torch.float32, device=x.device)
    nu = len(inc['ud'])
    for c in range(nc):
        for k, Bm in enumerate(gf.B):
            lo, hi = gf.cuts[k], gf.cuts[k + 1]
            if hi > lo:
                _gf_face[(triton.cdiv(hi - lo, fb),)](x, Yb, iT, Bm.contiguous(), hi - lo, ldT, lo, nc, nc, c, FB=fb, KC=kc,
                                                       num_warps=warps)
        _gf_reduce[(triton.cdiv(nu, db),)](Yb, y, inc['ud'], inc['ptr'], inc['slot'], nu, nc, c, DB=db,
                                           MAXD=triton.next_power_of_2(inc['maxd']), num_warps=4)
    return y
