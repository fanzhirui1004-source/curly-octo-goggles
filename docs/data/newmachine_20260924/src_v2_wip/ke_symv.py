"""Float32 element product y += sum_e P_e^T Ke_e P_e x from the packed upper triangle of Ke (3321 of 6561 entries per
element), one Triton program per element: gather of the element vector, symmetric product, atomic scatter into y
(OPL_KE_SYMV=1, default off; used by teacher._Kfp32 with ke_moments). Same arithmetic as the dense float32 bmm path
(float32 products, float32 sums; summation order differs, as index_add_'s atomics already do)."""
import os
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_KE_SYMV') == '1'
NUP = 81 * 82 // 2
COLS = 16                                                               # columns per kernel launch (register tile)


@triton.jit
def _symv(P_ptr, dofs_ptr, x_ptr, y_ptr, B, LD, BP: tl.constexpr, BI: tl.constexpr, NBLK: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, 128)
    mj = j < 81
    c = tl.arange(0, BP)
    mc = c < B
    dj = tl.load(dofs_ptr + e * 81 + j, mask=mj, other=0).to(tl.int64)
    X = tl.load(x_ptr + dj[:, None] * LD + c[None, :], mask=mj[:, None] & mc[None, :], other=0.0)      # [128, BP]
    for blk in tl.static_range(NBLK):
        i = blk * BI + tl.arange(0, BI)
        mi = i < 81
        a = tl.minimum(i[:, None], j[None, :])
        b = tl.maximum(i[:, None], j[None, :])
        idx = a * 81 - (a * (a - 1)) // 2 + (b - a)
        K = tl.load(P_ptr + e * 3321 + idx, mask=mi[:, None] & mj[None, :], other=0.0)            # [BI, 128]
        Y = tl.sum(K[:, :, None] * X[None, :, :], axis=1)                                            # [BI, BP]
        di = tl.load(dofs_ptr + e * 81 + i, mask=mi, other=0).to(tl.int64)
        tl.atomic_add(y_ptr + di[:, None] * LD + c[None, :], Y, mask=mi[:, None] & mc[None, :])


def symv_(y, P, dofs, x):
    """y[(E*81) scatter] += Ke(P) x[dofs] in place; y, x (nb, B) float32 / float64 contiguous, P (E, 3321), dofs (E, 81).
    More than COLS columns are processed in column blocks (the per-block register tile stays at [BI, 128, <=16])."""
    E = P.shape[0]
    if E == 0:
        return y
    B = x.shape[1]
    assert P.is_contiguous() and x.is_contiguous() and y.is_contiguous() and dofs.is_contiguous()
    for j in range(0, B, COLS):
        nbc = min(COLS, B - j)
        BP = triton.next_power_of_2(nbc)
        BI = 8 if BP <= 4 else 4                                        # tuned on the 5090 (ks_tune.py): 2 warps, small row blocks
        nblk = (81 + BI - 1) // BI
        _symv[(E,)](P, dofs, x[:, j:], y[:, j:], nbc, B, BP=BP, BI=BI, NBLK=nblk, num_warps=2)
    return y


def pack(Ke):
    """(E, 81, 81) -> (E, 3321) upper triangle, row-major (torch.triu_indices order)."""
    iu = torch.triu_indices(81, 81, device=Ke.device)
    return Ke[:, iu[0], iu[1]].contiguous()

