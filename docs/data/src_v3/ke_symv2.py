"""Single-read float32 element product (src_v3 P2, candidates; ke_symv._symv is the current kernel).

y += sum_e P_e^T K_e P_e x with K_e symmetric 81 x 81 stored as its packed upper triangle (row-major, 3321 entries).
ke_symv._symv reads, for each block of rows, all 81 columns through the symmetric index map: the upper part of a row is
contiguous, the lower part (K[i, j] = K[j, i], j < i) is a strided walk down a packed column, and every off-diagonal
entry is read twice. Here each row block reads only its upper part (contiguous in the packed array, one read per entry)
and uses every entry twice: for row i, the row contribution y_i += sum_{j >= i} K_ij x_j, and for the columns j > i the
column contribution y_j += K_ij x_i. Both are accumulated in registers and scattered once per element (variant 'onehot':
the row results are folded into the column accumulator, 81 atomics per element and column; 'twice': row results
scattered per block, 2 x 81 atomics). Same float32 products and sums; only the summation order differs."""
import torch
import triton
import triton.language as tl


@triton.jit
def _symv_up(P_ptr, dofs_ptr, x_ptr, y_ptr, B, LD, BP: tl.constexpr, BI: tl.constexpr, NBLK: tl.constexpr,
             ONEHOT: tl.constexpr, kmap_ptr=None, MAP: tl.constexpr = False):
    e = tl.program_id(0).to(tl.int64)
    if MAP:                                                                  # src_v3 P2b: row of P holding K_e
        ke = tl.load(kmap_ptr + e).to(tl.int64)
    else:
        ke = e
    j = tl.arange(0, 128)
    mj = j < 81
    c = tl.arange(0, BP)
    mc = c < B
    dj = tl.load(dofs_ptr + e * 81 + j, mask=mj, other=0).to(tl.int64)
    X = tl.load(x_ptr + dj[:, None] * LD + c[None, :], mask=mj[:, None] & mc[None, :], other=0.0)          # [128, BP]
    acc = tl.zeros([128, BP], dtype=tl.float32)
    for blk in tl.static_range(NBLK):
        i = blk * BI + tl.arange(0, BI)
        mi = i < 81
        off = i * 81 - (i * (i - 1)) // 2                                    # packed start of row i (entry (i, i))
        up = (j[None, :] >= i[:, None]) & mi[:, None] & mj[None, :]
        K = tl.load(P_ptr + ke * 3321 + off[:, None] + (j[None, :] - i[:, None]), mask=up, other=0.0)      # [BI, 128]
        Yr = tl.sum(K[:, :, None] * X[None, :, :], axis=1)                   # [BI, BP]: sum_{j >= i} K_ij x_j
        di = tl.load(dofs_ptr + e * 81 + i, mask=mi, other=0).to(tl.int64)
        Xi = tl.load(x_ptr + di[:, None] * LD + c[None, :], mask=mi[:, None] & mc[None, :], other=0.0)     # [BI, BP]
        Ks = tl.where(j[None, :] > i[:, None], K, 0.0)                       # strict upper
        acc += tl.sum(Ks[:, :, None] * Xi[:, None, :], axis=0)               # [128, BP]: y_j += K_ij x_i, j > i
        if ONEHOT:
            oh = j[None, :] == i[:, None]
            acc += tl.sum(tl.where(oh[:, :, None], Yr[:, None, :], 0.0), axis=0)
        else:
            tl.atomic_add(y_ptr + di[:, None] * LD + c[None, :], Yr, mask=mi[:, None] & mc[None, :])
    tl.atomic_add(y_ptr + dj[:, None] * LD + c[None, :], acc, mask=mj[:, None] & mc[None, :])


def symv_(y, P, dofs, x, BI=None, warps=None, onehot=True, cols=16):
    """y[(E*81) scatter] += Ke(P) x[dofs] in place (as ke_symv.symv_)."""
    E = P.shape[0]
    if E == 0:
        return y
    B = x.shape[1]
    assert P.is_contiguous() and x.is_contiguous() and y.is_contiguous() and dofs.is_contiguous()
    for j in range(0, B, cols):
        nbc = min(cols, B - j)
        BP = triton.next_power_of_2(nbc)
        bi = BI or (8 if BP <= 4 else 4)
        nblk = (81 + bi - 1) // bi
        _symv_up[(E,)](P, dofs, x[:, j:], y[:, j:], nbc, B, BP=BP, BI=bi, NBLK=nblk, ONEHOT=onehot,
                       num_warps=warps or (2 if BP <= 4 else 4))
    return y
