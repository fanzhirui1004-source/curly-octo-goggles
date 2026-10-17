"""Symmetric matrices stored as packed lower-triangle tiles (src_v3), for the cell coarse solve of the equilibrium
correction: the float32 matrix A (n x n, symmetric) is kept as its lower-triangle T x T tiles (I >= J), zero padded
at the edges, about half the memory of the dense matrix; y = A X is one Triton program per stored tile:
y_I += A_IJ X_J and, for I > J, y_J += A_IJ^T X_I (float32 products and sums, true IEEE float32 dot products; the
partial sums of a row block meet through atomic adds, so the summation order differs from a dense GEMM, as index_add_'s
atomics already do elsewhere)."""
import torch
import triton
import triton.language as tl

T = 64                                                                   # tile size


def ntiles(n):
    nt = triton.cdiv(n, T)
    return nt, nt * (nt + 1) // 2


def pack(A):
    """A (n x n) float32 symmetric (only its lower triangle is read) -> packed tiles (ntile, T, T); diagonal tiles hold
    the symmetric block (lower triangle mirrored)."""
    n = A.shape[0]
    nt, ntl = ntiles(n)
    N = nt * T
    Ap = torch.zeros((N, N), dtype=A.dtype, device=A.device)
    Ap[:n, :n] = A
    TI, TJ = _tile_index(nt, A.device)
    P = Ap.view(nt, T, nt, T).permute(0, 2, 1, 3)[TI.long(), TJ.long()].contiguous()
    del Ap
    d = torch.nonzero(TI == TJ).squeeze(1)
    D = P[d]
    P[d] = torch.tril(D) + torch.tril(D, -1).transpose(1, 2)
    return P


def unpack(P, n):
    """Dense symmetric matrix of packed tiles (tests)."""
    nt, _ = ntiles(n)
    A = torch.zeros((nt * T, nt * T), dtype=P.dtype, device=P.device)
    k = 0
    for I in range(nt):
        for J in range(I + 1):
            A[I * T:(I + 1) * T, J * T:(J + 1) * T] = P[k]
            if I != J:
                A[J * T:(J + 1) * T, I * T:(I + 1) * T] = P[k].T
            k += 1
    return A[:n, :n]


@triton.jit
def _symm(P_ptr, X_ptr, Y_ptr, n, B, TI_ptr, TJ_ptr, TT: tl.constexpr, BB: tl.constexpr):
    k = tl.program_id(0)
    I = tl.load(TI_ptr + k)
    J = tl.load(TJ_ptr + k)
    t = tl.arange(0, TT)
    c = tl.arange(0, BB)
    mc = c < B
    A = tl.load(P_ptr + k.to(tl.int64) * TT * TT + t[:, None] * TT + t[None, :])           # [TT, TT] rows of block I
    rI = I * TT + t
    rJ = J * TT + t
    mI = rI < n
    mJ = rJ < n
    XJ = tl.load(X_ptr + rJ[:, None].to(tl.int64) * B + c[None, :], mask=mJ[:, None] & mc[None, :], other=0.0)
    YI = tl.dot(A, XJ, input_precision="ieee")
    tl.atomic_add(Y_ptr + rI[:, None].to(tl.int64) * B + c[None, :], YI, mask=mI[:, None] & mc[None, :])
    if I != J:
        XI = tl.load(X_ptr + rI[:, None].to(tl.int64) * B + c[None, :], mask=mI[:, None] & mc[None, :], other=0.0)
        YJ = tl.dot(tl.trans(A), XI, input_precision="ieee")
        tl.atomic_add(Y_ptr + rJ[:, None].to(tl.int64) * B + c[None, :], YJ, mask=mJ[:, None] & mc[None, :])


_IDX = {}


def _tile_index(nt, device):
    key = (nt, str(device))
    if key not in _IDX:
        I = torch.repeat_interleave(torch.arange(nt, dtype=torch.int32), torch.arange(1, nt + 1))
        J = torch.cat([torch.arange(i + 1, dtype=torch.int32) for i in range(nt)]) if nt else torch.zeros(0, dtype=torch.int32)
        _IDX[key] = (I.to(device), J.to(device))
    return _IDX[key]


def matmul(P, n, X):
    """A X for packed tiles P of the symmetric n x n float32 matrix A; X (n, B) float32 (any B; blocks of 16 columns)."""
    nt, ntl = ntiles(n)
    if ntl == 0:
        return torch.zeros((n, X.shape[1]), dtype=torch.float32, device=X.device)
    TI, TJ = _tile_index(nt, X.device)
    return matmul_tiles(P, TI, TJ, n, X)


def matmul_tiles(P, TI, TJ, n, X):
    """A X for the symmetric n x n float32 matrix A given by its stored tiles P at the tile positions (TI, TJ) (int32,
    TI >= TJ, X's device), e.g. only the diagonal blocks' tiles of a block-diagonal matrix (coarse_blk); matmul is the
    full lower-triangle list."""
    X = X.contiguous()
    B = X.shape[1]
    Y = torch.zeros((n, B), dtype=torch.float32, device=X.device)
    ntl = len(TI)
    if ntl == 0 or B == 0:
        return Y
    for c0 in range(0, B, 16):
        nb = min(16, B - c0)
        Xc = X[:, c0:c0 + nb].contiguous() if B > 16 else X
        Yc = torch.zeros((n, nb), dtype=torch.float32, device=X.device) if B > 16 else Y
        _symm[(ntl,)](P, Xc, Yc, n, nb, TI, TJ, TT=T, BB=16, num_warps=4)
        if B > 16:
            Y[:, c0:c0 + nb] = Yc
    return Y
