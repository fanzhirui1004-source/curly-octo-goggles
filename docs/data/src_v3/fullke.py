"""Full elements share one element stiffness (src_v3, OPL_FULL_KE=1, deployment cells).

An element whose s^3 level-0 sub-cubes are all inside the material (moments_ad.full_rows: every psi >= 1e-9 at every
sub-cube corner) has the closed-form moments of the whole element, independent of tau, so these elements have the same
moments and the same K_e = M T. Their part of K x is one GEMM with that single K_e on the gathered element vectors,
    (K_full x) = sum_e P_e^T K_full P_e x = scatter( K_full [x_e1, x_e2, ...] ),
instead of forming (and, in the float32 correction, holding) one 81 x 81 matrix per element. The other ("partial")
elements keep the per-element path (packed upper-triangle Triton product with OPL_KE_SYMV / OPL_KE_SYMV64, else bmm).
Share of full elements: about 1% at tau = 0.18, 32% at 0.40, 45% at 0.55, 56% at 0.69.

Exactness: the shared matrix must be the very matrix the per-element path forms for these elements, else thousands of
elements carry the same rounding difference (a systematic perturbation of the operator: 1.9e-7 in the float32 K product
gave a 3e-6 change of the lattice compliance on hlat222). So
  - an element shares only if its moment row is bitwise equal to the representative's (float64 row for the float64 K,
    float32-rounded row for the float32 correction); the others stay partial;
  - the shared matrix is read from the same chunked GEMM the per-element path runs (rows lo:lo+chunk of M times the
    upper triangle T_up, float64 / float32), at the representative's row, then mirrored.
Only the summation order of the products differs (data-dependent rounding, like the atomics elsewhere).

Measured on hlat222 (docs/speedup_v3/CHANGES_CN.md, S4): every full element has its own last-bit moment pattern, so
any single shared float32 matrix differs from the per-element float32 matrices by about one unit in the last place in
many entries, coherently over thousands of elements; the float32 correction product then moves by ~1e-7 and the lattice
compliance by ~3e-6 (PCG iterations unchanged). In float64 the same sharing moves the K product by ~1e-15. Hence the
default: sharing in the float64 K product only (OPL_FULL_KE=1), the float32 correction per element (OPL_FULL_KE32=0).

C._fk = dict(f64=..., f32=...) with per precision: full, part (int64 element indices), Ke (81 x 81), dft (81 x n_full
element DOFs, transposed), n_full, n_part; built on first use after assemble_deploy (teacher resets it with the design)."""
import os
import torch
import fastidx as FI

ON = os.environ.get('OPL_FULL_KE', '1') == '1'                      # float64 K product (and hold64)
ON32 = os.environ.get('OPL_FULL_KE32', '0') == '1'                  # float32 correction: off by default (see below)
TOL = 1e-12


def _sym81(device):
    iu = torch.triu_indices(81, 81, device=device)
    pos = torch.full((81, 81), -1, dtype=torch.long, device=device)
    pos[iu[0], iu[1]] = torch.arange(iu.shape[1], device=device)
    return torch.where(pos >= 0, pos, pos.t()).reshape(-1)


def _split(C, full, rows, T, ch, packed):
    """Sharing set of one precision. The full elements' stored moment rows agree only to the summation order of their
    64 sub-cube terms (the last bits); the representative is the most frequent row pattern among them (rows of this
    precision: float64, or the float32-rounded rows), every full element within TOL of it shares, and the shared matrix
    is the representative's row of the chunked GEMM rows[lo:lo+ch] @ T as the per-element path computes it (T: upper
    triangle, packed and mirrored, or the 81 x 81 layout). The sharing therefore replaces each full element's own
    last-bit pattern by the most common one: no systematic offset beyond that."""
    E = len(rows)
    fi = torch.nonzero(full).squeeze(1)
    if len(fi) == 0:
        return dict(n_full=0, n_part=E, full=fi, part=torch.arange(E, device=rows.device))
    R = rows[fi]
    uq, inv, cnt = torch.unique(R, dim=0, return_inverse=True, return_counts=True)
    k = int(torch.argmax(cnt))
    e0 = int(fi[torch.nonzero(inv == k)[0, 0]])
    rep = rows[e0]
    dev = float(((R - rep[None]).abs().amax(1) / rep.abs().max()).max())
    if dev > TOL:                                                       # not one closed form: no sharing
        return dict(n_full=0, n_part=E, full=fi[:0], part=torch.arange(E, device=rows.device), spread=dev)
    pi = torch.nonzero(~full).squeeze(1)
    lo = (e0 // ch) * ch                                                # the per-element path's GEMM chunk
    row = (rows[lo:lo + ch] @ T)[e0 - lo]
    Ke = (row[_sym81(rows.device)] if packed else row).reshape(81, 81).contiguous()
    return dict(n_full=int(len(fi)), n_part=int(len(pi)), full=fi, part=pi, Ke=Ke, dft=C.dofs[fi].t().contiguous(),
                spread=dev, patterns=int(len(uq)), mode_count=int(cnt[k]))


def setup(C):
    fk = getattr(C, '_fk', None)
    if fk is not None:
        return fk
    import moments_ad as MA
    cells, taus, nrm, off = MA.cell_rows(C)
    full = MA.full_rows(cells, C.n, taus, nrm, off, s=C.s, surface=getattr(C, 'surface', 'P'))
    ch = C._lean_chunk
    f32 = torch.float32
    sym32 = os.environ.get('OPL_KE_SYMV') == '1'                        # float32 per-element path: packed (Triton)
    T32 = C.Tm_up.reshape(125, -1).to(f32) if sym32 else C.Tm.reshape(125, 81 * 81).to(f32)   # or held dense K_e
    fk = dict(f64=_split(C, full, C.M, C.Tm_up, ch, True),            # float64: both paths run the T_up GEMM
              f32=_split(C, full, C.M.to(f32), T32, ch, sym32))
    C._fk = fk
    return fk


def full_product(fs, x, y):
    """y += sum over the shared elements of P_e^T Ke P_e x (x, y: nb x B in fs['Ke']'s dtype), one GEMM."""
    if fs['n_full'] == 0:
        return y
    dft = fs['dft']
    B = x.shape[1]
    Xt = FI.rows(x, dft) if B > 1 else x[dft]                           # 81 x n_full x B
    Yt = fs['Ke'] @ Xt.reshape(81, -1)                                  # 81 x (n_full B)
    y.index_add_(0, dft.reshape(-1), Yt.reshape(-1, B))
    return y
