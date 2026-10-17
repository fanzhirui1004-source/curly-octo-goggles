"""Matrix-free float32 element product (src_v3 P13, OPL_V3_KMF): y += sum_e P_e^T K_e P_e x for the float32 correction
product (teacher._Kfp32) without forming the element matrices.

A deployment element is a Q2 hexahedron (27 nodes, 81 DOFs) whose stiffness is the moment-fitted sum
    K_e = sum_m M_e[m] T_m,   M_e[m] = int_(cut part of e) xi^a eta^b zeta^c dV   (m = (a, b, c), a, b, c <= 4)
(element_moments.pattern_operators: T_m holds the monomial coefficients of grad N_i . C . grad N_j). The integrands are
polynomials of degree <= 4 in each local coordinate, which the 5-point Gauss rule per axis integrates exactly, so with
the weights w_e = V^-1 M_e (V[m, q] = xi_q^a eta_q^b zeta_q^c at the 125 tensor Gauss points; float64, then rounded)
    K_e = sum_q w_e[q] B_q^T C B_q      (the same matrix in exact arithmetic; float64 check: 8e-15 relative)
and K_e x_e = sum_q w_e[q] sum_d dN/dx_d(q) sigma_.d(q), sigma = lam tr(eps) I + 2 mu eps, eps from the gradient
of x_e at q. The product then reads 125 weights and 27 node indices per element (~0.6 kB) instead of the 3321 packed
float32 entries (13.3 kB) of the formed matrix, and needs no formation of the matrices (the E x 125 x 3321 GEMM of
_Kfp32.hold). Arithmetic: per tile of EB elements and PB quadrature points, the nine gradient components as float32
dots (input_precision 'ieee': no TF32) of the shared derivative table (3 x 128 x 32, zero padded) with the gathered
element displacements, the stress, the weighted back projection as nine more dots; atomic adds into y as before.
Rounding differs from the formed-matrix product (float32 error against float64 of the same order, 3e-7 median on
random cut boxes, see scripts/t_kmf.py); the flop count grows with the number of columns while the formed-matrix
product reads the matrices once for all columns, so it is used for narrow products only (B <= MAXB).
Two implementations: the CUDA C++ kernel kmf_kernel.cu (default, OPL_V3_KMF_CUDA=1: sum factorisation, ~12.4k
multiply-adds per element; compiled once with nvcc into a cached shared library, launched through ctypes on torch's
current stream), and the Triton kernel below (OPL_V3_KMF_CUDA=0, or no nvcc: dense 125 x 27 derivative tables as fp32
dots, ~74k multiply-adds per element; measured 413 us per product on the plate cell, slower than the formed-matrix
product's 183 us, so not useful by itself)."""
import os
import ctypes
import hashlib
import shutil
import subprocess
import numpy as np
import torch
import triton
import triton.language as tl

ON = os.environ.get('OPL_V3_KMF', '0') == '1'
CUDA = os.environ.get('OPL_V3_KMF_CUDA', '1') == '1'
MAXB = int(os.environ.get('OPL_V3_KMF_MAXB', '16'))
EPB = int(os.environ.get('OPL_V3_KMF_EPB', '4'))                       # CUDA kernel: elements per block (2, 4, 8)
GF = int(os.environ.get('OPL_V3_KMF_GF', '1') or 0)                   # CUDA: the ghost faces matrix-free too
ON64 = os.environ.get('OPL_V3_KMF64', '1') == '1'                      # src_v3 P14: the float64 K product as well
PTS = os.environ.get('OPL_V3_KMF_PTS', 'equi')                          # quadrature points: 'equi' | 'gauss'


def points():
    """The 5 points per axis of the fitted rule: equispaced (-1, -1/2, 0, 1/2, 1: the Q2 shape function values and
    derivatives there are dyadic, exact in float32, so the tables carry no rounding common to all elements) or the
    Gauss-Legendre points. Any 5 distinct points integrate the element's integrands exactly with the fitted weights."""
    if PTS == 'gauss':
        return np.polynomial.legendre.leggauss(5)[0]
    return np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
FPB = int(os.environ.get('OPL_V3_KMF_FPB', '8'))                       # faces per group (4, 8)
GF_BLOCKS = int(os.environ.get('OPL_V3_KMF_GFBLK', '0') or 0)          # blocks of the face kernel (0: 8 per SM)
NB = int(os.environ.get('OPL_V3_KMF_NB', '0') or 0)                    # columns per block (1, 2, 4; 0: by B)
GF2 = os.environ.get('OPL_V3_KMF_GF2', 'auto')                         # ghost faces: DOF-parallel kernel (gf2): '1' always,
                                                                        # 'auto' from 2 columns on (faster there), '0' never
FPB2 = int(os.environ.get('OPL_V3_KMF_FPB2', '4'))                     # gf2: faces per group (2, 4)
EF = os.environ.get('OPL_V3_KMF_EF', '0') == '1'                        # P13b: elements and faces in one kernel (kef)
EB = int(os.environ.get('OPL_V3_KMF_EB', '32'))
PB = int(os.environ.get('OPL_V3_KMF_PB', '16'))
WARPS = int(os.environ.get('OPL_V3_KMF_WARPS', '4'))
_TAB = {}


def _lag(k, x):
    return {-1: x * (x - 1) / 2, 0: 1 - x * x, 1: x * (x + 1) / 2}[int(k)]


def _dlag(k, x):
    return {-1: x - 0.5, 0: -2 * x, 1: x + 0.5}[int(k)]


def tables(xi_nodes, n, device):
    """(DN, Vinv3T) for one node ordering xi_nodes (27 x 3 local coordinates in {-1, 0, 1}, the ordering the templates
    use) and grid parameter n: DN (3, 128, 32) float32, DN[d, q, i] = dN_i/dx_d at Gauss point q (physical: 2n d/dxi);
    Vinv3T (125, 125) float64 with w = M @ Vinv3T."""
    key = (np.asarray(xi_nodes, np.int64).tobytes(), int(n), str(device), PTS)
    t = _TAB.get(key)
    if t is None:
        g = points()
        V1i = np.linalg.inv(np.vander(g, 5, increasing=True).T)               # w1 = V1i m1 (V1[a, p] = g_p^a)
        Vinv3 = np.kron(np.kron(V1i, V1i), V1i)                               # [(p, q, r), (a, b, c)], row-major
        xi = np.asarray(xi_nodes).reshape(27, 3)
        DN = np.zeros((3, 128, 32))
        for qi, (p, q, r) in enumerate(np.ndindex(5, 5, 5)):
            x = (g[p], g[q], g[r])
            for i in range(27):
                v = [_lag(xi[i, d], x[d]) for d in range(3)]
                dv = [_dlag(xi[i, d], x[d]) for d in range(3)]
                DN[0, qi, i] = dv[0] * v[1] * v[2] * 2 * n
                DN[1, qi, i] = v[0] * dv[1] * v[2] * 2 * n
                DN[2, qi, i] = v[0] * v[1] * dv[2] * 2 * n
        t = _TAB[key] = (torch.tensor(DN, dtype=torch.float32, device=device).contiguous(),
                         torch.tensor(Vinv3.T, dtype=torch.float64, device=device).contiguous())
    return t


def weights(M, Vinv3T, chunk=8192):
    """Quadrature weights of the elements (float64 moments M, E x 125): float32 (128, E), rows 125.. zero."""
    E = M.shape[0]
    WT = torch.zeros((128, E), dtype=torch.float32, device=M.device)
    for lo in range(0, E, chunk):
        WT[:125, lo:lo + chunk] = (M[lo:lo + chunk] @ Vinv3T).t().to(torch.float32)
    return WT


def nodes(dofs):
    """Element node indices (E x 27 int32) of node-major element DOFs (dofs[e, 3 i + c] = 3 node + c)."""
    return (dofs[:, 0::3] // 3).to(torch.int32).contiguous()


@triton.jit
def _kmf(x_ptr, y_ptr, nd_ptr, w_ptr, dn_ptr, E, LD, lam, mu, EB: tl.constexpr, PB: tl.constexpr):
    """Elements pid * EB ..; one column c = program_id(1) of x (row stride LD)."""
    e = tl.program_id(0) * EB + tl.arange(0, EB)
    me = e < E
    col = tl.program_id(1)
    i = tl.arange(0, 32)
    mi = i < 27
    m2 = mi[:, None] & me[None, :]
    nd = tl.load(nd_ptr + e[None, :].to(tl.int64) * 27 + i[:, None], mask=m2, other=0).to(tl.int64)     # [32, EB]
    r0 = (3 * nd) * LD + col
    u0 = tl.load(x_ptr + r0, mask=m2, other=0.0)
    u1 = tl.load(x_ptr + r0 + LD, mask=m2, other=0.0)
    u2 = tl.load(x_ptr + r0 + 2 * LD, mask=m2, other=0.0)
    y0 = tl.zeros((32, EB), dtype=tl.float32)
    y1 = tl.zeros((32, EB), dtype=tl.float32)
    y2 = tl.zeros((32, EB), dtype=tl.float32)
    for pb in tl.static_range(0, 128, PB):
        q = pb + tl.arange(0, PB)
        a0 = tl.load(dn_ptr + (0 * 128 + q[:, None]) * 32 + i[None, :])                                  # [PB, 32]
        a1 = tl.load(dn_ptr + (1 * 128 + q[:, None]) * 32 + i[None, :])
        a2 = tl.load(dn_ptr + (2 * 128 + q[:, None]) * 32 + i[None, :])
        g00 = tl.dot(a0, u0, input_precision='ieee')                        # g_dc = d u_c / d x_d at the points [PB, EB]
        g01 = tl.dot(a0, u1, input_precision='ieee')
        g02 = tl.dot(a0, u2, input_precision='ieee')
        g10 = tl.dot(a1, u0, input_precision='ieee')
        g11 = tl.dot(a1, u1, input_precision='ieee')
        g12 = tl.dot(a1, u2, input_precision='ieee')
        g20 = tl.dot(a2, u0, input_precision='ieee')
        g21 = tl.dot(a2, u1, input_precision='ieee')
        g22 = tl.dot(a2, u2, input_precision='ieee')
        w = tl.load(w_ptr + q[:, None].to(tl.int64) * E + e[None, :], mask=me[None, :], other=0.0)       # [PB, EB]
        ld = lam * (g00 + g11 + g22)
        s00 = w * (ld + 2 * mu * g00)
        s11 = w * (ld + 2 * mu * g11)
        s22 = w * (ld + 2 * mu * g22)
        s01 = w * (mu * (g01 + g10))
        s02 = w * (mu * (g02 + g20))
        s12 = w * (mu * (g12 + g21))
        b0 = tl.trans(a0)                                                   # [32, PB]
        b1 = tl.trans(a1)
        b2 = tl.trans(a2)
        y0 = tl.dot(b0, s00, y0, input_precision='ieee')                    # y_p[i] += sum_d dN_i/dx_d sigma_pd
        y0 = tl.dot(b1, s01, y0, input_precision='ieee')
        y0 = tl.dot(b2, s02, y0, input_precision='ieee')
        y1 = tl.dot(b0, s01, y1, input_precision='ieee')
        y1 = tl.dot(b1, s11, y1, input_precision='ieee')
        y1 = tl.dot(b2, s12, y1, input_precision='ieee')
        y2 = tl.dot(b0, s02, y2, input_precision='ieee')
        y2 = tl.dot(b1, s12, y2, input_precision='ieee')
        y2 = tl.dot(b2, s22, y2, input_precision='ieee')
    tl.atomic_add(y_ptr + r0, y0, mask=m2)
    tl.atomic_add(y_ptr + r0 + LD, y1, mask=m2)
    tl.atomic_add(y_ptr + r0 + 2 * LD, y2, mask=m2)


def matvec_(y, nd, WT, DN, x, lam, mu, eb=None, pb=None, warps=None):
    """y += K x (float32, in place): nd (E x 27 int32 element nodes), WT (128 x E float32 weights), DN (3 x 128 x 32),
    x, y (nb x B) float32 contiguous; lam, mu the Lame constants of the templates."""
    E = nd.shape[0]
    if E == 0:
        return y
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32
    B = x.shape[1]
    eb, pb, warps = eb or EB, pb or PB, warps or WARPS
    _kmf[(triton.cdiv(E, eb), B)](x, y, nd, WT, DN, E, B, float(lam), float(mu), EB=eb, PB=pb, num_warps=warps)
    return y


# ---------------------------------------------------------------------- CUDA C++ kernel (sum factorisation)
_LIB = None
_LIB_ERR = None
_TABLES = [None]                                                        # the point set in the library's constant memory


def kind():
    """'cuda' (switch on, library not known to fail) or 'triton'."""
    return 'cuda' if CUDA and _LIB_ERR is None else 'triton'


def _lib():
    """The compiled kmf_kernel.cu (cached by source hash and architecture), initialised with the 1-D tables."""
    global _LIB
    if _LIB is not None:
        if _TABLES[0] != PTS:                                               # (a test switched the points)
            _init_tables(_LIB)
        return _LIB
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kmf_kernel.cu')
    code = open(src, 'rb').read()
    maj, mnr = torch.cuda.get_device_capability()
    arch = f'sm_{maj}{mnr}'
    tag = hashlib.sha1(code + arch.encode()).hexdigest()[:16]
    d = os.environ.get('OPL_V3_KMF_DIR') or os.path.join(os.path.expanduser('~'), '.cache', 'opl_v3_kmf')
    os.makedirs(d, exist_ok=True)
    so = os.path.join(d, f'libkmf_{tag}.so')
    if not os.path.exists(so):
        nvcc = os.environ.get('OPL_V3_NVCC') or shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
        tmp = f'{so}.{os.getpid()}.tmp'
        r = subprocess.run([nvcc, '-O3', '-std=c++17', f'-arch={arch}', '-Xcompiler', '-fPIC', '-shared', '-o', tmp, src],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError('NVCC:' + (r.stderr or r.stdout)[-1500:])
        os.replace(tmp, so)
    lib = ctypes.CDLL(so)
    lib.kmf_init.argtypes = [ctypes.c_void_p] * 4
    lib.kmf_launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.kmf_launch64.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.gf_launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
    lib.gf_launch64.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    lib.gf2_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
    lib.kef_launch.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.gf2_launch64.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    _init_tables(lib)
    _LIB = lib
    return lib


def _init_tables(lib):
    """The 1-D shape function tables at points() into the library's constant memory (float32 and float64)."""
    g = points()
    L64 = np.array([[_lag(k, gq) for k in (-1, 0, 1)] for gq in g], dtype=np.float64)
    D64 = np.array([[_dlag(k, gq) for k in (-1, 0, 1)] for gq in g], dtype=np.float64)
    L, D = L64.astype(np.float32), D64.astype(np.float32)
    rc = lib.kmf_init(L.ctypes.data, D.ctypes.data, L64.ctypes.data, D64.ctypes.data)
    if rc != 0:
        raise RuntimeError(f'KMF_INIT:{rc}')
    _TABLES[0] = PTS                                                        # the tables it holds (constant memory)


def cuda_ok():
    """CUDA path usable (switch on and the library builds; a failure is remembered: the Triton path then)."""
    global _LIB_ERR
    if kind() != 'cuda':
        return False
    try:
        _lib()
        return True
    except Exception as ex:                                                 # noqa: BLE001 (no nvcc: Triton path)
        _LIB_ERR = repr(ex)[:1600]
        return False


def nodes_tensor(C):
    """Element node indices in tensor order (E x 27 int32), cached on the cell (C._kmf_nd; reset with the design)."""
    nd = getattr(C, '_kmf_nd', None)
    if nd is None:
        xi = np.asarray(C._kmf_spec[0]).reshape(27, 3)
        t = (9 * (xi[:, 0] + 1) + 3 * (xi[:, 1] + 1) + (xi[:, 2] + 1)).astype(np.int64)
        inv = np.empty(27, np.int64); inv[t] = np.arange(27)                 # tensor slot t <- element node inv[t]
        nd = C._kmf_nd = nodes(C.dofs)[:, torch.as_tensor(inv, device=C.dofs.device)].contiguous()
    return nd


def state(C):
    """Per-cell data of the product for the current implementation (float64 moments C.M, element DOFs C.dofs,
    C._kmf_spec = (node ordering, lam, mu)): dict(kind='cuda': nd in tensor order (E x 27 int32), w (E x 125 float32,
    times (2n)^2); kind='triton': nd (E x 27), WT (128 x E), DN (3 x 128 x 32))."""
    xi, lam, mu = C._kmf_spec
    DN, Vi = tables(xi, C.n, C.M.device)
    if cuda_ok():
        nd = nodes_tensor(C)
        wl, wm = _weights_lm(C, Vi, torch.arange(C.M.shape[0], device=C.M.device), lam, mu, torch.float32)
        return dict(kind='cuda', nd=nd, wl=wl, wm=wm, lam=lam, mu=mu, pts=PTS)
    return dict(kind='triton', nd=nodes(C.dofs), WT=weights(C.M, Vi), DN=DN, lam=lam, mu=mu)


def _weights_lm(C, Vi, elements, lam, mu, dtype, chunk=8192):
    """(lam w, mu w) of the given elements, w = M Vi (2n)^2 in float64, then rounded to dtype (E x 125 each)."""
    E = len(elements)
    wl = torch.empty((E, 125), dtype=dtype, device=C.M.device)
    wm = torch.empty((E, 125), dtype=dtype, device=C.M.device)
    for lo in range(0, E, chunk):
        w = (C.M[elements[lo:lo + chunk]] @ Vi) * (2.0 * C.n) ** 2
        wl[lo:lo + chunk] = (w * lam).to(dtype)
        wm[lo:lo + chunk] = (w * mu).to(dtype)
    return wl, wm


def _check_pts(st):
    _lib()
    if _TABLES[0] != st['pts']:
        raise RuntimeError('KMF_PTS: the library tables and the cell weights use other points')


def apply_(y, st, x, epb=None, nb=None):
    """y += K x through state(C) (x, y: nb x B float32 contiguous)."""
    if st['kind'] == 'triton':
        return matvec_(y, st['nd'], st['WT'], st['DN'], x, st['lam'], st['mu'])
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32 and y.dtype == torch.float32
    _check_pts(st)
    E, B = st['nd'].shape[0], x.shape[1]
    rc = _lib().kmf_launch(x.data_ptr(), y.data_ptr(), st['nd'].data_ptr(), st['wl'].data_ptr(), st['wm'].data_ptr(), E, B,
                           int(epb or EPB), int(NB if nb is None else nb), torch.cuda.current_stream(x.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'KMF_LAUNCH:{rc}')
    return y


def gf_state(C, nd, dtype=torch.float32):
    """Ghost faces of the cell for gf_kernel (CUDA path): fe, fn (element and +1 neighbour along the face direction,
    int32, faces grouped by direction), cuts c1, c2, fmap (3 x 45 int32: face node -> 32 * neighbour + tensor slot),
    G (3 x 45 x 45 float32: gamma b_d^T b_d from the float64 templates, b_d the scalar block of B_d = b_d (x) I_3);
    fnode (faces x 45 int32: the face nodes, for gf2_kernel). The integer parts are kept once per cell (C._kmf_gfi,
    shared by the float32 and float64 states; reset with the design)."""
    gi = getattr(C, '_kmf_gfi', None)
    if gi is None:
        gi = C._kmf_gfi = _gf_int(C, nd)
    return dict(gi, G=gi['G64'].to(dtype), G64=None)


def _gf_int(C, nd):
    from pathlib import Path
    d = Path(C._body_dir) / C.case
    faces = np.load(d / 'GP_FACES.npy').astype(np.int64)
    tpl = np.load(Path(C._body_dir) / f'GP_TEMPLATES_n{C.n}.npz')
    offs, Bc = tpl['offsets'].astype(np.int64), tpl['canonical'].astype(np.float64)
    nT = len(offs)
    if nT != 3 or Bc.shape[1:] != (54, 135):
        raise ValueError('GF_TEMPLATES')
    order = np.concatenate([np.flatnonzero(faces[:, 2] == k) for k in range(3)])
    cuts = np.cumsum([0] + [int((faces[:, 2] == k).sum()) for k in range(3)])
    cells = np.asarray(C.cells)
    ek = np.eye(3, dtype=np.int64)
    if not (cells[faces[:, 1]] - cells[faces[:, 0]] == ek[faces[:, 2]]).all():
        raise ValueError('GF_NEIGHBOUR')
    fmap = np.zeros((3, 45), np.int32)
    G = np.zeros((3, 45, 45))
    for k in range(3):
        o = offs[k]
        for j in range(45):
            if o[j, k] <= 2:
                fmap[k, j] = 9 * o[j, 0] + 3 * o[j, 1] + o[j, 2]
            else:
                i = o[j].copy(); i[k] -= 2
                fmap[k, j] = 32 + 9 * i[0] + 3 * i[1] + i[2]
        b = Bc[k][0::3, 0::3]
        Bk = np.kron(b, np.eye(3))
        if np.abs(Bk - Bc[k]).max() > 1e-12 * np.abs(Bc[k]).max():
            raise ValueError('GF_NOT_SCALAR_KRON')
        G[k] = C.gamma * (b.T @ b)
    dev = C.M.device
    t = lambda a, dt_=torch.int32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt_, device=dev)
    fe, fn, fm = t(faces[order, 0]), t(faces[order, 1]), t(fmap)
    nf = int(cuts[3])
    dd = (torch.arange(nf, device=dev) >= int(cuts[1])).long() + (torch.arange(nf, device=dev) >= int(cuts[2])).long()
    m = fm.long()[dd]                                                       # nf x 45
    el = torch.where(m >= 32, fn.long()[:, None], fe.long()[:, None])
    fnode = nd.long()[el, m & 31].to(torch.int32).contiguous()
    return dict(fe=fe, fn=fn, c1=int(cuts[1]), c2=int(cuts[2]), nf=nf, fmap=fm, nd=nd, fnode=fnode,
                G64=torch.as_tensor(G, dtype=torch.float64, device=dev))


_SMS = {}


def _use_gf2(gf2, B):
    if gf2 is not None:
        return bool(gf2)
    return GF2 in (True, 1, '1') or (GF2 == 'auto' and B >= 2)


def _blocks(dev):
    if dev not in _SMS:
        _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return GF_BLOCKS or 8 * _SMS[dev]


def gf_apply_(y, gst, x, fpb=None, blocks=None, nb=None, gf2=None):
    """y += gamma G x (ghost faces, float32) through gf_state (x, y: nb x B float32 contiguous)."""
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32 and y.dtype == torch.float32
    if blocks is None:
        blocks = _blocks(x.device)
    if _use_gf2(gf2, x.shape[1]):
        rc = _lib().gf2_launch(x.data_ptr(), y.data_ptr(), gst['fnode'].data_ptr(), gst['G'].data_ptr(), gst['nf'], gst['c1'],
                               gst['c2'], x.shape[1], int(fpb or FPB2), int(NB if nb is None else nb), int(blocks),
                               torch.cuda.current_stream(x.device).cuda_stream)
        if rc != 0:
            raise RuntimeError(f'GF2_LAUNCH:{rc}')
        return y
    rc = _lib().gf_launch(x.data_ptr(), y.data_ptr(), gst['nd'].data_ptr(), gst['fe'].data_ptr(), gst['fn'].data_ptr(),
                          gst['fmap'].data_ptr(), gst['G'].data_ptr(), gst['nf'], gst['c1'], gst['c2'], x.shape[1],
                          int(fpb or FPB), int(NB if nb is None else nb), int(blocks),
                          torch.cuda.current_stream(x.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'GF_LAUNCH:{rc}')
    return y


# ---------------------------------------------------------------------- src_v3 P14: the float64 K product
def state64(C, elements):
    """Float64 data of the matrix-free product over the given elements (int64 indices; the partial elements when the
    full ones keep fullke's shared-matrix GEMM): nd (tensor order) and w (float64 weights times (2n)^2), plus the ghost
    faces (float64 Gram matrices, all faces; nd of all elements for their node lists)."""
    xi, lam, mu = C._kmf_spec
    _, Vi = tables(xi, C.n, C.M.device)
    nd_all = nodes_tensor(C)
    nd = nd_all[elements].contiguous()
    wl, wm = _weights_lm(C, Vi, elements, lam, mu, torch.float64)
    return dict(kind='cuda64', nd=nd, wl=wl, wm=wm, lam=float(lam), mu=float(mu), pts=PTS,
                gf=gf_state(C, nd_all, torch.float64))


def apply64_(y, st, x, epb=None, nb=None):
    """y += K_e x over state64's elements, float64 (x, y: nb x B float64 contiguous)."""
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float64 and y.dtype == torch.float64
    _check_pts(st)
    E, B = st['nd'].shape[0], x.shape[1]
    rc = _lib().kmf_launch64(x.data_ptr(), y.data_ptr(), st['nd'].data_ptr(), st['wl'].data_ptr(), st['wm'].data_ptr(),
                             E, B, int(epb or 4), int(nb or 0), torch.cuda.current_stream(x.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'KMF64_LAUNCH:{rc}')
    return y


def gf_apply64_(y, gst, x, blocks=None, nb=None, gf2=None):
    """y += gamma G x (ghost faces, float64) through gf_state(..., float64)."""
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float64 and y.dtype == torch.float64
    if blocks is None:
        blocks = _blocks(x.device)
    if _use_gf2(gf2, x.shape[1]):
        rc = _lib().gf2_launch64(x.data_ptr(), y.data_ptr(), gst['fnode'].data_ptr(), gst['G'].data_ptr(), gst['nf'],
                                 gst['c1'], gst['c2'], x.shape[1], int(nb or 0), int(blocks),
                                 torch.cuda.current_stream(x.device).cuda_stream)
        if rc != 0:
            raise RuntimeError(f'GF2_LAUNCH64:{rc}')
        return y
    rc = _lib().gf_launch64(x.data_ptr(), y.data_ptr(), gst['nd'].data_ptr(), gst['fe'].data_ptr(), gst['fn'].data_ptr(),
                            gst['fmap'].data_ptr(), gst['G'].data_ptr(), gst['nf'], gst['c1'], gst['c2'], x.shape[1],
                            int(nb or 0), int(blocks), torch.cuda.current_stream(x.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'GF64_LAUNCH:{rc}')
    return y


# ---------------------------------------------------------------------- src_v3 P13b: elements and faces in one kernel
def efn_table(gst, E):
    """(E x 3 int32) the neighbour element of each element's ghost face along axis d (-1: none), from gf_state."""
    efn = gst.get('efn')
    if efn is None:
        nf, c1, c2 = gst['nf'], gst['c1'], gst['c2']
        dev = gst['fe'].device
        d = (torch.arange(nf, device=dev) >= c1).long() + (torch.arange(nf, device=dev) >= c2).long()
        efn = torch.full((E * 3,), -1, dtype=torch.int32, device=dev)
        key = gst['fe'].long() * 3 + d
        if len(torch.unique(key)) != nf:
            raise ValueError('EF_FACE_TWICE')
        efn[key] = gst['fn']
        efn = gst['efn'] = efn.view(E, 3).contiguous()
    return efn


def apply_ef_(y, st, x, nb=None, blocks=None):
    """y += (K_elements + gamma G_faces) x in one kernel (state(C) with its 'gf' tables; x, y float32 contiguous)."""
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == torch.float32 and y.dtype == torch.float32
    _check_pts(st)
    gst = st['gf']
    E, B = st['nd'].shape[0], x.shape[1]
    efn = efn_table(gst, E)
    rc = _lib().kef_launch(x.data_ptr(), y.data_ptr(), st['nd'].data_ptr(), st['wl'].data_ptr(), st['wm'].data_ptr(),
                           efn.data_ptr(), gst['fmap'].data_ptr(), gst['G'].data_ptr(), E, B, int(NB if nb is None else nb),
                           int(blocks if blocks is not None else _blocks(x.device)),
                           torch.cuda.current_stream(x.device).cuda_stream)
    if rc != 0:
        raise RuntimeError(f'KEF_LAUNCH:{rc}')
    return y
