"""V0: representation-capacity test of the learned two-scale super-element on one saved label.

Structure (all free coefficients, no geometry network):
  K = L^T L on (Gamma + I), L block upper-triangular:
    L_GG : sparse band on the fine face grid (pairs within radius r_near)
    L_GI : sparse couplings from fine face nodes to coarse interior nodes within rho
    L_II : dense upper-triangular on the coarse interior
  S_hat = K_GG - K_GI K_II^-1 K_IG + V diag(exp(a)) V^T   (PSD by construction)
  A_hat = B S_hat B^T   (rigid modes annihilated by the compiler quotient)
Loss: energy metric  sum_probes ||R*^-T (A_hat - A) z||^2 / ||R* z||^2  over physical probe ensembles.
Read-only on all labels; outputs go to a new directory.
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
from stage_cutfem_neural_a.data import load_teacher_dense, write_json
from stage_cutfem_neural_a.dense_fit import FactorSample, make_directions, json_finite
from stage_cutfem_neural_a.dense_row_fit import append_json
from stage_cutfem_neural_a.elimination_reference import DenseUpperFactor, load_upper_factor, quotient_dense, evaluate
from stage_cutfem_neural_a.offdiagonal_diagnostic import exact_offdiagonal_alignment

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device('cuda')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sync():
    torch.cuda.synchronize(); return time.perf_counter()


# ----------------------------------------------------------------------------- geometry helpers
def face_ids(ijk, N):
    face = np.full(len(ijk), -1)
    for ax in range(3):
        face[(ijk[:, ax] == 0) & (face < 0)] = 2 * ax
        face[(ijk[:, ax] == N - 1) & (face < 0)] = 2 * ax + 1
    return face


def support_weights(ijk, N, tau_corners, cut_plane, sub=16):
    """Material area fraction of each face node's Q2 basis support, from the analytic level set only."""
    n = (N - 1) // 2
    weights = np.zeros(len(ijk))
    axis = np.linspace(0, 1, 2 * n * sub + 1)
    corners = list(__import__('itertools').product((0, 1), repeat=3))
    normal, offset = np.asarray(cut_plane[:3], dtype=float), float(cut_plane[3])
    nrm = np.linalg.norm(normal)
    for f in range(6):
        ax, side = f // 2, f % 2
        free = [a for a in range(3) if a != ax]
        g = np.meshgrid(axis, axis, indexing='ij')
        xyz = np.zeros(g[0].shape + (3,)); xyz[..., free[0]] = g[0]; xyz[..., free[1]] = g[1]; xyz[..., ax] = float(side)
        tau = np.zeros(g[0].shape)
        for value, corner in zip(tau_corners, corners):
            tau += value * np.prod([xyz[..., d] if corner[d] else 1 - xyz[..., d] for d in range(3)], axis=0)
        phi = np.cos(2 * np.pi * xyz).sum(axis=-1)
        material = (np.abs(phi) <= tau)
        if nrm > 0:
            material &= ((offset - xyz @ normal) / nrm >= 0)
        cum = np.zeros((material.shape[0] + 1, material.shape[1] + 1)); cum[1:, 1:] = material.cumsum(0).cumsum(1)
        on = np.nonzero(face_ids(ijk, N) == f)[0]
        for idx in on:
            a, b = ijk[idx, free[0]], ijk[idx, free[1]]
            ha = 2 if a % 2 == 0 else 1; hb = 2 if b % 2 == 0 else 1     # support half-width in fine steps
            a0, a1 = max(a - ha, 0), min(a + ha, 2 * n); b0, b1 = max(b - hb, 0), min(b + hb, 2 * n)
            A0, A1, B0, B1 = a0 * sub, a1 * sub, b0 * sub, b1 * sub
            area = (A1 - A0) * (B1 - B0)
            filled = cum[A1, B1] - cum[A0, B1] - cum[A1, B0] + cum[A0, B0]
            weights[idx] = max(weights[idx], filled / area if area > 0 else 0.0)
    return weights


def shape1d(t):
    return np.stack((2 * (t - .5) * (t - 1), 4 * t * (1 - t), 2 * t * (t - .5)), axis=-1)


def coarse_interpolation(ijk, N, stride):
    """P: coarse face nodes (all, nested) -> fine face nodes, 2D Q2 interpolation. Returns dense (n_fine, n_coarse)."""
    rows, cols, vals = [], [], []; cid = {}
    faces = face_ids(ijk, N)
    for f, (i, j, k) in enumerate(ijk):
        fixed = faces[f] // 2; free = [a for a in range(3) if a != fixed]
        w = np.ones(1); nc_ = [np.array([i, j, k])]
        for a in free:
            v = (i, j, k)[a]; e = 2 * stride; base = (v // e) * e
            if base == N - 1: base -= e
            sh = shape1d((v - base) / e); locs = base + np.array([0, stride, 2 * stride])
            nc_ = [c.copy() for c in nc_ for _ in range(3)]
            for m, c in enumerate(nc_): c[a] = locs[m % 3]
            w = (w[:, None] * sh[None, :]).ravel()
        for c, wv in zip(nc_, w):
            if abs(wv) < 1e-14: continue
            key = tuple(int(v) for v in c); cc = cid.setdefault(key, len(cid)); rows.append(f); cols.append(cc); vals.append(wv)
    P = np.zeros((len(ijk), len(cid))); P[rows, cols] = vals
    return P


def pairs_within(xa, xb, radius, block=2048, upper=False):
    """Index pairs (i, j) with |xa_i - xb_j| <= radius; if upper, only j >= i (same set)."""
    out_i, out_j = [], []
    for b in range(0, len(xa), block):
        d = torch.cdist(xa[b:b + block], xb)
        m = d <= radius
        if upper:
            ii = torch.arange(b, min(b + block, len(xa)), device=xa.device)[:, None]
            m &= (torch.arange(len(xb), device=xa.device)[None, :] >= ii)
        i, j = torch.nonzero(m, as_tuple=True)
        out_i.append(i + b); out_j.append(j)
    return torch.cat(out_i), torch.cat(out_j)


def expand_dofs(node_i, node_j, upper):
    """Node pairs -> 3x3 DOF blocks. For upper (i<=j): diagonal node blocks keep only the upper triangle."""
    ci = node_i[:, None] * 3 + torch.arange(3, device=node_i.device)[None, :]
    cj = node_j[:, None] * 3 + torch.arange(3, device=node_j.device)[None, :]
    I = ci[:, :, None].expand(-1, 3, 3).reshape(-1); J = cj[:, None, :].expand(-1, 3, 3).reshape(-1)
    if upper:
        keep = J >= I; I, J = I[keep], J[keep]
    return I, J


# ----------------------------------------------------------------------------- model
class SuperElement(torch.nn.Module):
    """K = [[L_GG^T L_GG, L_GG^T L_GI], [L_GI^T L_GG, C_II^T C_II]] with ||L_GI C_II^-1||_2 <= 1 enforced,
    so K is PSD and S_hat = L_GG^T (I - L_GI C_II^-1 C_II^-T L_GI^T) L_GG + V diag(exp a) V^T is PSD."""
    def __init__(self, xyz_face, xyz_int, r_near, rho, rank, w_face, s_unit, seed, decay=0.03, off_scale=0.3, gi_scale=0.1, ii_scale=0.01):
        super().__init__()
        g = torch.Generator(device='cpu').manual_seed(seed)
        q, nI = 3 * len(xyz_face), 3 * len(xyz_int)
        self.q, self.nI, self.root = q, nI, math.sqrt(s_unit)
        # L_GG: upper band, coalesced index order
        ni, nj = pairs_within(xyz_face, xyz_face, r_near, upper=True)
        I, J = expand_dofs(ni, nj, upper=True)
        order = torch.argsort(I * q + J); I, J = I[order], J[order]
        diag = I == J
        dist = torch.linalg.vector_norm(xyz_face[I // 3] - xyz_face[J // 3], dim=1)
        self.register_buffer('gg_index', torch.stack((I, J)))
        self.register_buffer('gg_index_t', torch.stack((J, I)))
        self.register_buffer('gg_diag', diag)
        self.register_buffer('gg_scale', (off_scale * self.root * torch.exp(-dist / decay))[~diag])
        self.gg_logdiag = torch.nn.Parameter(0.5 * torch.log(s_unit * w_face.repeat_interleave(3)).to(DEV))
        self.gg_theta = torch.nn.Parameter(torch.zeros(int((~diag).sum()), dtype=torch.float64, device=DEV))
        # L_GI: face DOF -> interior DOF couplings
        mi, mj = pairs_within(xyz_face, xyz_int, rho)
        Ig, Ji = expand_dofs(mi, mj, upper=False)
        order = torch.argsort(Ig * nI + Ji); Ig, Ji = Ig[order], Ji[order]
        self.register_buffer('gi_index', torch.stack((Ig, Ji)))
        self.register_buffer('gi_index_t', torch.stack((Ji, Ig)))
        self.gi_scale = gi_scale * self.root
        self.gi_theta = torch.nn.Parameter((1e-2 * torch.randn(len(Ig), dtype=torch.float64, generator=g)).to(DEV))
        # C_II: dense upper Cholesky factor of the interior block
        self.ii_logdiag = torch.nn.Parameter(torch.full((nI,), math.log(self.root), dtype=torch.float64, device=DEV))
        self.ii_theta = torch.nn.Parameter((1e-2 * torch.randn(nI, nI, dtype=torch.float64, generator=g)).to(DEV))
        self.ii_scale = ii_scale * self.root
        self.register_buffer('ii_mask', torch.triu(torch.ones((nI, nI), dtype=torch.bool, device=DEV), diagonal=1))
        self.register_buffer('power_vec', torch.randn(nI, 1, dtype=torch.float64, generator=g).to(DEV))
        self.sigma = 0.0
        # low-rank head
        self.rank = rank
        if rank:
            self.V = torch.nn.Parameter((1e-2 * torch.randn(q, rank, dtype=torch.float64, generator=g)).to(DEV))
            self.lam_log = torch.nn.Parameter(torch.full((rank,), math.log(1e-4 * s_unit), dtype=torch.float64, device=DEV))
        self.counts = dict(gg_diag=int(diag.sum()), gg_off=int((~diag).sum()), gi=len(Ig), ii=int(nI * (nI - 1) // 2 + nI), lowrank=int(rank * (q + 1)))

    def gg_values(self):
        v = torch.zeros(self.gg_index.shape[1], dtype=torch.float64, device=DEV)
        return v.masked_scatter(self.gg_diag, self.gg_logdiag.exp()).masked_scatter(~self.gg_diag, self.gg_scale * self.gg_theta)

    def sparse_gg(self, transpose=False):
        return torch.sparse_coo_tensor(self.gg_index_t if transpose else self.gg_index, self.gg_values(), (self.q, self.q), is_coalesced=not transpose)

    def C_ii(self, dominance=0.5):
        """Upper factor with strict row diagonal dominance, so triangular solves stay bounded."""
        dg = self.ii_logdiag.exp()
        off = (self.ii_scale * self.ii_theta) * self.ii_mask
        rowsum = off.abs().sum(1)
        off = off * torch.clamp(dominance * dg / rowsum.clamp(min=1e-300), max=1.0)[:, None]
        return torch.diag(dg) + off

    def sparse_gi(self, values, transpose=False):
        if transpose: return torch.sparse_coo_tensor(self.gi_index_t, values, (self.nI, self.q))
        return torch.sparse_coo_tensor(self.gi_index, values, (self.q, self.nI), is_coalesced=True)

    def coupling(self, C, iters=6):
        """L_GI values scaled so that ||L_GI C^-1||_2 <= 0.999 (power iteration, detached)."""
        raw = self.gi_scale * self.gi_theta
        with torch.no_grad():
            Cd = C.detach(); v = self.power_vec
            for _ in range(iters):
                y = torch.linalg.solve_triangular(Cd, v, upper=True)                       # C^-1 v
                u = torch.sparse.mm(self.sparse_gi(raw.detach()), y)                      # L v'
                t = torch.sparse.mm(self.sparse_gi(raw.detach(), transpose=True), u)      # L^T
                v = torch.linalg.solve_triangular(Cd.T, t, upper=False)                   # C^-T
                nrm = torch.linalg.vector_norm(v); v = v / nrm
            self.power_vec.copy_(v); self.sigma = float(nrm.sqrt())
        factor = min(1.0, 0.999 / max(self.sigma, 1e-300))
        return raw * factor

    def apply(self, x, state=None):
        """S_hat @ x for dense x (q x m)."""
        if state is None: state = self.state()
        C, gi = state
        u = torch.sparse.mm(self.sparse_gg(), x)
        t = torch.sparse.mm(self.sparse_gi(gi, transpose=True), u)
        y = torch.linalg.solve_triangular(C, torch.linalg.solve_triangular(C.T, t, upper=False), upper=True)
        u = u - torch.sparse.mm(self.sparse_gi(gi), y)
        out = torch.sparse.mm(self.sparse_gg(transpose=True), u)
        if self.rank:
            out = out + self.V @ (self.lam_log.exp()[:, None] * (self.V.T @ x))
        return out

    def state(self):
        C = self.C_ii(); return C, self.coupling(C)

    @torch.no_grad()
    def materialize(self):
        C, gi = self.state()
        LGI = self.sparse_gi(gi).to_dense()
        Y = torch.linalg.solve_triangular(C.T, LGI.T, upper=False); del LGI                 # nI x q  (C^-T L^T)
        T = -(Y.T @ Y); del Y; T.diagonal().add_(1.0)
        Q1 = torch.sparse.mm(self.sparse_gg(transpose=True), T); del T
        S = torch.sparse.mm(self.sparse_gg(transpose=True), Q1.T.contiguous()); del Q1
        if self.rank:
            S = S + (self.V * self.lam_log.exp()) @ self.V.T
        return 0.5 * (S + S.T)



class ContractionSuperElement(torch.nn.Module):
    """Exact-PSD form:  S_hat = L_GG^T (I + M M^T)^-1 L_GG,  M = [M_sparse (face->coarse interior), U (dense global columns)].
    Applied by Woodbury: (I + M M^T)^-1 u = u - M (I + M^T M)^-1 M^T u, with a dense Cholesky of G = I + M^T M.
    Equivalent to the super-element with interior block K_II = I + M^T M; strictly positive definite for any M."""
    def __init__(self, xyz_face, xyz_int, r_near, rho, rank, log_diag_init, s_unit, seed, decay=0.03, off_scale=0.3, gi_scale=0.1):
        super().__init__()
        g = torch.Generator(device='cpu').manual_seed(seed)
        q, nI = 3 * len(xyz_face), 3 * len(xyz_int)
        self.q, self.nI, self.root, self.rank = q, nI, math.sqrt(s_unit), rank
        ni, nj = pairs_within(xyz_face, xyz_face, r_near, upper=True)
        I, J = expand_dofs(ni, nj, upper=True)
        order = torch.argsort(I * q + J); I, J = I[order], J[order]
        diag = I == J
        dist = torch.linalg.vector_norm(xyz_face[I // 3] - xyz_face[J // 3], dim=1)
        self.register_buffer('gg_index', torch.stack((I, J))); self.register_buffer('gg_index_t', torch.stack((J, I)))
        self.register_buffer('gg_diag', diag)
        self.register_buffer('gg_scale', (off_scale * self.root * torch.exp(-dist / decay))[~diag])
        self.gg_logdiag = torch.nn.Parameter(log_diag_init.to(DEV).clone())
        self.gg_theta = torch.nn.Parameter(torch.zeros(int((~diag).sum()), dtype=torch.float64, device=DEV))
        self.use_interior = rho > 0
        if self.use_interior:
            mi, mj = pairs_within(xyz_face, xyz_int, rho)
            Ig, Ji = expand_dofs(mi, mj, upper=False)
            order = torch.argsort(Ig * nI + Ji); Ig, Ji = Ig[order], Ji[order]
        else:
            Ig = Ji = torch.zeros(0, dtype=torch.long, device=DEV); self.nI = nI = 0
        self.register_buffer('gi_index', torch.stack((Ig, Ji))); self.register_buffer('gi_index_t', torch.stack((Ji, Ig)))
        self.gi_theta = torch.nn.Parameter((1e-2 * torch.randn(len(Ig), dtype=torch.float64, generator=g)).to(DEV))
        self.gi_logscale = torch.nn.Parameter(torch.full((max(nI, 1),), math.log(gi_scale), dtype=torch.float64, device=DEV))      # per interior column
        self.U = torch.nn.Parameter((1e-2 * torch.randn(q, rank, dtype=torch.float64, generator=g)).to(DEV)) if rank else None
        self.U_logscale = torch.nn.Parameter(torch.full((rank,), math.log(0.1), dtype=torch.float64, device=DEV)) if rank else None
        self.sigma = 0.0
        self.counts = dict(gg_diag=int(diag.sum()), gg_off=int((~diag).sum()), gi=len(Ig), U=int(q * rank), interior_used=self.use_interior)

    def gg_values(self):
        v = torch.zeros(self.gg_index.shape[1], dtype=torch.float64, device=DEV)
        return v.masked_scatter(self.gg_diag, self.gg_logdiag.exp()).masked_scatter(~self.gg_diag, self.gg_scale * self.gg_theta)

    def sparse_gg(self, transpose=False):
        return torch.sparse_coo_tensor(self.gg_index_t if transpose else self.gg_index, self.gg_values(), (self.q, self.q), is_coalesced=not transpose)

    def gi_values(self):
        return self.gi_theta * self.gi_logscale.exp()[self.gi_index[1]]

    def sparse_gi(self, values, transpose=False):
        if transpose: return torch.sparse_coo_tensor(self.gi_index_t, values, (self.nI, self.q))
        return torch.sparse_coo_tensor(self.gi_index, values, (self.q, self.nI), is_coalesced=True)

    def Ucol(self):
        return self.U * self.U_logscale.exp()[None, :]

    def M_apply_T(self, gi, u):
        if not self.use_interior: return self.Ucol().T @ u
        t = torch.sparse.mm(self.sparse_gi(gi, transpose=True), u)
        return torch.cat((t, self.Ucol().T @ u), 0) if self.rank else t

    def M_apply(self, gi, t):
        if not self.use_interior: return self.Ucol() @ t
        out = torch.sparse.mm(self.sparse_gi(gi), t[:self.nI])
        return out + self.Ucol() @ t[self.nI:] if self.rank else out

    def state(self):
        """Cholesky factor of G = I + M^T M (dense, nI + rank)."""
        gi = self.gi_values()
        if not self.use_interior:
            Uc = self.Ucol(); G = Uc.T @ Uc
        else:
            Ms = self.sparse_gi(gi).to_dense()
            G11 = torch.sparse.mm(self.sparse_gi(gi, transpose=True), Ms)
            if self.rank:
                Uc = self.Ucol(); G12 = Ms.T @ Uc; G22 = Uc.T @ Uc
                G = torch.cat((torch.cat((G11, G12), 1), torch.cat((G12.T, G22), 1)), 0)
            else: G = G11
        G = G + torch.eye(G.shape[0], dtype=torch.float64, device=DEV)
        C = torch.linalg.cholesky(G)
        with torch.no_grad(): self.sigma = float(G.diagonal().max().sqrt())
        return gi, C

    def project(self): pass

    def apply(self, x, state=None):
        gi, C = self.state() if state is None else state
        u = torch.sparse.mm(self.sparse_gg(), x)
        t = self.M_apply_T(gi, u)
        u = u - self.M_apply(gi, torch.cholesky_solve(t, C))
        return torch.sparse.mm(self.sparse_gg(transpose=True), u)

    @torch.no_grad()
    def materialize(self):
        gi, C = self.state()
        if self.use_interior:
            M = self.sparse_gi(gi).to_dense()
            if self.rank: M = torch.cat((M, self.Ucol()), dim=1)
        else: M = self.Ucol()
        Y = torch.linalg.solve_triangular(C, M.T, upper=False); del M                      # C^-1 M^T
        T = -(Y.T @ Y); del Y; T.diagonal().add_(1.0)
        Q1 = torch.sparse.mm(self.sparse_gg(transpose=True), T); del T
        S = torch.sparse.mm(self.sparse_gg(transpose=True), Q1.T.contiguous()); del Q1
        return 0.5 * (S + S.T)


class LUFactor:
    """action/solve interface for a dense symmetric operator via pivoted LU; no PSD assumption for evaluation."""
    def __init__(self, M):
        self.M = M; self.lu, self.piv = torch.linalg.lu_factor(M)
    def action(self, x): return self.M @ x
    def solve(self, f): return torch.linalg.lu_solve(self.lu, self.piv, f)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True); ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--r-near', type=float, default=0.2); ap.add_argument('--rho', type=float, default=0.15)
    ap.add_argument('--coarse-n', type=int, default=8); ap.add_argument('--rank', type=int, default=128)
    ap.add_argument('--probes', type=int, default=32); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--eval-every', type=int, default=500); ap.add_argument('--seed', type=int, default=2026091206)
    ap.add_argument('--seat', type=int, default=415); ap.add_argument('--no-material-filter', action='store_true')
    ap.add_argument('--reevaluate', default=None, help='directory with CHECKPOINT_*.pt to re-evaluate; no training')
    ap.add_argument('--action-weight', type=float, default=1.0, help='weight of the isotropic relative action term on white probes')
    ap.add_argument('--action-probes', type=int, default=16)
    ap.add_argument('--white-probes', type=int, default=0, help='probes z = R*^-1 xi, uniform over modes in the energy metric')
    ap.add_argument('--off-scale', type=float, default=0.3); ap.add_argument('--gi-scale', type=float, default=0.1); ap.add_argument('--ii-scale', type=float, default=0.05)
    ap.add_argument('--lr-floor', type=float, default=0.1, help='cosine schedule floor as a fraction of lr'); ap.add_argument('--gg-lr-mult', type=float, default=3.0)
    ap.add_argument('--init-checkpoint', default=None)
    ap.add_argument('--form', default='contraction', choices=['contraction', 'interior'])
    ap.add_argument('--diag-init', default='teacher', choices=['teacher', 'geometry'])
    args = ap.parse_args()
    out = __import__('pathlib').Path(args.output); out.mkdir(parents=True, exist_ok=args.reevaluate is not None)
    t_start = time.perf_counter(); torch.manual_seed(args.seed)
    protocol = dict(vars(args), schema='CUTFEM_SUPERELEMENT_V0', loss='energy metric ||R*^-T (A_hat-A) z||^2 / ||R* z||^2 + action_weight * isotropic relative action error on white probes',
                    probes='half coarse-space random fields P z, half support-weighted fine random fields', script_sha256=sha256(__file__))
    write_json(out / 'PROTOCOL.json', protocol)
    records = json.load(open('/root/autodl-tmp/CUTFEM_NEURAL_DENSE_20260911_P01/INPUT_MANIFEST.json'))['samples']
    record = next(r for r in records if int(r['seat']) == args.seat)
    sample = FactorSample(record, 2026091197); data = sample.data.cuda()
    cache = np.load(record['trace_cache']); n = 32; N = 2 * n + 1
    assert np.all(np.diff(cache['indptr']) == 1) and np.allclose(cache['coefficients'], 1.0), 'V0 assumes nodal trace functionals'
    ijk = np.stack(np.unravel_index(cache['background_nodes'][cache['indices']], (N, N, N)), axis=1)
    S, _ = load_teacher_dense(record['packet'], data.quotient.dimension, DEV)
    S_diag = S.diagonal().clone().cpu()
    A, construction = quotient_dense(S, data.quotient); del S; torch.cuda.empty_cache()
    Rstar = load_upper_factor(sample.reference / 'R_UPPER.npy', sample.n, DEV)
    q, d = 3 * len(ijk), A.shape[0]
    x_dirs, names, force = make_directions(sample.data, sample.cache); fref = DenseUpperFactor(Rstar).solve(force)
    # geometry-only quantities
    w = support_weights(ijk, N, cache['tau_corners'], cache['cut_plane']); w = np.clip(w, 1e-4, 1.0)
    P = coarse_interpolation(ijk, N, stride=(2 * n) // (2 * args.coarse_n))
    P3 = torch.from_numpy(np.kron(P, np.eye(3))).to(DEV)
    xyz_face = torch.from_numpy(ijk / (N - 1)).to(DEV).double()
    cn = args.coarse_n; stride = (2 * n) // (2 * cn)
    grid = np.stack(np.meshgrid(*[np.arange(1, 2 * cn)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)   # interior coarse nodes
    if not args.no_material_filter:
        material = data.grid[0, 10].cpu().numpy() > 0.5
        keep = []
        for c in grid:
            lo = (c - 1) * stride; hi = (c + 1) * stride
            keep.append(material[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].any())
        grid = grid[np.array(keep)]
    xyz_int = torch.from_numpy(grid * stride / (N - 1)).to(DEV).double()
    s_unit = float(A.diagonal().median())
    if args.form == 'interior':
        model = SuperElement(xyz_face, xyz_int, args.r_near, args.rho, args.rank, torch.from_numpy(w).double(), s_unit, args.seed,
                             off_scale=args.off_scale, gi_scale=args.gi_scale, ii_scale=args.ii_scale)
    else:
        if args.diag_init == 'teacher':
            log_diag = 0.5 * torch.log(S_diag)
        else:
            log_diag = 0.5 * torch.log(s_unit * torch.from_numpy(w).double().repeat_interleave(3))
        model = ContractionSuperElement(xyz_face, xyz_int, args.r_near, args.rho, args.rank, log_diag, s_unit, args.seed,
                                        off_scale=args.off_scale, gi_scale=args.gi_scale)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location='cuda', weights_only=False)['model'])
    nparam = sum(p.numel() for p in model.parameters())
    write_json(out / 'MODEL.json', dict(counts=model.counts, parameters=nparam, interior_nodes=len(grid), interior_dofs=3 * len(grid),
        coarse_face_dofs=P3.shape[1], face_dofs=q, quotient_dim=d, s_unit=s_unit, support_weight_quantiles=[float(v) for v in np.quantile(w, [0, .1, .5, .9, 1])],
        sliver_fraction_w_below_1e_2=float((w < 1e-2).mean())))
    write_json(out / 'RUN.json', dict(source_commit='14301bc56b9032cb3c1c282ed9afc94b01300083', script_sha256=protocol['script_sha256'], pid=os.getpid(),
        started_unix=time.time(), gpu=torch.cuda.get_device_name(), torch=torch.__version__, label_sha256=sample.binding['teacher_sha256']))
    print(json.dumps(dict(stage='setup', parameters=nparam, counts=model.counts, interior_dofs=3 * len(grid), seconds=time.perf_counter() - t_start)), flush=True)
    # probes
    wq = torch.from_numpy(w).to(DEV).double().repeat_interleave(3)
    gen = torch.Generator(device=DEV).manual_seed(args.seed + 1)
    def draw_probes(m, generator, white=0):
        mc = m // 2
        zc = torch.randn(P3.shape[1], mc, dtype=torch.float64, device=DEV, generator=generator)
        xf = torch.randn(q, m - mc, dtype=torch.float64, device=DEV, generator=generator) * wq.sqrt()[:, None]
        z = data.quotient(torch.cat((P3 @ zc, xf), dim=1))                 # B x, quotient coordinates
        if white:
            xi = torch.randn(d, white, dtype=torch.float64, device=DEV, generator=generator)
            z = torch.cat((z, torch.linalg.solve_triangular(Rstar, xi, upper=True)), dim=1)
        return z
    held = draw_probes(64, torch.Generator(device=DEV).manual_seed(args.seed + 2), white=32)   # 32 coarse, 32 fine, 32 white
    def energy_loss(z, state=None, reduce=True):
        ref = A @ z
        xq = data.quotient.lift(z)
        pred = data.quotient(model.apply(xq, state))
        num = torch.linalg.solve_triangular(Rstar.T, pred - ref, upper=False).square().sum(0)
        den = (Rstar @ z).square().sum(0)
        e = num / den
        return e.mean() if reduce else e
    def action_loss(z, state=None):
        """Isotropic relative action error on white probes: E||(A_hat-A)z||^2/||Az||^2 ~ ||A_hat-A||_F^2/||A||_F^2."""
        ref = A @ z; pred = data.quotient(model.apply(data.quotient.lift(z), state))
        return ((pred - ref).square().sum(0) / ref.square().sum(0)).mean()
    scale_names = ('gi_logscale', 'U_logscale')
    groups = [dict(params=[model.gg_theta, model.gg_logdiag], lr=args.lr * args.gg_lr_mult),
              dict(params=[p_ for n_, p_ in model.named_parameters() if n_ in scale_names], lr=args.lr * 3),
              dict(params=[p_ for n_, p_ in model.named_parameters() if n_ not in ('gg_theta', 'gg_logdiag') + scale_names], lr=args.lr)]
    opt = torch.optim.Adam(groups, lr=args.lr); base_lrs = [g_['lr'] for g_ in opt.param_groups]
    def energy_norm_error(xm, xr):
        return (torch.linalg.vector_norm(Rstar @ (xm - xr), dim=0) / torch.linalg.vector_norm(Rstar @ xr, dim=0))
    def evaluate_full(step, tag='EVALUATION'):
        tick = sync()
        with torch.no_grad():
            e_held = energy_loss(held, reduce=False)
            Shat = model.materialize(); Ahat, _ = quotient_dense(Shat, data.quotient); del Shat
            _, info = torch.linalg.cholesky_ex(Ahat, upper=True)
            res = dict(step=step, held_energy_error=float(e_held[:64].mean()), held_energy_error_coarse=float(e_held[:32].mean()),
                       held_energy_error_fine=float(e_held[32:64].mean()), held_energy_error_white=float(e_held[64:].mean()), cholesky_info=int(info), sigma=model.sigma)
            fac = LUFactor(Ahat)
            # exact Frobenius components without a factor of A_hat
            D = Ahat - A; res['schur_relative_error'] = float(torch.linalg.vector_norm(D) / torch.linalg.vector_norm(A))
            off = D.clone(); off.diagonal().zero_(); Aoff = A.clone(); Aoff.diagonal().zero_()
            res['offdiagonal_relative_error'] = float(torch.linalg.vector_norm(off) / torch.linalg.vector_norm(Aoff)); del off, Aoff, D
            mech = evaluate(fac, lambda v: A @ v, x_dirs, names, force, fref)
            res['groups'] = mech['groups']; res['independent_force_summary'] = mech['independent_force_summary']
            # energy-norm versions of the inverse metrics (sliver displacements weighted by their stiffness)
            g = A @ x_dirs; xh = fac.solve(g); e_inv = energy_norm_error(xh, x_dirs)
            res['inverse_energy_norm_rms'] = {grp: float(e_inv[[i for i, nm in enumerate(names) if nm.startswith(grp)]].square().mean().sqrt())
                                              for grp in ('regression/random', 'regression/cos', 'independent_random/', 'independent_smooth/', 'independent_local/')}
            xg = fac.solve(force); res['gauss_force_energy_norm_rms'] = float(energy_norm_error(xg, fref).square().mean().sqrt())
            gf = torch.Generator(device=DEV).manual_seed(args.seed + 3)
            zc = torch.randn(P3.shape[1], 16, dtype=torch.float64, device=DEV, generator=gf)
            f = data.quotient(wq[:, None] * (P3 @ zc))
            xr = DenseUpperFactor(Rstar).solve(f); xm = fac.solve(f)
            disp = torch.linalg.vector_norm(xm - xr, dim=0) / torch.linalg.vector_norm(xr, dim=0)
            en = energy_norm_error(xm, xr); comp = ((f * xm).sum(0) / (f * xr).sum(0) - 1).abs()
            res['physical_force'] = dict(displacement_relative_rms=float(disp.square().mean().sqrt()), displacement_energy_norm_rms=float(en.square().mean().sqrt()),
                                         displacement_energy_norm_max=float(en.max()), compliance_relative_max=float(comp.max()))
            del Ahat, fac
        res['evaluation_seconds'] = sync() - tick
        write_json(out / f'{tag}_{step:06d}.json', json_finite(res))
        brief = dict(step=step, held=res['held_energy_error'], white=res['held_energy_error_white'], eA=res['schur_relative_error'], eO=res['offdiagonal_relative_error'], chol=res['cholesky_info'],
                     smooth_E=res['groups']['independent_smooth/']['energy_relative_max'], smooth_inv_energy=res['inverse_energy_norm_rms']['independent_smooth/'],
                     local_inv_energy=res['inverse_energy_norm_rms']['independent_local/'], phys_force_energy=res['physical_force']['displacement_energy_norm_rms'],
                     phys_force_compl=res['physical_force']['compliance_relative_max'], gauss_force_energy=res['gauss_force_energy_norm_rms'], sec=res['evaluation_seconds'])
        print(json.dumps(dict(stage='evaluation', **brief)), flush=True)
        torch.cuda.empty_cache(); return res
    if args.reevaluate:
        import glob
        for ck in sorted(glob.glob(str(__import__('pathlib').Path(args.reevaluate) / 'CHECKPOINT_*.pt'))):
            saved = torch.load(ck, map_location='cuda', weights_only=False); model.load_state_dict(saved['model'])
            evaluate_full(int(saved['step']), tag='REEVALUATION')
        return
    evaluations = [evaluate_full(0)]
    history = []; lr_scale = 1.0; bad = 0
    for step in range(1, args.steps + 1):
        tick = sync()
        sched = min(1.0, step / args.warmup) * (args.lr_floor + (1 - args.lr_floor) * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(args.steps - 1, 1)))) * lr_scale
        lr = args.lr * sched
        for g_, b_ in zip(opt.param_groups, base_lrs): g_['lr'] = b_ * sched
        z = draw_probes(args.probes, gen, white=args.white_probes)
        opt.zero_grad(set_to_none=True)
        state = model.state()
        loss_e = energy_loss(z, state); loss_a = torch.zeros((), dtype=torch.float64, device=DEV)
        if args.action_weight > 0:
            zw = torch.randn(d, args.action_probes, dtype=torch.float64, device=DEV, generator=gen)
            loss_a = action_loss(zw, state)
        loss = loss_e + args.action_weight * loss_a
        if not torch.isfinite(loss) or model.sigma > 1e6:
            bad += 1; lr_scale *= 0.5; print(json.dumps(dict(stage='guard', step=step, loss=float(loss.detach()), sigma=model.sigma, lr_scale=lr_scale)), flush=True)
            if bad > 8: raise FloatingPointError('Repeated nonfinite loss or divergent coupling')
            continue
        loss.backward(); opt.step()
        if hasattr(model, 'project'): model.project()
        elif model.sigma > 1.0:
            with torch.no_grad(): model.gi_theta.mul_(0.999 / model.sigma)
        row = dict(step=step, lr=lr, loss=float(loss.detach()), energy_loss=float(loss_e.detach()), action_loss=float(loss_a.detach()), sigma=model.sigma, seconds=sync() - tick)
        history.append(row); append_json(out / 'HISTORY.jsonl', row)
        if step <= 5 or step % 50 == 0:
            print(json.dumps(dict(**row, mean_loss_50=float(np.mean([r['loss'] for r in history[-50:]])), elapsed=time.perf_counter() - t_start, peak_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            torch.save(dict(model=model.state_dict(), step=step, protocol=protocol), out / f'CHECKPOINT_{step:06d}.pt')
            evaluations.append(evaluate_full(step))
    result = dict(status='COMPLETED_FIXED_BUDGET', steps=args.steps, elapsed_seconds=time.perf_counter() - t_start, parameters=nparam,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, initial=evaluations[0], final=evaluations[-1])
    write_json(out / 'RESULT.json', json_finite(result))
    print(json.dumps(dict(stage='done', elapsed=result['elapsed_seconds'], final={k: evaluations[-1].get(k) for k in ('held_energy_error', 'schur_relative_error')})), flush=True)


if __name__ == '__main__':
    main()
