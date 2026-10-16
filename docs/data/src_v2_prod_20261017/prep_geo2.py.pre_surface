"""Step-2 data v2 (review C1): lattice-context q classes with traction-CONSISTENT loads, exact port reactions F, and the
fp64 fix of the NaN support banks. prep_geo.py is unchanged; this writes new files next to its output (or, with --out,
into a new data root that hard-links the old files: the source tree is never modified).

Classes (every direction rigid-free and at unit exact energy, exact 8-corner sensitivities, reactions F = S q):
  force_c    multiscale random smooth tractions on every box face with material (10% also on the cut surface), integrated
             consistently over the MATERIAL part of the face, f_i = sum_qp h^2 t(x_qp) N_i(x_qp) (lattice3 quadrature);
             the traction itself is self-equilibrated (below), so f = A t is consistent AND has zero resultant force and
             moment; fictitious-fringe nodes get only their consistent share, and 90% of the samples leave the cut free;
             Neumann solve (its rigid projection is a no-op up to rounding)
  face_c     one box face at a time (cycled): consistent self-equilibrated traction on that face only (6 constraints on the
             face's own quadrature points), other ports free; Neumann solve
  support_k  one box face on springs k_i = alpha diag(K)_ii on its port DOFs (alpha log-uniform in [0.3, 3], one per face),
             consistent tractions on the other box faces, cut band free; fp64 spring factor, non-finite solutions raise
  glued      the cell glued at full-DOF level (nodes at the same absolute position share DOFs) to its family FULL parent
             across one of its box faces that carries material; neighbour offsets +x, +y, +z, -z only (the acceptance gate
             uses -x and -y, lattice3 configs x / y: never those); the neighbour's far face clamped or on springs
             k = alpha diag, drawn per chunk of samples (50/50); consistent tractions on the free box faces of both cells,
             25% of the samples load ONLY the neighbour (the test cell is dragged almost rigidly); q = the test cell's port
             values; every solve is checked against the fp64 glued matrix (below)
  --F-old        reactions F for the existing classes force, macro, grf, support, face
  --fix-support  recompute the old 'support' class (same definition as prep_geo) with fp64 spring factors where its bank
                 has non-finite values (or where an earlier run left it half renamed). In place: the new files are written
                 under temporary names first, then the old ones (q, _sens and _F) become *.nan_bak.npy and the new ones
                 take their names. With --out: written as the new class 'support64' (the old 'support' stays as it is)
Files per class: {split}_{cls}.npy (m x np fp32), {split}_{cls}_sens.npy (m x 8), {split}_{cls}_F.npy (m x np fp32);
DONE2.json (per-class stats, seeds, timings). Bank sizes from BANK_SPLITS like prep_geo (e.g. "512,64,64").
Usage: prep_geo2.py <body_dir> <data_dir> [--out <dir>] [--classes force_c,face_c,support_k,glued] [--F-old]
                    [--fix-support] [--force] [--legacy-seed] <case ...>

Audit fixes (docs/AUDIT_V2_20260925_CN.md, 2026-09-25; VERSION below marks banks made with them):
  DATA-1 seeds     the old seed (little-endian bytes of the name mod 2^31) depended only on 'fres', so every fresh_* case
                   drew the same load functions. Now case seed = int(sha256(case)[:8], 16) and every class has its own
                   torch.Generator seeded with (case seed + CLASS_OFFSET[cls] * 0x9E3779B1) mod 2^32 (kept to 32 bits: the
                   CPU generator ignores higher seed bits), so a class can be regenerated alone and reproducibly. Seeds are recorded in DONE2 ('seed', and per class).
                   --legacy-seed: the old seed formula and ONE generator shared by all classes in the old order (the
                   other fixes below still apply, so banks are not bit-identical to the pre-fix ones).
  DATA-2 traction  force_c / face_c no longer equilibrate the NODAL load by the Euclidean rigid projection over all port
                   nodes (which added uniform / linear nodal loads on every port incl. fringe and cut band). With
                   G = Q^T A (6 x 3m, Q the port rigid basis, A the consistent quadrature map) the traction samples are
                   corrected t <- t - G^T (G G^T)^-1 G t on the LOADED quadrature points (per sample: faces, plus the cut
                   points when that sample loads the cut; face_c: that face only). The correction is h^2 times a rigid-type
                   field a + b x (x - x_c) at the quadrature points (Q2 reproduces linear fields), so f = A t stays
                   consistent. DONE2 per class: rigid_frac_raw_median (||QQ^T f||/||f|| before the correction; ~0.35 in the
                   audit) and rigid_frac_median / rigid_frac_max after it (should be <~1e-12). cond(G G^T) > EQ_MAX_COND:
                   force_c raises EQUILIBRATION_DEGENERATE (class failed); face_c drops that face (faces_degenerate) and
                   raises only when no face is left.
  DATA-5 --out     read the existing banks from <data>/<case>, write everything new to <out>/<case>. Before that every
                   existing file of <data>/<case> is hard-linked into <out>/<case> if missing (os.link, copy on failure),
                   except bank files of the v2 classes whose source DONE2 record is not valid for this VERSION (old pilot
                   banks are regenerated, not inherited). Every write goes to a temporary name + os.replace, so a
                   hard-linked source inode is never truncated. The fixed support bank is the class 'support64'.
  DATA-4/9         classes are finalised (normalised, sensitivities, F) and saved as soon as their stage allows: force_c,
                   face_c, support_k, support(64) right after the spring stage (one interior factor), glued after the
                   glued stage (a second interior factor, only when glued runs together with other classes). Generation
                   and finalisation are wrapped per class: a failure is recorded in DONE2 as {'failed': repr, 'stage'}
                   (event CLASS_FAILED2) and the other classes proceed; DONE2 is rewritten after every class. FAILED2 is
                   logged only when a case produced nothing (setup failure or every class failed); PARTIAL2 otherwise.
                   Exit status 1 only when no case produced anything and something failed. Reruns are idempotent: a class
                   whose DONE2 record is valid (this VERSION, seed scheme and BANK_SPLITS, files present) is skipped unless
                   --force; deterministic skips (no FULL parent, no glue face, <= 1 face with material) count as done.
  DATA-3 glued     every glued solve is checked with the fp64 (Jacobi-scaled) glued upper CSR: r = F - K u via a symmetric
                   matvec from the upper CSR (UpperSym); if the relative residual ||r||/||F|| > REFINE_TARGET (the fp32 /
                   fp64_retry factors) up to REFINE_STEPS steps of iterative refinement u += K^-1 r. The per-group residual
                   (max / median over the samples, resid_unrefined_max before refinement) and the refinement steps are
                   recorded; an offset whose residual stays
                   above RESID_TOL is rejected like a memory skip (its quota moves to the remaining offsets). After an
                   fp32 / fp64_retry factor the fp64 matrix is kept on the host and streamed to the device per row chunk.
  DATA-6 glued BC  clamped vs springs for the neighbour's far face is drawn per chunk (50/50) from the glued generator
                   (at most two factors per offset, one alive at a time) instead of by the offset's list index; DONE2
                   glued.bc_counts = per axis {'clamped': samples, 'springs': samples} of the final bank.
  DATA-8           force_c records cut_qp (cut quadrature points), cut_loaded and traction_dropped (forces on non-port
                   nodes per face / cut); every regenerated class gets a fresh DONE2 record (no stale 'skipped' keys);
                   when a class fails or is skipped, stale bank files of that class (from a record that is not valid for
                   this VERSION) are retired: unlinked in the --out tree, renamed *.stale_bak.npy in place. A valid earlier
                   bank is kept (its record gets 'last_failure').
"""
import sys, json, time, gc, re, os, hashlib, shutil
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import prep_data as PD
import prep_geo as PG

_SPD0 = TE.SPDSolver


def _spd_clean(*a, **k):
    """Every cuDSS factor of this script (teacher.Cell.factor, prep_geo.spd_safe, _spd64, glued): cuDSS allocates outside
    torch, so return torch's cached blocks first (after dmoments the cache holds ~5 GB it no longer uses), and on a memory
    failure clean up and try once more."""
    gc.collect(); torch.cuda.empty_cache()
    try:
        return _SPD0(*a, **k)
    except Exception as e:
        if not PG._mem_error(e):
            raise
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        return _SPD0(*a, **k)


TE.SPDSolver = _spd_clean                                                     # this process only (data preparation)

dev, dt = TE.dev, TE.dt
SPLITS = PD.SPLITS
NEW = ('force_c', 'face_c', 'support_k', 'glued')
OLD = ('force', 'macro', 'grf', 'support', 'face')
V2_BANKS = NEW + ('support64',)                                               # classes this script owns in an --out tree
GLUE_OFFSETS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1))                  # never (-1,0,0) / (0,-1,0): the gate's
if os.environ.get('GLUED_ALL_FACES', '0') == '1':                            # new independent-field cells: the gate is evaluated
    GLUE_OFFSETS = GLUE_OFFSETS + ((-1, 0, 0), (0, -1, 0))                    # on held-out cells, and a heavy cut's only loaded
                                                                              # face is its intact -x / -y face
VERSION = 'prep_geo2/audit-2026-09-25'
GLUED_NEIGHBOURS = os.environ.get('GLUED_NEIGHBOURS', 'family')             # 'family' (default: the family FULL parent,
                                                                              # translated) | 'explicit' (<case>_nb<tag> packets:
                                                                              # continuous thickness across the glued face)
NB_TAG = {(1, 0, 0): 'px', (-1, 0, 0): 'mx', (0, 1, 0): 'py', (0, -1, 0): 'my', (0, 0, 1): 'pz', (0, 0, -1): 'mz'}
GLUED_SAFE = os.environ.get('GLUED_SAFE', '0') == '1'                         # default off: the old glued path, unchanged
GLUED_HEADROOM = float(os.environ.get('GLUED_HEADROOM_GB', '4')) * 2 ** 30     # free device memory wanted after the factor
GLUED_FP32_STEPS = int(os.environ.get('GLUED_FP32_STEPS', '12'))              # safe mode: refinement steps for an fp32 factor
GLUED_PCG_ITERS = int(os.environ.get('GLUED_PCG_ITERS', '60'))                # safe mode: PCG iterations for an fp32 factor
GLUED_FP32_DOFS = int(float(os.environ.get('GLUED_FP32_DOFS', '7e5')))       # safe mode: two-cell systems above this size go
                                                                              # straight to an fp32 factor (an fp64 factor of two
                                                                              # FULL cells fills the card and the solver's memory is
                                                                              # not returned in time for the second BC group)
CLASS_OFFSET = dict(force_c=1, face_c=2, support_k=3, glued=4, support=5, support64=5)
EQ_MAX_COND = 1e10                                                            # cond(G G^T) of the traction equilibration
REFINE_STEPS, REFINE_TARGET, RESID_TOL = 3, 1e-10, 1e-6                       # glued solves (DATA-3)
SUFFIXES = ('', '_sens', '_F')


# --------------------------------------------------------------------------------------------------------- seeds (DATA-1)
def case_seed(case, legacy=False):
    """Seed of a case: sha256 of the full name (legacy: the old little-endian form, which collides for all fresh_*)."""
    if legacy:
        return (int.from_bytes(case.encode(), 'little') + 7919) % (2 ** 31)
    return int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)


def class_seed(case, cls, legacy=False):
    s = case_seed(case, legacy)
    return s if legacy else (s + CLASS_OFFSET[cls] * 0x9E3779B1) % 2 ** 32   # 32 bits: torch's CPU generator ignores higher bits


class ClassGens:
    """gens(cls) -> the torch.Generator of that class: independent per class (seeded on first use), or, with legacy=True,
    one generator shared by all classes (the old behaviour; draws then depend on which classes run and in which order)."""

    def __init__(self, case, legacy=False):
        self.case, self.legacy, self._g = case, legacy, {}
        self.scheme = 'legacy' if legacy else 'sha256'

    def seed(self, cls):
        return class_seed(self.case, cls, self.legacy)

    def __call__(self, cls):
        key = '_shared' if self.legacy else cls
        if key not in self._g:
            self._g[key] = torch.Generator(device=dev).manual_seed(self.seed(cls))
        return self._g[key]

    def record(self, classes):
        return dict(scheme=self.scheme, case=case_seed(self.case, self.legacy), classes={c: self.seed(c) for c in classes})


# --------------------------------------------------------------------------------------------------------- quadrature
def _material(C, P):
    """Quadrature points inside the material: |phi| <= tau (trilinear corner thickness) and in the retained half-space."""
    from element_polyref import CUBE
    f = np.cos(2 * np.pi * P).sum(1)
    cube = np.asarray(CUBE, float)
    w8 = np.where(cube[:, None, :].astype(bool), P[None], 1 - P[None]).prod(-1)
    keep = np.abs(f) <= np.asarray(C.taus, float) @ w8
    if C.normal is not None and C.is_cut.any():
        keep &= (P @ np.asarray(C.normal, float)) <= C.offset
    return P[keep]


def _quad_nodes(C, P):
    """Q2 shape values N (m x 27) and node indices into C.nodes (m x 27) of the active element containing each point."""
    n = C.n
    c = np.minimum(np.floor(P * n).astype(np.int64), n - 1)
    key = (c[:, 0] * n + c[:, 1]) * n + c[:, 2]
    ok = np.isin(key, (C.cells[:, 0] * n + C.cells[:, 1]) * n + C.cells[:, 2])
    P, c = P[ok], c[ok]
    xi = P * n - c
    L = np.stack([2 * (xi - .5) * (xi - 1), -4 * xi * (xi - 1), 2 * xi * (xi - .5)], -1)       # m x 3(axis) x 3(o)
    M = 2 * n + 1
    idx, N = [], []
    for o in np.ndindex(3, 3, 3):
        g = 2 * c + np.asarray(o)
        ids = (g[:, 0] * M + g[:, 1]) * M + g[:, 2]
        loc = np.searchsorted(C.nodes, ids)
        if not (C.nodes[np.minimum(loc, len(C.nodes) - 1)] == ids).all():
            raise ValueError('QUAD_NODE_MISSING')
        idx.append(loc); N.append(L[:, 0, o[0]] * L[:, 1, o[1]] * L[:, 2, o[2]])
    return P, np.stack(idx, 1), np.stack(N, 1)


class Traction:
    """Consistent nodal forces of a traction field sampled at quadrature points: f (port DOFs, node-major xyz) = A t,
    A[(port node p), qp] = h^2 N_p(x_qp). Built for a box face (axis, value in {0, 1}) or for the cut surface."""

    def __init__(self, C, axis=None, value=None, cut=False, pts=6):
        n = C.n
        h = 1.0 / (n * pts)
        if cut:
            nrm = np.asarray(C.normal, float); nrm = nrm / np.linalg.norm(nrm)
            a = np.eye(3)[np.argmin(np.abs(nrm))]
            t1 = np.cross(nrm, a); t1 /= np.linalg.norm(t1); t2 = np.cross(nrm, t1)
            s = np.arange(-np.sqrt(3.0), np.sqrt(3.0) + h, h) + h / 2
            S1, S2 = np.meshgrid(s, s, indexing='ij')
            x0 = nrm * C.offset / np.linalg.norm(C.normal)                   # on the plane normal . x = offset (normal not unit)
            P = x0[None] + S1.reshape(-1, 1) * t1[None] + S2.reshape(-1, 1) * t2[None]
            P = P[(P >= 0).all(1) & (P <= 1).all(1)]
            P = _material_plane(C, P)
        else:
            s = np.arange(0, 1, h) + h / 2
            A, B = np.meshgrid(s, s, indexing='ij')
            P = np.zeros((A.size, 3)); o = [d for d in range(3) if d != axis]
            P[:, axis] = value; P[:, o[0]] = A.reshape(-1); P[:, o[1]] = B.reshape(-1)
            P = _material(C, P)
        P, idx, N = _quad_nodes(C, P)
        nz = N != 0
        q_i, k_i = np.nonzero(nz)
        pos = np.searchsorted(C.port_node_ids, C.nodes[idx[q_i, k_i]])
        pos_ok = C.port_node_ids[np.minimum(pos, len(C.port_node_ids) - 1)] == C.nodes[idx[q_i, k_i]]
        self.dropped = int((~pos_ok).sum())                                  # (face loads: none; cut surface: none expected)
        q_i, k_i, pos = q_i[pos_ok], k_i[pos_ok], pos[pos_ok]
        self.nodes = np.unique(pos)                                          # port-node positions that receive force
        self.m = len(P)
        self.X = torch.as_tensor(P, dtype=dt, device=dev)
        self.A = torch.sparse_coo_tensor(torch.as_tensor(np.stack([pos, q_i]), device=dev),
                                         torch.as_tensor(h * h * N[q_i, k_i], dtype=dt, device=dev),
                                         (len(C.port_node_ids), self.m)).coalesce().to_sparse_csr()
        self.area = h * h * self.m

    def forces(self, T):
        """T (m, 3, k) traction samples at the quadrature points -> (port DOFs, k)."""
        k = T.shape[2]
        return (self.A @ T.reshape(self.m, 3 * k)).reshape(-1, 3, k).reshape(-1, k)

    def random(self, k, gen):
        nw = k - k // 4
        return torch.cat([PD.plane_waves(self.X, nw, 0.5, 8.0, gen), PD.patches(self.X, k - nw, gen)], 2)

    def rigid_map(self, Q):
        """G (6, m, 3) with G t = Q^T (A t): the 6 rigid resultants (in the orthonormal port rigid basis Q, np x 6) of the
        consistent nodal forces of a traction field t (m, 3, k). Built from A, so forces dropped on non-port nodes are
        excluded exactly as in forces()."""
        Ac = self.A.to_sparse_coo().coalesce()
        pos, qi = Ac.indices()
        w = Ac.values()
        Qn = Q.reshape(-1, 3, Q.shape[1])                                    # port nodes x 3 x 6
        G = torch.zeros((self.m, 3, Q.shape[1]), dtype=dt, device=dev)
        for c in range(3):
            G[:, c].index_add_(0, qi, w[:, None] * Qn[pos, c])
        return G.permute(2, 0, 1).contiguous()


def _material_plane(C, P):
    from element_polyref import CUBE
    f = np.cos(2 * np.pi * P).sum(1)
    cube = np.asarray(CUBE, float)
    w8 = np.where(cube[:, None, :].astype(bool), P[None], 1 - P[None]).prod(-1)
    return P[np.abs(f) <= np.asarray(C.taus, float) @ w8]


def box_tractions(C, min_pts=64):
    """Traction operators of the six box faces that carry material: {(axis, side 0/2n): Traction}."""
    out = {}
    for a in range(3):
        for v in (0, 1):
            T = Traction(C, a, v)
            if T.m >= min_pts:
                out[(a, v * 2 * C.n)] = T
    return out


# --------------------------------------------------------------------------------------------------------- equilibration (DATA-2)
def _gram(G):
    Gf = G.reshape(G.shape[0], -1)
    return Gf @ Gf.T


def _gram_cond(M):
    ev = torch.linalg.eigvalsh(M)
    return float(ev[-1] / ev[0]) if float(ev[0]) > 0 else float('inf')


def _resultant(G, t):
    return torch.einsum('lmc,mck->lk', G, t)


def equilibrate(parts, Ms):
    """Self-equilibrate traction samples in traction space. parts: list of (G_j (6, m_j, 3), t_j (m_j, 3, k), on_j (k,) in
    {0, 1} or None = loaded in every sample). Ms: the Gram matrix sum_j on_jk G_j G_j^T per sample, (k, 6, 6). Returns the
    corrected t_j: t_j - on_j G_j^T lam_k with lam_k = M_k^-1 sum_j on_jk G_j t_jk, i.e. the least-squares (in the sum over
    the loaded quadrature points) removal of the rigid resultants; afterwards sum_j on_j G_j t_j = 0 (Q^T f = 0)."""
    r = sum(_resultant(G, t) * (1 if on is None else on[None, :]) for G, t, on in parts)          # 6 x k
    lam = torch.linalg.solve(Ms, r.T.unsqueeze(-1)).squeeze(-1).T                                 # 6 x k
    return [t - torch.einsum('lmc,lk->mck', G, lam) * (1 if on is None else on[None, None, :]) for G, t, on in parts]


def _frac(Q, f):
    """||Q Q^T f|| / ||f|| per column (Q orthonormal)."""
    return ((Q.T @ f).norm(dim=0) / f.norm(dim=0).clamp_min(1e-300))


def _frac_stats(raw, after):
    raw, after = torch.cat(raw), torch.cat(after)
    return dict(rigid_frac_raw_median=float(raw.median()), rigid_frac_median=float(after.median()),
                rigid_frac_max=float(after.max()))


# --------------------------------------------------------------------------------------------------------- classes
def force_c(C, faces, total, gen, cut_frac=0.1, chunk=32):
    """Returns (q (np, total), info). Draw order per chunk as before: every face's traction, the cut mask, the cut traction."""
    cut = Traction(C, cut=True) if (C.normal is not None and C.is_cut.any()) else None
    use_cut = cut is not None and cut.m > 0
    keys = list(faces)
    Gs = [faces[k_].rigid_map(C.Q) for k_ in keys]
    M0 = sum(_gram(G) for G in Gs)
    cond = _gram_cond(M0)
    if not cond <= EQ_MAX_COND:
        raise ValueError(f'EQUILIBRATION_DEGENERATE force_c: cond(G G^T) = {cond:.3g} over {len(keys)} face(s)')
    Gc = cut.rigid_map(C.Q) if use_cut else None
    Mc = _gram(Gc) if use_cut else None
    Qs, fr_raw, fr = [], [], []
    n_on = 0
    for j in range(0, total, chunk):
        k = min(chunk, total - j)
        ts = [faces[k_].random(k, gen) for k_ in keys]
        parts = [(G, t, None) for G, t in zip(Gs, ts)]
        Ms = M0[None].expand(k, 6, 6)
        if use_cut:
            on = (torch.rand(k, device=dev, generator=gen) < cut_frac).to(dt)
            parts.append((Gc, cut.random(k, gen), on))
            Ms = Ms + on[:, None, None] * Mc[None]
            n_on += int(on.sum())
        F_raw = sum(faces[k_].forces(t) for k_, t in zip(keys, ts))
        if use_cut:
            F_raw = F_raw + cut.forces(parts[-1][1]) * parts[-1][2][None, :]
        tc = equilibrate(parts, Ms)
        F = sum(faces[k_].forces(t) for k_, t in zip(keys, tc[:len(keys)]))
        if use_cut:
            F = F + cut.forces(tc[-1]) * parts[-1][2][None, :]
        fr_raw.append(_frac(C.Q, F_raw)); fr.append(_frac(C.Q, F))
        Qs.append(C.neumann(F))
    info = dict(cut_qp=(cut.m if cut is not None else 0), cut_loaded=n_on, eq_cond=cond,
                traction_dropped=dict({f'{a}:{s}': faces[(a, s)].dropped for a, s in keys}, **({'cut': cut.dropped} if cut is not None else {})),
                **_frac_stats(fr_raw, fr))
    return torch.cat(Qs, 1), info


def face_c(C, faces, total, gen, chunk=32):
    """Returns (q (np, total), info). Faces whose own 6 x 6 Gram matrix is degenerate are left out (faces_degenerate)."""
    keys, Gs, bad = [], {}, {}
    for key in faces:
        G = faces[key].rigid_map(C.Q)
        cond = _gram_cond(_gram(G))
        if cond <= EQ_MAX_COND:
            keys.append(key); Gs[key] = G
        else:
            bad[f'{key[0]}:{key[1]}'] = cond
    if not keys:
        raise ValueError(f'EQUILIBRATION_DEGENERATE face_c: no face with a regular G G^T ({bad})')
    per = int(np.ceil(total / len(keys)))
    Qs, fr_raw, fr = [], [], []
    for key in keys:
        T, G = faces[key], Gs[key]
        Ms = _gram(G)
        for j in range(0, per, chunk):
            k = min(chunk, per - j)
            t = T.random(k, gen)
            F_raw = T.forces(t)
            F = T.forces(equilibrate([(G, t, None)], Ms[None].expand(k, 6, 6))[0])
            fr_raw.append(_frac(C.Q, F_raw)); fr.append(_frac(C.Q, F))
            Qs.append(C.neumann(F))
    q = torch.cat(Qs, 1)
    info = dict(faces_used=len(keys), faces_degenerate=bad, **_frac_stats(fr_raw, fr))
    return q[:, torch.randperm(q.shape[1], generator=gen, device=dev)[:total]], info


def _spd64(crow, col, vals, n):
    return TE.SPDSolver(crow, col, vals, n)                                  # fp64 (no fp32 attempt: see support_k)


def _finite(x, what):
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f'NONFINITE_{what}')
    return x


def support_k(C, faces, total, gen, chunk=32):
    g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
    keys = list(faces) if len(faces) > 1 else []                            # springs on one face, loads on another
    if not keys:
        return None
    per = int(np.ceil(total / len(keys)))
    ru, cu = C.ru.long(), C.cu.long()
    Qs = []
    for key in keys:
        a, side = key
        m = C.port_is_box & (g[:, a] == side)
        fdofs = C.P[torch.nonzero(torch.as_tensor(m, device=dev).repeat_interleave(3)).squeeze(1)]
        alpha = float(0.3 * 10 ** (torch.rand((), generator=gen, device=dev) * np.log10(10.0)))
        v = C.vals.clone(); v[C.diag[fdofs]] *= 1 + alpha                   # k_i = alpha diag(K)_ii
        s = 1 / torch.sqrt(v[C.diag])
        sol = _spd64(C.crow.int(), C.cu, (v * s[ru] * s[cu]).contiguous(), C.nb)
        others = [T for kk, T in faces.items() if kk != key]
        for j in range(0, per, chunk):
            k = min(chunk, per - j)
            f = sum(T.forces(T.random(k, gen)) for T in others)
            Fb = torch.zeros((C.nb, k), dtype=dt, device=dev); Fb[C.P] = f
            u = _finite(s[:, None] * sol.solve(s[:, None] * Fb), 'SUPPORT_K')
            Qs.append(u[C.P])
        sol.free(); del sol, v; gc.collect(); torch.cuda.empty_cache()
    q = torch.cat(Qs, 1)
    return q[:, torch.randperm(q.shape[1], generator=gen, device=dev)[:total]]


def family_full(case, packets):
    """The FULL parent of a case's family ('<fresh_split_id>_full'; rotated packets: the rotated FULL if it exists)."""
    m = re.match(r'^(fresh_[a-z]+_\d{4})_(.*)$', case)
    if m is None:
        return None
    fam, rest = m.groups()
    rot = re.search(r'(_rot[a-z0-9]+)$', rest)
    name = f'{fam}_full' + (rot.group(1) if rot else '')
    return name if ((Path(packets) / name / 'FRESH_CONTEXT.json').exists() or (TE.packet_dir(name) / 'FRESH_CONTEXT.json').exists()) else None


def _mem(tag, **kw):
    """PREP_MEMLOG=1: device memory at a stage boundary (torch allocated / reserved, device free)."""
    if os.environ.get('PREP_MEMLOG'):
        free, tot = torch.cuda.mem_get_info()
        print(json.dumps(dict(event='MEM', tag=tag, alloc_GB=round(torch.cuda.memory_allocated() / 2 ** 30, 2),
                              reserved_GB=round(torch.cuda.memory_reserved() / 2 ** 30, 2), free_GB=round(free / 2 ** 30, 2), **kw)), flush=True)


def _nbytes(v):
    if v.layout == torch.sparse_csr:
        return v.values().numel() * v.values().element_size() + v.col_indices().numel() * v.col_indices().element_size()
    return v.numel() * v.element_size()


def _offload(obj, min_bytes=64 << 20):
    """Move the large device tensors held directly by obj (a teacher.Cell) to the host; returns their names (_restore).
    Only where the data lives changes, never its value."""
    keys = [k for k, v in vars(obj).items() if torch.is_tensor(v) and v.is_cuda and _nbytes(v) >= min_bytes]
    for k in keys:
        setattr(obj, k, getattr(obj, k).to('cpu'))
    gc.collect(); torch.cuda.empty_cache()
    return keys


def _restore(obj, keys):
    for k in keys:
        setattr(obj, k, getattr(obj, k).to(dev))


def _spd_glued(crow, col, vals, n):
    """The two-cell factor: fp64; on a memory failure (two FULL cells: ~8e5 DOFs) free the cache and retry fp64, then fall back
    to an fp32 factor of the (diagonally scaled) system. Only the DIRECTIONS q come from this solve: their reactions,
    sensitivities and unit-energy normalisation are computed afterwards with the test cell's own exact factor. Every solve
    is checked (and, when needed, refined) against the fp64 matrix: see UpperSym / refine."""
    for attempt in ('fp64', 'fp64_retry', 'fp32'):
        try:
            return TE.SPDSolver(crow, col, vals, n, **(dict(fdt=torch.float32) if attempt == 'fp32' else {})), attempt
        except Exception as e:
            if not PG._mem_error(e) or attempt == 'fp32':
                raise
            gc.collect(); torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------- fp64 check / refinement (DATA-3)
class UpperSym:
    """y = A x for a symmetric A stored as its upper CSR (crow, col, vals; diagonal stored once): A x = U x + U^T x - D x in
    the dtype of vals (fp64), chunked over rows so that the gathered temporaries stay ~ 2^25 values. The matrix lives on
    the device, or with host=True on the host with each row chunk copied to x's device when used (the fp32 / fp64_retry
    glued path, where device memory is short); a device memory error switches to host storage for good."""

    def __init__(self, crow, col, vals, host=False):
        self.crow, self.col, self.vals = crow.long(), col, vals
        if host:
            self._to_host()

    def _to_host(self):
        self.crow, self.col, self.vals = self.crow.cpu(), self.col.cpu(), self.vals.cpu()
        gc.collect(); torch.cuda.empty_cache()

    def _mv(self, x):
        d_ = x.device
        k = x.shape[1]
        step = max(1 << 14, (1 << 25) // max(k, 1))
        y = torch.zeros_like(x)
        crow = self.crow
        n = len(crow) - 1
        lo = 0
        while lo < n:
            hi = int(torch.searchsorted(crow, int(crow[lo]) + step, right=True)) - 1
            hi = min(max(hi, lo + 1), n)
            a_, b_ = int(crow[lo]), int(crow[hi])
            rows = torch.repeat_interleave(torch.arange(lo, hi, device=crow.device), crow[lo + 1:hi + 1] - crow[lo:hi]).to(d_)
            cols = self.col[a_:b_].to(d_).long(); v = self.vals[a_:b_].to(d_)
            y.index_add_(0, rows, v[:, None] * x[cols])
            off = rows != cols
            y.index_add_(0, cols[off], v[off][:, None] * x[rows[off]])
            del rows, cols, v, off
            lo = hi
        return y

    def __call__(self, x):
        try:
            return self._mv(x)
        except Exception as e:
            if not PG._mem_error(e) or self.vals.device.type == 'cpu':
                raise
            self._to_host()
            return self._mv(x)


def refine(matvec, solve, b, y, unscale=None, steps=REFINE_STEPS, target=REFINE_TARGET):
    """Iterative refinement of y ~ A^-1 b (A given by the fp64 matvec, solve by the possibly fp32 factor): while the relative
    residual exceeds target, at most `steps` times y += solve(b - A y). The residual is measured as ||w r|| / ||w b|| per
    column with w = unscale (the Jacobi scaling undone: the residual of the unscaled system). Returns y, rel (k,), steps,
    rel0 (the residual before refinement)."""
    w = (lambda z: z) if unscale is None else (lambda z: z * unscale[:, None])
    bn = w(b).norm(dim=0).clamp_min(1e-300)
    r = b - matvec(y)
    rel = rel0 = w(r).norm(dim=0) / bn
    it = 0
    while it < steps and float(rel.max()) > target:
        y = y + solve(r)
        r = b - matvec(y)
        rel = w(r).norm(dim=0) / bn
        it += 1
    return y, rel, it, rel0


def refine_pcg(matvec, solve, b, unscale=None, maxit=None, target=REFINE_TARGET):
    """Safe mode, fp32 factor: CG on the fp64 system preconditioned by the (fp32) factor, column by column (vectorised
    scalars; converged columns frozen). Much faster than stationary refinement when the fp32 factor contracts slowly
    (weakly connected two-cell systems). Same outputs as refine(): y, rel (true residual, recomputed), iterations, rel0."""
    maxit = GLUED_PCG_ITERS if maxit is None else maxit
    w = (lambda z: z) if unscale is None else (lambda z: z * unscale[:, None])
    bn = w(b).norm(dim=0).clamp_min(1e-300)
    x = solve(b)
    r = b - matvec(x)
    rel = rel0 = w(r).norm(dim=0) / bn
    z = solve(r); p = z.clone(); rz = (r * z).sum(0)
    it = 0
    while it < maxit and float(rel.max()) > target:
        act = (rel > target).to(b.dtype)
        Ap = matvec(p)
        alpha = act * rz / (p * Ap).sum(0).clamp_min(1e-300)
        x = x + alpha[None] * p
        r = r - alpha[None] * Ap
        rel = torch.where(act > 0, w(r).norm(dim=0) / bn, rel)
        z = solve(r)
        rz_new = (r * z).sum(0)
        beta = rz_new / rz.clamp_min(1e-300)
        p = z + beta[None] * p
        rz = rz_new
        it += 1
    r = b - matvec(x)
    return x, w(r).norm(dim=0) / bn, it, rel0


# --------------------------------------------------------------------------------------------------------- glued
def neighbour_name(case, d):
    return f'{case}_nb{NB_TAG[tuple(int(x) for x in d)]}'


def _glued_explicit(Ct, body, total, gen, chunk, log):
    """GLUED_NEIGHBOURS=explicit: offset d glues the neighbour packet <case>_nb<tag(d)> (a FULL cell whose four corners on the
    glued face equal the test cell's, far corners drawn independently: gen_new.py), one neighbour per offset, built when
    its offset starts and freed after it. Everything else as the family path."""
    n = Ct.n
    gp = np.stack(np.unravel_index(Ct.port_node_ids, (2 * n + 1,) * 3), 1)
    offs = [d for d in GLUE_OFFSETS if (Ct.port_is_box & (gp[:, np.flatnonzero(d)[0]] == (2 * n if sum(d) > 0 else 0))).sum() > 8]
    if not offs:
        return None, dict(skipped='no glue face with material', skip_final=True)
    have = [d for d in offs if (TE.packet_dir(neighbour_name(Ct.case, d)) / 'FRESH_CONTEXT.json').exists()]
    if not have:                                                           # the planned glue face(s) (gen_new) have no material
        return None, dict(skipped=f'no neighbour packet on a glue face with material {[_axis_name(d) for d in offs]}', skip_final=True)
    offs = have
    ft = box_tractions(Ct)
    cur = {}

    def drop():
        if cur:
            cur['Cn']._free(); cur.clear()
            gc.collect(); torch.cuda.empty_cache()

    def nbr(d):
        drop()
        name = neighbour_name(Ct.case, d)
        Cn = TE.Cell(name, body, log=lambda s_: None); Cn.assemble()
        gn = np.stack(np.unravel_index(Cn.nodes, (2 * n + 1,) * 3), 1)
        fn_ = box_tractions(Cn)
        _offload(Cn)
        cur['Cn'] = Cn
        return Cn, gn, fn_, name

    off_t = _offload(Ct)
    try:
        return _glued_offsets(Ct, None, n, None, ft, None, 'explicit', offs, total, gen, chunk, log, nbr=nbr)
    finally:
        _restore(Ct, off_t)
        drop()


def glued(Ct, body, total, gen, chunk=32, log=print):
    """Two-cell samples (see the module docstring). Returns (q (np_t, total), info) or (None, info)."""
    if GLUED_NEIGHBOURS == 'explicit':
        return _glued_explicit(Ct, body, total, gen, chunk, log)
    parent = family_full(Ct.case, TE.ROOT / 'packets')
    if parent is None:
        return None, dict(skipped='no family FULL parent', skip_final=True)
    n = Ct.n
    gp = np.stack(np.unravel_index(Ct.port_node_ids, (2 * n + 1,) * 3), 1)
    offs = [d for d in GLUE_OFFSETS if (Ct.port_is_box & (gp[:, np.flatnonzero(d)[0]] == (2 * n if sum(d) > 0 else 0))).sum() > 8]
    if not offs:
        return None, dict(skipped='no glue face with material', parent=parent, skip_final=True)
    Cn = TE.Cell(parent, body, log=lambda s_: None); Cn.assemble()
    gn = np.stack(np.unravel_index(Cn.nodes, (2 * n + 1,) * 3), 1)
    ft, fn_ = box_tractions(Ct), box_tractions(Cn)
    off_t, off_n = _offload(Ct), _offload(Cn)                                # both K on the host during the two-cell factors
    try:
        return _glued_offsets(Ct, Cn, n, gn, ft, fn_, parent, offs, total, gen, chunk, log)
    finally:
        _restore(Ct, off_t)
        Cn._free(); del Cn; gc.collect(); torch.cuda.empty_cache()


def _axis_name(d):
    a = int(np.flatnonzero(d)[0])
    return ('+' if sum(d) > 0 else '-') + 'xyz'[a]


def _glued_upper(Ct, Cn, dmap, nb):
    """Glued upper CSR entries (row <= col) of the two cells: test entries as they are, neighbour entries renumbered (upper
    after renumbering), summed. Returns [ru, cu, vals] (int64, int64, fp64) as a list the caller hands over (consumed)."""
    r = torch.cat([Ct.ru.to(dev).long(), dmap[Cn.ru.to(dev).long()]]); c = torch.cat([Ct.cu.to(dev).long(), dmap[Cn.cu.to(dev).long()]])
    v = torch.cat([Ct.vals.to(dev), Cn.vals.to(dev)])
    r, c = torch.minimum(r, c), torch.maximum(r, c)
    key = r * nb + c
    del r, c
    key, inv = torch.unique(key, return_inverse=True)
    vals = torch.zeros(len(key), dtype=dt, device=dev).index_add_(0, inv, v)
    del v, inv
    ru, cu = key // nb, key % nb
    del key
    return [ru, cu, vals]


def _glued_system(buf, nb, far, clamp, alpha):
    """Far face of the neighbour clamped (its DOFs dropped) or on springs k = alpha diag; Jacobi-scaled upper CSR of the
    kept DOFs. buf = [ru, cu, vals] is consumed (freed as early as before). Returns crow, col (int32), vals (scaled, fp64),
    keep (nb bool), sA (scaling of the kept DOFs), nf."""
    ru, cu, vals = buf; buf.clear()
    diag = torch.nonzero(ru == cu).squeeze(1)
    dvals = torch.zeros(nb, dtype=dt, device=dev); dvals[ru[diag]] = vals[diag]
    keep = torch.ones(nb, dtype=torch.bool, device=dev)
    if clamp:
        keep[far] = False
    else:
        spring = torch.zeros(nb, dtype=dt, device=dev); spring[far] = alpha * dvals[far]
        vals = vals + torch.where(ru == cu, spring[ru], torch.zeros((), dtype=dt, device=dev))
        dvals = dvals + spring
    new = torch.full((nb,), -1, dtype=torch.long, device=dev); new[keep] = torch.arange(int(keep.sum()), device=dev)
    sel = keep[ru] & keep[cu]
    rA, cA, vA = new[ru[sel]], new[cu[sel]], vals[sel]
    del ru, cu, vals, sel, new, diag
    nf = int(keep.sum())
    sA = 1 / torch.sqrt(dvals[keep])
    del dvals
    order = torch.argsort(rA * nf + cA)
    rA, cA, vA = rA[order], cA[order], vA[order]
    del order
    vA.mul_(sA[rA]).mul_(sA[cA])                                            # diagonal scaling in place
    crow = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(torch.bincount(rA, minlength=nf), 0)]).int()
    cA = cA.int(); del rA
    gc.collect(); torch.cuda.empty_cache()
    return crow, cA, vA.contiguous(), keep, sA, nf


def _glued_group_safe(Ct, Cn, dmap, nb, far, clamp, alpha, ks, tf, nf_, ptd, pnd, gen, log, d):
    """GLUED_SAFE: one (offset, far-face BC) group like the default path, but a device memory failure anywhere in the group
    (factor, fp64 matvec, solves, refinement; the default path only catches the factor call) does not end the class:
      attempt 1  the default factor ladder (fp64, fp64_retry, fp32); an fp64 factor that leaves less than GLUED_HEADROOM of
                 free device memory is replaced at once by an fp32 factor (half the size)
      attempt 2  fp32 factor, fp64 matrix on the host
    Both attempts replay the same random loads (generator state restored), every solve is refined against the fp64 matrix
    and must pass the same residual gate (RESID_TOL). Returns (group info, list of q blocks) or (None, None) when both fail."""
    state = gen.get_state()
    for attempt in ('auto', 'fp32'):
        gen.set_state(state)
        sol = Kg = None
        try:
            crow, cA, vA, keep, sA, nf = _glued_system(_glued_upper(Ct, Cn, dmap, nb), nb, far, clamp, alpha)
            nnz_ = int(vA.numel())
            _mem('glued_before_factor', offset=d, clamped=clamp, dofs=nf, nnz=nnz_, attempt=attempt)
            if attempt == 'auto' and nf > GLUED_FP32_DOFS:
                sol, prec = TE.SPDSolver(crow, cA, vA, nf, fdt=torch.float32), 'fp32_large'
            elif attempt == 'auto':
                sol, prec = _spd_glued(crow, cA, vA, nf)
                free_ = torch.cuda.mem_get_info()[0]
                if prec == 'fp64' and free_ < GLUED_HEADROOM:
                    sol.free(); sol = None; gc.collect(); torch.cuda.empty_cache()
                    log(json.dumps(dict(event='GLUED_SAFE_FP32', case=Ct.case, offset=d, clamped=clamp, free_GB=free_ / 2 ** 30)))
                    sol, prec = TE.SPDSolver(crow, cA, vA, nf, fdt=torch.float32), 'fp32_headroom'
            else:
                sol, prec = TE.SPDSolver(crow, cA, vA, nf, fdt=torch.float32), 'fp32_retry'
            _mem('glued_after_factor', offset=d, precision=prec)
            Kg = UpperSym(crow, cA, vA, host=prec != 'fp64')
            del crow, cA, vA; gc.collect(); torch.cuda.empty_cache()
            Qg, rels, rels0, steps = [], [], [], []
            for k in ks:
                only_n = (torch.rand(k, device=dev, generator=gen) < 0.25).to(dt)
                F = torch.zeros((nb, k), dtype=dt, device=dev)
                if tf:
                    F.index_add_(0, ptd, sum(T.forces(T.random(k, gen)) for T in tf) * (1 - only_n)[None, :])
                if nf_:
                    F.index_add_(0, pnd, sum(T.forces(T.random(k, gen)) for T in nf_))
                b = sA[:, None] * F[keep]
                if prec == 'fp64':
                    y, rel, it, rel0 = refine(Kg, sol.solve, b, sol.solve(b), unscale=1 / sA)
                else:                                                   # fp32 factor: PCG on the fp64 system
                    y, rel, it, rel0 = refine_pcg(Kg, sol.solve, b, unscale=1 / sA)
                u = torch.zeros((nb, k), dtype=dt, device=dev)
                u[keep] = sA[:, None] * y
                Qg.append(_finite(u[ptd], 'GLUED'))
                rels.append(rel); rels0.append(rel0); steps.append(it)
                del F, b, y, u
            sol.free(); sol = None
            del Kg, keep, sA; gc.collect(); torch.cuda.empty_cache()
            rel = torch.cat(rels)
            return dict(clamped=clamp, alpha=alpha, samples=int(sum(ks)), dofs=nf, nnz=nnz_, precision=prec, safe_attempt=attempt,
                        refine_steps=int(max(steps)), resid_max=float(rel.max()), resid_median=float(rel.median()),
                        resid_unrefined_max=float(torch.cat(rels0).max())), Qg
        except Exception as e:
            if not PG._mem_error(e):
                raise
            if sol is not None:
                try:
                    sol.free()
                except Exception:
                    pass
            sol = Kg = None
            crow = cA = vA = keep = sA = F = b = y = u = None
            gc.collect(); torch.cuda.empty_cache()
            log(json.dumps(dict(event='GLUED_SAFE_RETRY' if attempt == 'auto' else 'GLUED_OFFSET_SKIPPED', case=Ct.case, offset=d,
                                clamped=clamp, attempt=attempt, error=repr(e)[:160])))
    return None, None


def _glued_offsets(Ct, Cn, n, gn, ft, fn_, parent, offs, total, gen, chunk, log, nbr=None):
    gt = np.stack(np.unravel_index(Ct.nodes, (2 * n + 1,) * 3), 1)
    made = 0                                                                # samples so far (offsets that do not fit or fail the
                                                                            # residual check are skipped and their quota goes to the
                                                                            # remaining offsets)
    W = 6 * n + 1                                                           # absolute key over [-2n, 4n]^3
    kt = ((gt[:, 0] + 2 * n) * W + gt[:, 1] + 2 * n) * W + gt[:, 2] + 2 * n
    ordt = np.argsort(kt)
    Qs, labels, info = [], [], dict(parent=parent, offsets=[])
    for oi, d in enumerate(offs):
        per = int(np.ceil((total - made) / (len(offs) - oi)))
        if nbr is not None:                                                 # explicit neighbours: one cell per offset
            Cn, gn, fn_, parent = nbr(d)
        a = int(np.flatnonzero(d)[0]); sgn = int(sum(d))
        ga = gn + 2 * n * np.asarray(d)[None]
        kn = ((ga[:, 0] + 2 * n) * W + ga[:, 1] + 2 * n) * W + ga[:, 2] + 2 * n
        pos = np.searchsorted(kt[ordt], kn)
        shared = kt[ordt][np.minimum(pos, len(kt) - 1)] == kn
        tnode = ordt[np.minimum(pos, len(kt) - 1)]
        nnode = np.full(len(Cn.nodes), -1, np.int64)
        nnode[shared] = tnode[shared]
        own = np.flatnonzero(~shared)
        nnode[own] = len(Ct.nodes) + np.arange(len(own))
        Ntot = len(Ct.nodes) + len(own)
        dmap = torch.as_tensor((3 * nnode[:, None] + np.arange(3)[None]).reshape(-1), device=dev)
        nb = 3 * Ntot
        far_local = 2 * n if sgn > 0 else 0
        far_nodes = np.flatnonzero(gn[:, a] == far_local)
        far = torch.as_tensor(np.unique((3 * nnode[far_nodes][:, None] + np.arange(3)[None]).reshape(-1)), device=dev)
        # far-face BC per chunk (DATA-6), drawn before any load of this offset
        sizes = [min(chunk, per - j) for j in range(0, per, chunk)]
        clampc = (torch.rand(len(sizes), generator=gen, device=dev) < 0.5).tolist()
        # loads: free box faces of both cells (test: not the glued face; neighbour: neither glued nor far face)
        tface = (a, 2 * n if sgn > 0 else 0)
        nglue, nfar = (a, 0 if sgn > 0 else 2 * n), (a, far_local)
        tf = [T for kk, T in ft.items() if kk != tface]
        nf_ = [T for kk, T in fn_.items() if kk not in (nglue, nfar)]
        ptd = Ct.P                                                          # test port DOFs in the glued numbering
        pnd = dmap[Cn.P]                                                    # neighbour port DOFs in the glued numbering
        ro = dict(offset=list(d), axis=_axis_name(d), shared_nodes=int(shared.sum()), groups=[], **({'neighbour': parent} if nbr is not None else {}))
        Qo, Lo, ok = [], [], True
        for clamp in (True, False):
            ks = [k for k, c_ in zip(sizes, clampc) if bool(c_) == clamp]
            if not ks:
                continue
            alpha = None if clamp else float(0.3 * 10 ** torch.rand((), generator=gen, device=dev))
            if GLUED_SAFE:
                g, Qg = _glued_group_safe(Ct, Cn, dmap, nb, far, clamp, alpha, ks, tf, nf_, ptd, pnd, gen, log, d)
                if g is None:
                    ro.update(skipped='memory (safe mode: fp64 and fp32 attempts)')
                    ok = False
                    break
                Qo += Qg; Lo += [clamp] * sum(ks)
                ro['groups'].append(g)
                log(json.dumps(dict(event='GLUED_OFFSET', case=Ct.case, parent=parent, offset=d, shared_nodes=ro['shared_nodes'], **g)))
                if not g['resid_max'] <= RESID_TOL:
                    ro['skipped'] = f"residual {g['resid_max']:.3g} > {RESID_TOL:g} ({g['precision']}, {g['refine_steps']} refinement steps)"
                    log(json.dumps(dict(event='GLUED_OFFSET_REJECTED', case=Ct.case, offset=d, clamped=clamp, resid_max=g['resid_max'],
                                        precision=g['precision'])))
                    ok = False
                    break
                continue
            crow, cA, vA, keep, sA, nf = _glued_system(_glued_upper(Ct, Cn, dmap, nb), nb, far, clamp, alpha)
            nnz_ = int(vA.numel())
            _mem('glued_before_factor', offset=d, clamped=clamp, dofs=nf, nnz=nnz_)
            try:
                sol, prec = _spd_glued(crow, cA, vA, nf)
            except Exception as e:
                if not PG._mem_error(e):
                    raise
                crow = cA = vA = keep = sA = None; gc.collect(); torch.cuda.empty_cache()
                ro.update(skipped='memory', dofs=nf, nnz=nnz_)
                log(json.dumps(dict(event='GLUED_OFFSET_SKIPPED', case=Ct.case, offset=d, clamped=clamp, dofs=nf, nnz=nnz_)))
                ok = False
                break
            _mem('glued_after_factor', offset=d, precision=prec)
            Kg = UpperSym(crow, cA, vA, host=prec != 'fp64')                # the fp64 scaled glued matrix (residual / refinement)
            del crow, cA, vA; gc.collect(); torch.cuda.empty_cache()
            rels, rels0, steps = [], [], []
            for k in ks:
                only_n = (torch.rand(k, device=dev, generator=gen) < 0.25).to(dt)
                F = torch.zeros((nb, k), dtype=dt, device=dev)
                if tf:
                    F.index_add_(0, ptd, sum(T.forces(T.random(k, gen)) for T in tf) * (1 - only_n)[None, :])
                if nf_:
                    F.index_add_(0, pnd, sum(T.forces(T.random(k, gen)) for T in nf_))
                b = sA[:, None] * F[keep]
                y, rel, it, rel0 = refine(Kg, sol.solve, b, sol.solve(b), unscale=1 / sA)
                u = torch.zeros((nb, k), dtype=dt, device=dev)
                u[keep] = sA[:, None] * y
                Qo.append(_finite(u[ptd], 'GLUED')); Lo += [clamp] * k
                rels.append(rel); rels0.append(rel0); steps.append(it)
                del F, b, y, u
            sol.free(); del sol, Kg, keep, sA; gc.collect(); torch.cuda.empty_cache()
            rel = torch.cat(rels)
            g = dict(clamped=clamp, alpha=alpha, samples=int(sum(ks)), dofs=nf, nnz=nnz_, precision=prec,
                     refine_steps=int(max(steps)), resid_max=float(rel.max()), resid_median=float(rel.median()),
                     resid_unrefined_max=float(torch.cat(rels0).max()))
            ro['groups'].append(g)
            log(json.dumps(dict(event='GLUED_OFFSET', case=Ct.case, parent=parent, offset=d, shared_nodes=ro['shared_nodes'], **g)))
            if not g['resid_max'] <= RESID_TOL:
                ro['skipped'] = f"residual {g['resid_max']:.3g} > {RESID_TOL:g} ({prec}, {g['refine_steps']} refinement steps)"
                log(json.dumps(dict(event='GLUED_OFFSET_REJECTED', case=Ct.case, offset=d, clamped=clamp, resid_max=g['resid_max'],
                                    precision=prec)))
                ok = False
                break
        info['offsets'].append(ro)
        if not ok:
            continue
        Qs += Qo; labels += [(ro['axis'], c_) for c_ in Lo]; made += per
    if made < total:                                                        # no offset fits at all
        info['skipped'] = f'memory or residual: {made} of {total} samples'
        return None, info
    q = torch.cat(Qs, 1)
    perm = torch.randperm(q.shape[1], generator=gen, device=dev)[:total]
    bc = {}
    for i in perm.tolist():
        ax, c_ = labels[i]
        e = bc.setdefault(ax, dict(clamped=0, springs=0)); e['clamped' if c_ else 'springs'] += 1
    groups = [g for o in info['offsets'] for g in o['groups']]
    info.update(bc_counts=bc, resid_max=max(g['resid_max'] for g in groups), precisions=sorted({g['precision'] for g in groups}),
                refine_steps_max=max(g['refine_steps'] for g in groups))
    return q[:, perm], info


# --------------------------------------------------------------------------------------------------------- output
def sens_F(C, q, chunk=128):
    S, F = [], []
    for j in range(0, q.shape[1], chunk):
        u = C.extend(q[:, j:j + chunk])
        S.append(C.sens2(u)); F.append((C.K @ u)[C.P])
    return torch.cat(S, 1), torch.cat(F, 1)


def _npsave(path, arr):
    """np.save to a temporary name next to path, then os.replace: never writes into an existing inode (a hard link into a
    frozen source tree stays untouched) and a crash leaves either the old or the new file."""
    path = Path(path)
    tmp = path.with_name(f'.{path.stem}.tmp{os.getpid()}.npy')
    np.save(tmp, arr)
    os.replace(tmp, path)


def _write_text(path, s):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    tmp.write_text(s)
    os.replace(tmp, path)


def _stage(d, cls, q, S, F):
    """Write all files of a class under temporary names; returns [(tmp, final)] for _commit."""
    out, lo = [], 0
    try:
        for name, n_ in SPLITS:
            for suf, arr in (('', q[:, lo:lo + n_].T.contiguous().to(torch.float32)), ('_sens', S[:, lo:lo + n_].T.contiguous()),
                             ('_F', F[:, lo:lo + n_].T.contiguous().to(torch.float32))):
                final = d / f'{name}_{cls}{suf}.npy'
                tmp = d / f'.{name}_{cls}{suf}.new{os.getpid()}.npy'
                np.save(tmp, arr.cpu().numpy())
                out.append((tmp, final))
            lo += n_
    except BaseException:
        for tmp, _ in out:
            tmp.unlink(missing_ok=True)
        raise
    return out


def _commit(staged, backup=False):
    """Move staged files to their names. backup=True (in-place --fix-support): every existing final file (q, _sens, _F) is
    first renamed to *.nan_bak.npy, unless a backup already exists (a half-done earlier run: the backup is the original)."""
    if backup:
        for _, final in staged:
            bak = final.with_name(final.stem + '.nan_bak.npy')
            if final.exists() and not bak.exists():
                final.rename(bak)
    for tmp, final in staged:
        os.replace(tmp, final)


def save(d, cls, q, S, F):
    _commit(_stage(Path(d), cls, q, S, F))


def _nonfinite_bank(d, cls):
    return any(not np.isfinite(np.load(d / f'{s}_{cls}.npy', mmap_mode='r')).all() for s, _ in SPLITS if (d / f'{s}_{cls}.npy').exists())


def _bank_names(cls, splits=('train', 'val', 'test')):
    return {f'{s}_{cls}{suf}.npy' for s in splits for suf in SUFFIXES}


def _splits_json():
    return [list(x) for x in SPLITS]


def _valid(rec, d, cls, scheme):
    """A DONE2 class record made by this VERSION with the same seed scheme and BANK_SPLITS, whose files are all present
    (or a deterministic skip)."""
    r = (rec or {}).get('banks', {}).get(cls)
    if not isinstance(r, dict) or r.get('version') != VERSION or r.get('seed_scheme') != scheme or r.get('splits') != _splits_json():
        return False
    if r.get('skip_final'):
        return True
    return 'count' in r and all((d / f'{s}_{cls}{suf}.npy').exists() for s, _ in SPLITS for suf in SUFFIXES)


def _link_tree(src, d, skip=()):
    """Hard-link (copy when linking fails) every file of src missing in d; names in skip are left out."""
    d.mkdir(parents=True, exist_ok=True)
    n_link = n_copy = 0
    if src.is_dir():
        for f in sorted(src.iterdir()):
            if not f.is_file() or f.name in skip or f.name.startswith('.'):
                continue
            t = d / f.name
            if t.exists() or t.is_symlink():
                continue
            try:
                os.link(f, t); n_link += 1
            except OSError:
                shutil.copy2(f, t); n_copy += 1
    return dict(linked=n_link, copied=n_copy)


def _retire(d, cls, sep):
    """Stale bank files of a class that failed / was skipped: unlink in an --out tree (the source keeps its files), rename
    to *.stale_bak.npy in place. Returns the names."""
    out = []
    for name in sorted(_bank_names(cls)):
        f = d / name
        if f.exists():
            if sep:
                f.unlink()
            else:
                f.rename(d / (f.stem + '.stale_bak.npy'))
            out.append(name)
    return out


def _load_rec(src, d, sep, scheme, log):
    """DONE2 of the target (d); with --out and no DONE2 there yet, the source's, without the records of v2 classes that
    are not valid for this VERSION (their files were not linked). Also links the source files (see the docstring)."""
    rd = lambda p_: json.loads(p_.read_text()) if p_.exists() else None
    if not sep:
        return rd(d / 'DONE2.json') or dict(case=d.name, banks={}, seconds={})
    src_rec = rd(src / 'DONE2.json')
    ok_src = {c for c in V2_BANKS if _valid(src_rec, src, c, scheme)}
    skip = {'DONE2.json'} | set().union(*[_bank_names(c) for c in V2_BANKS if c not in ok_src])
    links = _link_tree(src, d, skip)
    rec = rd(d / 'DONE2.json')
    if rec is None:
        rec = dict(src_rec or dict(case=d.name, banks={}, seconds={}))
        rec['banks'] = {c: v for c, v in rec.get('banks', {}).items() if c not in V2_BANKS or c in ok_src}
        rec['source'] = str(src)
    rec.setdefault('links', {}).update(links)
    log(json.dumps(dict(event='OUT_TREE', case=d.name, source=str(src), out=str(d), **links, inherited_v2=sorted(ok_src))))
    return rec


def one(body, data, case, classes, F_old=False, fix_support=False, log=print, out=None, force=False, legacy_seed=False):
    """One case. Returns dict(status='done' | 'partial' | 'skipped', produced=[...], failed={...}); raises only when nothing
    could be produced (setup failure, or every requested class failed)."""
    total = sum(n_ for _, n_ in SPLITS)
    data = Path(data); out = data if out is None else Path(out)
    src, d = data / case, out / case
    sep = out.resolve() != data.resolve()
    gens = ClassGens(case, legacy_seed)
    scheme = gens.scheme
    d.mkdir(parents=True, exist_ok=True)
    rec = _load_rec(src, d, sep, scheme, log)
    rec.setdefault('banks', {}); rec.setdefault('seconds', {})
    tt = rec['seconds']
    sup_cls = 'support64' if sep else 'support'
    rec.update(version=VERSION, seed=gens.record(list(NEW) + [sup_cls]), out_mode=sep)
    produced, failed = [], {}
    base = lambda cls: dict(version=VERSION, seed_scheme=scheme, seed=gens.seed(cls), splits=_splits_json())
    write = lambda: _write_text(d / 'DONE2.json', json.dumps(rec))

    def settle(cls, entry, event):
        """A class that failed or was skipped: a valid earlier bank stays (record gets last_failure); otherwise a fresh
        record and the stale files are retired."""
        old = rec['banks'].get(cls)
        if _valid(rec, d, cls, scheme) and isinstance(old, dict) and 'count' in old:
            old['last_failure'] = entry
        else:
            r = dict(base(cls), **entry)
            if cls in V2_BANKS:
                moved = _retire(d, cls, sep)
                if moved:
                    r['retired_files'] = moved
            rec['banks'][cls] = r
        if 'failed' in entry:
            failed[cls] = entry['failed']
        log(json.dumps(dict(event=event, case=case, cls=cls, **entry)))
        write()

    def fail(cls, e, stage):
        settle(cls, dict(failed=repr(e)[:300], stage=stage), 'CLASS_FAILED2')

    todo = [c for c in NEW if c in classes and (force or not _valid(rec, d, c, scheme))]
    for c in classes:
        if c not in todo:
            log(json.dumps(dict(event='CLASS_VALID2', case=case, cls=c)))
    if 'glued' in todo and GLUED_NEIGHBOURS != 'explicit' and family_full(case, TE.ROOT / 'packets') is None:
        todo.remove('glued')
        settle('glued', dict(skipped='no family FULL parent', skip_final=True), 'CLASS_SKIPPED2')
    fix = False
    if fix_support:
        half_done = (d / 'train_support.nan_bak.npy').exists() and not (d / 'train_support.npy').exists()
        fix = half_done or ((d / 'train_support.npy').exists() and _nonfinite_bank(d, 'support'))
        if sep and fix and not force and _valid(rec, d, 'support64', scheme):
            fix = False
    if not todo and not fix and not F_old:
        write()
        log(json.dumps(dict(event='SKIP2', case=case, reason='all requested classes valid')))
        return dict(status='skipped', produced=[], failed={})
    t0 = time.perf_counter()
    C = TE.Cell(case, body, log=lambda s_: None)
    try:
        if C.ni == 0:
            raise ValueError('NO_INTERIOR')
        _mem('start', case=case)
        C.assemble(); _mem('assembled'); C.dmoments(); _mem('dmoments')
        faces = box_tractions(C)
        rec['faces'] = [list(k) for k in faces]
        raw, info = {}, {}

        def gen_class(cls, fn, stage):
            t = time.perf_counter()
            try:
                q = fn()
            except Exception as e:
                fail(cls, e, stage)
                return
            if isinstance(q, tuple):
                q, info[cls] = q
            info.setdefault(cls, {})
            if q is None:
                settle(cls, dict(info[cls]), 'CLASS_SKIPPED2')
                return
            info[cls]['gen_seconds'] = time.perf_counter() - t
            raw[cls] = q

        def finalize():
            """Interior factor (if not alive), then normalise / sensitivities / reactions and save every pending class."""
            if not raw:
                return
            if C.sol_I is None:
                try:
                    tt['interior_precision'] = PG.factor_safe(C, neumann=False, fp32=True)
                except Exception as e:
                    for cls in list(raw):
                        fail(cls, e, 'interior_factor')
                    raw.clear()
                    return
            for cls in list(raw):
                t = time.perf_counter()
                try:
                    q = _finite(raw.pop(cls), cls.upper())
                    qn = (q - C.Q @ (C.Q.T @ q)).norm(dim=0)
                    if bool((qn <= 1e-12 * qn.max()).any()):
                        raise ValueError(f'ZERO_DIRECTION_{cls.upper()}: {int((qn <= 1e-12 * qn.max()).sum())} columns')
                    q = PG.normalize(C, q)
                    rq = (q * q).sum(0)
                    S, F = sens_F(C, q)
                    _finite(S, cls.upper() + '_SENS'); _finite(F, cls.upper() + '_F')
                    r = dict(base(cls), rayleigh_quantiles=np.quantile((1 / rq).cpu().numpy(), [0, .1, .5, .9, 1]).tolist(),
                             unit_energy_check=float(((F[:, :16] * q[:, :16]).sum(0) - 1).abs().max()), count=q.shape[1],
                             **info.get(cls, {}))
                    _commit(_stage(d, cls, q, S, F), backup=(cls == 'support'))
                    r['finalize_seconds'] = time.perf_counter() - t
                    rec['banks'][cls] = r
                    produced.append(cls)
                    write()
                    log(json.dumps(dict(event='CLASS_DONE2', case=case, cls=cls, count=r['count'],
                                        **{k_: v_ for k_, v_ in r.items() if k_.startswith(('rigid_frac', 'resid', 'bc_', 'precisions'))})))
                    del q, S, F
                except Exception as e:
                    fail(cls, e, 'finalize')

        # stage A: Neumann factor (force_c, face_c)
        t = time.perf_counter()
        A = [c for c in ('force_c', 'face_c') if c in todo]
        if A and not faces:
            for c in A:
                fail(c, ValueError('NO_BOX_FACE_WITH_MATERIAL'), 'faces')
        elif A:
            try:
                tt['neumann_precision'] = PG.factor_safe(C, neumann=True, interior=False, fp32_neumann=True)
            except Exception as e:
                for c in A:
                    fail(c, e, 'neumann_factor')
            else:
                if 'force_c' in A:
                    gen_class('force_c', lambda: force_c(C, faces, total, gens('force_c')), 'generate')
                if 'face_c' in A:
                    gen_class('face_c', lambda: face_c(C, faces, total, gens('face_c')), 'generate')
            finally:
                C._free()
        tt['neumann_dirs2'] = time.perf_counter() - t; t = time.perf_counter()
        # stage B: spring factors (support_k, the support fix)
        if 'support_k' in todo:
            if not faces:
                fail('support_k', ValueError('NO_BOX_FACE_WITH_MATERIAL'), 'faces')
            else:
                def _sk():
                    q = support_k(C, faces, total, gens('support_k'))
                    return (q, {}) if q is not None else (None, dict(skipped=f'{len(faces)} box face(s) with material', skip_final=True))
                gen_class('support_k', _sk, 'generate')
        if fix:
            def _fix():
                orig = PG.spd_safe
                PG.spd_safe = _spd64
                try:
                    g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
                    X = torch.as_tensor(g / (2 * C.n), dtype=dt, device=dev)
                    return _finite(PG.support_bank(C, X, g, total, gens(sup_cls)), 'SUPPORT_FP64'), dict(fixes='support', fp64=True)
                finally:
                    PG.spd_safe = orig
            gen_class(sup_cls, _fix, 'generate')
        tt['spring_dirs2'] = time.perf_counter() - t; t = time.perf_counter()
        # stage C: save everything so far (DATA-9: a glued failure cannot take these with it)
        finalize()
        tt['normalize_sens_F2'] = time.perf_counter() - t; t = time.perf_counter()
        # stage D: glued (its own two-cell factors; the interior factor is freed first)
        if 'glued' in todo:
            C._free()
            _mem('before_glued', case=case)
            gen_class('glued', lambda: glued(C, body, total, gens('glued'), log=log), 'generate')
            tt['glued_dirs'] = time.perf_counter() - t; t = time.perf_counter()
            finalize()
            tt['glued_normalize_sens_F2'] = time.perf_counter() - t; t = time.perf_counter()
        # stage E: reactions of the old classes
        if F_old:
            try:
                if C.sol_I is None:
                    tt['interior_precision'] = PG.factor_safe(C, neumann=False, fp32=True)
                for cls in OLD:
                    if not (d / f'train_{cls}.npy').exists():
                        continue
                    for s_, _ in SPLITS:
                        qb = torch.as_tensor(np.load(d / f'{s_}_{cls}.npy'), device=dev).T.to(dt)
                        ok = torch.isfinite(qb).all(0)
                        F = torch.full_like(qb, float('nan'))
                        if ok.any():
                            F[:, ok] = torch.cat([C.apply(qb[:, ok][:, j:j + 128]) for j in range(0, int(ok.sum()), 128)], 1)
                        _npsave(d / f'{s_}_{cls}_F.npy', F.T.contiguous().to(torch.float32).cpu().numpy())
                    rec['banks'].setdefault(cls + '_F', {})['written'] = True
                    produced.append(cls + '_F')
                    write()
            except Exception as e:
                failed['F_old'] = repr(e)[:300]
                rec['banks']['F_old'] = dict(failed=repr(e)[:300], version=VERSION)
                log(json.dumps(dict(event='CLASS_FAILED2', case=case, cls='F_old', failed=repr(e)[:300])))
            tt['F_old2'] = time.perf_counter() - t
        rec.update(splits=SPLITS, classes=sorted(set(rec.get('classes', [])) | set(classes)), total_seconds2=time.perf_counter() - t0,
                   peak_GB2=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0)
        write()
        if failed and not produced:
            raise RuntimeError(f'NOTHING_PRODUCED: {failed}')
        status = 'partial' if failed else 'done'
        log(json.dumps(dict(event='DONE2' if not failed else 'PARTIAL2', case=case, seconds=rec['total_seconds2'], produced=produced,
                            failed=failed, banks=rec['banks'])))
        return dict(status=status, produced=produced, failed=failed)
    finally:
        C._free()


def main(argv):
    import argparse, traceback
    ap = argparse.ArgumentParser()
    ap.add_argument('body'); ap.add_argument('data'); ap.add_argument('cases', nargs='+')
    ap.add_argument('--classes', default=','.join(NEW)); ap.add_argument('--F-old', action='store_true')
    ap.add_argument('--fix-support', action='store_true')
    ap.add_argument('--out', default=None, help='new data root (default: the data dir, in place)')
    ap.add_argument('--force', action='store_true', help='regenerate classes whose DONE2 record is already valid')
    ap.add_argument('--legacy-seed', action='store_true', help='old seed formula and one shared generator (DATA-1)')
    a = ap.parse_args(argv)
    classes = [c for c in a.classes.split(',') if c]
    bad = [c for c in classes if c not in NEW]
    if bad:
        raise SystemExit(f'unknown classes {bad}')
    Q = Path(os.environ['PREP_LOCK']) if os.environ.get('PREP_LOCK') else None
    n_prod = n_fail = 0
    for ci, case in enumerate(a.cases):
        tag = f'prep2_{os.getpid()}_{ci}'
        if Q is not None:
            PG._acquire(Q, tag)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            r = one(a.body, Path(a.data), case, classes, a.F_old, a.fix_support, out=a.out, force=a.force, legacy_seed=a.legacy_seed)
            n_prod += bool(r['produced'])
        except Exception as e:
            n_fail += 1
            own = [f'{fr.name}:{fr.lineno}' for fr in traceback.extract_tb(e.__traceback__) if fr.filename.endswith(('prep_geo2.py', 'prep_geo.py'))]
            print(json.dumps(dict(event='FAILED2', case=case, error=repr(e)[:300], where=own, trace=traceback.format_exc()[-1500:])), flush=True)
        gc.collect(); torch.cuda.empty_cache()
        if Q is not None:
            PG._release(Q, tag)
    print(json.dumps(dict(event='SUMMARY2', cases=len(a.cases), produced=n_prod, failed=n_fail)), flush=True)
    return 1 if (n_fail and not n_prod) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
