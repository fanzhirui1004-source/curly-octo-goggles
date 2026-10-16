"""Section 6.11 homogenisation comparison, step 2: macroscale model and its MMA optimisation.  New script (CPU only).

Material: the effective tensor C^H(tau) of the Schwarz-P sheet cell with uniform thickness (homog_cell.py), cubic
symmetry (C11, C12, C44), and its volume fraction rho(tau); cubic splines in tau (derivatives from the splines).  Graded
designs use the local thickness: at every quadrature point tau(x) is the trilinear interpolation of its cell's eight
corner parameters (the lattice vertex field), and C^H(tau(x)), rho(tau(x)) are evaluated there (the usual graded-lattice
homogenisation).  Geometry: the lattice box of the layout (cell units), a Q1 hexahedral mesh with --m elements per cell and
axis; the planar cut n . X <= b of the layout is integrated with a finite-cell indicator (4^3 sub-points per element that
the plane intersects, 2^3 Gauss points otherwise); void parts carry --eps x C^H(0.4) to keep the matrix regular.
Supports and loads as opt_design.py: the clamp face fully fixed (opt-in --clamp cut: the lattice bonded to a wall along
the cut plane, u = 0 imposed by a penalty --pen x C11(0.4) / h on the plane section n . X = b of the lattice box, 7-point
triangle rule on each element's section polygon; the counterpart of opt_design.py --clamp cut); the load face carries a uniform unit traction in
--load-dir over its material part (face quadrature with the same indicator), normalised to unit total force.
Compliance C = f^T u; adjoint sensitivity dC/dtau_v = -sum_q w_q phi_q eps_q^T dC^H/dtau(tau_q) eps_q N_v(x_q);
volume V = sum_q w_q phi_q rho(tau_q), dV/dtau_v likewise.  Design variables, fixed load-face vertices, constraints
(volume V/V* - 1, span, gradient norm, bounds) and the MMA settings are those of opt_design.py.
Outputs in <root>: history.jsonl (per iteration C, V, constraint maxima, dx), final_tv.npy, final_layout.json (the layout
with the final corner parameters, for the fine-scale evaluation with the exact model), meta.json.
Usage: homog_macro.py <root> <layout.json> <homog_cells.json> [--m 6] [--clamp x,min | --clamp cut] [--pen 1e6] [--load x,max] [--load-dir y]
       [--vfrac 0.8] [--move 0.05] [--maxit 60] [--tmin 0.18] [--tmax 0.69] [--span 0.45] [--grad 0.45]
       homog_macro.py --selftest     (finite-difference check of the adjoint gradient on a small synthetic problem)
"""
import json, sys, time, argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline

CUBE = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)])     # corner index 4x + 2y + z
AX = {'x': 0, 'y': 1, 'z': 2}


# ------------------------------------------------------------------------------------------------ material law
class Material:
    def __init__(self, taus, C11, C12, C44, rho):
        o = np.argsort(taus)
        self.s = {k: CubicSpline(np.asarray(taus)[o], np.asarray(v)[o]) for k, v in
                  dict(C11=C11, C12=C12, C44=C44, rho=rho).items()}
        self.tmin, self.tmax = float(np.min(taus)), float(np.max(taus))

    @classmethod
    def from_json(cls, path):
        d = json.loads(Path(path).read_text())['cells']
        t = [c['tau'] for c in d]
        CH = [np.asarray(c['CH']) for c in d]
        C11 = [np.mean(np.diag(M)[:3]) for M in CH]
        C12 = [np.mean([M[0, 1], M[0, 2], M[1, 2]]) for M in CH]
        C44 = [np.mean(np.diag(M)[3:]) for M in CH]
        return cls(t, C11, C12, C44, [c['rho'] for c in d])

    def tensor(self, tau, der=0):
        """(q, 6, 6) cubic Voigt tensors at thicknesses tau (q,), or their tau-derivatives (der=1)."""
        c11, c12, c44 = (self.s[k](tau, der) for k in ('C11', 'C12', 'C44'))
        D = np.zeros((len(tau), 6, 6))
        for i in range(3):
            for j in range(3):
                D[:, i, j] = c11 if i == j else c12
            D[:, 3 + i, 3 + i] = c44
        return D

    def rho(self, tau, der=0):
        return self.s['rho'](tau, der)


# ------------------------------------------------------------------------------------------------ Q1 hexahedron
def shape_q1(xi):
    """xi (q, 3) in [0,1]^3 -> N (q, 8), dN/dxi (q, 8, 3); node order = CUBE."""
    N = np.ones((len(xi), 8)); dN = np.ones((len(xi), 8, 3))
    for a in range(8):
        for d in range(3):
            f = xi[:, d] if CUBE[a, d] else 1 - xi[:, d]
            g = 1.0 if CUBE[a, d] else -1.0
            N[:, a] *= f
            for e in range(3):
                dN[:, a, e] *= g if e == d else f
    return N, dN


def bmat(dNdx):
    """dNdx (q, 8, 3) -> B (q, 6, 24), Voigt order xx, yy, zz, yz, xz, xy (engineering shear)."""
    q = dNdx.shape[0]
    B = np.zeros((q, 6, 24))
    for a in range(8):
        dx, dy, dz = dNdx[:, a, 0], dNdx[:, a, 1], dNdx[:, a, 2]
        B[:, 0, 3 * a] = dx; B[:, 1, 3 * a + 1] = dy; B[:, 2, 3 * a + 2] = dz
        B[:, 3, 3 * a + 1] = dz; B[:, 3, 3 * a + 2] = dy
        B[:, 4, 3 * a] = dz; B[:, 4, 3 * a + 2] = dx
        B[:, 5, 3 * a] = dy; B[:, 5, 3 * a + 1] = dx
    return B


class Macro:
    """Q1 mesh of the lattice box, finite-cell cut, graded C^H; compliance, adjoint gradient and volume per vertex field."""

    def __init__(self, shape, cells, normal, b, m, clamp, load, load_dir, mat, eps=1e-6, pen=1e6):
        self.shape, self.m, self.mat, self.eps = tuple(shape), m, mat, eps
        nx, ny, nz = shape
        self.h = 1.0 / m
        ne = (nx * m, ny * m, nz * m)
        self.nn = tuple(k + 1 for k in ne)
        nodes = np.stack(np.meshgrid(*[np.arange(k) for k in self.nn], indexing='ij'), -1).reshape(-1, 3)
        self.X = nodes * self.h
        nid = lambda g: (g[..., 0] * self.nn[1] + g[..., 1]) * self.nn[2] + g[..., 2]
        present = {tuple(int(v) for v in c['position']) for c in cells}
        normal = np.asarray(normal, float); self.normal, self.b = normal, float(b)
        # quadrature: 2^3 Gauss or 4^3 midpoint sub-points for cut elements
        g2 = (1 + np.array([-1, 1]) / np.sqrt(3)) / 2
        G2 = np.stack(np.meshgrid(g2, g2, g2, indexing='ij'), -1).reshape(-1, 3); W2 = np.full(8, 1 / 8)
        s4 = (np.arange(4) + 0.5) / 4
        G4 = np.stack(np.meshgrid(s4, s4, s4, indexing='ij'), -1).reshape(-1, 3); W4 = np.full(64, 1 / 64)
        self.rules = {2: (G2, W2) + shape_q1(G2), 4: (G4, W4) + shape_q1(G4)}
        els, rule, phi_list = [], [], []
        for i in range(ne[0]):
            for j in range(ne[1]):
                for k in range(ne[2]):
                    cell = (i // m, j // m, k // m)
                    if cell not in present:
                        continue
                    corner = np.array([i, j, k]) + CUBE
                    xc = corner * self.h
                    s = xc @ normal[:3]
                    if s.max() <= self.b:
                        els.append((i, j, k, cell)); rule.append(2); phi_list.append(None)
                    elif s.min() >= self.b:
                        continue
                    else:
                        pts = (np.array([i, j, k]) + G4) * self.h
                        phi = (pts @ normal[:3] <= self.b).astype(float)
                        if phi.sum() == 0:
                            continue
                        els.append((i, j, k, cell)); rule.append(4); phi_list.append(phi)
        self.els, self.rule, self.phi = els, rule, phi_list
        conn = np.array([nid(np.array(e[:3]) + CUBE) for e in els])
        used = np.unique(conn)
        self.node_map = -np.ones(len(nodes), np.int64); self.node_map[used] = np.arange(len(used))
        self.conn = self.node_map[conn]
        self.nnode = len(used)
        self.dof = (3 * self.conn[:, :, None] + np.arange(3)).reshape(len(els), 24)
        Xu = self.X[used]
        # vertex map of the lattice (same keys as r1x3_common.vertex_map: position + CUBE)
        self.cell_index = {tuple(int(v) for v in c['position']): n for n, c in enumerate(cells)}
        keys, vid = {}, np.zeros((len(cells), 8), np.int64)
        for n, c in enumerate(cells):
            for a in range(8):
                v = tuple(int(p) + int(q) for p, q in zip(c['position'], CUBE[a]))
                keys.setdefault(v, len(keys)); vid[n, a] = keys[v]
        self.vid, self.vkeys, self.nv = vid, np.array(list(keys)), len(keys)
        # supports
        la = AX[load[0]]
        self.Kpen = None
        if clamp[0] == 'cut':
            self.Kpen, self.pen_area, self.pen_missed = self._plane_penalty(pen * mat.tensor(np.array([0.4]))[0, 0, 0] / self.h)
            self.fixed_dofs = np.zeros(0, np.int64)
        else:
            ca = AX[clamp[0]]
            cplane = 0.0 if clamp[1] == 'min' else shape[ca]
            fixed_nodes = np.flatnonzero(np.isclose(Xu[:, ca], cplane))
            self.fixed_dofs = (3 * fixed_nodes[:, None] + np.arange(3)).reshape(-1)
        self.free_dofs = np.setdiff1d(np.arange(3 * self.nnode), self.fixed_dofs)
        # load: uniform traction on the material part of the load face, unit total force in load_dir
        lplane = 0.0 if load[1] == 'min' else shape[la]
        f = np.zeros(3 * self.nnode)
        other = [d for d in range(3) if d != la]
        s4f = (np.arange(4) + 0.5) / 4
        for e, (i, j, k, cell) in enumerate(els):
            idx = np.array([i, j, k])
            if not np.isclose((idx[la] + (1 if load[1] == 'max' else 0)) * self.h, lplane):
                continue
            face_corners = [a for a in range(8) if CUBE[a, la] == (1 if load[1] == 'max' else 0)]
            for u_ in s4f:
                for v_ in s4f:
                    xi = np.zeros(3); xi[la] = 1.0 if load[1] == 'max' else 0.0; xi[other[0]] = u_; xi[other[1]] = v_
                    x = (idx + xi) * self.h
                    if x @ normal[:3] > self.b:
                        continue
                    N, _ = shape_q1(xi[None])
                    for a in face_corners:
                        f[3 * self.conn[e, a] + AX[load_dir]] += N[0, a] / 16 * self.h ** 2
        if not f.sum() > 0:
            raise ValueError('EMPTY_LOAD')
        self.f = f / f.sum()
        self.Xu = Xu

    def _plane_penalty(self, alpha):
        """alpha * int_G N^T N dG (each component) over the plane section G = {n . X = b} of the meshed elements; G is cut
        into one polygon per element (edge intersections, convex), fanned from its centroid, 7-point degree-5 rule.
        Returns the sparse penalty matrix, the section area and the section area in elements not meshed (no material
        sub-point; those parts of G are not clamped)."""
        a1, b1 = 0.0597158717, 0.4701420641; a2, b2 = 0.7974269853, 0.1012865073
        TB = np.array([[1 / 3, 1 / 3], [a1, b1], [b1, a1], [b1, b1], [a2, b2], [b2, a2], [b2, b2]])
        TW = np.array([0.225] + [0.1323941527] * 3 + [0.1259391805] * 3)               # sum 1 (area-normalised)
        edges = [(a, a ^ (1 << d)) for a in range(8) for d in range(3) if not a & (1 << d)]
        n = self.normal[:3] / np.linalg.norm(self.normal[:3]); bb = self.b / np.linalg.norm(self.normal[:3])
        t1 = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0]); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)

        def polygon(i, j, k):
            xc = (np.array([i, j, k]) + CUBE) * self.h
            s = xc @ n - bb
            if s.min() >= 0 or s.max() <= 0:
                return None
            pts = [xc[a] + s[a] / (s[a] - s[c]) * (xc[c] - xc[a]) for a, c in edges if s[a] * s[c] < 0]
            pts += [xc[a] for a in range(8) if s[a] == 0]
            P = np.array(pts); c0 = P.mean(0)
            ang = np.arctan2((P - c0) @ t2, (P - c0) @ t1)
            return P[np.argsort(ang)], c0

        rows, cols, vals, area = [], [], [], 0.0
        for e, (i, j, k, cell) in enumerate(self.els):
            pg = polygon(i, j, k)
            if pg is None:
                continue
            P, c0 = pg
            M = np.zeros((8, 8))
            for q in range(len(P)):
                A_, B_ = P[q], P[(q + 1) % len(P)]
                ar = 0.5 * np.linalg.norm(np.cross(A_ - c0, B_ - c0))
                if ar == 0:
                    continue
                X = c0 + TB[:, :1] * (A_ - c0) + TB[:, 1:] * (B_ - c0)
                N, _ = shape_q1(X / self.h - np.array([i, j, k]))
                M += ar * np.einsum('q,qa,qb->ab', TW, N, N); area += ar * TW.sum()
            for d in range(3):
                dd = self.dof[e][d::3]
                rows.append(np.repeat(dd, 8)); cols.append(np.tile(dd, 8)); vals.append(alpha * M.reshape(-1))
        meshed = {e[:3] for e in self.els}
        missed = 0.0
        nx, ny, nz = (s_ * self.m for s_ in self.shape)
        present = {e[3] for e in self.els}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (i, j, k) in meshed or (i // self.m, j // self.m, k // self.m) not in present:
                        continue
                    pg = polygon(i, j, k)
                    if pg is not None:
                        P, c0 = pg
                        missed += sum(0.5 * np.linalg.norm(np.cross(P[q] - c0, P[(q + 1) % len(P)] - c0)) for q in range(len(P)))
        if not rows:
            raise ValueError('EMPTY_CUT_SECTION')
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * self.nnode,) * 2)
        return K, area, missed

    def _tau_at(self, tv, e, xi):
        i, j, k, cell = self.els[e]
        n = self.cell_index[cell]
        xc = (np.array([i, j, k]) % self.m + xi) / self.m                   # coordinates in the cell, [0,1]^3
        Nc, _ = shape_q1(xc)
        return Nc @ tv[self.vid[n]], Nc, n

    def solve(self, tv, want_grad=True):
        mat = self.mat
        rows, cols, vals = [], [], []
        cache = []
        V, dV = 0.0, np.zeros(self.nv)
        detJ = self.h ** 3
        for e in range(len(self.els)):
            r = self.rule[e]
            G, W, N, dN = self.rules[r]
            w = W * (self.phi[e] if self.phi[e] is not None else 1.0)
            tau, Nc, n = self._tau_at(tv, e, G)
            D = mat.tensor(tau)
            B = bmat(dN / self.h)
            wv = w * detJ
            Dv = D * wv[:, None, None]
            if self.phi[e] is not None:                                     # void part: eps * C^H(0.4)
                Dv = Dv + (self.eps * mat.tensor(np.full(len(tau), 0.4)) * (W * (1 - self.phi[e]) * detJ)[:, None, None])
            Ke = np.einsum('qia,qij,qjb->ab', B, Dv, B)
            rows.append(np.repeat(self.dof[e], 24)); cols.append(np.tile(self.dof[e], 24)); vals.append(Ke.reshape(-1))
            rq = mat.rho(tau)
            V += float((rq * wv).sum())
            np.add.at(dV, self.vid[n], (mat.rho(tau, 1) * wv) @ Nc)
            cache.append((B, tau, Nc, n, wv))
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * self.nnode,) * 2)
        if self.Kpen is not None:                                           # design-independent: the adjoint is unchanged
            K = K + self.Kpen
        fr = self.free_dofs
        u = np.zeros(3 * self.nnode)
        u[fr] = spla.spsolve(K[fr][:, fr].tocsc(), self.f[fr])
        C = float(self.f @ u)
        g = np.zeros(self.nv)
        if want_grad:
            for e, (B, tau, Nc, n, wv) in enumerate(cache):
                eps_q = B @ u[self.dof[e]]                                  # (q, 6)
                dD = self.mat.tensor(tau, 1)
                en = np.einsum('qi,qij,qj->q', eps_q, dD, eps_q) * wv
                np.add.at(g, self.vid[n], -(en @ Nc))
        return C, g, V, dV, u


# ------------------------------------------------------------------------------------------------ optimisation
def constraints(tv, free, V, dV, Vstar, pairs, stencils, span, grad):
    nv = len(tv); col = np.full(nv, -1); col[free] = np.arange(len(free))
    f, G = [V / Vstar - 1.0], [dV[free] / Vstar]
    for va, vb in pairs:
        g = np.zeros(len(free))
        if col[va] >= 0:
            g[col[va]] += 1 / span
        if col[vb] >= 0:
            g[col[vb]] -= 1 / span
        if g.any():
            f.append((tv[va] - tv[vb]) / span - 1.0); G.append(g)
    for vc, vx, vy, vz in stencils:
        dd = np.array([tv[vx] - tv[vc], tv[vy] - tv[vc], tv[vz] - tv[vc]])
        g = np.zeros(len(free))
        for vn, di in zip((vx, vy, vz), dd):
            if col[vn] >= 0:
                g[col[vn]] += 2 * di / grad ** 2
            if col[vc] >= 0:
                g[col[vc]] -= 2 * di / grad ** 2
        if g.any():
            f.append((dd ** 2).sum() / grad ** 2 - 1.0); G.append(g)
    return np.asarray(f), np.asarray(G)


def structure(vid):
    pairs, stencils = set(), set()
    for row in vid:
        for a in range(8):
            for b in range(8):
                if a != b and row[a] != row[b]:
                    pairs.add((int(row[a]), int(row[b])))
            stencils.add((int(row[a]), int(row[a ^ 4]), int(row[a ^ 2]), int(row[a ^ 1])))
    return sorted(pairs), sorted(stencils)


def optimise(A):
    import mma as MMA
    root = Path(A.root); root.mkdir(parents=True, exist_ok=True)
    L = json.loads(Path(A.layout).read_text())
    mat = Material.from_json(A.cells)
    M = Macro(L['shape'], L['cells'], L['normal'], L['b_global'], A.m, (A.clamp + ',').split(',')[:2], A.load.split(','),
              A.load_dir, mat, pen=A.pen)
    tv = np.zeros(M.nv)
    for n, c in enumerate(L['cells']):
        tv[M.vid[n]] = c['tau_corners']
    la = A.load.split(',')
    coord = M.vkeys[:, AX[la[0]]]
    fixed = coord == (coord.max() if la[1] == 'max' else coord.min())
    free = np.flatnonzero(~fixed)
    pairs, stencils = structure(M.vid)
    (root / 'meta.json').write_text(json.dumps(dict(args=vars(A), elements=len(M.els), nodes=M.nnode, vertices=M.nv,
                                                    free=len(free), material_range=[mat.tmin, mat.tmax],
                                                    cut_section=dict(area=M.pen_area, unmeshed_area=M.pen_missed)
                                                    if M.Kpen is not None else None), indent=1))
    hist = open(root / 'history.jsonl', 'a')
    state, Vstar, C0, fh = None, None, None, []
    for k in range(A.maxit):
        t = time.perf_counter()
        C, g, V, dV, _ = M.solve(tv)
        if Vstar is None:
            Vstar, C0 = A.vfrac * V, C
        fval, dfdx = constraints(tv, free, V, dV, Vstar, pairs, stencils, A.span, A.grad)
        xnew, state, info = MMA.mma_update(tv[free], C / C0, g[free] / C0, fval, dfdx, A.tmin, A.tmax, state, move=A.move)
        dx = float(np.abs(xnew - tv[free]).max())
        rec = dict(k=k, C=C, f0=C / C0, V=V, V_rel=V / Vstar, g_max=float(fval.max()), dx=dx, seconds=time.perf_counter() - t,
                   tau_min=float(tv.min()), tau_max=float(tv.max()), tv=tv.tolist(), g=g.tolist())
        hist.write(json.dumps(rec) + '\n'); hist.flush()
        print(json.dumps({k_: v for k_, v in rec.items() if k_ not in ('tv', 'g')}), flush=True)
        fh.append(C / C0)
        tv = tv.copy(); tv[free] = xnew
        small = len(fh) >= 4 and all(abs(fh[-i] - fh[-i - 1]) / abs(fh[-i - 1]) < A.ftol for i in (1, 2, 3))
        if dx < A.xtol or small:
            break
    np.save(root / 'final_tv.npy', tv)
    out = json.loads(json.dumps(L))
    out['name'] = L['name'] + '_homog'
    for n, c in enumerate(out['cells']):
        c['tau_corners'] = [float(x) for x in tv[M.vid[n]]]
    (root / 'final_layout.json').write_text(json.dumps(out, indent=1))


def selftest():
    """Adjoint gradient against central differences on a 2 x 2 x 1 cut lattice with a synthetic smooth material law."""
    t = np.linspace(0.15, 0.72, 12)
    rho = 0.1 + 0.55 * (t - 0.15)
    mat = Material(t, 2.0 * rho ** 1.8, 0.7 * rho ** 1.9, 0.5 * rho ** 1.7, rho)
    cells = [dict(position=[i, j, 0]) for i in range(2) for j in range(2)]
    n = np.array([np.cos(0.5), np.sin(0.5), 0.0]); b = 2.2
    ok = True
    for ld, cl, lo in (('y', ('x', 'min'), ('x', 'max')), ('z', ('x', 'min'), ('x', 'max')), ('x', ('cut', ''), ('y', 'min'))):
        M = Macro((2, 2, 1), cells, n, b, 3, cl, lo, ld, mat)
        rng = np.random.default_rng(1)
        tv = 0.35 + 0.1 * rng.uniform(-1, 1, M.nv)
        C, g, V, dV, _ = M.solve(tv)
        errs, verrs = [], []
        for v in rng.choice(M.nv, 5, replace=False):
            hstep = 1e-6
            tp = tv.copy(); tp[v] += hstep; tm = tv.copy(); tm[v] -= hstep
            Cp, _, Vp, _, _ = M.solve(tp, False); Cm, _, Vm, _, _ = M.solve(tm, False)
            errs.append(abs((Cp - Cm) / (2 * hstep) - g[v]) / np.abs(g).max())
            verrs.append(abs((Vp - Vm) / (2 * hstep) - dV[v]) / np.abs(dV).max())
        e, ev = max(errs), max(verrs)
        ok &= e < 1e-5 and ev < 1e-6
        print(f'clamp {cl[0]}, load {ld}: elements {len(M.els)}, C = {C:.6g}, V = {V:.6g}, adjoint vs FD {e:.1e}, volume gradient {ev:.1e}')
    print('SELFTEST', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('layout'); ap.add_argument('cells')
    ap.add_argument('--m', type=int, default=6)
    ap.add_argument('--clamp', default='x,min'); ap.add_argument('--load', default='x,max'); ap.add_argument('--load-dir', default='y')
    ap.add_argument('--vfrac', type=float, default=0.8); ap.add_argument('--move', type=float, default=0.05)
    ap.add_argument('--maxit', type=int, default=60)
    ap.add_argument('--tmin', type=float, default=0.18); ap.add_argument('--tmax', type=float, default=0.69)
    ap.add_argument('--span', type=float, default=0.45); ap.add_argument('--grad', type=float, default=0.45)
    ap.add_argument('--xtol', type=float, default=1e-3); ap.add_argument('--ftol', type=float, default=1e-4)
    ap.add_argument('--pen', type=float, default=1e6, help='--clamp cut: penalty factor (x C11(0.4) / h)')
    optimise(ap.parse_args())
