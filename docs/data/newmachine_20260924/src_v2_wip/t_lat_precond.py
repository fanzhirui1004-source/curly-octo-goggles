"""CPU tests of lat_multi / lat_precond (tiny, < 1 min): synthetic cells = Q1 linear elasticity on an m^3 element grid of
the unit cell (node grid (m+1)^3 = (2n+1)^3 with n = m / 2, the teacher node-id convention), random element moduli
(contrast), ports = box nodes (+ a few private 'cut-band' interior nodes, + a few box nodes flagged cut = shared), exact
dense Schur complement S = K_PP - K_PI K_II^-1 K_IP as the cell operator (DenseOp), K_PP triplets from the same K.
Checks: gluing identical to lattice3 on its two-cell x configuration; gather / scatter adjointness; assembled operator
symmetric PSD and SPD on the free DOFs; batched == unbatched matvec (one apply per operator object); matmat_sparse == dense;
K_PP >= A (Loewner); coarse spaces contain the global rigid / affine fields and W^T A W = I; every preconditioner
symmetric positive definite; PCG / deflated PCG converge to the same solution and the two-level methods cut the
iterations on an ill-conditioned slender lattice; optional smoke test on the real fixture cell (teacher.Cell K_PP path).
Usage: OPL_DEV=cpu python t_lat_precond.py      (or pytest t_lat_precond.py)
"""
import os
os.environ.setdefault('OPL_DEV', 'cpu')
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lat_multi as LM
import lat_precond as PR

dt = torch.float64
CPU = torch.device('cpu')
SPECS = ['jacobi', 'kpp', 'add:jac:q1r', 'add:jac:pu6', 'add:kpp:q1r', 'bnn:kpp:q1r', 'bnn:kpp:pu12', 'bnn:jac:q1t',
         'bnn:kpp:q1a', 'defl:jac:q1r', 'defl:kpp:q1r']


# ---------------------------------------------------------------------- synthetic cells
def q1_ke(nu=0.3):
    lam = nu / ((1 + nu) * (1 - 2 * nu)); mu = 1 / (2 * (1 + nu))
    D = np.zeros((6, 6)); D[:3, :3] = lam; D[range(3), range(3)] += 2 * mu; D[range(3, 6), range(3, 6)] = mu
    corners = np.asarray(list(np.ndindex(2, 2, 2)), float)
    g = (1 + np.array([-1, 1]) / np.sqrt(3)) / 2
    Ke = np.zeros((24, 24))
    for xi in np.ndindex(2, 2, 2):
        p = g[list(xi)]
        f = np.where(corners == 1, p, 1 - p)                                          # 8 x 3 factors
        s = np.where(corners == 1, 1.0, -1.0)
        dN = np.stack([s[:, d] * np.prod(np.delete(f, d, 1), 1) for d in range(3)], 1)    # 8 x 3
        B = np.zeros((6, 24))
        for a in range(8):
            x, y, z = dN[a]
            B[0, 3 * a] = x; B[1, 3 * a + 1] = y; B[2, 3 * a + 2] = z
            B[3, 3 * a] = y; B[3, 3 * a + 1] = x
            B[4, 3 * a + 1] = z; B[4, 3 * a + 2] = y
            B[5, 3 * a] = z; B[5, 3 * a + 2] = x
        Ke += B.T @ D @ B / 8
    return Ke


class DenseOp:
    def __init__(self, S):
        self.S, self.calls, self.cols = S, 0, 0

    def apply(self, q):
        self.calls += 1; self.cols += q.shape[1]
        return self.S @ q


def synth_cell(m=4, seed=0, contrast=0.0, n_priv=0, n_box_cut=0, case=None):
    """(CellGeom, DenseOp, K scipy, port DOFs) of a Q1 elasticity cell."""
    rng = np.random.default_rng(seed)
    M = m + 1
    Ke = q1_ke() / m                                                                  # element size h = 1/m: K_e = h Ke
    E = 10 ** (-contrast * rng.random(m ** 3))
    corners = np.asarray(list(np.ndindex(2, 2, 2)))
    el = np.asarray(list(np.ndindex(m, m, m)))
    nd = el[:, None, :] + corners[None]                                               # E x 8 x 3
    nid = (nd[..., 0] * M + nd[..., 1]) * M + nd[..., 2]
    dofs = (3 * nid[:, :, None] + np.arange(3)).reshape(len(el), 24)
    rows = np.repeat(dofs, 24, 1).reshape(-1); cols = np.tile(dofs, (1, 24)).reshape(-1)
    vals = (E[:, None, None] * Ke[None]).reshape(-1)
    K = sp.csr_matrix((vals, (rows, cols)), shape=(3 * M ** 3,) * 2)
    g = np.stack(np.unravel_index(np.arange(M ** 3), (M,) * 3), 1)
    box = ((g == 0) | (g == m)).any(1)
    cut = np.zeros(M ** 3, bool)
    inner = np.flatnonzero(~box)
    cut[rng.choice(inner, n_priv, replace=False)] = True
    cut[rng.choice(np.flatnonzero(box), n_box_cut, replace=False)] = True
    onport = box | cut
    pn = np.flatnonzero(onport)
    P = (3 * pn[:, None] + np.arange(3)).reshape(-1)
    I = np.setdiff1d(np.arange(3 * M ** 3), P)
    Kd = K.toarray()
    S = Kd[np.ix_(P, P)] - Kd[np.ix_(P, I)] @ np.linalg.solve(Kd[np.ix_(I, I)], Kd[np.ix_(I, P)])
    S = (S + S.T) / 2
    KPP = sp.triu(K[P][:, P]).tocoo()

    def kpp():
        return (torch.as_tensor(KPP.row, dtype=torch.long), torch.as_tensor(KPP.col, dtype=torch.long),
                torch.as_tensor(KPP.data, dtype=dt))

    G = LM.CellGeom(case or f'syn{seed}', m // 2, pn, box[onport], cut[onport], kpp_fn=kpp)
    return G, DenseOp(torch.as_tensor(S, dtype=dt)), Kd[np.ix_(P, P)]


def dense_A(lat, ops):
    return lat.matvec(ops, torch.eye(lat.nfree, dtype=dt))


def lattice(layout_ops, **kw):
    lay = {p: g for p, (g, _) in layout_ops.items()}
    kw.setdefault('loads', 'uniform'); kw.setdefault('device', CPU); kw.setdefault('log', lambda s_: None)
    return LM.MultiLattice(lay, **kw), [o for (_, o) in layout_ops.values()]


# ---------------------------------------------------------------------- tests
def test_glue_matches_lattice3():
    """The two-cell x configuration: same free DOFs and numbering as lattice3.Lattice (uniform loads: same supports)."""
    import lattice3 as LT
    G0, _, _ = synth_cell(4, 0, n_box_cut=5)
    G1, _, _ = synth_cell(4, 1, n_priv=4, n_box_cut=5)

    def fake(G, label, off):
        C = types.SimpleNamespace(port_node_ids=G.port_node_ids, port_is_cut=G.priv | np.zeros(len(G.priv), bool),
                                  port_is_box=~G.priv, is_cut=np.zeros(1, bool), n=G.n)
        return dict(label=label, cell=C, offset=off, T=None)
    os.environ.pop('LAT_LOADS', None)
    l3 = LT.Lattice([fake(G0, 'test', (0, 0, 0)), fake(G1, 'nbr', (-1, 0, 0))], 'x', n=G0.n, log=lambda s_: None)
    lat = LM.MultiLattice({(0, 0, 0): G0, (-1, 0, 0): G1}, clamp=('x', 'min'), load=('y', 'min'), loads='uniform',
                          n_random=0, device=CPU)
    assert lat.nfree == len(l3.free) and lat.N == l3.N
    for a, b in zip(lat.gather_idx, l3.gather_idx):
        assert torch.equal(a, b.cpu())
    F3 = l3.F.cpu().numpy() != 0; F = lat.F.numpy() != 0
    for d in range(3):
        assert np.array_equal(F[:, d], F3[:, d] | F3[:, 3 + d])
    assert lat.priv.sum() == 12 and (lat.mult.numpy()[lat.priv] == 1).all()
    return dict(free=lat.nfree, glued=lat.N)


def _mixed(shape=(3, 2, 1), m=4, contrast=1.0, seeds=(0, 1)):
    """Two distinct synthetic cell kinds in a checkerboard (shared operator objects per kind)."""
    kinds = [synth_cell(m, s, contrast, n_priv=2, n_box_cut=3) for s in seeds]
    lo = {}
    for p in np.ndindex(*shape):
        G, op, _ = kinds[sum(p) % len(kinds)]
        lo[p] = (G, op)
    return lo, kinds


def test_adjoint_symmetry_batching():
    lo, kinds = _mixed()
    lat, ops = lattice(lo, n_random=2)
    gen = torch.Generator().manual_seed(0)
    U = torch.randn(lat.nfree, 3, generator=gen, dtype=dt)
    err = 0.0
    for i, G in enumerate(lat.geoms):
        q = torch.randn(G.nport, 3, generator=gen, dtype=dt)
        Y = torch.zeros_like(U); lat.scatter_add(Y, q, i)
        a, b = (lat.gather(U, i) * q).sum(), (U * Y).sum()
        err = max(err, float(abs(a - b) / abs(a)))
    assert err < 1e-13
    for _, op, _ in kinds:
        op.calls = 0
    Yb = lat.matvec(ops, U, batch=True)
    calls_b = sum(op.calls for _, op, _ in kinds)
    Yu = lat.matvec(ops, U, batch=False)
    Yc = lat.matvec(ops, U, batch=True, max_cols=4)                                   # chunked columns
    bat = float((Yb - Yu).norm() / Yu.norm()); chk = float((Yc - Yu).norm() / Yu.norm())
    assert calls_b == len(kinds) and bat < 1e-14 and chk < 1e-14
    A = dense_A(lat, ops)
    sym = float((A - A.T).norm() / A.norm())
    ev = torch.linalg.eigvalsh((A + A.T) / 2)
    assert sym < 1e-13 and float(ev.min()) > 0
    Z, _ = PR.coarse_basis(lat, 'q1r')
    AZs = lat.matmat_sparse(ops, Z)
    mm = float((AZs - A @ Z).norm() / (A @ Z).norm())
    assert mm < 1e-13
    K = lat.kpp_dense()
    kmin = float(torch.linalg.eigvalsh(K - A).min() / torch.linalg.eigvalsh(K).max())
    assert kmin > -1e-12                                                               # K_PP >= A
    # independent K_PP assembly from the synthetic dense blocks
    Kd = torch.zeros_like(K)
    for i, (p, (G, _)) in enumerate(lo.items()):
        kd = [k for k in kinds if k[0] is G][0][2]
        f = lat.gather_idx[i]; kp = torch.nonzero(f >= 0).squeeze(1)
        Kd[f[kp][:, None], f[kp][None, :]] += torch.as_tensor(kd)[kp][:, kp]
    assert float((Kd - K).norm() / K.norm()) < 1e-14
    return dict(free=lat.nfree, adjoint=err, batched_vs_unbatched=bat, chunked=chk, apply_calls_batched=calls_b,
                cells=len(ops), A_sym=sym, A_min_eig=float(ev.min()), A_cond=float(ev.max() / ev.min()),
                matmat_sparse=mm, kpp_minus_A_min_eig_rel=kmin)


def test_coarse_content():
    lo, _ = _mixed((2, 2, 2))
    lat, ops = lattice(lo, n_random=0)
    x, c = lat.xyz, lat.comp
    fields = PR.mode_values(x - 0.3, c, 'a')                                           # global rigid + affine fields
    out = {}
    for kind, ncheck in (('q1t', 12), ('q1r', 12), ('q1a', 12), ('pu6', 6), ('pu12', 12)):
        Z, desc = PR.coarse_basis(lat, kind)
        Qz, _ = torch.linalg.qr(Z)
        r = fields[:, :ncheck] - Qz @ (Qz.T @ fields[:, :ncheck])
        out[kind] = dict(cols=desc['columns'], resid=float(r.norm() / fields[:, :ncheck].norm()))
        assert out[kind]['resid'] < 1e-10, (kind, out[kind])
    co = PR.Coarse(lat, ops, 'q1r')
    I = co.W.T @ co.AW
    out['WtAW_minus_I'] = float((I - torch.eye(I.shape[0], dtype=dt)).norm() / I.shape[0] ** .5)
    assert out['WtAW_minus_I'] < 1e-8
    return out


def test_preconditioners_spd():
    lo, _ = _mixed((2, 2, 1), contrast=1.0)
    lat, ops = lattice(lo, n_random=0)
    fac = PR.Factory(lat, ops)
    A = dense_A(lat, ops)
    I = torch.eye(lat.nfree, dtype=dt)
    out = {}
    for spec in SPECS:
        pc, st, _ = fac.build(spec)
        if isinstance(pc, PR.Deflated):
            # deflated operator: (I - QA) B restricted to the A-orthogonal complement is self-adjoint in A: check M A is
            # A-symmetric on range(I - QA)
            Pm = pc.coarse.proj_T(I)
            MA = pc(A @ Pm)
            S = Pm.T @ A @ MA
            out[spec] = dict(A_sym=float((S - S.T).norm() / S.norm()))
            assert out[spec]['A_sym'] < 1e-9
            continue
        M = pc(I)
        sym = float((M - M.T).norm() / M.norm())
        ev = torch.linalg.eigvalsh((M + M.T) / 2)
        out[spec] = dict(sym=sym, min_eig=float(ev.min()), max_eig=float(ev.max()),
                         cond_MA=float(_cond_MA(M, A)))
        assert sym < 1e-10 and float(ev.min()) > 0, (spec, out[spec])
    return out


def _cond_MA(M, A):
    L = torch.linalg.cholesky((A + A.T) / 2)
    ev = torch.linalg.eigvalsh(L.T @ ((M + M.T) / 2) @ L)
    return ev.max() / ev.min()


def test_pcg_slender():
    """Ill-conditioned: a slender 8 x 1 x 1 beam of heterogeneous cells (moduli over 2 decades), clamped at x = min,
    traction on x = max, + 3 random loads."""
    kinds = [synth_cell(4, s, contrast=2.0, n_priv=2, n_box_cut=2) for s in (3, 4, 5)]
    lo = {(i, 0, 0): kinds[i % 3][:2] for i in range(8)}
    lat, ops = lattice(lo, n_random=3)
    fac = PR.Factory(lat, ops)
    A = dense_A(lat, ops)
    X0 = torch.linalg.solve(A, lat.F)
    c0 = (lat.F * X0).sum(0)
    out = dict(free=lat.nfree, A_cond=float(_cond_MA(torch.eye(lat.nfree, dtype=dt), A)), runs={})
    for spec in SPECS:
        pc, st, desc = fac.build(spec)
        r = PR.solve(lat, ops, pc, tol=1e-8, maxit=3000)
        tr = PR.true_residual(lat, ops, r['X'])
        ce = float(((lat.F * r['X']).sum(0) / c0 - 1).abs().max())
        out['runs'][spec] = dict(it=r['iterations'], true_res=tr, compl_err=ce, coarse=desc.get('coarse', {}).get('columns'))
        assert tr < 1e-7 and ce < 1e-7, (spec, out['runs'][spec])
    it = {k: v['it'] for k, v in out['runs'].items()}
    assert it['bnn:kpp:q1r'] < it['kpp'] < it['jacobi']
    assert it['add:jac:q1r'] < it['jacobi'] and it['defl:jac:q1r'] < it['jacobi']
    return out


def test_real_fixture_cell():
    """Real CutFEM cell (fixture fresh_train_0010_cover01_r1): K_PP extraction from teacher.Cell, a one-cell lattice
    with a CPU exact operator (scipy LU of K_II), jacobi vs two-level iterations (real weak / ghost-penalty nodes)."""
    try:
        import fixture as FX
        geo = FX.small()
    except Exception as e:                                                             # fixture not present: skip
        return dict(skipped=repr(e)[:120])
    C = geo.C
    G = LM.from_teacher(C)
    K = FX.scipy_K(C).tocsr()
    P, I = C.P.numpy(), C.I.numpy()
    r, c, v = G.kpp()
    Kpp = K[P][:, P]
    Ku = sp.triu(Kpp).tocoo()
    Kt = sp.coo_matrix((v.numpy(), (r.numpy(), c.numpy())), shape=Kpp.shape).tocsr()
    kpp_err = float(abs(Kt - Ku.tocsr()).max() / abs(Ku.data).max())               # (stored pattern may hold zeros)
    assert kpp_err < 1e-14

    class LUOp:
        def __init__(self):
            self.lu = spl.splu(K[I][:, I].tocsc()); self.KPI = K[P][:, I]; self.KIP = K[I][:, P]; self.Kpp = Kpp

        def apply(self, q):
            qn = q.numpy()
            return torch.as_tensor(self.Kpp @ qn - self.KPI @ self.lu.solve(np.asarray(self.KIP @ qn)), dtype=dt)
    op = LUOp()
    # this cut cell has shared box nodes only on x = 0 and y = 0 (it cannot be glued to itself): one cell, clamped on
    # x = 0, loaded on y = 0
    lat = LM.MultiLattice({(0, 0, 0): G}, clamp=('x', 'min'), load=('y', 'min'), loads='uniform', n_random=1,
                          device=CPU, log=lambda s_: None)
    ops = [op]
    fac = PR.Factory(lat, ops)
    out = dict(free=lat.nfree, kpp_err=kpp_err, runs={})
    for spec in ('jacobi', 'kpp', 'add:jac:q1r', 'bnn:kpp:q1r', 'bnn:kpp:pu12', 'defl:jac:q1r'):
        t = time.perf_counter()
        pc, st, desc = fac.build(spec)
        r = PR.solve(lat, ops, pc, tol=1e-8, maxit=1500, max_seconds=4)
        out['runs'][spec] = dict(it=r['iterations'], res=r['residual'], true_res=PR.true_residual(lat, ops, r['X']),
                                 s=round(time.perf_counter() - t, 2))
    return out


def test_harness_smoke():
    """bench_precond.run_opset on a synthetic lattice: exact = the Schur complements, 'learned' = a perturbed SPD copy
    (upper bound: S + 1% of K_PP); rows complete, exact rows compared with the final reference."""
    import json as _json
    import tempfile
    import bench_precond as BP
    BP.BD.sync = lambda: None                                                         # no CUDA here
    kinds = [synth_cell(4, s, contrast=1.0, n_priv=2, n_box_cut=2) for s in (7, 8)]
    lo = {p: kinds[sum(p) % 2][:2] for p in np.ndindex(3, 2, 1)}
    lat, _ = lattice(lo, n_random=2)
    ex = {G.case: op for G, op, _ in kinds}
    le = {G.case: DenseOp(op.S + 0.01 * torch.as_tensor(kd)) for G, op, kd in kinds}
    with tempfile.TemporaryDirectory() as td:
        args = types.SimpleNamespace(precs='jacobi,kpp,add:jac:q1r,bnn:kpp:q1r,defl:jac:q1r', tol=1e-8, maxit=3000,
                                     max_seconds=None, kpp_backend='auto', out=str(Path(td) / 'o.json'))
        rec = dict(spec='synthetic', runs=[]); out = dict(lattices=[rec]); shared, ref = {}, {}
        BP.run_opset(lat, 'exact', ex, args, shared, ref, rec, out, log=lambda d: None)
        BP.run_opset(lat, 'learned', le, args, shared, ref, rec, out, log=lambda d: None)
        _json.loads(Path(args.out).read_text())
    bad = [r for r in rec['runs'] if 'error' in r or not r['converged']]
    assert not bad, bad
    ex_err = max(r['compliance_rel_err_max'] for r in rec['runs'] if r['operators'] == 'exact')
    le_err = [r['compliance_rel_err_max'] for r in rec['runs'] if r['operators'] == 'learned']
    assert ex_err < 1e-7 and min(le_err) > 1e-4                                       # the perturbation is visible
    return {f"{r['operators']}/{r['preconditioner']}": dict(it=r['iterations'], setup_s=round(r['setup_s'], 4),
                                                            apply_s=round(r['apply_s'], 5), cerr=r['compliance_rel_err_max'])
            for r in rec['runs']}


def main():
    t0 = time.perf_counter()
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith('test_')]:
        t = time.perf_counter()
        res = fn()
        print(f'PASS {name} ({time.perf_counter() - t:.1f}s)')
        import json
        print('  ' + json.dumps(res, default=float)[:3000])
    print(f'ALL PASS in {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
