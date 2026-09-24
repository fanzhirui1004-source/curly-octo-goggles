"""Tests of cert.py (D1 residual certificate).
Local (CPU, scipy stand-ins for every factorization, in the test only):
  (1) random SPD toys: Delta_lb <= Delta = r^T K_II^-1 r, monotone in m, exact when W contains K_II^-1 r (m = 0 on a
      block-diagonal K_II, m = ni - 1, galerkin with K_II^-1 r in W), krylov == cg, mu / eps bounds, column blocks, and the
      lattice bound sum_c Delta_lb,c <= C - C_hat on a two-cell toy lattice with a random linear extension;
  (2) fixture small geometry, every val class: u* + delta (delta_P = 0: smooth, soft = K_II^-1 smooth load, rough, on weak
      nodes) and the zero-shot network fields (FIX/mgno2_r2_d2_best.pt; FIX/s2v1_snap_10000.pt reported too);
      Delta_true = r_I^T K_II^-1 r_I (splu of K_II) cross-checked with e^T K e and u^T K u - q^T S q (q^T S q from the exact
      port reactions; the '- 1' of the unit-energy banks is off by the fp32 rounding of q); assert
      Delta_lb <= Delta_true (1 + 1e-9) for m = 0, 4, 8, 16 (krylov and cg); efficiency Delta_lb / Delta_true and timing;
  (3) r_I = K_II e_I numerically;
  (4) one-cell lattice on the fixture (face x = 0 clamped, s2v1 network, PCG with the exact preconditioner):
      C >= 2 f^T U - U^T K_hat U + D exactly, D <= C - C_hat, and C - C_hat = sum Delta + |U* - U_hat|_S^2.
  Usage: OPL_DEV=cpu python3 t_cert.py [toy|fixture|all] [out.json]
Remote (GPU; not run locally): exact extension by the teacher's interior factor on slot-cache geometries and the lattice
  bound on lattice3 with the learned test cell (see remote()). Pass criteria:
  R1  Delta_lb <= Delta_true (1 + 1e-9) for every column, class and m (Delta_true = e^T K e, e = u - C.extend(u_P));
  R2a lattice, exact for any U: C_ref >= 2 f^T U - U^T K_hat U + D - 1e-8 C_ref, evaluated at U = fp32(U_hat) so that the
      learned cell's energy is u^T K u of its own field and the exact neighbour's q^T S q is the dense teacher's
      (1e-8: accuracy of the dense-Cholesky reference);
  R2b D <= (C_ref - C_hat) + 1e-6 C_ref per load (fp32 variational readout inside the PCG, PCG residual <= 1e-10);
  R3  timing: certify(m = 8) on 16 columns < 0.25 s per 1e5 DOFs (the review's 'a few ms per cell and query').
  Usage: python3 t_cert.py remote <out.json> <checkpoint.pt> <slot_dir> <body> <case> [<case> ...]
         (env LAT_NBR: the lattice neighbour, default fresh_train_0020_full; LAT_SKIP=1 skips the lattice part)
"""
import os
import sys
import json
import time
import numpy as np
import torch

if len(sys.argv) < 2 or sys.argv[1] != 'remote':
    os.environ.setdefault('OPL_DEV', 'cpu')
import cert as CE

f64 = torch.float64


def stats(x):
    x = np.asarray(x, float)
    return dict(min=float(x.min()), med=float(np.median(x)), max=float(x.max()))


# ------------------------------------------------------------------------------------------------------------ (1) toys
class ToyCell:
    """Dense SPD K with the teacher.Cell interface used by cert (C @ x, nb, upper CSR ru / cu / vals / crow)."""

    def __init__(self, K):
        import scipy.sparse as sp
        self.Kd = torch.as_tensor(K); self.nb = K.shape[0]
        U = sp.triu(sp.csr_matrix(K)).tocsr()
        self.crow = torch.as_tensor(U.indptr, dtype=torch.long)
        self.cu = torch.as_tensor(U.indices, dtype=torch.int32); self.vals = torch.as_tensor(U.data, dtype=f64)
        self.ru = torch.repeat_interleave(torch.arange(self.nb, dtype=torch.int32), self.crow[1:] - self.crow[:-1])

    def __matmul__(self, x):
        return self.Kd @ x


def toy_K(N, ports, rng, blockdiag=False):
    """High-contrast element-assembled SPD matrix on N nodes (3 DOFs each); blockdiag: no element couples two interior
    nodes (K_II block diagonal)."""
    K = np.zeros((3 * N, 3 * N))
    inner = np.setdiff1d(np.arange(N), ports)
    for e in range(4 * N):
        if blockdiag:
            nodes = np.concatenate([rng.choice(ports, 3, replace=False), rng.choice(inner, 1)])
        else:
            nodes = rng.choice(N, 4, replace=False)
        d = (3 * nodes[:, None] + np.arange(3)).ravel()
        Bm = rng.standard_normal((6, 12)) * 10 ** rng.uniform(-3, 0)
        K[np.ix_(d, d)] += Bm.T @ Bm
    weak = rng.choice(inner, max(1, len(inner) // 8), replace=False)             # fictitious-fringe-like soft nodes
    for j in weak:
        d = 3 * j + np.arange(3)
        K[d, :] *= 1e-3; K[:, d] *= 1e-3
    return K + 1e-6 * np.trace(K) / len(K) * np.eye(3 * N)


def toy_split(N, ports):
    pm = np.zeros(N, bool); pm[ports] = True
    pm = np.repeat(pm, 3)
    return torch.as_tensor(np.flatnonzero(pm)), torch.as_tensor(np.flatnonzero(~pm))


def test_toys(rec):
    rng = np.random.default_rng(0)
    out = rec.setdefault('toy', {})
    N = 48
    for trial in range(3):
        ports = np.sort(rng.choice(N, 14, replace=False))
        K = toy_K(N, ports, rng)
        C = ToyCell(K); P, I = toy_split(N, ports)
        cert = CE.Cert(C, P, I)
        Kt = torch.as_tensor(K)
        KII, KIP = Kt[I][:, I], Kt[I][:, P]
        S = Kt[P][:, P] - KIP.T @ torch.linalg.solve(KII, KIP)
        B = 8
        u = torch.randn((C.nb, B), dtype=f64, generator=torch.Generator().manual_seed(trial))
        q = u[P]
        # (3) r_I = K_II e_I and the identity Delta = u^T K u - q^T S q = r^T K_II^-1 r
        us = torch.zeros_like(u); us[P] = q; us[I] = -torch.linalg.solve(KII, KIP @ q)
        e = u - us
        r = cert.residual(u)
        assert float(e[P].abs().max()) == 0.0
        rel_r = float(((r - KII @ e[I]).norm(dim=0) / r.norm(dim=0)).max())
        Dex = (r * torch.linalg.solve(KII, r)).sum(0)
        Dq = (u * (Kt @ u)).sum(0) - (q * (S @ q)).sum(0)
        rel_id = float(((Dq - Dex).abs() / Dex).max())
        assert rel_r < 1e-12 and rel_id < 1e-8, (rel_r, rel_id)
        # node blocks: exact, and the diag3 path gives the same certificate
        Dn = CE.node_blocks(C)
        blk = torch.stack([Kt[3 * j:3 * j + 3, 3 * j:3 * j + 3] for j in range(N)])
        assert float((Dn - blk).abs().max()) == 0.0
        cert3 = CE.Cert(C, P, I, diag3=Dn.numpy())
        # bounds: <= exact, monotone in m, exact at m = ni - 1
        ms = [0, 1, 2, 4, 8, 16, 32, len(I) - 1]
        lb = {m: cert.lower(u, m) for m in ms}
        for m in ms:
            assert bool((lb[m] <= Dex * (1 + 1e-10)).all()), (m, lb[m] / Dex)
        for a, b in zip(ms[:-1], ms[1:]):
            assert bool((lb[b] >= lb[a] * (1 - 1e-10)).all()), (a, b)
        exact_full = float(((lb[ms[-1]] - Dex).abs() / Dex).max())
        assert exact_full < 1e-8, exact_full
        assert float((cert3.lower(u, 4) - lb[4]).abs().max() / lb[4].min()) < 1e-12
        # krylov == cg (same Krylov space), column blocks
        kc = max(float(((cert.lower(u, m, method='cg') - lb[m]).abs() / lb[m]).max()) for m in (1, 2, 4, 8))
        assert kc < 1e-6, kc
        for m in (0, 4):
            assert float(((cert.lower(u, m, bs=3) - lb[m]).abs() / lb[m]).max()) < 1e-12
        # Galerkin over an arbitrary basis containing K_II^-1 r: exact; without it: a bound
        W = torch.cat([torch.linalg.solve(KII, r), torch.randn((len(I), 5), dtype=f64, generator=torch.Generator().manual_seed(9))], 1)
        gx = float(((cert.galerkin(r, W) - Dex).abs() / Dex).max())
        assert gx < 1e-8, gx
        assert bool((cert.galerkin(r, W[:, B:]) <= Dex * (1 + 1e-10)).all())
        # mu / eps
        eh = (u * (Kt @ u)).sum(0); es = (q * (S @ q)).sum(0)
        mu_lb, eps_lb = cert.mu_lower(u, m=8)
        assert bool((mu_lb <= eh / es * (1 + 1e-10)).all() and (eps_lb <= (eh - es) / es * (1 + 1e-10)).all())
        # eigenvalue floor path (any SPD block preconditioner keeps the bound rigorous) and degenerate columns
        cf = CE.Cert(C, P, I, floor=0.5)
        assert cf.floored > 0
        for m in (0, 4, 16, len(I) - 1):
            assert bool((cf.lower(u, m) <= Dex * (1 + 1e-10)).all())
        assert float(((cf.lower(u, len(I) - 1) - Dex).abs() / Dex).max()) < 1e-8
        uz = torch.cat([torch.zeros((C.nb, 1), dtype=f64), us[:, :1], u[:, :1]], 1)   # zero, exact extension, generic
        cz = cert.certify(uz, m=8)
        assert bool(torch.isfinite(cz['mu_lb']).all()) and float(cz['delta_lb'][0]) == 0.0 and float(cz['mu_lb'][0]) == 1.0
        assert float(cz['eps_lb'][1]) < 1e-20 and float(cz['delta_lb'][2]) > 0
        for meth in ('krylov', 'cg'):
            assert bool(torch.isfinite(cert.lower(uz, 8, method=meth)).all())
        eff = {m: stats((lb[m] / Dex).numpy()) for m in ms}
        out[f'trial{trial}'] = dict(ni=len(I), floored=cert.floored, rel_rI_KIIeI=rel_r, rel_identity=rel_id,
                                    exact_at_full_krylov=exact_full, krylov_vs_cg=kc, galerkin_exact=gx,
                                    efficiency={str(m): eff[m] for m in ms})
        print(json.dumps(dict(toy=trial, **{k: v for k, v in out[f'trial{trial}'].items() if k != 'efficiency'},
                              eff_med={m: round(eff[m]['med'], 4) for m in ms})), flush=True)
    # block-diagonal K_II: the Jacobi vector is K_II^-1 r, m = 0 is exact
    ports = np.sort(rng.choice(N, 20, replace=False))
    K = toy_K(N, ports, rng, blockdiag=True)
    C = ToyCell(K); P, I = toy_split(N, ports)
    cert = CE.Cert(C, P, I)
    Kt = torch.as_tensor(K); KII = Kt[I][:, I]
    off = KII.clone()
    for j in range(len(I) // 3):
        off[3 * j:3 * j + 3, 3 * j:3 * j + 3] = 0
    assert float(off.abs().max()) == 0.0
    u = torch.randn((C.nb, 8), dtype=f64, generator=torch.Generator().manual_seed(5))
    r = cert.residual(u)
    Dex = (r * torch.linalg.solve(KII, r)).sum(0)
    bd = float(((cert.lower(u, 0) - Dex).abs() / Dex).max())
    assert bd < 1e-12, bd
    out['blockdiag_m0_exact'] = bd
    print(json.dumps(dict(toy='blockdiag', m0_rel_err=bd)), flush=True)
    test_toy_lattice(out, rng)


def test_toy_lattice(out, rng):
    """Two cells glued on shared port DOFs, a random linear extension with exact port values (variational readout):
    C - C_hat >= sum_c Delta_c(q_hat_c) >= lattice_lower."""
    N = 40
    cells = []
    for c in range(2):
        ports = np.sort(rng.choice(N, 16, replace=False))
        K = toy_K(N, ports, rng)
        Cc = ToyCell(K); P, I = toy_split(N, ports)
        Kt = torch.as_tensor(K)
        KII, KIP = Kt[I][:, I], Kt[I][:, P]
        S = Kt[P][:, P] - KIP.T @ torch.linalg.solve(KII, KIP)
        X = -torch.linalg.solve(KII, KIP) + 0.3 * torch.randn(KIP.shape, dtype=f64, generator=torch.Generator().manual_seed(c)) \
            * KIP.abs().max() / KII.diagonal().max()                             # perturbed extension on the interior
        E = torch.zeros((3 * N, len(P)), dtype=f64); E[P] = torch.eye(len(P), dtype=f64); E[I] = X
        cells.append(dict(cert=CE.Cert(Cc, P, I), K=Kt, S=S, Sh=E.T @ Kt @ E, E=E, np=len(P)))
    # lattice DOFs: cell 0 ports -> 0..np0-1; cell 1: its first 12 port DOFs glued to cell 0's last 12, the rest private
    n0, n1 = cells[0]['np'], cells[1]['np']
    g1 = np.concatenate([np.arange(n0 - 12, n0), n0 + np.arange(n1 - 12)])
    maps = [np.arange(n0), g1]
    Nl = n0 + n1 - 12
    A = []
    for mp, cd in zip(maps, cells):
        a = torch.zeros((cd['np'], Nl), dtype=f64); a[np.arange(cd['np']), mp] = 1; A.append(a)
    KL = sum(a.T @ cd['S'] @ a for a, cd in zip(A, cells))
    KH = sum(a.T @ cd['Sh'] @ a for a, cd in zip(A, cells))
    f = torch.randn((Nl, 6), dtype=f64, generator=torch.Generator().manual_seed(11))
    Cref = (f * torch.linalg.solve(KL, f)).sum(0)
    Uh = torch.linalg.solve(KH, f)
    Chat = (f * Uh).sum(0)
    pairs = [(cd['cert'], cd['E'] @ (a @ Uh)) for a, cd in zip(A, cells)]
    Dtrue = sum(((a @ Uh) * ((cd['Sh'] - cd['S']) @ (a @ Uh))).sum(0) for a, cd in zip(A, cells))
    assert bool((Chat <= Cref).all()) and bool((Cref - Chat >= Dtrue * (1 - 1e-10)).all())
    res = {}
    for m in (0, 4, 8):
        D, rel = CE.lattice_lower(pairs, m=m, c_hat=Chat)
        assert bool((D <= (Cref - Chat) * (1 + 1e-10)).all()) and bool((rel <= (Cref - Chat) / Cref * (1 + 1e-10)).all())
        res[m] = stats((D / (Cref - Chat)).numpy())
    out['lattice'] = dict(gap_over_Dtrue=stats(((Cref - Chat) / Dtrue).numpy()), efficiency_vs_gap={str(k): v for k, v in res.items()})
    print(json.dumps(dict(toy='lattice', **out['lattice'])), flush=True)


# ------------------------------------------------------------------------------------------------------- (2), (3) fixture
def test_fixture(rec, nets=('mgno2_r2_d2_best.pt', 's2v1_snap_10000.pt')):
    import scipy.sparse.linalg as sla
    import fixture as FX
    geo = FX.small()
    C, nd = geo.C, geo.nd
    K = FX.scipy_K(C)
    Pn, In = geo.P.numpy(), geo.I.numpy()
    t = time.perf_counter()
    lu = sla.splu(K[In][:, In].tocsc())
    out = rec.setdefault('fixture', dict(case=geo.case, dofs=int(C.nb), ports=int(len(Pn)), interior=int(len(In)),
                                         splu_seconds=time.perf_counter() - t))
    t = time.perf_counter(); cert = CE.from_geo(geo); out['cert_setup_seconds'] = time.perf_counter() - t
    d3 = np.asarray(nd['diag3'])[cert.inode.numpy()]
    out['diag3_vs_C_blocks_maxrel'] = float(np.abs(cert.D.numpy() - d3).max() / np.abs(d3).max())
    out['floored'] = cert.floored
    inode = cert.inode.numpy()
    X = nd['grid'].astype(float) / 64.0
    Xi = torch.as_tensor(X[inode])
    dist = torch.cdist(Xi, torch.as_tensor(X[nd['is_port'].astype(bool)])).min(1).values
    phi = (dist / (4 / 64)).clamp(max=1) ** 2                                      # smooth ramp from 0 on the ports
    weak_i = np.asarray(nd['weak']).astype(bool)[inode]
    out['interior_weak_nodes'] = int(weak_i.sum())
    gen = torch.Generator().manual_seed(0)

    def waves(Y, B, kmax):
        k = torch.randn((B, 3, 8, 3), dtype=f64, generator=gen) * kmax
        ph = 2 * np.pi * torch.rand((B, 3, 8), dtype=f64, generator=gen)
        a = torch.randn((B, 3, 8), dtype=f64, generator=gen)
        v = torch.cos(2 * np.pi * torch.einsum('nd,bcwd->nbcw', Y, k) + ph[None]) * a[None]
        return v.sum(-1).permute(0, 2, 1).reshape(-1, B)                         # (ni, B) node-major xyz

    def Kmul(x):
        return torch.as_tensor(K @ x.numpy())

    def kii_inv(r):
        return torch.as_tensor(lu.solve(r.numpy()))

    ms = (0, 4, 8, 16)
    targets = torch.tensor([1e-3, 1e-2, 1e-1, 3e-1] * 4, dtype=f64)
    worst, eff, tim, checks, krcg = [], {}, {}, {}, []
    nets_ok = [n for n in nets if (FX.FIX / n).exists()]
    models = {n: FX.model_for(geo, n) for n in nets_ok}
    for cls in geo.classes:
        q = geo.banks['val'][cls].to(f64)                                           # fp32 bank, exact in fp64
        us = torch.as_tensor(np.asarray(geo.fix_ustar_val[cls]), dtype=f64)
        react = torch.as_tensor(np.asarray(geo.fix_react_val[cls]), dtype=f64)
        B = q.shape[1]
        Kus = Kmul(us)
        es = (q * react).sum(0)
        checks[cls] = dict(ustar_ports_maxabs=float((us[Pn] - q).abs().max()),
                           ustar_interior_residual_rel=float((Kus[In].norm(dim=0) / Kus[Pn].norm(dim=0)).max()),
                           react_vs_Kustar_rel=float(((Kus[Pn] - react).norm(dim=0) / react.norm(dim=0)).max()),
                           qSq_minus_1=stats((es - 1).numpy()))
        fields = {}
        ni = len(In)
        sm = waves(Xi, B, 1.5) * phi.repeat_interleave(3)[:, None]
        soft = kii_inv(waves(Xi, B, 1.5))
        rough = torch.randn((ni, B), dtype=f64, generator=gen)
        weak = rough * torch.as_tensor(np.repeat(weak_i, 3), dtype=f64)[:, None] if weak_i.any() else None
        for name, dI in (('smooth', sm), ('soft', soft), ('rough', rough), ('weak', weak)):
            if dI is None:
                continue
            dI = dI * torch.sqrt(targets[:B] / (dI * Kmul_I(K, In, dI)).sum(0))[None, :]
            d = torch.zeros_like(us); d[In] = dI
            fields[name] = us + d
        with torch.no_grad():
            for n, mdl in models.items():
                fields['net:' + n.split('.')[0]] = geo.field(mdl, geo.banks['val'][cls]).to(f64)
        for name, u in fields.items():
            Ku = Kmul(u)
            r = Ku[In]
            Dr = (r * kii_inv(r)).sum(0)                                             # Delta_true
            e = u - us
            De = (e * Kmul(e)).sum(0)
            eh = (u * Ku).sum(0)
            DE, D1 = eh - es, eh - 1
            rI = cert.residual(u)
            c = dict(De_vs_Dr=float(((De - Dr).abs() / Dr).max()), DE_vs_Dr=float(((DE - Dr).abs() / Dr).max()),
                     D1_vs_Dr=float(((D1 - Dr).abs() / Dr).max()),
                     rI_vs_KIIeI=float(((rI - cert.kii(e[In])).norm(dim=0) / rI.norm(dim=0)).max()),
                     rI_cert_vs_scipy=float(((rI - r).norm(dim=0) / r.norm(dim=0)).max()),
                     Delta_true=stats(Dr.numpy()), eps_true=stats((Dr / es).numpy()))
            lbs = {}
            for method in ('krylov', 'cg'):
                for m in ms:
                    t = time.perf_counter(); lb = cert.lower(u, m, method=method); dtm = time.perf_counter() - t
                    lbs[method, m] = lb
                    ratio = (lb / Dr).numpy()
                    worst.append(float(ratio.max()))
                    assert bool((lb <= Dr * (1 + 1e-9)).all()), (cls, name, method, m, float(ratio.max()))
                    assert bool((lb <= De * (1 + 1e-9) + 1e-9 * Dr).all()) if c['De_vs_Dr'] < 1e-9 else True
                    eff.setdefault(f'{name}|{method}|m{m}', []).extend(ratio.tolist())
                    tim.setdefault(f'{method}|m{m}', []).append(dtm)
                    c[f'{method}_m{m}'] = stats(ratio)
            for m in ms[1:]:
                a = lbs['krylov', m] >= lbs['krylov', ms[ms.index(m) - 1]] * (1 - 1e-9)
                assert bool(a.all()), ('monotone', cls, name, m)
                kc = float(((lbs['cg', m] - lbs['krylov', m]).abs() / lbs['krylov', m]).max())
                c[f'krylov_vs_cg_m{m}'] = kc
                krcg.append(kc)
            cc = cert.certify(u, m=8, bs=5)                                        # column blocks; torch vs scipy K
            c['certify_ehat_vs_scipy'] = float(((cc['e_hat'] - eh).abs() / eh).max())
            c['certify_bs5_vs_full'] = float(((cc['delta_lb'] - lbs['krylov', 8]).abs() / lbs['krylov', 8]).max())
            assert c['certify_ehat_vs_scipy'] < 1e-9 and c['certify_bs5_vs_full'] < 1e-9, (c['certify_ehat_vs_scipy'], c['certify_bs5_vs_full'])
            mu_lb, eps_lb = cert.mu_lower(u, m=8)
            mu_true = eh / es
            assert bool((mu_lb <= mu_true * (1 + 1e-9)).all()) and bool((eps_lb <= Dr / es * (1 + 1e-9)).all())
            c['mu_true'] = stats(mu_true.numpy()); c['mu_lb_m8'] = stats(mu_lb.numpy())
            checks[f'{cls}|{name}'] = c
            print(json.dumps({f'{cls}|{name}': dict(Delta_true_med=round(c['Delta_true']['med'], 6),
                                                    De_vs_Dr=c['De_vs_Dr'], DE_vs_Dr=c['DE_vs_Dr'], rI_vs_KIIeI=c['rI_vs_KIIeI'],
                                                    eff_med={m: round(c[f'krylov_m{m}']['med'], 4) for m in ms},
                                                    eff_cg_m16=round(c['cg_m16']['med'], 4))}), flush=True)
    out['checks'] = checks
    out['efficiency'] = {k: stats(v) for k, v in eff.items()}
    out['seconds_per_call_B16'] = {k: float(np.median(v)) for k, v in tim.items()}
    out['max_ratio_lb_over_true'] = max(worst)
    out['krylov_vs_cg_maxrel'] = max(krcg)
    out['nets'] = nets_ok
    print(json.dumps(dict(max_ratio_lb_over_true=max(worst), krylov_vs_cg_maxrel=max(krcg),
                          seconds_per_call_B16=out['seconds_per_call_B16'],
                          diag3_vs_C_blocks_maxrel=out['diag3_vs_C_blocks_maxrel'], floored=cert.floored,
                          interior_weak_nodes=out['interior_weak_nodes'])), flush=True)
    lat_net = 's2v1_snap_10000.pt' if 's2v1_snap_10000.pt' in models else None     # r2-d2 zero-shot: mu ~ 1e11, no PCG
    if lat_net:
        out['lattice_net'] = lat_net
        test_fixture_lattice(out, geo, K, cert, models[lat_net], lu)


def Kmul_I(K, In, dI):
    """K_II dI via scipy (for the test's own scaling)."""
    x = np.zeros((K.shape[0], dI.shape[1])); x[In] = dI.numpy()
    return torch.as_tensor((K @ x)[In])


def test_fixture_lattice(out, geo, K, cert, model, lu_I, tol=1e-10, maxit=300):
    """One-cell 'lattice' on the fixture: box face x = 0 clamped, the other ports free (lattice unknowns), loads: uniform
    nodal x / y / z on the cut band and 3 smooth random forces on the free ports. Reference C = f^T S_ff^-1 f (scipy splu of
    the free-DOF K); learned: PCG on S_hat_ff U = f (variational readout of the fp32 network, exact preconditioner).
    Checks: (a) exact for any U: C >= 2 f^T U' - e_hat(u) + Delta_lb(u), u = field(U), U' = u_P (the fp32 port values);
            (b) at the PCG solution: Delta_lb <= C - C_hat (+ 1e-6 C for the fp32 readout and the solver residual);
            (c) the identity C - (2 f^T U' - e_hat) = Delta(u) + |U* - U'|_S^2 (cell energy excess + interface error)."""
    import scipy.sparse.linalg as sla
    nd = geo.nd
    pn = np.flatnonzero(nd['is_port'])
    gp = nd['grid'][pn].astype(int)
    clamp_n = gp[:, 0] == 0
    cutload = np.asarray(nd['is_cut'])[pn] & ~np.asarray(nd['is_box'])[pn] & ~clamp_n
    fp = np.flatnonzero(~np.repeat(clamp_n, 3))                                   # free port DOFs (lattice unknowns)
    Pn = geo.P.numpy()
    F = np.zeros((len(Pn), 6))
    for d in range(3):
        F[3 * np.flatnonzero(cutload) + d, d] = 1.0 / cutload.sum()
    rng = np.random.default_rng(3)
    X = gp[~clamp_n] / 64.0
    for j in range(3, 6):
        k = rng.standard_normal((3, 6, 3)) * 2
        v = np.stack([(np.cos(2 * np.pi * X @ k[c].T + rng.uniform(0, 6.3, 6)) * rng.standard_normal(6)).sum(1) for c in range(3)], 1)
        F[fp, j] = v.reshape(-1) / np.abs(v).sum()
    free = np.setdiff1d(np.arange(K.shape[0]), Pn[np.setdiff1d(np.arange(len(Pn)), fp)])
    pos = np.searchsorted(free, Pn[fp])                                            # free ports inside the free-DOF system
    t = time.perf_counter()
    lu = sla.splu(K[free][:, free].tocsc())
    out['lattice_splu_seconds'] = time.perf_counter() - t

    def Sinv(r):                                                                   # S_ff^-1 r: condensation by the full solve
        b = np.zeros((len(free), r.shape[1])); b[pos] = r.numpy()
        return torch.as_tensor(lu.solve(b)[pos])

    f = torch.as_tensor(F[fp])
    Uref = Sinv(f)
    Cref = (f * Uref).sum(0)

    def embed(x):
        q = torch.zeros((len(Pn), x.shape[1]), dtype=f64); q[fp] = x
        return q

    def Sh(x):
        return geo.s_hat_apply(model, embed(x))[fp]

    t = time.perf_counter()
    Xs = torch.zeros_like(f); R = f.clone(); Z = Sinv(R); Pd = Z.clone(); rz = (R * Z).sum(0); r0 = R.norm(dim=0)
    it = 0
    for it in range(1, maxit + 1):
        AP = Sh(Pd)
        a = rz / (Pd * AP).sum(0)
        Xs += a * Pd; R -= a * AP
        if float((R.norm(dim=0) / r0).max()) < tol:
            break
        Z = Sinv(R); rz1 = (R * Z).sum(0); Pd = Z + (rz1 / rz) * Pd; rz = rz1
    res = float((R.norm(dim=0) / r0).max())
    out['lattice_pcg'] = dict(iterations=it, residual=res, seconds=time.perf_counter() - t)
    Chat = (f * Xs).sum(0)
    with torch.no_grad():
        u = geo.field(model, embed(Xs)).to(f64)
    U1 = u[geo.P][fp]
    e_hat = (u * torch.as_tensor(K @ u.numpy())).sum(0)
    gap = Cref - Chat
    In, Pd_ = geo.I.numpy(), geo.P.numpy()

    def S_apply(q):                                                                # exact S q (K_II splu, test only)
        w = np.zeros((K.shape[0], q.shape[1])); w[Pd_] = q.numpy()
        w[In] = lu_I.solve(-(K @ w)[In])
        return torch.as_tensor((K @ w)[Pd_])

    qS = (u[geo.P] * S_apply(u[geo.P])).sum(0)
    Dtrue = e_hat - qS
    dU = embed(Uref) - u[geo.P]
    iface = (dU * S_apply(dU)).sum(0)
    gap_eff = Cref - (2 * (f * U1).sum(0) - e_hat)
    ident = float(((gap_eff - Dtrue - iface).abs() / gap_eff).max())
    assert ident < 1e-8, ident
    rec = dict(gap_rel=stats((gap / Cref).numpy()), loads=6, identity_rel_err=ident,
               cell_share_of_gap=stats((Dtrue / gap_eff).numpy()))
    for m in (0, 8, 16):
        D, rel = CE.lattice_lower([(cert, u)], m=m, c_hat=Chat)
        lhs = 2 * (f * U1).sum(0) - e_hat + D
        assert bool((Cref >= lhs - 1e-12 * Cref.abs()).all()), ('R2a', m, ((lhs - Cref) / Cref).tolist())
        if res < 1e-8:
            assert bool((D <= gap + 1e-6 * Cref).all()), ('R2b', m, (D / gap).tolist())
            assert bool((rel <= gap / Cref + 1e-6).all())
        rec[f'm{m}'] = dict(eff_vs_gap=stats((D / gap).numpy()), eff_vs_Delta=stats((D / Dtrue).numpy()), rel_lb=stats(rel.numpy()),
                            R2a_slack_rel=stats(((Cref - lhs) / Cref).numpy()))
    out['lattice'] = rec
    print(json.dumps(dict(fixture_lattice=rec, pcg=out['lattice_pcg'])), flush=True)


# ------------------------------------------------------------------------------------------------------------ remote
def remote(out, ck_path, slot_dir, body, cases):
    """GPU checks R1-R3 (module docstring) on slot-cache geometries of the remote layout."""
    from pathlib import Path
    import trainlib as TL
    import models as MD
    import train2 as T2
    dev = TL.dev
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    cfg = ck['cfg']
    rec = dict(ckpt=ck_path, results={}, lattice={})
    ok = True
    for case in cases:
        g = torch.load(Path(slot_dir) / f'{case}.pt', map_location='cpu', weights_only=False)
        T2.move(g, 'cuda'); g.C.K = g.C
        model = MD.build(cfg['model'], [g], **cfg.get('model_args', {})).to(dev)
        model.load_state_dict(ck['model'], strict=False); model.eval()
        C = g.C
        C.factor(neumann=False, fp32=True)                                        # exact reference (test only)
        cert = CE.from_geo(g)
        r = {}
        for cls in g.classes:
            Q = g.banks['val'][cls][:, :64]
            Q = Q[:, torch.isfinite(Q).all(0)]                                  # the 5 NaN support banks (C6)
            for j in range(0, Q.shape[1], 16):
                q = Q[:, j:j + 16]
                with torch.no_grad():
                    u = g.field(model, q).to(f64)
                e = u - C.extend(q.to(f64))
                Dt = (e * (C @ e)).sum(0)
                for m in (0, 8, 16):
                    torch.cuda.synchronize(); t = time.perf_counter()
                    cc = cert.certify(u, m=m)
                    torch.cuda.synchronize(); dtm = time.perf_counter() - t
                    ratio = (cc['delta_lb'] / Dt).cpu().numpy()
                    ok &= bool((cc['delta_lb'] <= Dt * (1 + 1e-9)).all())
                    d = r.setdefault(f'{cls}|m{m}', dict(eff=[], eps_true=[], eps_lb=[], seconds=[]))
                    d['eff'] += ratio.tolist(); d['seconds'].append(dtm)
                    d['eps_true'] += (Dt / (cc['e_hat'] - Dt)).cpu().tolist(); d['eps_lb'] += cc['eps_lb'].cpu().tolist()
        res = {k: dict(eff=stats(v['eff']), eps_true=stats(v['eps_true']), eps_lb=stats(v['eps_lb']),
                       seconds_B16=float(np.median(v['seconds']))) for k, v in r.items()}
        t8 = max(v['seconds_B16'] for k, v in res.items() if k.endswith('|m8'))
        res['R3_timing_pass'] = bool(t8 < 0.25 * C.nb / 1e5)
        ok &= res['R3_timing_pass']
        rec['results'][case] = res
        print(json.dumps({case: res}), flush=True)
        C._free(); del model, g, C, cert; torch.cuda.empty_cache()
        if os.environ.get('LAT_SKIP') != '1':
            ok &= remote_lattice(rec, ck, cfg, body, case)
    rec['pass'] = bool(ok)
    print(json.dumps(dict(PASS=rec['pass'])), flush=True)
    open(out, 'w').write(json.dumps(rec, indent=1))


def remote_lattice(rec, ck, cfg, body, case):
    """R2 on lattice3 (traction-consistent loads, configurations x / y), learned test cell, exact neighbour."""
    import lattice3 as LT
    import trainlib as TL
    import models as MD
    import ops as OP
    import evalnet as EN
    os.environ['LAT_LOADS'] = 'consistent'
    nbr = os.environ.get('LAT_NBR', 'fresh_train_0020_full')
    C, _ = LT.prepared(case, body)
    geo = TL.Geo(case, body, cfg['data'], neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).to(TL.dev)
    model.load_state_dict(ck['model'], strict=False); model.eval()
    op = EN.NetOp(geo, model)
    cert = CE.from_geo(geo)
    ok = True
    for conf in ('x', 'y'):
        lat = LT.build(case, nbr, conf, body)
        ref = lat.reference()
        nb_op = OP.ExactOp(lat.cells[1]['cell'], lat.cells[1]['T'])
        res = lat.evaluate([op, nb_op], maxit=400)
        U = res['U']
        Cref = torch.as_tensor(ref['compliance'], dtype=f64)
        Chat = (lat.F * U).sum(0).cpu()
        U2 = U.to(torch.float32).to(f64)                                          # fp32-exact lattice vector
        u = op.field(lat.gather(U2, 0))                                           # port values = gather(U2, 0) exactly
        q1 = lat.gather(U2, 1)
        UKU = ((u * (C @ u)).sum(0) + (q1 * nb_op.apply(q1)).sum(0)).cpu()       # U2^T K_hat U2
        fU2 = (lat.F * U2).sum(0).cpu()
        out = {}
        for m in (0, 8, 16):
            D = CE.lattice_lower([(cert, u)], m=m).cpu()
            r1 = bool((Cref >= 2 * fU2 - UKU + D - 1e-8 * Cref).all())
            r2 = bool((D <= (Cref - Chat) + 1e-6 * Cref).all())
            ok &= r1 and r2
            out[f'm{m}'] = dict(R2a=r1, R2b=r2, eff_vs_gap=stats((D / (Cref - Chat)).numpy()),
                                rel_lb=stats((D / (Chat + D)).numpy()),
                                R2a_slack_rel=stats(((Cref - 2 * fU2 + UKU - D) / Cref).numpy()))
        rec['lattice'][f'{case}|{conf}'] = dict(nbr=nbr, pcg_residual=res['pcg_residual'], loads=lat.labels,
                                                gap_rel=stats((1 - Chat / Cref).numpy()), **out)
        print(json.dumps({f'{case}|{conf}': rec['lattice'][f'{case}|{conf}']}), flush=True)
        del lat; torch.cuda.empty_cache()
    return ok


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'remote':
        remote(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])
        sys.exit(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rec = {}
    if mode in ('toy', 'all'):
        test_toys(rec)
    if mode in ('fixture', 'all'):
        test_fixture(rec)
    if len(sys.argv) > 2:
        open(sys.argv[2], 'w').write(json.dumps(rec, indent=1))
    print('PASS', flush=True)
