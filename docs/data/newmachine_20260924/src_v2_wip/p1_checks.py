"""P1 local checks of the two-grid predictor (A3) on single cells, validation directions of the val bank:
 (a) operator properties in the deployed precision (fastnet): symmetry of Q^T S_hat Q, rigid-body null, action vs field
     energy, fastnet vs trainlib path;
 (b) Chebyshev interval coverage: the estimated upper end b = 1.05 x (40 power steps) against lambda_max(D^-1 K_II)
     (Lanczos, scipy eigsh) and the Gershgorin bound;
 (c) what the network contributes: the same correction (tail k - Q1_17 - tail k) from the untrained B network, a
     zero interior and a graph-harmonic interior (both with the exact rigid split), k sweep, vs A3;
 (d) delta^2 kappa: eps = delta^2 kappa exactly with delta^2 = d_I^T D d_I / u_I^T D u_I, kappa = R(d) / R(u),
     R(d) = d^T K d / d_I^T D d_I, R(u) = u^T K u / u_I^T D u_I (D = diag K_II), per direction for B and A3;
 (e) ghost-penalty share of the exact field energy (1 - bulk element energy / total);
 (f) --dump <case>: npz with node coordinates, port / cut-band masks, exact / B / A3 fields and bulk element error
     energies for the first 4 directions of the first class (field figure).
Usage: p1_checks.py <a3 ckpt> <b ckpt> <out.json> <case>[,...] [--classes force_c,force] [--m 32] [--ks 8,16,32,64]
       [--dump case] [--dump-out file.npz]"""
import os, sys, json, time, argparse
from pathlib import Path
if os.environ.get('OPL_DEV') == 'cpu':
    import diag_sens as DS                                              # first: CPU environment (PARDISO factors)
import numpy as np
import torch
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import trainlib as TL
import teacher as TE
import ops as OP
import diag_cert as DC
import fastnet as FN

dev, dt = TL.dev, TL.dt
if dev.type == 'cpu':
    class _CpuSPD:
        """teacher.SPDSolver stand-in on the host (upper CSR -> symmetric, MKL PARDISO), for GraphLift."""
        def __init__(self, crow, col, vals, n):
            import scipy.sparse as sp, pypardiso
            U = sp.csr_matrix((vals.cpu().numpy(), col.cpu().numpy(), crow.cpu().numpy()), shape=(n, n))
            self.A = (U + sp.triu(U, 1).T).tocsr()
            self.ps = pypardiso.PyPardisoSolver(); self.ps.factorize(self.A)

        def solve(self, r):
            return torch.as_tensor(self.ps.solve(self.A, np.ascontiguousarray(r.cpu().numpy())), dtype=dt).reshape(r.shape)

        def free(self):
            self.ps.free_memory(everything=True)
    TE.SPDSolver = _CpuSPD
    TE.Cell.factor = lambda self, neumann=True, fp32=False, **kw: DS.cpu_factor(self)


class _W:
    """Model stand-in carrying only the wrapper settings (trainlib.wrap reads smooth_k / smooth_alpha / coarse_space)."""
    def __init__(self, k, space='Q1_17', alpha=30.0):
        self.smooth_k, self.smooth_alpha, self.coarse_space = k, alpha, space


def lam_checks(C, alpha=30.0):
    import scipy.sparse.linalg as sla
    lmin, b = TL.tail_bounds(C, alpha)
    dI = C.dK[C.I]
    s = 1 / torch.sqrt(dI)
    full = torch.zeros((C.nb, 1), dtype=dt, device=dev)

    def mv(x):
        full.zero_(); full[C.I, 0] = s * torch.as_tensor(np.asarray(x).ravel(), dtype=dt, device=dev)
        return (s * (C.K @ full)[C.I, 0]).cpu().numpy()
    op = sla.LinearOperator((C.ni, C.ni), matvec=mv, dtype=np.float64)
    lam = float(sla.eigsh(op, k=1, which='LA', tol=1e-10, return_eigenvectors=False, maxiter=5000)[0])
    # Gershgorin for D^-1 K_II: max_i sum_j |a_ij| / a_ii (upper storage: off-diagonals count for both rows)
    pm = torch.zeros(C.nb, dtype=torch.bool, device=dev); pm[C.P] = True
    ru, cu, v = C.ru.long().to(dev), C.cu.long().to(dev), C.vals.to(dev)
    sel = ~pm[ru] & ~pm[cu]
    rs = torch.zeros(C.nb, dtype=dt, device=dev)
    off = sel & (ru != cu)
    rs.index_add_(0, ru[off], v[off].abs()); rs.index_add_(0, cu[off], v[off].abs())
    dg = sel & (ru == cu); rs.index_add_(0, ru[dg], v[dg].abs())
    gers = float((rs[C.I] / dI).max())
    return dict(b=b, a=lmin, lanczos_lmax=lam, gershgorin=gers, covered=bool(b >= lam), margin=b / lam - 1)


def delta_kappa(u, x, C):
    d = x - u
    d[C.P] = 0                                                                  # ports agree (fp32 rounding only)
    D = C.dK[C.I][:, None]
    dI, uI = d[C.I], u[C.I]
    Ed, Eu = TL.energy(d, C.K), TL.energy(u, C.K)
    dD, uD = (D * dI * dI).sum(0), (D * uI * uI).sum(0)
    delta2 = dD / uD
    kappa = (Ed / dD) / (Eu / uD)
    return delta2.cpu().numpy(), kappa.cpu().numpy(), (Ed / Eu).cpu().numpy()


def elem_energy(C, x):
    """Bulk element energies x_e^T K_e x_e (ghost penalty excluded), x (nb, k) -> (E, k)."""
    iu = torch.triu_indices(81, 81, device=dev)
    w = torch.where(iu[0] == iu[1], 1.0, 2.0).to(dt)
    out = []
    dofs = torch.as_tensor(C.dofs, device=dev).long()
    for lo in range(0, len(C.M), 2048):
        ke = C.M[lo:lo + 2048] @ C.Tm_up                                                  # (e, 3321)
        xe = x[dofs[lo:lo + 2048]]                                                        # (e, 81, k)
        out.append(torch.einsum('ep,epk->ek', ke * w, xe[:, iu[0]] * xe[:, iu[1]]))
    return torch.cat(out)


def st(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), p90=float(np.quantile(v, .9)), max=float(v.max()))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('a3'); ap.add_argument('b'); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--classes', default='force_c,force'); ap.add_argument('--m', type=int, default=32)
    ap.add_argument('--ks', default='8,16,32,64'); ap.add_argument('--dump', default=''); ap.add_argument('--dump-out', default='')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data_v2')
    a = ap.parse_args(argv)
    cka = torch.load(a.a3, map_location=dev, weights_only=False); ckb = torch.load(a.b, map_location=dev, weights_only=False)
    ks = [int(k) for k in a.ks.split(',')]
    rec = dict(a3=a.a3, b=a.b, conv=TL.conv_precision(), per_case={})
    for case in a.cases.split(','):
        t0 = time.perf_counter()
        geo = TL.Geo(case, a.body, a.data, neumann=False, log=lambda s_: None)
        C = geo.C
        C.factor(neumann=False)
        A3, missa = DC.net_for(cka, geo); Bm, missb = DC.net_for(ckb, geo)
        fa3, fb = FN.FastNet(A3, geo), FN.FastNet(Bm, geo)
        r = dict(interior=int(C.ni), ports=int(C.np_), missing_keys=[missa, missb], lam=lam_checks(C))
        gl = OP.GraphLift(C)
        RP, RA, RPp = geo.RP.to(dt), geo.RA.to(dt), geo.RPpinv.to(dt)

        def rigid_split(ext, Q):
            c = RPp @ Q
            u = ext(Q - RP @ c) + RA @ c
            u[C.P] = Q
            return u

        for cls in [c for c in a.classes.split(',') if c in geo.classes]:
            Q = geo.banks['val'][cls][:, :a.m].to(dt)
            with torch.no_grad():
                u = C.extend(Q); eu = TL.energy(u, C.K)
                ex = lambda x: (TL.energy(x, C.K) / eu - 1).cpu().numpy()
                xa3 = fa3.field(Q).to(dt)
                xb = fb.field(Q).to(dt)
                z = rigid_split(lambda q: torch.zeros((C.nb, q.shape[1]), dtype=dt, device=dev), Q)
                h = rigid_split(gl.ext, Q)
                if not torch.allclose(h[C.P], Q):
                    raise AssertionError('GRAPHLIFT_PORT_ORDER')
                out = dict(A3=st(ex(xa3)), B=st(ex(xb)), zero=st(ex(z)), harmonic=st(ex(h)))
                for k in ks:
                    w = _W(k)
                    for nm, x0 in (('B', xb), ('zero', z), ('harmonic', h)):
                        out[f'{nm}+tail{k}+Q1_17+tail{k}'] = st(ex(TL.wrap(C, x0, w)))
                # (a) operator properties on the first 8 directions
                q8 = Q[:, :8]
                S8 = fa3.s_hat(q8)
                G = q8.T @ S8
                dg = torch.sqrt(torch.outer(G.diagonal().abs(), G.diagonal().abs()))
                f8 = fa3.field(q8).to(dt)
                eF = TL.energy(f8, C.K)
                xt = geo.field(A3, q8).to(dt)
                Rn = RP / RP.norm(dim=0, keepdim=True); qn = q8 / q8.norm(dim=0, keepdim=True)
                SR = fa3.s_hat(Rn)
                ops = dict(sym_rel_max=float(((G - G.T).abs() / dg).max()),
                           action_vs_field_energy_rel_max=float(((G.diagonal() - eF).abs() / eF).max()),
                           fastnet_vs_trainlib_field_rel=float((f8 - xt).norm() / f8.norm()),
                           rigid_energy_ratio_max=float((Rn * SR).sum(0).abs().max() / (qn * fa3.s_hat(qn)).sum(0).mean()),
                           rigid_force_rel_max=float(SR.norm(dim=0).max() / fa3.s_hat(qn).norm(dim=0).mean()),
                           exact_S_psd_min_eps_A3=float(ex(xa3).min()))
                dB = delta_kappa(u, xb, C); dA = delta_kappa(u, xa3, C)
                dk = dict(B=dict(delta=st(np.sqrt(dB[0])), kappa=st(dB[1]), eps_check=float(np.abs(dB[0] * dB[1] - dB[2]).max())),
                          A3=dict(delta=st(np.sqrt(dA[0])), kappa=st(dA[1]), eps_check=float(np.abs(dA[0] * dA[1] - dA[2]).max())))
                ee = elem_energy(C, u)
                ghost = dict(ghost_share=st((1 - ee.sum(0) / eu).cpu().numpy()))
            r[cls] = dict(eps=out, ops=ops, delta_kappa=dk, ghost=ghost, m=int(Q.shape[1]))
            print(json.dumps(dict(case=case, cls=cls, A3=round(100 * out['A3']['mean'], 4), B=round(100 * out['B']['mean'], 3),
                                  **{k_: round(100 * v['mean'], 4) for k_, v in out.items() if 'tail8' in k_ or 'tail32' in k_},
                                  ops={k_: float(f'{v:.3g}') for k_, v in ops.items()},
                                  dkB=(round(dk['B']['delta']['mean'], 6), round(dk['B']['kappa']['mean'], 1)),
                                  dkA3=(round(dk['A3']['delta']['mean'], 6), round(dk['A3']['kappa']['mean'], 1)),
                                  ghost=round(ghost['ghost_share']['mean'], 6))), flush=True)
            if a.dump == case and not Path(a.dump_out or 'x').exists():
                with torch.no_grad():
                    q4 = Q[:, :4]; u4 = u[:, :4]; b4 = xb[:, :4]; a4 = xa3[:, :4]
                    n2 = 2 * C.n + 1
                    np.savez_compressed(a.dump_out, case=case, cls=cls, n=C.n, nodes=np.asarray(C.nodes),
                                        xyz=np.stack(np.unravel_index(np.asarray(C.nodes), (n2,) * 3), 1) / (n2 - 1),
                                        port_dofs=C.P.cpu().numpy(), port_node_ids=np.asarray(C.port_node_ids),
                                        cells=np.asarray(C.cells), vf=C.M[:, 0].cpu().numpy(),
                                        u_exact=u4.cpu().numpy(), u_B=b4.cpu().numpy(), u_A3=a4.cpu().numpy(), q=q4.cpu().numpy(),
                                        ee_exact=elem_energy(C, u4).cpu().numpy(), ee_err_B=elem_energy(C, b4 - u4).cpu().numpy(),
                                        ee_err_A3=elem_energy(C, a4 - u4).cpu().numpy())
        r['seconds'] = time.perf_counter() - t0
        rec['per_case'][case] = r
        print(json.dumps(dict(case=case, lam=r['lam'], seconds=round(r['seconds'], 1))), flush=True)
        A3.caches.pop(case, None); Bm.caches.pop(case, None)
        if hasattr(gl.sol, 'free'):
            gl.sol.free()
        del fa3, fb, gl, geo, A3, Bm
        C._free(); del C
        import gc; gc.collect(); torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
