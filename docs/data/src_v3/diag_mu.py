"""A6: does the mu bound drive the lattice interface error? (one learned test cell, exact neighbour, consistent loads)

  mu       top eigenvalue of the pencil (S_hat, S) on the test cell's ports: block power iteration with Rayleigh-Ritz and
           the fp32 Neumann factor (the math of trainlib.Geo.adversarial, one S_hat product per iteration), fixed seed,
           iterated until the top Ritz value changes by < tol
  mu_-F    the same on V_F = {q : q_F = 0}, F = the test cell's box face shared with the neighbour (lattice3.build: config x
           puts the neighbour at (-1, 0, 0) -> F = the x = 0 face; config y at (0, -1, 0) -> y = 0); S_GG^-1 from one fp32
           factorization of K with the F DOFs clamped (G = the other ports)
  exact    optional final Rayleigh-Ritz of both blocks on the exact pencil (X^T S_hat X, X^T S X), S X from the interior
           factor: Ritz values of the exact pencil are rigorous lower bounds of mu and mu_-F
  eps_q    q^T S_hat q / q^T S q - 1 at the exact lattice port data q of the test cell, per load (= diag_lat's
           extension-only energy error)
  bound    ||dU||_{K_L} / ||q||_S <= ||dU||_{K_hat_L} / ||q||_S <= sqrt((1 - 1/mu_eff) eps_q),  mu_-F <= mu_eff <= mu
           dU = U - U_hat the lattice port error, K_L / K_hat_L the exact / learned lattice stiffness, ||q||_S^2 = q^T S q.
  Proof: K_hat_L = K_L + D, D = G^T (S_hat - S) G >= 0 (G gathers the test cell), dU = K_hat_L^-1 D U, so
  ||dU||^2_{K_hat} = z^T D^1/2 (K_L + D)^-1 D^1/2 z with z = D^1/2 U, z^T z = q^T (S_hat - S) q = eps_q ||q||_S^2, and the
  eigenvalues of D^1/2 (K_L + D)^-1 D^1/2 are nu / (1 + nu) <= 1 - 1/mu_eff (nu: pencil (D, K_L), top value mu_eff - 1).
  mu_eff >= mu_-F: a test-cell q with q_F = 0 and the neighbour at rest is a lattice vector of energy q^T S q (the far
  face clamp only touches the neighbour); mu_eff <= mu: K_L >= G^T S G.
Measured per load: r = ||dU||_{K_L} / ||q||_S, r_hat (K_hat_L norm), the implied lower bound of mu_eff
1 / (1 - r_hat^2 / eps_q), and diag_lat's sensitivity decomposition (total / extension / interface, relative 8-corner
errors of the test cell). Ordering: Spearman over the gate loads of the bounds against the interface error and r.
Usage: diag_mu.py <checkpoint.pt> <out.json> <case> [--nbr <case>] [--configs x,y] [--k 16] [--tol 1e-4] ... (--help)
Needs LT.prepared (dense port operators, cached T64.npy) like evalnet; run after training (two factorizations at a time).
"""
import json, sys, os, time, argparse
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import diag_cert as DC

dev, dt = TL.dev, TL.dt
F_AXIS = {'x': 0, 'y': 1}                                                          # shared face = test-cell <axis> = 0 (lattice3.build)


def face_ports(C, axis, side=0):
    """Port-vector positions of the DOFs of the box port nodes on the face grid[axis] == side * 2n."""
    g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
    on = C.port_is_box & (g[:, axis] == side * 2 * C.n)
    return torch.as_tensor(np.flatnonzero(np.repeat(on, 3)), device=dev)


def rigid_proj(Q):
    return lambda Y: Y - Q @ (Q.T @ Y)


def zero_proj(fp):
    def pr(Y):
        Y = Y.clone(); Y[fp] = 0
        return Y
    return pr


def block_ritz(shat, solve, proj, X, tol=1e-4, maxit=60):
    """Top Ritz pairs of the pencil (S_hat, S) on a subspace V of the ports by block power iteration + Rayleigh-Ritz:
    Z = solve(proj(S_hat X)) (= S_V^-1 S_hat X), S-orthonormalized with S Z = proj(S_hat X) (no S product), H = Z^T S_hat Z,
    X = Z W (Ritz vectors, descending). proj: the load projection onto V's dual (rigid-free: I - Q Q^T; q_F = 0: zero F rows),
    also applied to the start block. S_hat X of the next iteration is S_hat Z W (one S_hat product per iteration).
    Returns X, Ritz values (descending), S_hat X, history of the top value."""
    X = proj(X.to(dt))
    Y = shat(X)
    hist = []
    for _ in range(maxit):
        PY = proj(Y)
        Z = solve(PY)
        G = Z.T @ PY; G = 0.5 * (G + G.T)
        ev, V = torch.linalg.eigh(G)
        keep = ev > ev.max() * 1e-12
        T = V[:, keep] / torch.sqrt(ev[keep])[None, :]
        Z = Z @ T
        HY = shat(Z)
        H = Z.T @ HY; H = 0.5 * (H + H.T)
        ritz, Wv = torch.linalg.eigh(H)
        o = torch.argsort(ritz, descending=True)
        ritz, Wv = ritz[o], Wv[:, o]
        X, Y = Z @ Wv, HY @ Wv
        hist.append(float(ritz[0]))
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) <= tol * abs(hist[-1]):
            break
    return X, ritz, Y, hist


def exact_ritz(X, SX, SHX):
    """Ritz values of the exact pencil (X^T S_hat X, X^T S X) (descending; rigorous lower bounds of its top eigenvalues)."""
    Gs = X.T @ SX; Gs = 0.5 * (Gs + Gs.T)
    H = X.T @ SHX; H = 0.5 * (H + H.T)
    ev, V = torch.linalg.eigh(Gs)
    keep = ev > ev.max() * 1e-12
    T = V[:, keep] / torch.sqrt(ev[keep])[None, :]
    return torch.linalg.eigvalsh(T.T @ H @ T).flip(0)


def dirichlet_solver(C, fp, fp32=True, refine=3):
    """solve(y) = S_GG^-1 y_G (F rows zero) from K with the F DOFs clamped (rows / columns zeroed, unit diagonal), Jacobi
    scaled like teacher.Cell._factor_neumann; fp32 (fp64 fallback) plus `refine` fp64 iterative-refinement steps with the
    clamped K (as teacher.Cell.extend). Returns (solve, free)."""
    import teacher as TE
    fd = C.P[fp]
    ru, cu = C.ru.long(), C.cu.long()
    pin = torch.zeros(C.nb, dtype=torch.bool, device=dev); pin[fd] = True
    v = C.vals.clone(); v[pin[ru] | pin[cu]] = 0; v[C.diag[fd]] = 1.0
    s = 1 / torch.sqrt(v[C.diag])
    vs = (v * s[ru] * s[cu]).contiguous()
    try:
        sol = TE.SPDSolver(C.crow.int(), C.cu, vs, C.nb, fdt=torch.float32 if fp32 else None)
    except Exception:
        sol = TE.SPDSolver(C.crow.int(), C.cu, vs, C.nb)
    del v, vs, pin

    def kc(u):                                                                      # clamped K: F rows / columns -> identity
        x = u.clone(); x[fd] = 0
        y = C @ x; y[fd] = u[fd]
        return y

    def solve(y):
        F = torch.zeros((C.nb, y.shape[1]), dtype=dt, device=dev); F[C.P] = y; F[fd] = 0
        u = s[:, None] * sol.solve(s[:, None] * F)
        for _ in range(refine if sol.fdt != dt else 0):
            u += s[:, None] * sol.solve(s[:, None] * (F - kc(u)))
        q = u[C.P]
        q[fp] = 0
        return q
    return solve, sol.free


def bound_table(eps_q, mu, mu_F, r, r_hat, interface=None, gate=None):
    """Per load: bound_hi = sqrt((1 - 1/mu) eps_q) (valid), bound_lo = sqrt((1 - 1/mu_-F) eps_q) (the bound with the
    smallest admissible mu_eff), the implied mu_eff >= 1 / (1 - r_hat^2 / eps_q); checks and rank correlations."""
    eps_q, r, r_hat = (np.asarray(x, float) for x in (eps_q, r, r_hat))
    mu, mu_F = np.broadcast_to(np.asarray(mu, float), eps_q.shape), np.broadcast_to(np.asarray(mu_F, float), eps_q.shape)
    e = np.maximum(eps_q, 0)
    hi, lo = np.sqrt((1 - 1 / mu) * e), np.sqrt((1 - 1 / mu_F) * e)
    x = np.clip(r_hat ** 2 / np.maximum(e, 1e-300), 0, 1 - 1e-15)
    out = dict(bound_hi=hi.tolist(), bound_lo=lo.tolist(), implied_mu_eff=(1 / (1 - x)).tolist(),
               r_le_bound_hi=bool((r <= hi * (1 + 1e-6) + 1e-12).all()), r_hat_le_bound_hi=bool((r_hat <= hi * (1 + 1e-6) + 1e-12).all()),
               implied_le_mu=bool(((1 / (1 - x)) <= mu * (1 + 1e-6)).all()), tightness_hi=(r_hat / np.maximum(hi, 1e-300)).tolist())
    g = np.ones(len(e), bool) if gate is None else np.asarray(gate, bool)
    out['spearman_gate'] = dict(bound_hi_vs_r=DC.spearman(hi[g], r_hat[g]), bound_lo_vs_r=DC.spearman(lo[g], r_hat[g]),
                                eps_q_vs_r=DC.spearman(e[g], r_hat[g]))
    if interface is not None:
        it = np.asarray(interface, float)
        out['spearman_gate'].update(bound_hi_vs_interface=DC.spearman(hi[g], it[g]), bound_lo_vs_interface=DC.spearman(lo[g], it[g]),
                                    r_vs_interface=DC.spearman(r_hat[g], it[g]), eps_q_vs_interface=DC.spearman(e[g], it[g]))
    return out


@torch.no_grad()
def lattice_terms(lat, ref, res, op_hat, op_exact_test, op_nbr):
    """eps_q, the energy norms of dU (exact and learned lattice stiffness) relative to ||q||_S, per load."""
    q = lat.gather(ref['U'], 0)
    E = torch.as_tensor(ref['energy'][0], dtype=dt, device=dev)                      # q^T S q (dense exact port operator)
    u = op_hat.field(q)
    eh = (u * (lat.cells[0]['cell'] @ u)).sum(0)                                      # q^T S_hat q = u^T K u (variational readout)
    dU = ref['U'] - res['U']
    nK = (dU * lat.matvec([op_exact_test, op_nbr], dU)).sum(0)
    nKh = (dU * lat.matvec([op_hat, op_nbr], dU)).sum(0)
    return dict(eps_q=(eh / E - 1).tolist(), q_S2=E.tolist(), r=torch.sqrt(nK.clamp_min(0) / E).tolist(),
                r_hat=torch.sqrt(nKh.clamp_min(0) / E).tolist(), q_rel=((lat.gather(res['U'], 0) - q).norm(dim=0) / q.norm(dim=0)).tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('case')
    ap.add_argument('--nbr', default='', help='neighbour case (default: the family FULL parent, fresh_train_0020_full for 0020)')
    ap.add_argument('--configs', default='x,y'); ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--tol', type=float, default=1e-4); ap.add_argument('--maxit', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--no-exact-ritz', action='store_true')
    ap.add_argument('--body', default=''); ap.add_argument('--data', default='')
    args = ap.parse_args(argv)
    import lattice3 as LT
    import ops as OP
    import evalnet as EN
    import fastnet as FN
    os.environ['LAT_LOADS'] = 'consistent'
    ck = DC.load_ckpt(args.ckpt); cfg = ck['cfg']
    body, data = args.body or cfg['body'], args.data or cfg['data']
    nbr = args.nbr or DC.family_full(args.case)
    configs = args.configs.split(',')
    C, _ = LT.prepared(args.case, body)                                               # one teacher cell shared with the lattice
    geo = TL.Geo(args.case, body, data, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    model, miss = DC.net_for(ck, geo)
    fast = FN.FastNet(model, geo); op = EN.FastOp(fast)
    rec = dict(ckpt=args.ckpt, step=ck.get('step'), case=args.case, nbr=nbr, missing_keys=miss, args=vars(args), mu={}, results=[])
    log = lambda d: print(json.dumps(d), flush=True)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    X0 = torch.randn((C.np_, args.k), dtype=dt, device=dev, generator=gen)
    blocks = {}
    # 1. mu: Neumann factor (fp32), rigid-free ports
    t = time.perf_counter()
    C.factor(neumann=True, interior=False, fp32_neumann=True)
    X, ritz, SHX, hist = block_ritz(op.apply, C.neumann, rigid_proj(C.Q), X0, args.tol, args.maxit)
    C._free()
    blocks['all'] = (X, SHX)
    rec['mu']['all'] = dict(ritz=ritz[:4].tolist(), iterations=len(hist), history=hist, seconds=time.perf_counter() - t)
    log(dict(event='MU', face='all', mu=float(ritz[0]), it=len(hist)))
    # 2. mu_-F per configuration: K with the shared face clamped (fp32)
    for conf in configs:
        t = time.perf_counter()
        fp = face_ports(C, F_AXIS[conf])
        solve, free = dirichlet_solver(C, fp)
        X, ritz, SHX, hist = block_ritz(op.apply, solve, zero_proj(fp), X0, args.tol, args.maxit)
        free(); DC.free_mem()
        blocks[conf] = (X, SHX)
        rec['mu'][conf] = dict(face_dofs=int(len(fp)), ritz=ritz[:4].tolist(), iterations=len(hist), history=hist,
                               seconds=time.perf_counter() - t)
        log(dict(event='MU', face=conf, mu_F=float(ritz[0]), it=len(hist)))
    # 3. exact Rayleigh-Ritz (interior factor): rigorous lower bounds
    if not args.no_exact_ritz:
        C.factor(neumann=False, fp32=True)
        for k_, (X, SHX) in blocks.items():
            rec['mu'][k_]['exact_ritz'] = exact_ritz(X, C.apply(X), SHX)[:4].tolist()
        C._free()
    mu = rec['mu']['all'].get('exact_ritz', rec['mu']['all']['ritz'])[0]
    # 4. lattice: eps_q, dU norms, diag_lat's sensitivity decomposition
    for conf in configs:
        t = time.perf_counter()
        lat = LT.build(args.case, nbr, conf, body)
        ref = lat.reference()
        ex_n = OP.ExactOp(lat.cells[1]['cell'], lat.cells[1]['T'])
        res = lat.evaluate([op, ex_n], maxit=400)
        cmp_ = lat.compare(res)
        lt = lattice_terms(lat, ref, res, op, OP.ExactOp(lat.cells[0]['cell'], lat.cells[0]['T']), ex_n)
        q_ref, q_net = lat.gather(ref['U'], 0), lat.gather(res['U'], 0)
        C.factor(neumann=False, fp32=True)
        s_ref = C.sens(C.extend(q_ref)); s_ext = C.sens(op.field(q_ref))
        s_int = C.sens(C.extend(q_net)); s_tot = C.sens(op.field(q_net))
        C._free()
        nrm = s_ref.norm(dim=0)
        sd = {k: ((s - s_ref).norm(dim=0) / nrm).tolist() for k, s in (('total', s_tot), ('extension', s_ext), ('interface', s_int))}
        mu_F = rec['mu'][conf].get('exact_ritz', rec['mu'][conf]['ritz'])[0]
        bt = bound_table(lt['eps_q'], mu, mu_F, lt['r'], lt['r_hat'], sd['interface'], lat.gate)
        r = dict(config=conf, loads=lat.labels, gate=lat.gate.tolist(), mu=mu, mu_F=mu_F, **lt, sens=sd, bounds=bt,
                 compliance_rel=cmp_['compliance_rel_err'], energy_share_test=cmp_['energy_share_test_cell'],
                 pcg=res.get('pcg_iterations'), seconds=time.perf_counter() - t)
        rec['results'].append(r)
        for j, lab in enumerate(lat.labels):
            log(dict(config=conf, load=lab, eps_q=round(lt['eps_q'][j], 5), r=round(lt['r_hat'][j], 5),
                     bound_lo=round(bt['bound_lo'][j], 5), bound_hi=round(bt['bound_hi'][j], 5),
                     interface=round(sd['interface'][j], 5), extension=round(sd['extension'][j], 5), total=round(sd['total'][j], 5)))
        del lat; DC.free_mem()
        Path(args.out).write_text(json.dumps(rec, indent=1))
    # 5. ordering over all gate loads of all configurations
    cat = lambda k, f=lambda r_: r_: np.concatenate([np.asarray(f(r_)[k], float) for r_ in rec['results']])
    gate = cat('gate')
    rec['overall'] = bound_table(cat('eps_q'), np.concatenate([np.full(len(r_['loads']), r_['mu']) for r_ in rec['results']]),
                                 np.concatenate([np.full(len(r_['loads']), r_['mu_F']) for r_ in rec['results']]),
                                 cat('r'), cat('r_hat'), cat('interface', lambda r_: r_['sens']), gate)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    log(dict(event='OVERALL', **{k: v for k, v in rec['overall'].items() if not isinstance(v, list)}))
    return rec


if __name__ == '__main__':
    main()
