"""Revision round 1, E3 + E4 on the two-cell configurations (gate protocol of lat_full.py --nb-mode explicit --sets test:
test cell learned through fastnet, its continuous-thickness neighbour packet <case>_nbm<x|y> exact, traction-consistent
loads, LAT_CPU=1 recommended). One process per test cell (both configurations), new script; lat_full / lattice3 unchanged.

Per configuration (E4 instrumentation, same solve as the gate):
  exact reference U, C = f^T U, s (8 x loads, test cell, -u^T K_c u); learned solve U_bar (PCG, exact lattice factor as
  preconditioner, tol 1e-10, maxit 400), s_tilde (field estimate at the learned solution), gate metrics as lat_full;
  rho = f - y(U_bar) (y = the applied lattice action), U_bar^T rho, C_bar = f^T U_bar, J = C_bar + U_bar^T rho,
  e_act = U_bar^T y(U_bar), rho^T Kex^-1 rho by the exact dense lattice Cholesky, dual-norm bound
  |U_bar^T rho| <= sqrt(e_act) sqrt(rho^T Kex^-1 rho), omega = q_hat^T S_hat q_hat - u_bar^T K u_bar (learned cell),
  a(u_bar - u, u_bar - u) (from the dense exact S: e_fld - 2 q_hat^T S q + q^T S q, plus dq^T S dq for the exact
  neighbour) and the residual of the identity C - C_bar = a + U_bar^T rho + omega (Appendix J.6).
E3 (complete surrogate derivative, fixed-q_hat central differences; loads held at the evaluated design):
  for every corner c and step h (relative, h_c = h tau_c) the test cell is rebuilt at tau +- h_c e_c (r1x3_common.Rebuilder)
  and applied to the solved traces q_hat of both configurations at once, no re-solve:
    D_act  = -(q^T S_hat(+) q - q^T S_hat(-) q) / 2h_c          (action energy; the requested estimate of C_hat_,c)
    D_fld  = -(u(+)^T K(+) u(+) - u(-)^T K(-) u(-)) / 2h_c      (field energy of the recovered field)
    D_ext  = s_tilde_c - 2 ((u(+) - u(-)) / 2h_c)^T K u_bar     (Eq. 14 with F_,c q by central differences)
  plus the rebuild at the unperturbed tau (reproducibility against the solve's operator), switch indicators at every
  build, and (--fixb-steps) the same with the Chebyshev interval frozen at the base design.
  --resolve case:conf:corners: full re-solve central differences of C_hat (and of J) for those corners at --resolve-steps
  (the loads and the exact preconditioner of the base design; the neighbour exact).
Outputs: <outdir>/pair_<case>.json (metrics) and pair_<case>.npz (vectors: s, s_tilde, D_* per step, q_hat, rho, ...).
Usage: r1_pairs.py <outdir> <case> [--ckpt A3] [--configs x,y] [--steps 1e-2,3e-3,1e-3,3e-4,1e-4] [--corners 0,...,7]
       [--fixb-steps 1e-3] [--resolve fresh_val_2003_d1_v1:x:auto2] [--resolve-steps 1e-2,3e-3] [--no-deriv] [--maxit 400]
"""
import json, time, argparse, os
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import trainlib as TL
import lattice3 as LT
import ops as OP
import diag_cert as DC
import lat_full as LF
import r1x3_common as RC

dev, dt = RC.dev, RC.dt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir'); ap.add_argument('case')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    ap.add_argument('--configs', default='x,y')
    ap.add_argument('--steps', default='1e-2,3e-3,1e-3,3e-4,1e-4')
    ap.add_argument('--corners', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--fixb-steps', default='1e-3')
    ap.add_argument('--resolve', default='')
    ap.add_argument('--resolve-steps', default='1e-2,3e-3')
    ap.add_argument('--no-deriv', action='store_true')
    ap.add_argument('--maxit', type=int, default=400)
    ap.add_argument('--tmp', default='/root/autodl-tmp/OPL/S1/V2/R1/X3/tmp')
    a = ap.parse_args(argv)
    os.environ['LAT_LOADS'] = 'consistent'
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    tag = f'pair_{a.case}'
    log = lambda d: print(json.dumps(RC.tojson(d), default=float), flush=True)
    rec = dict(args=vars(a), env=RC.env_flags(), case=a.case, configs={}, t0=time.strftime('%F %T'))
    vec = {}
    ck = DC.load_ckpt(a.ckpt)
    body, data = ck['cfg']['body'], ck['cfg']['data']
    Ct, _ = LT.prepared(a.case, body)
    opt, miss = LF.fast_op(ck, a.case, body, data, Ct)
    rec['missing_keys'] = miss
    taus0 = list(Ct.taus0)
    rec['taus0'] = taus0
    resolve = {}
    for spec in [s for s in a.resolve.split(',') if s]:
        cs, cf, cc = spec.split(':')
        if cs == a.case:
            resolve[cf] = cc
    Q, U0, KU0, s_ex, s_ti, cols = [], [], [], [], [], []
    keep_lat = {}
    for conf in a.configs.split(','):
        t = time.perf_counter()
        ncase = f'{a.case}_nbm{conf}'
        lat = LT.build(a.case, ncase, conf, body)
        ex = [OP.ExactOp(cd['cell'], cd['T']) for cd in lat.cells]
        ref = lat.reference()
        res = lat.evaluate([opt, ex[1]], maxit=a.maxit)
        cmp_ = lat.compare(res)
        eps, w, bound = LF.cell_metrics(lat, ref, [opt, ex[1]], (True, False))
        summ = LF.summarize(cmp_, eps, w, bound, lat.gate)
        # ---------------------------------------------------------------- E4 instrumentation
        ops = [opt, ex[1]]
        F = lat.F
        Xh = res['U'].to(dev)
        with torch.no_grad():
            y = lat.matvec(ops, Xh)
            rho = F - y
            Cbar = (F * Xh).sum(0); Ur = (Xh * rho).sum(0); J = Cbar + Ur; eact = (Xh * y).sum(0)
            z = lat._psolve(rho); rKr = (rho * z).sum(0)
            qh = lat.gather(Xh, 0); q0 = lat.gather(ref['U'].to(dev), 0)
            ub = opt.field(qh); Kub = Ct.K @ ub
            efld = (ub * Kub).sum(0); eact0 = (qh * opt.apply(qh)).sum(0)
            omega = eact0 - efld
            T0q0 = ex[0].apply(q0)
            a0 = efld - 2 * (qh * T0q0).sum(0) + (q0 * T0q0).sum(0)
            dq1 = lat.gather(Xh, 1) - lat.gather(ref['U'].to(dev), 1)
            a1 = (dq1 * ex[1].apply(dq1)).sum(0)
        C = torch.as_tensor(ref['compliance'], dtype=dt, device=dev)
        ins = dict(C=C, C_bar=Cbar, Ut_rho=Ur, J=J, e_act=eact, rho_Kinv_rho=rKr, bound_dual=torch.sqrt(eact * rKr.clamp_min(0)),
                   true_residual=rho.norm(dim=0) / F.norm(dim=0), omega=omega, a_err=a0 + a1, a_err_test=a0, a_err_nbr=a1,
                   err_C_bar=(C - Cbar) / C, err_J=(C - J) / C, Ut_rho_rel=Ur / C, bound_dual_rel=torch.sqrt(eact * rKr.clamp_min(0)) / C,
                   omega_rel=omega / C, identity_resid_rel=((C - Cbar) - (a0 + a1 + Ur + omega)) / C,
                   pcg_residual=res.get('pcg_residual'), pcg_iterations=res.get('pcg_iterations'))
        ins = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in ins.items()}
        s0 = np.asarray(ref['sens'][0]); s1 = np.asarray(res['sens'][0])
        rec['configs'][conf] = dict(gate=summ, loads=lat.labels, gate_mask=lat.gate.tolist(), instr=ins,
                                    sens_exact=s0, sens_tilde=s1, s_tilde_metrics=RC.grad_metrics(s0, s1),
                                    energy_share=(np.asarray(ref['energy']) / np.asarray(ref['compliance'])[None, :]),
                                    seconds=time.perf_counter() - t, ref_seconds=ref.get('seconds'))
        log(dict(event='CONFIG', case=a.case, config=conf, gate_compliance_max=summ['gate_compliance_max'],
                 gate_sens_max=summ['gate_sens_max'], err_C_bar=ins['err_C_bar'], Ut_rho_rel=ins['Ut_rho_rel'],
                 bound_dual_rel=ins['bound_dual_rel'], identity_resid_rel=ins['identity_resid_rel'], omega_rel=ins['omega_rel'],
                 true_residual=ins['true_residual'], gpu=RC.gpu_gb()))
        vec[f'{conf}_U_bar'] = Xh.cpu().numpy(); vec[f'{conf}_U'] = ref['U'].cpu().numpy(); vec[f'{conf}_rho'] = rho.cpu().numpy()
        vec[f'{conf}_q_hat'] = qh.cpu().numpy(); vec[f'{conf}_s'] = s0; vec[f'{conf}_s_tilde'] = s1
        Q.append(qh); U0.append(ub); KU0.append(Kub); s_ex.append(s0); s_ti.append(s1)
        cols += [(conf, j) for j in range(qh.shape[1])]
        if conf in resolve:
            keep_lat[conf] = (lat, ex, s0)
        else:
            del lat, ex
        del ref, res, y, rho, z
        RC.free()
        RC.dump(out / f'{tag}.json', rec)
    if a.no_deriv:
        np.savez(out / f'{tag}.npz', **vec)
        return rec
    # -------------------------------------------------------------------- E3: fixed-q_hat central differences
    Q = torch.cat(Q, 1); U0 = torch.cat(U0, 1); KU0 = torch.cat(KU0, 1)
    S_ex = np.concatenate(s_ex, 1); S_ti = np.concatenate(s_ti, 1)
    with torch.no_grad():
        e_base = (Q * opt.apply(Q)).sum(0)
        e_fld_base = (U0 * KU0).sum(0)
    rb = RC.Rebuilder(a.case, body, a.tmp, lambda geo: DC.net_for(ck, geo)[0], log=log)
    op0, info0 = rb.build(taus0)
    rb.set_reference(info0)
    e0, u0r = RC.energy_cols(op0, Q)
    repro = dict(e_act_rel=((e0 - e_base) / e_base).abs().max().item(), field_rel=((u0r - U0).norm() / U0.norm()).item(),
                 build_s=info0['seconds'], tail=info0['tail'], shift=info0['shift'])
    rb.release(op0); del op0, u0r; RC.free()
    log(dict(event='REPRO', case=a.case, **repro, gpu=RC.gpu_gb()))
    steps = [float(s) for s in a.steps.split(',') if s]
    fixb = [float(s) for s in a.fixb_steps.split(',') if s]
    corners = [int(c) for c in a.corners.split(',')]
    rsteps = [float(s) for s in a.resolve_steps.split(',') if s]
    rsel = {}
    for conf, spec in resolve.items():
        s0 = keep_lat[conf][2]
        g = np.linalg.norm(s0[:, np.asarray(rec['configs'][conf]['gate_mask'])], axis=1)
        if spec.startswith('auto'):
            k = int(spec[4:] or 2)
            o = np.argsort(-g)
            rsel[conf] = [int(o[0])] + ([int(o[len(o) // 2])] if k > 1 else [])
        else:
            rsel[conf] = [int(c) for c in spec.split('+')]
    D = {}                                                                       # key -> (8, K) arrays
    sw = []
    resolve_out = []
    for fb, hs in ((False, steps), (True, fixb)):
        for h in hs:
            key = f'{"fixb_" if fb else ""}{h:g}'
            Da, Df, De = np.full((8, Q.shape[1]), np.nan), np.full((8, Q.shape[1]), np.nan), np.full((8, Q.shape[1]), np.nan)
            for c in corners:
                hc = h * taus0[c]
                ea, ef, uu, rs = {}, {}, {}, {}
                for sg in (1, -1):
                    tp = list(taus0); tp[c] += sg * hc
                    op, info = rb.build(tp, fix_bounds=fb)
                    e, u = RC.energy_cols(op, Q)
                    ea[sg] = e
                    with torch.no_grad():
                        ef[sg] = (u * (rb.C.K @ u)).sum(0)
                    uu[sg] = u
                    s_ = rb.switches(info); s_.update(corner=c, step=h, sign=sg, fix_bounds=fb, build_s=info['seconds'])
                    sw.append(s_)
                    if not fb:
                        for conf, cl in rsel.items():
                            if c in cl and h in rsteps:
                                lat, ex, _ = keep_lat[conf]
                                with torch.no_grad():
                                    mv = lambda P: lat.matvec([op, ex[1]], P)
                                    X, it, rr = RC.pcg_fixed(mv, lat._psolve, lat.F, tol=1e-10, maxit=a.maxit)
                                    rho = lat.F - mv(X)
                                    rs[(conf, sg)] = dict(C_hat=(lat.F * X).sum(0).cpu().numpy(),
                                                          J=((lat.F * X).sum(0) + (X * rho).sum(0)).cpu().numpy(),
                                                          it=it, rec_res=rr, true_res=(rho.norm(dim=0) / lat.F.norm(dim=0)).cpu().numpy())
                    rb.release(op); del op
                with torch.no_grad():
                    Da[c] = (-(ea[1] - ea[-1]) / (2 * hc)).cpu().numpy()
                    Df[c] = (-(ef[1] - ef[-1]) / (2 * hc)).cpu().numpy()
                    Fc = (uu[1] - uu[-1]) / (2 * hc)
                    De[c] = S_ti[c] - 2 * (Fc * KU0).sum(0).cpu().numpy()
                del uu, Fc; RC.free()
                for conf in rsel:
                    if (conf, 1) in rs:
                        p, m = rs[(conf, 1)], rs[(conf, -1)]
                        jj = [j for j, (cf, _) in enumerate(cols) if cf == conf]
                        resolve_out.append(dict(config=conf, corner=c, step=h, hc=hc,
                                                dC_hat_resolve=(p['C_hat'] - m['C_hat']) / (2 * hc),
                                                dJ_resolve=(p['J'] - m['J']) / (2 * hc), D_act_fixed_q=Da[c, jj], D_fld_fixed_q=Df[c, jj],
                                                D_ext_fixed_q=De[c, jj], s_exact=S_ex[c, jj], s_tilde=S_ti[c, jj],
                                                pcg=[p['it'], m['it']], rec_res=[p['rec_res'], m['rec_res']],
                                                true_res_max=[float(p['true_res'].max()), float(m['true_res'].max())]))
                        log(dict(event='RESOLVE', config=conf, corner=c, step=h, dC=resolve_out[-1]['dC_hat_resolve'],
                                 dJ=resolve_out[-1]['dJ_resolve'], D_act=Da[c, jj]))
                log(dict(event='CORNER', case=a.case, key=key, corner=c, D_act=Da[c, :3], D_ext=De[c, :3], s=S_ex[c, :3],
                         s_tilde=S_ti[c, :3], builds=rb.n_builds, build_s_mean=rb.seconds / rb.n_builds, gpu=RC.gpu_gb()))
            D[f'D_act_{key}'], D[f'D_fld_{key}'], D[f'D_ext_{key}'] = Da, Df, De
            rec['deriv_partial'] = dict(done=key)
            RC.dump(out / f'{tag}.json', rec)
            np.savez(out / f'{tag}.npz', **vec, **D, S_ex=S_ex, S_tilde=S_ti, cols=np.asarray([f'{c}:{j}' for c, j in cols]))
    # -------------------------------------------------------------------- metrics
    per = {}
    for k, v in D.items():
        per[k] = {}
        for conf in rec['configs']:
            jj = [j for j, (cf, _) in enumerate(cols) if cf == conf]
            g = np.asarray(rec['configs'][conf]['gate_mask'])
            per[k][conf] = dict(all=RC.grad_metrics(S_ex[:, jj], v[:, jj]),
                                vs_tilde=RC.grad_metrics(S_ti[:, jj], v[:, jj]),
                                ext_over_norm=(np.linalg.norm(v[:, jj] - S_ti[:, jj], axis=0) / np.linalg.norm(S_ex[:, jj], axis=0)).tolist(),
                                gate_loads=g.tolist())
    step_study = {}
    for kind in ('act', 'fld', 'ext'):
        ks = [f'{h:g}' for h in steps]
        step_study[kind] = {f'{ks[i]}_vs_{ks[i + 1]}': (np.linalg.norm(D[f'D_{kind}_{ks[i]}'] - D[f'D_{kind}_{ks[i + 1]}'], axis=0)
                                                        / np.linalg.norm(S_ex, axis=0)).tolist() for i in range(len(ks) - 1)}
    rec.update(reproducibility=repro, deriv_metrics=per, step_study=step_study, switches=sw, resolve=resolve_out,
               rebuilds=dict(n=rb.n_builds, seconds=rb.seconds), cols=[f'{c}:{j}' for c, j in cols],
               e_base=e_base.cpu().numpy(), e_fld_base=e_fld_base.cpu().numpy(), t1=time.strftime('%F %T'))
    rec.pop('deriv_partial', None)
    RC.dump(out / f'{tag}.json', rec)
    np.savez(out / f'{tag}.npz', **vec, **D, S_ex=S_ex, S_tilde=S_ti, cols=np.asarray([f'{c}:{j}' for c, j in cols]))
    log(dict(event='DONE', case=a.case, rebuilds=rb.n_builds, seconds=rb.seconds))
    return rec


if __name__ == '__main__':
    main()
