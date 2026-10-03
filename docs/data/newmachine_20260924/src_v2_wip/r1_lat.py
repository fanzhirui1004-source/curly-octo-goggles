"""Revision round 1, E4 (+ E3 with --deriv) on the heterogeneous lattices of Section 6.9 (gen_hlat layouts; every cell
learned). New script: the lattice, loads, exact reference and learned operators are built exactly as lat_hetero.py
(--exact-dense reference, one --model, optional --lean / --deploy / --park); lat_hetero.py is unchanged.

E4 instrumentation per lattice (all loads: three consistent face loads + n_random random loads):
  exact: U, C = f^T U, per-cell s (8 x loads), energies q^T S q; learned: U_bar by PCG (--prec, --tol) with snapshots at
  the recursive-residual levels --snaps; rho = f - y(U_bar) (y = the applied learned lattice action), U_bar^T rho,
  C_bar = f^T U_bar, J = C_bar + U_bar^T rho, e_act = U_bar^T y(U_bar); rho^T Kex^-1 rho by PCG with the EXACT dense
  lattice operator (the reference's cell matrices) and the dual-norm bound |U_bar^T rho| <= sqrt(e_act) sqrt(rho^T Kex^-1 rho)
  (valid because S_hat >= S); per cell omega_m = q_hat^T S_hat q_hat - u_bar^T K u_bar and a_m = u_bar^T K u_bar -
  2 q_hat^T S q + q^T S q (energy of u_bar - u), the identity C - C_bar = sum a_m + U_bar^T rho + sum omega_m; the same
  scalars at every snapshot; per-cell s_tilde, eps (energy error at the exact traces) and the bound beta.
  Global gradient metrics of s_tilde (and of the complete derivative with --deriv) against s over all cell-corner
  variables and over the shared lattice vertices (sum over the cells' corners at each vertex; loads held fixed).
E3 (--deriv): every cell rebuilt at tau +- h tau_c e_c (r1x3_common.Rebuilder) and applied to its solved traces of EVERY
  layout processed in this run that contains it (e.g. hlat222 with its two 2x2x1 layers hlat221a / hlat221b), no
  re-solve: D_act, D_fld, D_ext as in r1_pairs.py; rebuild at the unperturbed tau for reproducibility; switch indicators.
Outputs: <out>.json (metrics), <out>.npz (vectors: U, U_bar, rho, per-cell s, s_tilde, D_*).
Usage: r1_lat.py <out_prefix> <layout.json>[,...] --model A3=<ckpt> [--prec bnn:kpp:q1r] [--tol 1e-10] [--maxit 3000]
       [--n-random 3] [--body /root/autodl-tmp/OPL/S4/body] [--max-cols 16] [--lean | --deploy] [--park --resident k]
       [--snaps 1e-3,...,1e-9] [--deriv --steps 3e-3,1e-3,3e-4]
"""
import json, time, gc, os, argparse, contextlib
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('layouts'); ap.add_argument('--model', required=True)
ap.add_argument('--prec', default='bnn:kpp:q1r'); ap.add_argument('--tol', type=float, default=1e-10)
ap.add_argument('--maxit', type=int, default=3000); ap.add_argument('--n-random', type=int, default=3)
ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
ap.add_argument('--max-cols', type=int, default=16)
ap.add_argument('--lean', action='store_true'); ap.add_argument('--deploy', action='store_true')
ap.add_argument('--park', action='store_true'); ap.add_argument('--resident', type=int, default=0)
ap.add_argument('--snaps', default='1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9')
ap.add_argument('--dual-tol', type=float, default=1e-10)
ap.add_argument('--deriv', action='store_true'); ap.add_argument('--steps', default='3e-3,1e-3,3e-4')
ap.add_argument('--deriv-cases', default='', help='comma list (default: every case)')
ap.add_argument('--deriv-corners', default='0,1,2,3,4,5,6,7')
ap.add_argument('--tmp', default='/root/autodl-tmp/OPL/S1/V2/R1/X3/tmp')
A = ap.parse_args()
if A.deploy:                                                             # as lat_hetero.py --deploy (before any correction)
    os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'

import models as MD                                                     # noqa: E402,F401  first: applies OPL_CONV_FP32
import teacher as TE                                                    # noqa: E402
import trainlib as TL                                                   # noqa: E402
import fastnet as FN                                                    # noqa: E402
import evalnet as EN                                                    # noqa: E402
import bench_deploy as BD                                               # noqa: E402
import lat_multi as LM                                                  # noqa: E402
import lat_precond as PR                                                # noqa: E402
import lat_hetero as LH                                                 # noqa: E402
import r1x3_common as RC                                                  # noqa: E402

dev, dt = TE.dev, TE.dt
log = lambda d: print(json.dumps(RC.tojson(d), default=float), flush=True)


def ctx(op):
    return op.active() if A.park else contextlib.nullcontext()


def scal(lat, ops, X, comp):
    """Solve-quality scalars of an approximate solution X (one learned lattice action)."""
    F = lat.F
    with torch.no_grad():
        y = lat.matvec(ops, X)
        rho = F - y
        Cb = (F * X).sum(0); Ur = (X * rho).sum(0)
    C = torch.as_tensor(comp, dtype=dt, device=F.device)
    return dict(true_residual=(rho.norm(dim=0) / F.norm(dim=0)).cpu().numpy(), C_bar=Cb.cpu().numpy(),
                Ut_rho=Ur.cpu().numpy(), J=(Cb + Ur).cpu().numpy(), e_act=(X * y).sum(0).cpu().numpy(),
                err_C_bar=((C - Cb) / C).cpu().numpy(), err_J=((C - Cb - Ur) / C).cpu().numpy(),
                Ut_rho_rel=(Ur / C).cpu().numpy()), rho


def main():
    res = dict(args=vars(A), env=RC.env_flags(), lattices={}, t0=time.strftime('%F %T'))
    BD.TMP.mkdir(parents=True, exist_ok=True)
    snaps = [float(s) for s in A.snaps.split(',') if s]
    percase = {}
    vec = {}
    name, h = LH.holder_for(A.model)
    for lp in A.layouts.split(','):
        L = json.loads(Path(lp).read_text())
        LN = L['name']
        rec = dict(layout=LN, cells=[c['case'] for c in L['cells']], kinds=[c['kind'] for c in L['cells']])
        res['lattices'][LN] = rec
        t = time.perf_counter()
        Cs, lay = [], {}
        for c in L['cells']:
            C = TE.Cell(c['case'], A.body, log=lambda s_: None); C.assemble()
            Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
            if A.park:
                LH._move(C, torch.device('cpu')); RC.free()
        lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=A.n_random, device=dev,
                              max_cols=A.max_cols, log=lambda s_: None)
        order = [g.case for g in lat.geoms]
        Cmap = {C.case: C for C in Cs}
        lcell = {c['case']: c for c in L['cells']}
        rec.update(setup_s=time.perf_counter() - t, **{k: v for k, v in lat.info().items() if k not in ('cases',)})
        ncons = lat.F.shape[1] - A.n_random
        kpp = lat.assemble_kpp() if A.park else None
        # ---------------------------------------------------------------- exact reference (as lat_hetero --exact-dense)
        ops = [LH.DenseExactOp(Cmap[c], A.body) for c in order]
        X, st = LH.solve(lat, ops, A.prec, A.tol, A.maxit, kpp)
        comp = (lat.F * X).sum(0).cpu().numpy()
        ops = None; RC.free()
        S, E = [], []
        for i, c in enumerate(order):
            C = Cmap[c]; q = lat.gather(X, i)
            if A.park:
                LH._move(C, dev, min_bytes=0)
            LH.exact_op_host(C)
            u = C.extend(q.to(dt)); C.dmoments()
            S.append(C.sens(u).cpu()); E.append((u * (C.K @ u)).sum(0).cpu())
            del u
            C._free(); RC.free()
            if A.park:
                LH._move(C, torch.device('cpu')); RC.free()
        Xref = X.cpu()
        rec['exact'] = dict(st, compliance=comp, energy_share=(torch.stack(E) / torch.as_tensor(comp)).numpy())
        log(dict(event='EXACT', lattice=LN, **st, compliance=comp))
        for C in Cs:
            C._free()
        RC.free()
        # ---------------------------------------------------------------- learned (as lat_hetero, non-stream)
        t = time.perf_counter()
        lops = []
        if A.park:
            for C in Cs:
                LH._move(C, torch.device('cpu'))
            RC.free()
        for j, c in enumerate(order):
            C = Cmap[c]
            if A.park:
                LH._move(C, dev)
            BD.netdata(C, A.body, BD.TMP / c)
            geo = TL.Geo(c, A.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
            op = EN.FastOp(FN.FastNet(h.add(geo), geo))
            if A.deploy and not getattr(C, '_lean', False):
                C.lean(deploy=True); RC.free()
            elif A.lean and not getattr(C, '_lean', False):
                C.lean(); RC.free()
            if A.park:
                op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))
                op = LH.ParkedOp(op, C, resident=j < A.resident)
                if not op.resident:
                    LH._move(C, torch.device('cpu')); RC.free()
            lops.append(op)
        prep = time.perf_counter() - t
        fac = PR.Factory(lat, lops, shared={} if kpp is None else {'kpp_triplets': kpp, 'kpp_triplets_s': 0.0},
                         kpp_backend='auto', log=lambda d: None)
        pc, pst, _ = fac.build(A.prec)
        r = PR.pcg(lat, lops, pc, tol=A.tol, maxit=A.maxit, snaps=snaps)
        fac.free(); del pc
        Xh = r['X']
        final, rho = scal(lat, lops, Xh, comp)
        snap_rows = []
        for sn in r['snaps']:
            s_, _ = scal(lat, lops, sn['X'], comp)
            snap_rows.append(dict(level=sn['level'], iterations=sn['iterations'], **s_))
        log(dict(event='LEARNED_SOLVE', lattice=LN, iterations=r['iterations'], seconds=r['seconds'],
                 true_residual=final['true_residual'], err_C_bar=final['err_C_bar'][:ncons], Ut_rho_rel=final['Ut_rho_rel'][:ncons],
                 snaps=[(s_['level'], s_['iterations'], float(s_['true_residual'].max())) for s_ in snap_rows], gpu=RC.gpu_gb()))
        cells = []
        sh_all = []
        for i, c in enumerate(order):
            C = Cmap[c]
            with ctx(lops[i]):
                if A.park:
                    LH._move(C, dev, min_bytes=0)
                with torch.no_grad():
                    qh = lat.gather(Xh, i)
                    u = lops[i].field(qh).to(dt)
                    Ku = C.K @ u
                    efld = (u * Ku).sum(0); eact = (qh * lops[i].apply(qh)).sum(0)
                    sh = C.sens(u).cpu()
                    qe = lat.gather(Xref.to(dev), i)
                    eps = ((qe * lops[i].apply(qe)).sum(0).cpu() / E[i] - 1).numpy()
                sh_all.append(sh)
                cells.append(dict(case=c, e_fld=efld.cpu().numpy(), e_act=eact.cpu().numpy(), omega=(eact - efld).cpu().numpy(),
                                  eps=eps, sens_rel_err=((sh - S[i]).norm(dim=0) / S[i].norm(dim=0)).numpy()))
                pc_ = percase.setdefault(c, dict(Q=[], U0=[], KU0=[], S_ex=[], S_ti=[], e_base=[], cols=[], taus0=list(C.taus0)))
                pc_['Q'].append(qh.cpu()); pc_['U0'].append(u.cpu()); pc_['KU0'].append(Ku.cpu())
                pc_['S_ex'].append(S[i].numpy()); pc_['S_ti'].append(sh.numpy()); pc_['e_base'].append(eact.cpu())
                pc_['cols'] += [f'{LN}:{j}' for j in range(qh.shape[1])]
                vec[f'{LN}__{c}__q_hat'] = qh.cpu().numpy()
                del u, Ku
            if A.park and not lops[i].resident:
                LH._move(C, torch.device('cpu')); RC.free()
        Ssum = torch.stack(S).numpy(); Sti = torch.stack(sh_all).numpy()                    # cells x 8 x loads
        cerr = np.abs(final['C_bar'] - comp) / np.abs(comp)
        w = np.asarray(torch.stack(E)) / comp[None]
        epsm = np.stack([c_['eps'] for c_ in cells])
        serr = np.stack([c_['sens_rel_err'] for c_ in cells])
        learned = dict(iterations=r['iterations'], seconds=r['seconds'], residual=r['residual'], setup_s=pst['total_s'], prep_s=prep,
                       history=r['history'][::10], snaps=snap_rows, final=final, cells=cells,
                       compliance_rel_err=cerr, sens_rel_err=serr, eps=epsm, bound=(w * epsm).sum(0),
                       solution_rel_err=float((Xh.cpu() - Xref).norm() / Xref.norm()),
                       gate_compliance_max=float(cerr[:ncons].max()), gate_sens_max=float(serr[:, :ncons].max()))
        # free the learned operators (as lat_hetero)
        for c in order:
            h.model.caches.pop(c, None)
        del lops; RC.free()
        for C in Cs:
            for k in ('_cV', '_cL', '_c_space', '_tail_bounds', '_cL32', '_cAi32', '_cV32', '_cV32t', '_Kc'):
                if hasattr(C, k):
                    delattr(C, k)
        RC.free()
        # ---------------------------------------------------------------- exact dual solve and a-terms
        t = time.perf_counter()
        eops = [LH.DenseExactOp(Cmap[c], A.body) for c in order]
        fac = PR.Factory(lat, eops, shared={} if kpp is None else {'kpp_triplets': kpp, 'kpp_triplets_s': 0.0},
                         kpp_backend='auto', log=lambda d: None)
        pc, _, _ = fac.build(A.prec)
        rd = PR.pcg(lat, eops, pc, F=rho, tol=A.dual_tol, maxit=A.maxit)
        fac.free(); del pc
        Z = rd['X']
        rKr = (rho * Z).sum(0).cpu().numpy()
        a_m = []
        with torch.no_grad():
            for i, c in enumerate(order):
                qh = lat.gather(Xh, i); q = lat.gather(Xref.to(dev), i)
                Tq = eops[i].apply(q)
                a_m.append((cells[i]['e_fld'] - 2 * (qh * Tq).sum(0).cpu().numpy() + (q * Tq).sum(0).cpu().numpy()))
        a_m = np.stack(a_m); om = np.stack([c_['omega'] for c_ in cells])
        del eops, Z; RC.free()
        bd = np.sqrt(final['e_act'] * np.maximum(rKr, 0))
        learned['dual'] = dict(rho_Kinv_rho=rKr, bound_dual=bd, bound_dual_rel=bd / comp, pcg_iterations=rd['iterations'],
                               pcg_residual=rd['residual'], seconds=time.perf_counter() - t,
                               a_err=a_m.sum(0), a_err_per_cell=a_m, omega=om.sum(0), omega_rel=om.sum(0) / comp,
                               identity_resid_rel=((comp - final['C_bar']) - (a_m.sum(0) + final['Ut_rho'] + om.sum(0))) / comp)
        # ---------------------------------------------------------------- global gradient metrics of s_tilde
        pos = [tuple(lat.positions[i]) for i in range(len(order))]
        taus = [lcell[c]['tau_corners'] for c in order]
        vid, nv, tv, dmax, vkeys = RC.vertex_map(pos, taus)
        tmax = max(float(np.max(np.abs(np.asarray(taus[i]) - np.asarray(Cmap[c].taus0)))) for i, c in enumerate(order))
        learned['grad_s_tilde'] = dict(cellcorner=RC.grad_metrics(Ssum.reshape(-1, Ssum.shape[2]), Sti.reshape(-1, Sti.shape[2])),
                                       vertex=RC.grad_metrics(RC.aggregate(Ssum, vid, nv), RC.aggregate(Sti, vid, nv)),
                                       cell_norm_share=(np.linalg.norm(Ssum, axis=1) / np.linalg.norm(Ssum.reshape(-1, Ssum.shape[2]), axis=0)))
        rec['order'] = order
        rec['vertices'] = dict(n=nv, vid=vid, tau=tv, shared_tau_max_diff=dmax, layout_vs_packet_tau_max_diff=tmax, positions=pos)
        rec[name] = learned
        vec[f'{LN}__U'] = Xref.numpy(); vec[f'{LN}__U_bar'] = Xh.cpu().numpy(); vec[f'{LN}__rho'] = rho.cpu().numpy()
        vec[f'{LN}__S'] = Ssum; vec[f'{LN}__S_tilde'] = Sti; vec[f'{LN}__vid'] = vid
        log(dict(event='LATTICE_DONE', lattice=LN, gate_compliance_max=learned['gate_compliance_max'],
                 gate_sens_max=learned['gate_sens_max'], Ut_rho_rel=final['Ut_rho_rel'], bound_dual_rel=bd / comp,
                 identity_resid_rel=learned['dual']['identity_resid_rel'], grad_rel_vertex=learned['grad_s_tilde']['vertex']['rel']))
        for C in Cs:
            C._free()
        del Cs, Cmap, lat, X, Xh, rho; RC.free()
        RC.dump(A.out + '.json', res)
        np.savez(A.out + '.npz', **vec)
    # -------------------------------------------------------------------- E3
    if A.deriv:
        steps = [float(s) for s in A.steps.split(',') if s]
        sel = [c for c in A.deriv_cases.split(',') if c] or list(percase)
        h2 = LH.holder_for(A.model)[1]

        def model_fn(geo):
            if h2.model is not None:
                h2.model.caches.pop(geo.case, None)
            return h2.add(geo)
        res['deriv'] = {}
        for case in sel:
            pc_ = percase[case]
            Q = torch.cat(pc_['Q'], 1).to(dev); KU0 = torch.cat(pc_['KU0'], 1).to(dev); U0 = torch.cat(pc_['U0'], 1).to(dev)
            S_ex = np.concatenate(pc_['S_ex'], 1); S_ti = np.concatenate(pc_['S_ti'], 1); e_base = torch.cat(pc_['e_base']).to(dev)
            taus0 = pc_['taus0']
            rb = RC.Rebuilder(case, A.body, A.tmp, model_fn, log=log)
            op0, info0 = rb.build(taus0)
            rb.set_reference(info0)
            e0, u0r = RC.energy_cols(op0, Q)
            e0b, u0b = RC.energy_cols(op0, Q)
            repro = dict(e_act_rel=float(((e0 - e_base) / e_base).abs().max()), field_rel=float((u0r - U0).norm() / U0.norm()),
                         build_s=info0['seconds'], tail=info0['tail'], shift=info0['shift'],
                         determinism=dict(e_act_rel=float(((e0b - e0) / e0).abs().max()), field_rel=float((u0b - u0r).norm() / u0r.norm())))
            del u0b
            rb.release(op0); del op0, u0r, U0; RC.free()
            log(dict(event='REPRO', case=case, **repro, gpu=RC.gpu_gb()))
            D, sw = {}, []
            for hh in steps:
                key = f'{hh:g}'
                Da, Df, De = (np.full((8, Q.shape[1]), np.nan) for _ in range(3))
                for c in [int(x) for x in A.deriv_corners.split(',')]:
                    hc = hh * taus0[c]
                    ea, ef, uu = {}, {}, {}
                    for sg in (1, -1):
                        tp = list(taus0); tp[c] += sg * hc
                        op, info = rb.build(tp)
                        e, u = RC.energy_cols(op, Q)
                        with torch.no_grad():
                            ef[sg] = (u * (rb.C.K @ u)).sum(0)
                        ea[sg], uu[sg] = e, u
                        s_ = rb.switches(info); s_.update(corner=c, step=hh, sign=sg, build_s=info['seconds']); sw.append(s_)
                        rb.release(op); del op
                    with torch.no_grad():
                        Da[c] = (-(ea[1] - ea[-1]) / (2 * hc)).cpu().numpy()
                        Df[c] = (-(ef[1] - ef[-1]) / (2 * hc)).cpu().numpy()
                        De[c] = S_ti[c] - 2 * (((uu[1] - uu[-1]) / (2 * hc)) * KU0).sum(0).cpu().numpy()
                    del uu; RC.free()
                D[f'D_act_{key}'], D[f'D_fld_{key}'], D[f'D_ext_{key}'] = Da, Df, De
                log(dict(event='STEP', case=case, step=hh, D_act=Da[:, 0], D_ext=De[:, 0], s=S_ex[:, 0], s_tilde=S_ti[:, 0],
                         builds=rb.n_builds, build_s_mean=rb.seconds / rb.n_builds, gpu=RC.gpu_gb()))
            res['deriv'][case] = dict(reproducibility=repro, switches=sw, cols=pc_['cols'], rebuilds=dict(n=rb.n_builds, seconds=rb.seconds),
                                      per_cell={k: dict(all=RC.grad_metrics(S_ex, v), vs_tilde=RC.grad_metrics(S_ti, v)) for k, v in D.items()},
                                      s_tilde=RC.grad_metrics(S_ex, S_ti))
            for k, v in D.items():
                vec[f'D__{case}__{k}'] = v
            pc_['D'] = D
            del rb, Q, KU0; RC.free()
            RC.dump(A.out + '.json', res)
            np.savez(A.out + '.npz', **vec)
        # lattice-level metrics of the complete derivative
        for LN, rec in res['lattices'].items():
            order, vid, nv = rec['order'], np.asarray(rec['vertices']['vid']), rec['vertices']['n']
            if not all(c in res['deriv'] for c in order):
                continue
            out = {}
            for key in percase[order[0]]['D']:
                G = []
                for c in order:
                    jj = [j for j, col in enumerate(percase[c]['cols']) if col.startswith(LN + ':')]
                    G.append(percase[c]['D'][key][:, jj])
                G = np.stack(G)
                Sx = vec[f'{LN}__S']; St = vec[f'{LN}__S_tilde']
                out[key] = dict(cellcorner=RC.grad_metrics(Sx.reshape(-1, Sx.shape[2]), G.reshape(-1, G.shape[2])),
                                vertex=RC.grad_metrics(RC.aggregate(Sx, vid, nv), RC.aggregate(G, vid, nv)),
                                vertex_vs_tilde=RC.grad_metrics(RC.aggregate(St, vid, nv), RC.aggregate(G, vid, nv)))
                vec[f'{LN}__{key}'] = G
            rec['grad_complete'] = out
    res['t1'] = time.strftime('%F %T')
    RC.dump(A.out + '.json', res)
    np.savez(A.out + '.npz', **vec)
    log(dict(event='DONE'))


if __name__ == '__main__':
    main()
