"""P1 item 9: heterogeneous lattices (gen_hlat.py layouts) with EVERY cell learned, against the exact lattice.
Lattice: lat_multi.MultiLattice, clamped on y = min, traction-consistent unit loads x / y / z on y = max (+ n_random random
loads); PCG with one deployable preconditioner (--prec, lat_precond.Factory) to --tol on all loads.
  exact     bench_deploy.SparseExactOp per cell (GPU interior factor fp32 + fp64 refinement): reference solution U, compliance,
            per-cell fields E_c q_c, 8-corner sensitivities s_c = -u_c^T K_c,tau u_c and energies q_c^T S_c q_c
  learned   one run per --model (name=ckpt[;json model_args override]): every cell through fastnet (FastOp); compliance error
            per load, per-cell sensitivity error ||s_hat - s|| / ||s|| from the recovered learned fields at the learned
            solution, eps_c = q_c^T (S_hat_c - S_c) q_c / q_c^T S_c q_c at the exact traces and the bound sum_c w_c eps_c,
            relative solution distance, PCG iterations / seconds, recomputed residual
Gate-style summary on the consistent loads: max compliance error, max over cells of the sensitivity error.
Usage: lat_hetero.py <out.json> <layout.json>[,...] --model A3=<ckpt> [--model 'B2grid=<ckpt>;{"smooth_k": 8, ...}']
       [--prec bnn:kpp:q1r] [--tol 1e-10] [--maxit 3000] [--n-random 3] [--body /root/autodl-tmp/OPL/S4/body]"""
import json, time, gc, os, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD
import lat_multi as LM
import lat_precond as PR

dev, dt = TE.dev, TE.dt


def holder_for(spec):
    name, rest = spec.split('=', 1)
    ck, ov = (rest.split(';', 1) + [None])[:2]
    h = BD.ModelHolder(ck)
    if ov:
        h.ck['cfg'] = dict(h.ck['cfg'], model_args=dict(h.ck['cfg'].get('model_args', {}), **json.loads(ov)))
    return name, h


def free():
    gc.collect(); torch.cuda.empty_cache()


def solve(lat, ops, spec, tol, maxit):
    fac = PR.Factory(lat, ops, shared={}, kpp_backend='auto', log=lambda d: None)
    pc, st, _ = fac.build(spec)
    r = PR.solve(lat, ops, pc, tol=tol, maxit=maxit)
    out = dict(iterations=r['iterations'], seconds=r['seconds'], residual=r['residual'], setup_s=st['total_s'],
               true_residual=PR.true_residual(lat, ops, r['X']))
    X = r['X']
    fac.free(); del pc, r
    return X, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layouts'); ap.add_argument('--model', action='append', default=[])
    ap.add_argument('--prec', default='bnn:kpp:q1r'); ap.add_argument('--tol', type=float, default=1e-10)
    ap.add_argument('--maxit', type=int, default=3000); ap.add_argument('--n-random', type=int, default=3)
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    res = dict(args=vars(a), gpu=torch.cuda.get_device_name(0), conv=TL.conv_precision(), lattices={})
    BD.TMP.mkdir(parents=True, exist_ok=True)
    for lp in a.layouts.split(','):
        L = json.loads(Path(lp).read_text())
        rec = dict(layout=L['name'], cells=[c['case'] for c in L['cells']], kinds=[c['kind'] for c in L['cells']])
        res['lattices'][L['name']] = rec
        t = time.perf_counter()
        Cs, lay = [], {}
        for c in L['cells']:
            C = TE.Cell(c['case'], a.body, log=lambda s_: None); C.assemble()
            Cs.append(C); lay[tuple(c['position'])] = LM.from_teacher(C)
        lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=a.n_random, device=dev,
                              max_cols=64, log=lambda s_: None)
        order = [lay_g.case for lay_g in lat.geoms]
        Cmap = {C.case: C for C in Cs}
        rec.update(setup_s=time.perf_counter() - t, **{k: v for k, v in lat.info().items() if k not in ('positions', 'cases')})
        ncons = lat.F.shape[1] - a.n_random
        # exact reference
        ops = [BD.SparseExactOp(Cmap[c]) for c in order]
        X, st = solve(lat, ops, a.prec, a.tol, a.maxit)
        comp = (lat.F * X).sum(0).cpu().numpy()
        S, E = [], []
        for i, c in enumerate(order):
            C = Cmap[c]; q = lat.gather(X, i)
            u = C.extend(q.to(dt)); C.dmoments()
            S.append(C.sens(u).cpu()); E.append((u * (C.K @ u)).sum(0).cpu())
            del u
        rec['exact'] = dict(st, compliance=comp.tolist(), energy_share=(torch.stack(E) / torch.as_tensor(comp)).tolist())
        log(dict(event='EXACT', lattice=L['name'], **st, compliance=comp.tolist()))
        Xref = X.cpu()
        for C in Cs:
            C._free()
        del ops; free()
        for spec in a.model:
            name, h = holder_for(spec)
            t = time.perf_counter()
            ops = []
            for c in order:
                C = Cmap[c]
                BD.netdata(C, a.body, BD.TMP / c)
                geo = TL.Geo(c, a.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
                ops.append(EN.FastOp(FN.FastNet(h.add(geo), geo)))
            prep = time.perf_counter() - t
            X, st = solve(lat, ops, a.prec, a.tol, a.maxit)
            ch = (lat.F * X).sum(0).cpu().numpy()
            cerr = np.abs(ch - comp) / np.abs(comp)
            serr, eps = [], []
            for i, c in enumerate(order):
                C = Cmap[c]
                u = ops[i].field(lat.gather(X, i)).to(dt)
                sh = C.sens(u).cpu(); del u
                serr.append(((sh - S[i]).norm(dim=0) / S[i].norm(dim=0)).numpy())
                qe = lat.gather(Xref.to(dev), i)
                eps.append(((qe * ops[i].apply(qe)).sum(0).cpu() / E[i] - 1).numpy())
            serr, eps = np.stack(serr), np.stack(eps)
            w = np.asarray(torch.stack(E)) / comp[None]
            row = dict(st, prep_s=prep, compliance_rel_err=cerr.tolist(), sens_rel_err=serr.tolist(), eps=eps.tolist(),
                       bound=(w * eps).sum(0).tolist(),
                       solution_rel_err=float((X.cpu() - Xref).norm() / Xref.norm()),
                       gate_compliance_max=float(cerr[:ncons].max()), gate_sens_max=float(serr[:, :ncons].max()),
                       gate_sens_max_per_cell=serr[:, :ncons].max(1).tolist(), random_compliance_max=float(cerr[ncons:].max()) if a.n_random else None)
            rec[name] = row
            log(dict(event='LEARNED', lattice=L['name'], model=name, gate_compliance_max=row['gate_compliance_max'],
                     gate_sens_max=row['gate_sens_max'], iterations=st['iterations'], seconds=st['seconds'], prep_s=prep,
                     true_residual=st['true_residual']))
            for c in order:
                h.model.caches.pop(c, None)
            del ops, h; free()
            Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        for C in Cs:
            C._free()
        del Cs, Cmap, lat; free()
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
