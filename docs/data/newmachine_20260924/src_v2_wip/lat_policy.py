"""P2 speed, solver policies on a lattice and a perturbed copy (one design step), exact cell operators from the cached
dense condensations (make_T_gpu / make_T_cpu): iterations and accuracy of the PCG solution at relative-residual levels
(tolerance study), for cold starts on both lattices and a warm start on the second from the first's converged solution
(matched by absolute grid position and component; DOFs without a match start at zero). Consistent face loads only.
Accuracy per level: compliance error against the converged solution and the energy-norm solution error
||X - X*||_A / ||X*||_A (first-order proxy of the field and sensitivity error caused by stopping early).
Usage: lat_policy.py <out.json> <layoutA.json> <layoutB.json> [--body S4/body] [--prec bnn:kpp:q1r] [--tol 1e-10]
       [--levels 1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]"""
import json, time, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # noqa: F401
import teacher as TE
import lat_multi as LM
import lat_precond as PR
import lat_hetero as LH

dev, dt = TE.dev, TE.dt


def build(layout, body):
    L = json.loads(Path(layout).read_text())
    Cs, lay = [], {}
    for c in L['cells']:
        C = TE.Cell(c['case'], body, log=lambda s_: None); C.assemble()
        lay[tuple(c['position'])] = LM.from_teacher(C)
        C.lean(); Cs.append(C); LH.free()
    lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=0, device=dev,
                          max_cols=64, log=lambda s_: None)
    Cmap = {C.case: C for C in Cs}
    ops = [LH.DenseExactOp(Cmap[g.case], body) for g in lat.geoms]
    kpp = lat.assemble_kpp()
    return L['name'], lat, ops, kpp, Cs


def keys(lat):
    f = lat.free
    return {(int(a), int(b), int(c), int(d)): i for i, (a, b, c, d) in
            enumerate(np.concatenate([lat.gpos[f], lat.gcomp[f][:, None]], 1).tolist())}


def run(lat, ops, kpp, spec, tol, levels, X0=None):
    fac = PR.Factory(lat, ops, shared={'kpp_triplets': kpp, 'kpp_triplets_s': 0.0}, kpp_backend='auto', log=lambda d: None)
    pc, st, _ = fac.build(spec)
    r = PR.pcg(lat, ops, pc, tol=tol, maxit=5000, X0=X0, snaps=levels)
    fac.free()
    return r, st


def accuracy(lat, ops, X, Xs, comp_s, AXs_norm):
    F = lat.F
    comp = (F * X).sum(0)
    D = X - Xs
    AD = lat.matvec(ops, D)
    return dict(compliance_rel_max=float(((comp - comp_s).abs() / comp_s.abs()).max()),
                solution_A_rel_max=float(((D * AD).sum(0).clamp_min(0).sqrt() / AXs_norm).max()))


def study(lat, ops, kpp, spec, tol, levels, X0=None):
    r, st = run(lat, ops, kpp, spec, tol, levels, X0)
    Xs = r['X']; comp_s = (lat.F * Xs).sum(0); AXs = (Xs * lat.matvec(ops, Xs)).sum(0).sqrt()
    rows = [dict(level=s['level'], iterations=s['iterations'], **accuracy(lat, ops, s['X'], Xs, comp_s, AXs)) for s in r['snaps']]
    return dict(iterations=r['iterations'], seconds=r['seconds'], setup_s=st['total_s'], residual=r['residual'],
                start_residual=r['history'][0] if r['history'] else None, levels=rows), Xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('A'); ap.add_argument('B')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body'); ap.add_argument('--prec', default='bnn:kpp:q1r')
    ap.add_argument('--tol', type=float, default=1e-10); ap.add_argument('--levels', default='1e-2,3e-3,1e-3,1e-4,1e-5,1e-6,1e-8')
    a = ap.parse_args()
    levels = [float(x) for x in a.levels.split(',')]
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    res = {}
    nameA, latA, opsA, kppA, CsA = build(a.A, a.body)
    res['A_cold'], XA = study(latA, opsA, kppA, a.prec, a.tol, levels); log(dict(run='A_cold', lattice=nameA, **res['A_cold']))
    kA = keys(latA); XA = XA.cpu()
    del opsA, latA, kppA, CsA; LH.free()
    nameB, latB, opsB, kppB, CsB = build(a.B, a.body)
    res['B_cold'], XB = study(latB, opsB, kppB, a.prec, a.tol, levels); log(dict(run='B_cold', lattice=nameB, **res['B_cold']))
    kB = keys(latB)
    X0 = torch.zeros_like(XB)
    hit = [(iB, kA[k]) for k, iB in kB.items() if k in kA]
    ib, ia = map(np.asarray, zip(*hit))
    X0[torch.as_tensor(ib, device=dev)] = XA[torch.as_tensor(ia)].to(dev)
    res['matched_fraction'] = len(hit) / len(kB)
    # unmatched DOFs (newly active nodes): value of the nearest matched DOF of the same component
    from scipy.spatial import cKDTree
    fB = latB.free; posB = latB.gpos[fB].astype(float); compB = latB.gcomp[fB]
    miss = np.setdiff1d(np.arange(len(fB)), ib)
    for c in range(3):
        mc = miss[compB[miss] == c]
        hc = ib[compB[ib] == c]
        if len(mc) and len(hc):
            _, j = cKDTree(posB[hc]).query(posB[mc])
            X0[torch.as_tensor(mc, device=dev)] = X0[torch.as_tensor(hc[j], device=dev)]
    res['filled_fraction'] = len(miss) / len(kB)
    comp_s = (latB.F * XB).sum(0); AXs = (XB * latB.matvec(opsB, XB)).sum(0).sqrt()
    res['warm_start_initial'] = accuracy(latB, opsB, X0, XB, comp_s, AXs)
    res['B_warm'], _ = study(latB, opsB, kppB, a.prec, a.tol, levels, X0=X0); log(dict(run='B_warm', lattice=nameB, matched=res['matched_fraction'], initial=res['warm_start_initial'], **res['B_warm']))
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
