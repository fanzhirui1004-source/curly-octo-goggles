"""Where does the design-sensitivity error of a trained model come from? (step-2 checkpoint, exact fields from the teacher)

Per validation direction q of the chosen classes: exact u = E q (interior factor), learned u_hat = geo.field(model, q);
per element e and thickness corner c the sensitivity contribution s_ce(u) = -u_e^T A_ce u_e (A_ce = sum_m dM_cem Tm_m, the
same element derivative as teacher.sens2). The error D_ce = s_ce(u_hat) - s_ce(u) = -(2 e_e^T A_ce u_e + e_e^T A_ce e_e)
(e = u_hat - u) is summed per corner (the 8-vector the gate compares, relative error ||D|| / ||S|| per direction) and split
  - by element group: volume fraction vf = M0 / h^3 (< 0.1 / 0.1-0.5 / 0.5-0.999 / full), elements touching a port node,
    elements with a weak node, and for corner c the elements whose centre is within 0.5 of that corner (trilinear weight
    of c > 1/8) vs the rest;
  - into the first-order (2 e A u) and second-order (e A e) parts.
Group shares are ||sum_{e in g} D_ce||_(c, samples) / sum_g ||...|| (signed within the group, so cancellation inside a group
counts), plus the plain share of sum |D_ce|. Energy error e_hat - 1 of the same directions for reference.
Usage: diag_sens.py <ckpt> <out.json> <case>[,<case>...] [--classes force_c,force] [--m 32] [--body S0] [--data S2/data_v2]"""
import sys, json, time, argparse
import os
if os.environ.get('OPL_DEV') == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''                               # CPU run must never touch the training GPU
from pathlib import Path
import numpy as np
import torch
import models as MD                                                   # first: applies OPL_CONV_FP32
import trainlib as TL
if os.environ.get('OPL_DEV') == 'cpu':                                       # polyref defaults to device='cuda'
    import inspect, polyref_torch_fast as _PT
    TL.TE.sync = lambda: None
    for _f in vars(_PT).values():
        if inspect.isfunction(_f) and _f.__defaults__ and 'cuda' in _f.__defaults__:
            _f.__defaults__ = tuple('cpu' if d == 'cuda' else d for d in _f.__defaults__)

dev, dt = TL.dev, TL.dt


def cpu_factor(C):
    """OPL_DEV=cpu: interior factor K_II by scipy SuperLU (the teacher's SPDSolver is GPU-only); same scaling as Cell.factor."""
    import scipy.sparse as sp, scipy.sparse.linalg as sla
    C._free()
    pm = torch.zeros(C.nb, dtype=torch.bool); pm[C.P] = True
    new = torch.full((C.nb,), -1, dtype=torch.long); new[C.I] = torch.arange(C.ni)
    ru, cu = C.ru.long(), C.cu.long()
    sel = torch.nonzero(~pm[ru] & ~pm[cu]).squeeze(1)
    rA, cA, vA = new[ru[sel]], new[cu[sel]], C.vals[sel]
    sA = torch.zeros(C.ni, dtype=dt); sA[rA[rA == cA]] = 1 / torch.sqrt(vA[rA == cA])
    U = sp.csc_matrix(((vA * sA[rA] * sA[cA]).numpy(), (rA.numpy(), cA.numpy())), shape=(C.ni, C.ni))   # upper CSR storage
    A = (U + sp.triu(U, 1).T).tocsc()
    try:                                                                      # MKL PARDISO when importable (large FULL cells)
        import pypardiso
        ps = pypardiso.PyPardisoSolver(); A = A.tocsr(); ps.factorize(A)
        sol = lambda b: ps.solve(A, np.ascontiguousarray(b))
        rel = lambda: ps.free_memory(everything=True)                         # MKL keeps the factor until told
    except ImportError:
        lu = sla.splu(A, permc_spec='MMD_AT_PLUS_A', options=dict(SymmetricMode=True)); sol = lu.solve
        rel = lambda: None

    class _S:
        def solve(self, r):
            return torch.as_tensor(sol(r.numpy()), dtype=dt).reshape(r.shape)

        def free(self):
            rel()
    C.sA, C.sol_I, C.fp32 = sA, _S(), False


def groups_of(C, nd, corner_xyz):
    n = C.n
    h3 = (1.0 / n) ** 3
    vf = (C.moments(getattr(C, 'taus', C.taus0))[:, 0] / h3).clamp(0, 1)
    node_of = (C.dofs[:, ::3] // 3)                                           # E x 27 local node indices
    is_port = torch.zeros(C.nb // 3, dtype=torch.bool, device=dev); is_port[C.P[::3] // 3] = True
    weak = torch.as_tensor(nd['weak'], device=dev).bool()
    g = {
        'vf<0.1': vf < 0.1, 'vf0.1-0.5': (vf >= 0.1) & (vf < 0.5), 'vf0.5-0.999': (vf >= 0.5) & (vf < 0.999), 'vf_full': vf >= 0.999,
        'touch_port': is_port[node_of].any(1), 'touch_weak': weak[node_of].any(1),
    }
    cells = torch.as_tensor(C.cells, device=dev).to(torch.float64)
    ctr = (cells + 0.5) / n                                                   # element centres in the unit cell
    near = torch.stack([(ctr - torch.as_tensor(cz, device=dev, dtype=torch.float64)).abs().max(1).values < 0.5 for cz in corner_xyz])
    return g, near, vf


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--classes', default='force_c,force'); ap.add_argument('--m', type=int, default=32)
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data_v2')
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = ck['cfg']
    corners = [((i >> 2) & 1, (i >> 1) & 1, i & 1) for i in range(8)]       # corner index = 4x + 2y + z
    rec = dict(ckpt=a.ckpt, classes=a.classes, m=a.m, per_case={})
    model = None
    for case in a.cases.split(','):
        t0 = time.perf_counter()
        geo = TL.Geo(case, a.body, a.data, neumann=False, log=lambda s_: None)
        C = geo.C
        if not hasattr(C, 'dM'):
            C.dmoments()
        cpu_factor(C) if dev.type == 'cpu' else C.factor(neumann=False)
        if model is None:
            model = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=dev.type != 'cpu')).to(dev)
            MD.load_compat(model, ck['model']); model.eval()
        else:
            model.add_geo(geo)
        grp, near, vf = groups_of(C, geo.nd, corners)
        out = {}
        for cls in [c for c in a.classes.split(',') if c in geo.classes]:
            Q = geo.banks['val'][cls][:, :a.m].to(dt)
            with torch.no_grad():
                u = C.extend(Q)
                uh = geo.field(model, Q).to(dt)
            e = uh - u
            eh = (TL.energy(uh, C.K) / TL.energy(u, C.K) - 1).cpu().numpy()
            E = len(C.cells); k = Q.shape[1]
            S = torch.zeros((8, k), dtype=dt, device=dev)
            Dg = {gname: torch.zeros((8, k), dtype=dt, device=dev) for gname in grp}
            Dnear = torch.zeros((8, k), dtype=dt, device=dev); Dfar = torch.zeros((8, k), dtype=dt, device=dev)
            D1 = torch.zeros((8, k), dtype=dt, device=dev); D2 = torch.zeros((8, k), dtype=dt, device=dev)
            absD = {gname: 0.0 for gname in grp}; absT = 0.0
            for lo in range(0, E, 512):
                hi = min(E, lo + 512)
                A = torch.einsum('cem,mij->ceij', C.dM[:, lo:hi], C.Tm)            # 8 x c x 81 x 81
                ue, ee = u[C.dofs[lo:hi]], e[C.dofs[lo:hi]]                       # c x 81 x k
                Au = torch.einsum('ceij,ejk->ceik', A, ue); Ae = torch.einsum('ceij,ejk->ceik', A, ee)
                s = -(Au * ue[None]).sum(2)                                        # 8 x c x k
                d1 = -2 * (Au * ee[None]).sum(2); d2 = -(Ae * ee[None]).sum(2)
                d = d1 + d2
                S += s.sum(1); D1 += d1.sum(1); D2 += d2.sum(1)
                for gname, msk in grp.items():
                    mk = msk[lo:hi]
                    Dg[gname] += d[:, mk].sum(1); absD[gname] += float(d[:, mk].abs().sum())
                nm = near[:, lo:hi]                                                # 8 x c
                Dnear += (d * nm[:, :, None]).sum(1); Dfar += (d * (~nm)[:, :, None]).sum(1)
                absT += float(d.abs().sum())
                del A, Au, Ae, s, d1, d2, d
            D = D1 + D2
            rel = (D.norm(dim=0) / S.norm(dim=0)).cpu().numpy()
            gn = {g_: float(v.norm()) for g_, v in Dg.items()}
            out[cls] = dict(
                sens_rel_mean=float(rel.mean()), sens_rel_p90=float(np.quantile(rel, .9)), energy_excess_mean=float(eh.mean()),
                first_order_share=float(D1.norm() / (D1.norm() + D2.norm())),
                group_signed_norm_share={g_: gn[g_] / max(sum(gn[x] for x in ('vf<0.1', 'vf0.1-0.5', 'vf0.5-0.999', 'vf_full')), 1e-300)
                                         for g_ in gn},
                group_abs_share={g_: absD[g_] / max(absT, 1e-300) for g_ in absD},
                element_fraction={g_: float(m_.float().mean()) for g_, m_ in grp.items()},
                near_corner_share=float(Dnear.norm() / (Dnear.norm() + Dfar.norm())),
                per_corner_rel=(D.norm(dim=1) / S.norm(dim=1)).cpu().numpy().tolist())
            print(json.dumps(dict(case=case, cls=cls, sens_rel=round(out[cls]['sens_rel_mean'], 4), energy=round(out[cls]['energy_excess_mean'], 4),
                                  first_order=round(out[cls]['first_order_share'], 3),
                                  signed={g_: round(v, 3) for g_, v in out[cls]['group_signed_norm_share'].items()},
                                  near_corner=round(out[cls]['near_corner_share'], 3))), flush=True)
        rec['per_case'][case] = dict(out, seconds=time.perf_counter() - t0, elements=int(len(C.cells)))
        model.caches.pop(case, None)
        C._free(); del geo, C
        import gc; gc.collect()
        if dev.type != 'cpu': torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
