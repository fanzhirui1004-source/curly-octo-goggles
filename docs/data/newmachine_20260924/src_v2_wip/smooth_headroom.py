"""Is the learned extension error high-frequency (removable by a few local equilibrium sweeps) or smooth / soft?
For a trained checkpoint and validation directions q: exact u = E q, learned u0 = geo.field(model, q); then k steps of a
fixed linear smoother on the interior equilibrium (ports held): Jacobi-preconditioned Chebyshev on [lmax / alpha, lmax]
(lmax of D^-1 K_II by power iteration), linear in q, energy-norm contractive. Reports per k the energy excess
e(u_k) / e(u) - 1 = ||u_k - u||_K^2 / ||u||_K^2 and the 8-corner sensitivity error, from the network start and from the
zero-interior start (what the smoother alone does). Usage: smooth_headroom.py <ckpt> <out.json> <case>[,...]
[--classes force_c,force] [--m 32] [--ks 0,1,2,4,8,16,32] [--alpha 30]     (OPL_DEV=cpu: exact solve by PARDISO/SuperLU)"""
import diag_sens as DS                                                   # first: CPU env (hides the GPU when OPL_DEV=cpu)
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD
import trainlib as TL

dev, dt = TL.dev, TL.dt


def kx_int(C, x):
    return (C.K @ x)[C.I]


def lmax_est(C, dinv, it=40):
    v = torch.randn((C.ni, 1), dtype=dt, device=dev); x = torch.zeros((C.nb, 1), dtype=dt, device=dev)
    lam = 0.0
    for _ in range(it):
        v = v / v.norm(); x[C.I] = v
        w = dinv * kx_int(C, x)
        lam = float((v * w).sum()); v = w
    return lam


def chebyshev(C, u0, dinv, lmin, lmax, ks):
    """Chebyshev semi-iteration on K_II x_I = -K_IP q from u0 (ports untouched); yields (k, u_k) for k in ks."""
    out = {}
    x = u0.clone()
    if 0 in ks:
        out[0] = x.clone()
    theta, delta = (lmax + lmin) / 2, (lmax - lmin) / 2
    sigma = theta / delta; rho = 1 / sigma
    r = -kx_int(C, x)
    d = dinv * r / theta
    for k in range(1, max(ks) + 1):
        x[C.I] += d
        if k in ks:
            out[k] = x.clone()
        r = -kx_int(C, x)
        rho_n = 1 / (2 * sigma - rho)
        d = rho_n * rho * d + (2 * rho_n / delta) * dinv * r
        rho = rho_n
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--classes', default='force_c,force'); ap.add_argument('--m', type=int, default=32)
    ap.add_argument('--ks', default='0,1,2,4,8,16,32'); ap.add_argument('--alpha', type=float, default=30.0)
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data_v2')
    a = ap.parse_args(argv)
    ks = sorted(int(k) for k in a.ks.split(','))
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    rec = dict(ckpt=a.ckpt, ks=ks, alpha=a.alpha, per_case={})
    model = None
    for case in a.cases.split(','):
        t0 = time.perf_counter()
        geo = TL.Geo(case, a.body, a.data, neumann=False, log=lambda s_: None)
        C = geo.C
        if not hasattr(C, 'dM'):
            C.dmoments()
        DS.cpu_factor(C) if dev.type == 'cpu' else C.factor(neumann=False)
        if model is None:
            model = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=dev.type != 'cpu')).to(dev)
            MD.load_compat(model, ck['model']); model.eval()
        else:
            model.add_geo(geo)
        dinv = (1 / C.dK[C.I])[:, None]
        lmax = 1.05 * lmax_est(C, dinv); lmin = lmax / a.alpha
        r = dict(lmax=lmax, interior=int(C.ni))
        for cls in [c for c in a.classes.split(',') if c in geo.classes]:
            Q = geo.banks['val'][cls][:, :a.m].to(dt)
            with torch.no_grad():
                u = C.extend(Q); uh = geo.field(model, Q).to(dt)
                z = torch.zeros_like(u); z[C.P] = Q
                eu = TL.energy(u, C.K); s0 = C.sens2(u)
                res = {}
                for start, x0 in (('net', uh), ('zero', z)):
                    seq = chebyshev(C, x0, dinv, lmin, lmax, ks)
                    rows = []
                    for k in ks:
                        xk = seq[k]
                        ex = (TL.energy(xk, C.K) / eu - 1).cpu().numpy()
                        row = dict(k=k, energy_mean=float(ex.mean()), energy_p90=float(np.quantile(ex, .9)))
                        if start == 'net' or k == max(ks):
                            sk = C.sens2(xk)
                            sr = ((sk - s0).norm(dim=0) / s0.norm(dim=0)).cpu().numpy()
                            row.update(sens_mean=float(sr.mean()), sens_p90=float(np.quantile(sr, .9)))
                        rows.append(row)
                    res[start] = rows
            r[cls] = res
            print(json.dumps(dict(case=case, cls=cls, net=[(x['k'], round(x['energy_mean'], 4), round(x.get('sens_mean', -1), 4)) for x in res['net']],
                                  zero_last=res['zero'][-1])), flush=True)
        r['seconds'] = time.perf_counter() - t0
        rec['per_case'][case] = r
        model.caches.pop(case, None); C._free(); del geo, C
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
