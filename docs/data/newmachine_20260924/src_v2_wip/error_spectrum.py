"""Spectral location of the learned extension error. Lowest nev eigenpairs of the pencil K_II v = lam D v (D = diag K_II,
shift-invert Lanczos with the exact interior factor), D-orthonormal. For validation directions q: error e_I = (u_hat - u)_I
and exact interior field u_I, expanded e = sum c_i v_i + rest; reports the share of the error energy ||e||_K^2 that lies
in the lowest m modes (m = 1, 2, 5, 10, 20, 50, 100, ..., nev) and the same for the exact field u (how soft the signal
is), plus the eigenvalues. Energy of the interior error is e_I^T K_II e_I (ports are exact, so this is the whole excess).
Usage: error_spectrum.py <ckpt> <out.json> <case>[,...] [--classes force_c,force] [--m 32] [--nev 200]
(OPL_DEV=cpu: exact factor by PARDISO/SuperLU, GPU hidden)"""
import diag_sens as DS                                                   # first: CPU env
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
import models as MD
import trainlib as TL

dev, dt = TL.dev, TL.dt


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--classes', default='force_c,force'); ap.add_argument('--m', type=int, default=32)
    ap.add_argument('--nev', type=int, default=200)
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S0'); ap.add_argument('--data', default='/root/autodl-tmp/OPL/S2/data_v2')
    a = ap.parse_args(argv)
    import scipy.sparse.linalg as sla
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    rec = dict(ckpt=a.ckpt, nev=a.nev, per_case={})
    model = None
    for case in a.cases.split(','):
        t0 = time.perf_counter()
        geo = TL.Geo(case, a.body, a.data, neumann=False, log=lambda s_: None)
        C = geo.C
        DS.cpu_factor(C) if dev.type == 'cpu' else C.factor(neumann=False)
        if model is None:
            model = MD.build(cfg['model'], [geo], **dict(cfg.get('model_args', {}), sparse=dev.type != 'cpu')).to(dev)
            MD.load_compat(model, ck['model']); model.eval()
        else:
            model.add_geo(geo)
        ni = C.ni
        sA = C.sA.cpu().numpy()                                           # D^-1/2 (Jacobi scaling of the factor)
        xbuf = torch.zeros((C.nb, 1), dtype=dt, device=dev)

        def kii(v):                                                       # scaled operator A = D^-1/2 K_II D^-1/2
            xbuf.zero_(); xbuf[C.I, 0] = torch.as_tensor(sA * v.ravel(), dtype=dt, device=dev)
            return sA * (C.K @ xbuf)[C.I, 0].cpu().numpy()

        def ainv(v):                                                      # A^-1 through the scaled factor
            return C.sol_I.solve(torch.as_tensor(v.reshape(-1, 1), dtype=dt, device=dev)).cpu().numpy().ravel()
        A = sla.LinearOperator((ni, ni), matvec=kii, dtype=np.float64)
        Ai = sla.LinearOperator((ni, ni), matvec=ainv, dtype=np.float64)
        t1 = time.perf_counter()
        lam, W = sla.eigsh(A, k=a.nev, sigma=0.0, which='LM', OPinv=Ai, tol=1e-8)
        o = np.argsort(lam); lam, W = lam[o], W[:, o]                     # W orthonormal: v_i = D^-1/2 w_i
        r = dict(interior=int(ni), eig_seconds=time.perf_counter() - t1, lam=lam.tolist())
        ms = [m for m in (1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500) if m <= a.nev]
        for cls in [c for c in a.classes.split(',') if c in geo.classes]:
            Q = geo.banks['val'][cls][:, :a.m].to(dt)
            with torch.no_grad():
                u = C.extend(Q); uh = geo.field(model, Q).to(dt)
            eI = ((uh - u)[C.I].cpu().numpy()) / sA[:, None]              # y = D^1/2 e: ||e||_K^2 = y^T A y
            uI = (u[C.I].cpu().numpy()) / sA[:, None]
            out = {}
            for name, Y in (('error', eI), ('field', uI)):
                c = W.T @ Y                                               # coefficients in the A-eigenbasis
                tot = np.einsum('ik,ik->k', Y, np.stack([kii(Y[:, j]) for j in range(Y.shape[1])], 1))
                cum = np.cumsum(lam[:, None] * c ** 2, 0) / tot[None, :]
                out[name] = {str(m): dict(mean=float(cum[m - 1].mean()), p10=float(np.quantile(cum[m - 1], .1)),
                                          p90=float(np.quantile(cum[m - 1], .9))) for m in ms}
            r[cls] = out
            print(json.dumps(dict(case=case, cls=cls, lam_min=float(lam[0]), lam_nev=float(lam[-1]),
                                  error_share={m: round(out['error'][m]['mean'], 3) for m in out['error']},
                                  field_share={m: round(out['field'][m]['mean'], 3) for m in out['field']})), flush=True)
        r['seconds'] = time.perf_counter() - t0
        rec['per_case'][case] = r
        model.caches.pop(case, None); C._free(); del geo, C
        Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
