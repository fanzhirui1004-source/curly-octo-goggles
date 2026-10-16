"""Chebyshev upper endpoint check: the operational endpoint (trainlib.tail_bounds: 40 power steps on D^-1 K_II, x1.05)
against lambda_max of D^-1 K_II from a converged Lanczos run (full reorthogonalisation, symmetric form
D^-1/2 K_II D^-1/2), with the Ritz residual, and the (strict, looser) Gershgorin bound max_i sum_j |K_ij| / K_ii over
interior rows. Usage: lam_check.py <out.json> <body> <case>[,...] [--steps 150]"""
import sys, json, time, argparse
import numpy as np
import torch
import models as MD                                                    # noqa: F401
import teacher as TE
import trainlib as TL

dev, dt = TE.dev, TE.dt


def lanczos_max(C, steps):
    I = C.I; d = C.dK[I]; s = 1 / torch.sqrt(d)
    x = torch.zeros((C.nb, 1), dtype=dt, device=dev)

    def M(v):
        x.zero_(); x[I, 0] = s * v
        return s * (C.K @ x)[I, 0]
    g = torch.Generator(device=dev); g.manual_seed(1)
    q = torch.randn(len(I), dtype=dt, device=dev, generator=g); q /= q.norm()
    Q = [q]; al, be = [], []
    b = 0.0; qp = torch.zeros_like(q)
    for j in range(steps):
        w = M(Q[-1]) - b * qp
        a = float(Q[-1] @ w); w = w - a * Q[-1]
        for qq in Q:                                                    # full reorthogonalisation (twice)
            w = w - (qq @ w) * qq
        for qq in Q:
            w = w - (qq @ w) * qq
        al.append(a); b = float(w.norm()); be.append(b)
        if b < 1e-14:
            break
        qp = Q[-1]; Q.append(w / b)
    m = len(al)
    T = np.diag(al) + np.diag(be[:m - 1], 1) + np.diag(be[:m - 1], -1)
    th, Y = np.linalg.eigh(T)
    return float(th[-1]), float(abs(be[m - 1] * Y[-1, -1])), m


def gershgorin(C):
    ru, cu, v = C.ru.long(), C.cu.long(), C.vals.abs()
    inner = torch.zeros(C.nb, dtype=torch.bool, device=dev); inner[C.I] = True
    keep = inner[ru] & inner[cu]
    rs = torch.zeros(C.nb, dtype=dt, device=dev)
    rs.index_add_(0, ru[keep], v[keep]); off = keep & (ru != cu)
    rs.index_add_(0, cu[off], v[off])
    return float((rs[C.I] / C.dK[C.I]).max())


def main(argv):
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('body'); ap.add_argument('cases')
    ap.add_argument('--steps', type=int, default=150); a = ap.parse_args(argv)
    rec = []
    for case in a.cases.split(','):
        t = time.perf_counter()
        C = TE.Cell(case, a.body, log=lambda s_: None); C.assemble()
        _, op = TL.tail_bounds(C, 30.0)
        lam, res, m = lanczos_max(C, a.steps)
        r = dict(case=case, interior=int(C.ni), op_lmax=op, lanczos_lmax=lam, ritz_residual=res, lanczos_steps=m,
                 margin=op / lam - 1, gershgorin=gershgorin(C), seconds=time.perf_counter() - t)
        r['gershgorin_over_op'] = r['gershgorin'] / op
        print(json.dumps(r), flush=True); rec.append(r)
        del C; torch.cuda.empty_cache()
        json.dump(rec, open(a.out, 'w'), indent=1)
    mg = [r['margin'] for r in rec]
    print(json.dumps(dict(cells=len(rec), margin_min=min(mg), margin_max=max(mg), all_positive=all(x > 0 for x in mg),
                          ritz_residual_rel_max=max(r['ritz_residual'] / r['lanczos_lmax'] for r in rec))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
