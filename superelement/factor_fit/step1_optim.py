#!/usr/bin/env python3
"""Step 1, optimisation pass: which parameterization can actually descend the free-factor objective?

The landscape pass established that the objective is clean (D(R*) = 0 exactly), that every natural
initialisation starts at D/d ~ 4-7, and that the raw parameterization imposes a curvature spread of
about 5e3 whose column dependence is exactly 2 (A^-1)_jj.  That last fact is the lever: scaling
column j by 1/sqrt((A^-1)_jj) equalises the quadratic curvature exactly, and preconditioning the
gradient on the right by A inverts it exactly.

    D(R_hat)   = tr(R_hat A^-1 R_hat^T) - 2 sum log R_hat_ii + logdet A - d
    grad       = 2 R_hat A^-1 - 2 diag(1/R_hat_ii)          (upper triangle only)

Four runs on the same budget and the same start, so the difference is the parameterization alone.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch


def load_packed(path, device, dtype=torch.float64):
    p = np.load(path, mmap_mode='r'); n = p.shape[0]
    d = int((math.isqrt(8*n+1)-1)//2)
    if d*(d+1)//2 != n: raise ValueError('PACKED_TRIANGLE_SHAPE')
    R = torch.zeros((d, d), dtype=dtype, device=device); off = 0
    for i in range(d):
        R[i, i:] = torch.from_numpy(np.array(p[off:off+d-i], dtype=np.float64)).to(device=device, dtype=dtype)
        off += d-i
    return R


class Problem:
    def __init__(self, Rstar):
        self.d = Rstar.shape[0]
        self.logdet_A = float(2.0*Rstar.diagonal().log().sum())
        I = torch.eye(self.d, dtype=torch.float64, device=Rstar.device)
        X = torch.linalg.solve_triangular(Rstar, I, upper=True)      # R*^-1
        self.Ainv64 = (X @ X.T).contiguous(); del X, I
        self.diagAinv = self.Ainv64.diagonal().clone()
        self.Ainv = self.Ainv64.float()
        self.A = (Rstar.T @ Rstar).float()
        self.Rstar = Rstar
        self.mask = torch.triu(torch.ones(self.d, self.d, dtype=torch.bool, device=Rstar.device))

    def D64(self, Rhat):
        R = Rhat.double()
        tr = (R @ self.Ainv64 * R).sum()
        return float(tr - 2.0*R.diagonal().log().sum() + self.logdet_A - self.d)

    def grad32(self, Rhat):
        G = 2.0*(Rhat @ self.Ainv)
        G.diagonal().sub_(2.0/Rhat.diagonal())
        return G*self.mask


def run(prob, kind, R0, steps, lr, every, log):
    d = prob.d; t0 = time.time(); hist = []
    if kind == 'precond_A':                                   # right-preconditioned: G A inverts the quadratic Hessian
        R = R0.clone()
        for s in range(steps+1):
            if s % every == 0: hist.append((s, prob.D64(R), time.time()-t0)); log(kind, hist[-1])
            if s == steps: break
            step = (prob.grad32(R) @ prob.A)*prob.mask
            R = R - lr*step
            R.diagonal().clamp_(min=1e-12)
        return R, hist
    if kind == 'raw':          P = R0.clone().requires_grad_(True); pack = lambda: P
    elif kind == 'logdiag':
        off = (R0*torch.triu(torch.ones_like(R0), 1)).clone().requires_grad_(True)
        u = R0.diagonal().log().clone().requires_grad_(True)
        P = [off, u]; pack = lambda: off*torch.triu(torch.ones_like(off), 1) + torch.diag(u.exp())
    elif kind == 'colscale':                                   # column j scaled so the curvature is 2 everywhere
        s_j = prob.diagAinv.rsqrt().float()
        Q = (R0/s_j[None, :]).clone().requires_grad_(True)
        P = Q; pack = lambda: Q*s_j[None, :]
    opt = torch.optim.Adam(P if isinstance(P, list) else [P], lr=lr)
    for s in range(steps+1):
        R = pack()*prob.mask
        if s % every == 0:
            with torch.no_grad(): hist.append((s, prob.D64(R), time.time()-t0)); log(kind, hist[-1])
        if s == steps: break
        Rc = R.clone(); Rc.diagonal().clamp_(min=1e-10)
        tr = (Rc @ prob.Ainv * Rc).sum()
        loss = tr - 2.0*Rc.diagonal().log().sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad(): return (pack()*prob.mask).detach(), hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=1000); ap.add_argument('--every', type=int, default=50)
    ap.add_argument('--spectrum', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda')
    Rstar = load_packed(a.factor, dev); prob = Problem(Rstar)
    d = prob.d; print('d %d  logdet_A %.6f' % (d, prob.logdet_A), flush=True)
    cstar = math.sqrt(d/float(prob.diagAinv.sum()))
    R0 = (cstar*torch.eye(d, dtype=torch.float64, device=dev)).float()
    print('start: c* I, D/d = %.6f' % (prob.D64(R0)/d), flush=True)

    def log(kind, h): print('  %-11s step %5d  D/d %14.8e  %7.1f s' % (kind, h[0], h[1]/d, h[2]), flush=True)
    out = dict(schema='STEP1_OPTIM_V1', d=d, start='c_star_I', start_D_per_mode=prob.D64(R0)/d, steps=a.steps, runs={})
    for kind, lr in (('raw', 1e-3), ('logdiag', 1e-3), ('colscale', 1e-3), ('precond_A', 0.5)):
        print('\n== %s (lr %.3g)' % (kind, lr), flush=True)
        try:
            R, hist = run(prob, kind, R0, a.steps, lr, a.every, log)
            rec = dict(lr=lr, history=[dict(step=s, D_per_mode=D/d, seconds=t) for s, D, t in hist],
                       final_D_per_mode=hist[-1][1]/d)
            if a.spectrum:
                W = torch.linalg.solve_triangular(prob.Rstar, R.double(), upper=True, left=False)
                sv = torch.linalg.svdvals(W); mu = (sv*sv)
                rec['mu_min'] = float(mu.min()); rec['mu_max'] = float(mu.max())
                rec['below_0.9'] = int((mu < 0.9).sum()); rec['above_1.1'] = int((mu > 1.1).sum())
                print('  spectrum mu in [%.6e, %.6e]  below0.9 %d  above1.1 %d' % (rec['mu_min'], rec['mu_max'], rec['below_0.9'], rec['above_1.1']), flush=True)
                del W, sv, mu
            out['runs'][kind] = rec; del R
        except Exception as exc:
            out['runs'][kind] = dict(lr=lr, error=str(exc)); print('  FAILED', exc, flush=True)
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=1)); print('\nwritten', a.out)


if __name__ == '__main__': main()
