#!/usr/bin/env python3
"""Step 1, third pass: a step size that respects the log barrier.

The comparison pass showed the column scaling is the only parameterization that descends, and that
the plain Newton direction blows up.  It blows up for a reason that is not about the direction:

    D(R_hat) = tr(R_hat A^-1 R_hat^T) - 2 sum log R_hat_ii + const

is a quadratic plus a log barrier on the diagonal.  The Newton step of the quadratic part alone is
Delta = -(1/2) grad A, and taking it whole walks the diagonal through zero.  Backtracking until the
diagonal stays positive and the value actually decreases is the standard fix, and it is what decides
whether this problem needs thousands of steps or tens.

Also runs the column-scaled Adam baseline for longer, to see whether its descent is flattening or
merely slow.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch


def load_packed(path, device):
    p = np.load(path, mmap_mode='r'); n = p.shape[0]
    d = int((math.isqrt(8*n+1)-1)//2)
    R = torch.zeros((d, d), dtype=torch.float64, device=device); off = 0
    for i in range(d):
        R[i, i:] = torch.from_numpy(np.array(p[off:off+d-i], dtype=np.float64)).to(device); off += d-i
    return R


class Problem:
    def __init__(self, Rstar):
        self.d = d = Rstar.shape[0]
        self.logdet_A = float(2.0*Rstar.diagonal().log().sum())
        X = torch.linalg.solve_triangular(Rstar, torch.eye(d, dtype=torch.float64, device=Rstar.device), upper=True)
        self.Ainv64 = (X @ X.T).contiguous(); del X
        self.diagAinv = self.Ainv64.diagonal().clone()
        self.Ainv = self.Ainv64.float(); self.A = (Rstar.T @ Rstar).float(); self.Rstar = Rstar
        self.mask = torch.triu(torch.ones(d, d, dtype=torch.bool, device=Rstar.device))
    def D64(self, R):
        R = R.double()
        return float((R @ self.Ainv64 * R).sum() - 2.0*R.diagonal().log().sum() + self.logdet_A - self.d)
    def D32(self, R):
        dg = R.diagonal()
        if not bool((dg > 0).all()): return float('inf')
        return float((R @ self.Ainv * R).sum() - 2.0*dg.log().sum()) + self.logdet_A - self.d
    def grad(self, R):
        G = 2.0*(R @ self.Ainv); G.diagonal().sub_(2.0/R.diagonal()); return G*self.mask


def newton(prob, R0, steps, log, shrink=0.5, max_back=40, c1=1e-4):
    R = R0.clone(); hist = []; t0 = time.time(); f = prob.D32(R)
    for s in range(steps):
        G = prob.grad(R)
        step = -(0.5*(G @ prob.A))*prob.mask                 # Newton step of the quadratic part
        slope = float((G*step).sum())                        # < 0 for a descent direction
        t, ok = 1.0, False
        for _ in range(max_back):
            Rt = R + t*step
            ft = prob.D32(Rt)
            if math.isfinite(ft) and ft <= f + c1*t*slope: ok = True; break
            t *= shrink
        if not ok:
            hist.append(dict(step=s, D_per_mode=prob.D64(R)/prob.d, t=0.0, seconds=time.time()-t0, stalled=True))
            log('newton', hist[-1]); break
        R, f = Rt, ft
        if s % 1 == 0 or s == steps-1:
            hist.append(dict(step=s+1, D_per_mode=prob.D64(R)/prob.d, t=t, seconds=time.time()-t0))
            if s < 12 or (s+1) % 10 == 0: log('newton', hist[-1])
    return R, hist


def colscale_adam(prob, R0, steps, lr, every, log):
    s_j = prob.diagAinv.rsqrt().float()
    Q = (R0/s_j[None, :]).clone().requires_grad_(True)
    opt = torch.optim.Adam([Q], lr=lr); hist = []; t0 = time.time()
    for s in range(steps+1):
        R = (Q*s_j[None, :])*prob.mask
        if s % every == 0:
            with torch.no_grad():
                hist.append(dict(step=s, D_per_mode=prob.D64(R)/prob.d, seconds=time.time()-t0)); log('colscale', hist[-1])
        if s == steps: break
        Rc = R.clone(); Rc.diagonal().clamp_(min=1e-10)
        loss = (Rc @ prob.Ainv * Rc).sum() - 2.0*Rc.diagonal().log().sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad(): return ((Q*s_j[None, :])*prob.mask).detach(), hist


def spectrum(prob, R):
    W = torch.linalg.solve_triangular(prob.Rstar, R.double(), upper=True, left=False)
    mu = torch.linalg.svdvals(W).square()
    return dict(mu_min=float(mu.min()), mu_max=float(mu.max()),
                below_09=int((mu < 0.9).sum()), above_11=int((mu > 1.1).sum()),
                inside_3pct=int(((mu > 0.97) & (mu < 1.03)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--newton-steps', type=int, default=60)
    ap.add_argument('--adam-steps', type=int, default=6000)
    a = ap.parse_args()
    dev = torch.device('cuda'); Rstar = load_packed(a.factor, dev); prob = Problem(Rstar); d = prob.d
    c = math.sqrt(d/float(prob.diagAinv.sum())); R0 = (c*torch.eye(d, dtype=torch.float64, device=dev)).float()
    def log(k, h): print('  %-9s step %5d  D/d %14.8e  t %-8.4g %7.1f s' % (k, h['step'], h['D_per_mode'], h.get('t', float('nan')), h['seconds']), flush=True)
    out = dict(schema='STEP1_NEWTON_V1', d=d, start_D_per_mode=prob.D64(R0)/d, runs={})
    print('d %d  start D/d %.6f' % (d, out['start_D_per_mode']), flush=True)

    print('\n== newton with backtracking', flush=True)
    Rn, hn = newton(prob, R0, a.newton_steps, log)
    out['runs']['newton'] = dict(history=hn, final_D_per_mode=hn[-1]['D_per_mode'], spectrum=spectrum(prob, Rn))
    print('  spectrum', json.dumps(out['runs']['newton']['spectrum']), flush=True)
    del Rn; torch.cuda.empty_cache()

    print('\n== colscale Adam, long budget', flush=True)
    Ra, ha = colscale_adam(prob, R0, a.adam_steps, 1e-3, max(1, a.adam_steps//12), log)
    out['runs']['colscale_long'] = dict(lr=1e-3, history=ha, final_D_per_mode=ha[-1]['D_per_mode'], spectrum=spectrum(prob, Ra))
    print('  spectrum', json.dumps(out['runs']['colscale_long']['spectrum']), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=1)); print('\nwritten', a.out)


if __name__ == '__main__': main()
