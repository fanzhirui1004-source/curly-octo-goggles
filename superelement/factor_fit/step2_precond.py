#!/usr/bin/env python3
"""Step 2's prerequisite: can the column scaling be obtained without the teacher?

Step 1 established that the free factor fits, and that the only parameterization that descends
scales column j by 1/sqrt((A^-1)_jj).  A network does not have A at prediction time, so that scale
has to come from somewhere else.  Two sources are possible and they are complementary:

  self      the student's own diag(A_hat^-1), refreshed every K steps.  Free, but at a flat start
            it carries no information, so this measures whether it can bootstrap at all.
  warm      the same refresh, started from the teacher's scale.  Isolates "does refreshing work"
            from "can it start cold", so a failure of `self` can be attributed correctly.

`teacher` is the step-1 reference.  All three share the start, the optimizer and the budget.
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
        self.d = d = Rstar.shape[0]; self.dev = Rstar.device
        self.logdet_A = float(2.0*Rstar.diagonal().log().sum())
        X = torch.linalg.solve_triangular(Rstar, torch.eye(d, dtype=torch.float64, device=self.dev), upper=True)
        self.Ainv64 = (X @ X.T).contiguous(); del X
        self.diagAinv = self.Ainv64.diagonal().clone()
        self.Ainv = self.Ainv64.float(); self.Rstar = Rstar
        self.mask = torch.triu(torch.ones(d, d, dtype=torch.bool, device=self.dev))
    def D64(self, R):
        R = R.double()
        return float((R @ self.Ainv64 * R).sum() - 2.0*R.diagonal().log().sum() + self.logdet_A - self.d)


@torch.no_grad()
def student_diag_inv(R):
    """diag(A_hat^-1) = squared row norms of R_hat^-1, for the current student."""
    d = R.shape[0]
    X = torch.linalg.solve_triangular(R.double(), torch.eye(d, dtype=torch.float64, device=R.device), upper=True)
    return X.square().sum(dim=1)


def run(prob, source, R0, steps, lr, every, refresh, log):
    d = prob.d
    if source == 'teacher': s = prob.diagAinv.rsqrt().float()
    elif source == 'warm':  s = prob.diagAinv.rsqrt().float()
    else:                   s = torch.ones(d, dtype=torch.float32, device=prob.dev)
    Q = (R0/s[None, :]).clone().requires_grad_(True)
    opt = torch.optim.Adam([Q], lr=lr); hist = []; t0 = time.time()
    for step in range(steps+1):
        R = (Q*s[None, :])*prob.mask
        if step % every == 0:
            with torch.no_grad():
                corr = float(torch.corrcoef(torch.stack((s.double().log(), prob.diagAinv.rsqrt().log())))[0, 1])
                hist.append(dict(step=step, D_per_mode=prob.D64(R)/d, scale_logcorr=corr, seconds=time.time()-t0))
                log(source, hist[-1])
        if step == steps: break
        if source != 'teacher' and refresh and step > 0 and step % refresh == 0:
            with torch.no_grad():
                new = student_diag_inv(R.detach()).rsqrt().float()
                if bool(torch.isfinite(new).all()) and float(new.min()) > 0:
                    r = (new/s)[None, :]                            # dL/dQ scales with the column scale
                    Q.data.div_(r)                                  # same R, new coordinates
                    st = opt.state.get(Q)
                    if st:                                          # carry Adam's moments into the new coordinates
                        st['exp_avg'].mul_(r); st['exp_avg_sq'].mul_(r*r)
                    s = new
        Rc = R.clone(); Rc.diagonal().clamp_(min=1e-10)
        loss = (Rc @ prob.Ainv * Rc).sum() - 2.0*Rc.diagonal().log().sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad(): return ((Q*s[None, :])*prob.mask).detach(), hist, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--factor', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--every', type=int, default=250)
    ap.add_argument('--refresh', type=int, default=100); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--sources', default='teacher,warm,self')
    a = ap.parse_args()
    dev = torch.device('cuda'); prob = Problem(load_packed(a.factor, dev)); d = prob.d
    c = math.sqrt(d/float(prob.diagAinv.sum())); R0 = (c*torch.eye(d, dtype=torch.float64, device=dev)).float()
    ds = prob.diagAinv.rsqrt()
    print('d %d  start D/d %.6f' % (d, prob.D64(R0)/d), flush=True)
    print('teacher scale 1/sqrt((A^-1)_jj): min %.4e max %.4e spread %.3e'
          % (float(ds.min()), float(ds.max()), float(ds.max()/ds.min())), flush=True)
    def log(k, h): print('  %-8s step %5d  D/d %13.7e  log-corr with teacher scale %+.4f  %6.1f s'
                         % (k, h['step'], h['D_per_mode'], h['scale_logcorr'], h['seconds']), flush=True)
    out = dict(schema='STEP2_PRECOND_SOURCE_V1', d=d, lr=a.lr, steps=a.steps, refresh=a.refresh,
               teacher_scale_spread=float(ds.max()/ds.min()), runs={})
    for source in tuple(a.sources.split(',')):
        print('\n== %s' % source, flush=True)
        R, hist, s = run(prob, source, R0, a.steps, a.lr, a.every, a.refresh, log)
        out['runs'][source] = dict(history=hist, final_D_per_mode=hist[-1]['D_per_mode'],
                                   final_scale_logcorr=hist[-1]['scale_logcorr'],
                                   final_scale_spread=float(s.max()/s.min()))
        del R; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=1)); print('\nwritten', a.out)


if __name__ == '__main__': main()
