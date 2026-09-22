"""Octree polyhedral reference on GPU: time and per-element Loewner deviation against exact 125 moments."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import element_moments as EM, polyref_torch as PT

ROOT = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921')
T0 = Path('/root/autodl-tmp/CLAUDE_TAKEOVER_20260923')
ap = argparse.ArgumentParser()
ap.add_argument('--case', required=True); ap.add_argument('--configs', nargs='+', default=['4:0', '4:1', '4:2'])
a = ap.parse_args()
ctx = json.loads((ROOT / 'packets' / a.case / 'FRESH_CONTEXT.json').read_text())
n = int(ctx['n']); E = float(ctx['material']['E']); nu = float(ctx['material']['nu'])
lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu_ = E / (2 * (1 + nu))
taus = [float(v) for v in ctx['case']['tau_corners']]
normal = None if ctx['case'].get('normal') is None else [float(v) for v in ctx['case']['normal']]
offset = None if normal is None else float(ctx['case']['offset'])
truth = np.load(T0 / 'ELEMENT_MOMENTS_01' / a.case / 'MOMENTS125.npz')
Mt, cells, keys = truth['moments'], truth['cells'], truth['orderings']
Tm = EM.pattern_operators(keys[0].reshape(27, 3), lam, mu_, n)[1]
Tt = torch.tensor(Tm, dtype=torch.float64, device='cuda').reshape(125, -1)
K = (torch.tensor(Mt, dtype=torch.float64, device='cuda') @ Tt).reshape(-1, 81, 81)
ev, U = torch.linalg.eigh(K)
keep = ev > ev[:, -1:] * 1e-12
Wh = U / torch.sqrt(ev.clamp_min(1e-300))[:, None, :] * keep[:, None, :]
out = T0 / 'POLYREF_01' / a.case; out.mkdir(parents=True, exist_ok=True)
vf = Mt[:, 0] * n ** 3
PT.cell_moments(cells[:32], n, taus, normal, offset, 4)
for cfg in a.configs:
    s, L = map(int, cfg.split(':'))
    torch.cuda.synchronize(); t = time.perf_counter()
    M = PT.cell_moments(cells, n, taus, normal, offset, s, levels=L)
    torch.cuda.synchronize(); sec = time.perf_counter() - t
    Kr = (torch.tensor(M, dtype=torch.float64, device='cuda') @ Tt).reshape(-1, 81, 81)
    G = Wh.transpose(1, 2) @ (Kr - K) @ Wh
    dev = torch.linalg.eigvalsh(G).abs().max(1).values.cpu().numpy()
    large = vf >= 1e-3
    rec = dict(case=a.case, s=s, levels=L, effective=s * 2 ** L, seconds=sec, elements=int(len(cells)),
               dev_median=float(np.median(dev)), dev_q90=float(np.quantile(dev, .9)), dev_q99=float(np.quantile(dev, .99)),
               dev_max=float(dev.max()), within_3pct=float(np.mean(dev <= .03)),
               large_dev_max=float(dev[large].max()), large_within_1pct=float(np.mean(dev[large] <= .01)))
    np.savez(out / f'POLYREF_TORCH_S{s}_L{L}.npz', moments=M, loewner_dev=dev)
    print(json.dumps(rec), flush=True)
