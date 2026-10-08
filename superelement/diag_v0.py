"""Whitened residual spectrum and component ablation of a trained V0 checkpoint. Read-only."""
import sys, json, time, torch, numpy as np
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0'); sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
import v0_superelement as V
from stage_cutfem_neural_a.data import load_teacher_dense
from stage_cutfem_neural_a.dense_fit import FactorSample
from stage_cutfem_neural_a.elimination_reference import load_upper_factor, quotient_dense
torch.backends.cuda.matmul.allow_tf32 = False; DEV = V.DEV
run = sys.argv[1]; ck = sys.argv[2]
proto = json.load(open(f'{run}/PROTOCOL.json'))
records = json.load(open('/root/autodl-tmp/CUTFEM_NEURAL_DENSE_20260911_P01/INPUT_MANIFEST.json'))['samples']
record = next(r for r in records if int(r['seat']) == 415)
sample = FactorSample(record, 2026091197); data = sample.data.cuda()
cache = np.load(record['trace_cache']); N = 65
ijk = np.stack(np.unravel_index(cache['background_nodes'][cache['indices']], (N, N, N)), axis=1)
S, _ = load_teacher_dense(record['packet'], data.quotient.dimension, DEV); diagS = S.diagonal().clone()
A, _ = quotient_dense(S, data.quotient); del S; torch.cuda.empty_cache()
S_diag_cpu = diagS.cpu()
R = load_upper_factor(sample.reference / 'R_UPPER.npy', sample.n, DEV)
w = np.clip(V.support_weights(ijk, N, cache['tau_corners'], cache['cut_plane']), 1e-4, 1)
P = V.coarse_interpolation(ijk, N, stride=4); P3 = torch.from_numpy(np.kron(P, np.eye(3))).to(DEV)
xyz_face = torch.from_numpy(ijk / 64).to(DEV).double()
grid = np.stack(np.meshgrid(*[np.arange(1, 16)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
material = data.grid[0, 10].cpu().numpy() > 0.5
grid = grid[np.array([material[(c[0]-1)*4:(c[0]+1)*4+1, (c[1]-1)*4:(c[1]+1)*4+1, (c[2]-1)*4:(c[2]+1)*4+1].any() for c in grid])]
xyz_int = torch.from_numpy(grid * 4 / 64).to(DEV).double()
s_unit = float(A.diagonal().median())
if proto.get('form', 'interior') == 'interior':
    model = V.SuperElement(xyz_face, xyz_int, proto['r_near'], proto['rho'], proto['rank'], torch.from_numpy(w).double(), s_unit, proto['seed'])
else:
    log_diag = 0.5 * torch.log(S_diag_cpu) if proto.get('diag_init', 'teacher') == 'teacher' else 0.5 * torch.log(s_unit * torch.from_numpy(w).double().repeat_interleave(3))
    model = V.ContractionSuperElement(xyz_face, xyz_int, proto['r_near'], proto['rho'], proto['rank'], log_diag, s_unit, proto['seed'], off_scale=proto.get('off_scale', 0.3), gi_scale=proto.get('gi_scale', 0.1))
saved = torch.load(ck, map_location='cuda', weights_only=False); model.load_state_dict(saved['model']); print('loaded step', saved['step'], flush=True)
wq = torch.from_numpy(w).to(DEV).double().repeat_interleave(3)
gen = torch.Generator(device=DEV).manual_seed(proto['seed'] + 2)
zc = torch.randn(P3.shape[1], 32, dtype=torch.float64, device=DEV, generator=gen); xf = torch.randn(model.q, 32, dtype=torch.float64, device=DEV, generator=gen) * wq.sqrt()[:, None]
held = data.quotient(torch.cat((P3 @ zc, xf), 1))
def err(z, apply):
    ref = A @ z; pred = data.quotient(apply(data.quotient.lift(z)))
    return torch.linalg.solve_triangular(R.T, pred - ref, upper=False).square().sum(0) / (R @ z).square().sum(0)
with torch.no_grad():
    e = err(held, model.apply); print(f'held: coarse {float(e[:32].mean()):.4f} fine {float(e[32:].mean()):.4f}', flush=True)
    xi = torch.randn(A.shape[0], 32, dtype=torch.float64, device=DEV, generator=gen); white = torch.linalg.solve_triangular(R, xi, upper=True)
    e = err(white, model.apply); print(f'held white (uniform in energy): {float(e.mean()):.4f}', flush=True)
    if hasattr(model, 'Ucol'):
        Uc = model.Ucol(); sv = torch.linalg.svdvals(Uc); print('softening columns: singular values top5', [round(float(v), 3) for v in sv[:5]], 'count > 1 / > 0.1:', int((sv > 1).sum()), int((sv > 0.1).sum()))
        no_soft = lambda x: torch.sparse.mm(model.sparse_gg(transpose=True), torch.sparse.mm(model.sparse_gg(), x))
        e = err(held, no_soft); e2 = err(white, no_soft); print(f'ablate softening: coarse {float(e[:32].mean()):.4f} fine {float(e[32:].mean()):.4f} white {float(e2.mean()):.4f}', flush=True)
    print('gg_theta abs quantiles', [round(float(v), 3) for v in torch.quantile(model.gg_theta.abs()[::97], torch.tensor([.5, .9, .99, 1.0], device=DEV, dtype=torch.float64))], flush=True)
    Shat = model.materialize(); Ahat, _ = quotient_dense(Shat, data.quotient); del Shat
    D = Ahat - A; del Ahat
    X = torch.linalg.solve_triangular(R.T, D, upper=False); del D
    W = torch.linalg.solve_triangular(R, X, upper=True, left=False); del X; W = 0.5 * (W + W.T); torch.cuda.empty_cache()
    lam, U = torch.linalg.eigh(W); a = lam.abs()
    print('whitened residual: ||W||_2', float(a.max()), '||W||_F', float(torch.linalg.vector_norm(lam)), '#|lambda|>0.3/0.1/0.03/0.01:', [int((a > t).sum()) for t in (0.3, 0.1, 0.03, 0.01)], flush=True)
    s = a.sort(descending=True).values; cum = (s * s).cumsum(0) / (s * s).sum(); print('share of ||W||_F^2 in top 16/64/256/1024 modes', [round(float(cum[k-1]), 3) for k in (16, 64, 256, 1024)])
    idx = a.argsort(descending=True)[:12]; Xq = torch.linalg.solve_triangular(R, U[:, idx], upper=True); Y = data.quotient.lift(Xq); del U
    small = diagS < 1e-3; Q8, _ = torch.linalg.qr(P3)
    print('top residual modes: lambda | A-Rayleigh | participation | sliver share | n=8 capture')
    for m in range(12):
        y = Y[:, m]; y = y / torch.linalg.vector_norm(y); y2 = y * y; xq = Xq[:, m]
        print(f'  {float(lam[idx[m]]):+8.3f} | {float((xq @ (A @ xq)) / (xq @ xq)):.2e} | {float(1 / (y2 * y2).sum() / len(y)):.4f} | {float(y2[small].sum()):.2f} | {float(torch.linalg.vector_norm(Q8.T @ y))**2:.2f}')
print('peak GiB', torch.cuda.max_memory_allocated() / 2**30)
