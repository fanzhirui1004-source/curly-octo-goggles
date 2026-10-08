import sys, json, time, torch
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_superelement as V1
DEV = torch.device('cuda')
def T(): torch.cuda.synchronize(); return time.perf_counter()
# 1. gradcheck small
torch.manual_seed(0); q = 40; I, J = torch.triu_indices(q, q); keep = (J - I) <= 5; I, J = I[keep].cuda(), J[keep].cuda()
vals = (torch.randn(len(I), dtype=torch.float64, device='cuda') * 0.1); vals[I == J] = 1.0 + torch.rand(int((I == J).sum()), dtype=torch.float64, device='cuda'); vals.requires_grad_(True)
B = torch.randn(q, 3, dtype=torch.float64, device='cuda', requires_grad=True); idx = torch.stack((I, J)); crow = torch.zeros(q + 1, dtype=torch.int64, device='cuda'); crow[1:] = torch.bincount(I, minlength=q).cumsum(0)
for mode in ('dense', 'sparse'):
    for up in (True, False):
        print(f'gradcheck {mode} upper={up}:', torch.autograd.gradcheck(lambda v, b: V1.BandTriSolve.apply(v, b, idx, crow, q, up, mode), (vals, B), eps=1e-6, atol=1e-6, rtol=1e-5, nondet_tol=1e-9), flush=True)
# 2. real band: dense vs sparse forward/backward agreement and timing
rec = next(r for r in json.load(open('/root/autodl-tmp/CUTFEM_NEURAL_DENSE_20260911_P01/INPUT_MANIFEST.json'))['samples'] if int(r['seat']) == 415)
lab = V1.Label(dict(rec, split='train'), 0.2, 0.03, 0.3); idx = lab.band_index.to(DEV); crow = lab.band_crow.to(DEV); q = lab.q
I, J = idx[0], idx[1]; d = torch.linalg.vector_norm(lab.xyz_nodes.to(DEV).double()[I // 3] - lab.xyz_nodes.to(DEV).double()[J // 3], dim=1)
vals = 0.03 * torch.exp(-d / 0.03) * torch.randn(len(I), dtype=torch.float64, device=DEV); vals[I == J] = 0.1 + 0.05 * torch.rand(int((I == J).sum()), dtype=torch.float64, device=DEV)
B = torch.randn(q, 22, dtype=torch.float64, device=DEV)
out = {}
for mode in ('dense', 'sparse'):
    v = vals.clone().requires_grad_(True); torch.cuda.reset_peak_memory_stats(); t = T()
    X = V1.BandTriSolve.apply(v, B, idx, crow, q, False, mode); Y = V1.BandTriSolve.apply(v, X, idx, crow, q, True, mode); (Y * B).sum().backward()
    el = T() - t; out[mode] = (Y.detach(), v.grad.clone()); print(f'{mode}: fwd+bwd two solves {el*1000:.0f} ms, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)
print('sparse vs dense: X rel diff', float((out['sparse'][0] - out['dense'][0]).abs().max() / out['dense'][0].abs().max()), 'grad rel diff', float((out['sparse'][1] - out['dense'][1]).abs().max() / out['dense'][1].abs().max()))
