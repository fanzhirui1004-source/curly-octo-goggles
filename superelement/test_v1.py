import sys, json, time, torch, numpy as np
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0')
import v1_superelement as V1
t0 = time.time()
rec = next(r for r in json.load(open('/root/autodl-tmp/CUTFEM_NEURAL_DENSE_20260911_P01/INPUT_MANIFEST.json'))['samples'] if int(r['seat']) == 415)
rec = dict(rec, split='train')
lab = V1.Label(rec, 0.2, 0.03, 0.3)
print('label built', round(time.time() - t0, 1), 's; q', lab.q, 'pairs', len(lab.pair_i), 'band entries', lab.band_index.shape[1], 'images', tuple(lab.images.shape), 'memberships', len(lab.mem_node), flush=True)
idx = lab.band_index; assert bool((idx[0] * lab.q + idx[1])[1:].gt((idx[0] * lab.q + idx[1])[:-1]).all()), 'band index not sorted'
assert bool((idx[1] >= idx[0]).all()), 'not upper'
g = dict(images=lab.images.to(V1.DEV), theta=lab.theta.to(V1.DEV), mem_node=lab.mem_node.to(V1.DEV), mem_face=lab.mem_face.to(V1.DEV), mem_pix=lab.mem_pix.to(V1.DEV),
         face_of_node=lab.face_of_node.to(V1.DEV), pair_i=lab.pair_i.to(V1.DEV), pair_j=lab.pair_j.to(V1.DEV), pair_dx=lab.pair_dx.to(V1.DEV), pair_scale=lab.pair_scale.to(V1.DEV),
         band_index=lab.band_index.to(V1.DEV), band_index_t=lab.band_index_t.to(V1.DEV), band_perm=lab.band_perm.to(V1.DEV), w=torch.from_numpy(lab.w).double().to(V1.DEV))
net = V1.GeometryNet(lab.images.shape[1]).to(V1.DEV); print('params', sum(p.numel() for p in net.parameters()))
torch.cuda.synchronize(); t1 = time.time()
values, M = net(g, g['w'].log()); torch.cuda.synchronize(); print('forward', round(time.time() - t1, 3), 's; values', tuple(values.shape), 'finite', bool(torch.isfinite(values).all()), 'M', tuple(M.shape))
# diagonal entries should be exp(log sqrt(S0 w)) at init (head output ~0)
diag_pos = (idx[0] == idx[1]).to(V1.DEV); dv = values[diag_pos]; expect = (V1.S0 * g['w'].repeat_interleave(3)).sqrt()
print('diag rel dev from prior', float(((dv / expect) - 1).abs().max()))
op = V1.Operator(g, values, M); x = torch.randn(lab.q, 4, dtype=torch.float64, device=V1.DEV)
y = op.apply(x); print('apply ok', tuple(y.shape), 'sym check', float((x[:, :1].T @ op.apply(x[:, 1:2]) - x[:, 1:2].T @ op.apply(x[:, :1])).abs().max()))
loss = (y.square().sum()); loss.backward(); print('backward ok; grad norms', {n: round(float(p.grad.norm()), 4) for n, p in net.named_parameters() if p.grad is not None and n.endswith('weight')} and 'present')
print('peak GiB', torch.cuda.max_memory_allocated() / 2**30, 'total s', round(time.time() - t0, 1))
