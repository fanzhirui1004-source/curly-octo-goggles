"""Do V1's output bounds admit the V0G solution? Compare V0G band coefficients with the V1 bound min(D_i,D_j)*exp(-d/0.03)*OFF_SCALE."""
import sys, json, torch, numpy as np
sys.path.insert(0, '/root/cutfem_neural_a_20260910/superelement_v0'); sys.path.insert(0, '/root/cutfem_neural_a_20260910/source_14301bc56')
import v0_superelement as V
DEV = V.DEV
ck = torch.load('/root/autodl-tmp/CUTFEM_SUPERELEMENT_20260912_V0G/CHECKPOINT_004000.pt', map_location='cuda', weights_only=False)
proto = ck['protocol']; sd = ck['model']
rec = next(r for r in json.load(open('/root/autodl-tmp/CUTFEM_NEURAL_DENSE_20260911_P01/INPUT_MANIFEST.json'))['samples'] if int(r['seat']) == 415)
cache = np.load(rec['trace_cache']); N = 65
ijk = np.stack(np.unravel_index(cache['background_nodes'][cache['indices']], (N, N, N)), axis=1)
xyz = torch.from_numpy(ijk / 64).to(DEV).double(); q = 3 * len(ijk)
ni, nj = V.pairs_within(xyz, xyz, proto['r_near'], upper=True); I, J = V.expand_dofs(ni, nj, upper=True)
order = torch.argsort(I * q + J); I, J = I[order], J[order]; diag = I == J
dist = torch.linalg.vector_norm(xyz[I // 3] - xyz[J // 3], dim=1)
s_unit = float(torch.exp(2 * sd['gg_logdiag']).median())   # not exactly V0's s_unit; use protocol-independent proxy
root = np.sqrt(0.0099)                                        # V0 s_unit was the median teacher diagonal (0.0099)
L_off = (proto['off_scale'] * root * torch.exp(-dist / 0.03))[~diag] * sd['gg_theta']
D = sd['gg_logdiag'].exp(); Dmin = torch.minimum(D[I[~diag]], D[J[~diag]]); decay = torch.exp(-dist[~diag] / 0.03)
for off_scale in (1.0, 3.0, 10.0):
    bound = off_scale * Dmin * decay; ratio = L_off.abs() / bound
    over = ratio > 1; e2 = L_off.square(); share = float(e2[over].sum() / e2.sum())
    print(f'OFF_SCALE={off_scale}: entries over bound {int(over.sum())} of {len(ratio)} ({100*float(over.float().mean()):.2f}%), share of off-diagonal L energy in over-bound entries {100*share:.2f}%, ratio quantiles 50/90/99/max: ' + ' '.join(f'{float(v):.3g}' for v in torch.quantile(ratio[::37], torch.tensor([.5, .9, .99, 1.0], device=DEV, dtype=torch.float64))), flush=True)
# by distance bin (OFF_SCALE=1)
bound = Dmin * decay; ratio = L_off.abs() / bound; d = dist[~diag]
edges = [0, 0.02, 0.05, 0.1, 0.15, 0.2]
print('distance bin | entries | over-bound % | energy share over-bound % | median ratio | 99% ratio')
for a, b in zip(edges[:-1], edges[1:]):
    m = (d >= a) & (d < b); r = ratio[m]; e2 = L_off[m].square()
    print(f'[{a},{b}) {int(m.sum()):9d} {100*float((r>1).float().mean()):7.2f} {100*float(e2[r>1].sum()/e2.sum()):7.2f} {float(r.median()):8.3g} {float(torch.quantile(r[::11], 0.99)):8.3g}')
# how is the V0G off-diagonal energy distributed by distance at all?
e2 = L_off.square()
print('share of V0G off-diagonal L energy by distance bin:', [f'[{a},{b}):{100*float(e2[(d>=a)&(d<b)].sum()/e2.sum()):.2f}%' for a, b in zip(edges[:-1], edges[1:])])
# alternative bound: geometric mean instead of min
Dgm = torch.sqrt(D[I[~diag]] * D[J[~diag]]); ratio_gm = L_off.abs() / (Dgm * decay)
print('geometric-mean bound (OFF_SCALE=1): over-bound entries %.2f%%, energy share %.2f%%' % (100*float((ratio_gm>1).float().mean()), 100*float(e2[ratio_gm>1].sum()/e2.sum())))
