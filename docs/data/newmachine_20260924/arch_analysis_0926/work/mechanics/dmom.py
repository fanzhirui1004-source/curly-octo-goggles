import sys; sys.path.insert(0, 'lib')
import numpy as np, torch, time
torch.set_num_threads(2)
import polyref_torch_fast as PT
case = sys.argv[1]
z = np.load('/tmp/claude-0/-home-user-curly-octo-goggles/76f53c89-e2a9-52bb-8817-9460fd780b76/scratchpad/fixdl/FIX/%s/NETDATA.npz' % case)
taus = list(z['taus']); out = []
mom = lambda t: PT.cell_moments(z['elem_cells'], 32, t, z['normal'], float(z['offset']), 4, device='cpu', levels=1)
for c in range(8):
    h = 1e-5 * taus[c]; tp = list(taus); tp[c] += h; tm = list(taus); tm[c] -= h
    out.append((mom(tp) - mom(tm)) / (2 * h)); print(c, time.strftime('%X'), flush=True)
np.save('dM_%s.npy' % case, np.stack(out))
