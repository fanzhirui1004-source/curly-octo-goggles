"""fast_tet vs moments_ad._quad: forward values, vertex gradients (reverse mode) and times."""
import json, time
import torch, numpy as np
import os
import moments_ad as MA, fast_tet as FT, fast_tet2 as FT2
BLK, WP = int(os.environ.get('B2', 128)), int(os.environ.get('W2', 4))
FT.tet_moments = lambda t, a, b: FT2.tet_moments2(t, a, b, BLK, WP)
FT.tet_moments_vjp = lambda t, a, b, g: FT2.tet_moments_vjp2(t, a, b, g, BLK, WP)
from element_polyref import tet_rule
dev, dt = torch.device('cuda'), torch.float64
def T():
    torch.cuda.synchronize(); return time.perf_counter()
tr = tet_rule(4)
tref = torch.tensor(tr[0], dtype=dt, device=dev); tw = torch.tensor(tr[1], dtype=dt, device=dev)
g = torch.Generator(device=dev).manual_seed(0)
N = 1 << 20
base = torch.rand((N, 1, 3), dtype=dt, device=dev, generator=g) * 2 - 1
tets = (base + 0.25 * torch.rand((N, 4, 3), dtype=dt, device=dev, generator=g)).contiguous()
r = dict(Q=int(tref.shape[0]), N=N)
ref = torch.cat([MA._quad(tets[i:i + 65536], tref, tw) for i in range(0, N, 65536)])
t = T(); ref = torch.cat([MA._quad(tets[i:i + 65536], tref, tw) for i in range(0, N, 65536)]); r['quad_torch_s'] = T() - t
out = FT.tet_moments(tets, tref, tw)
t = T(); out = FT.tet_moments(tets, tref, tw); r['quad_triton_s'] = T() - t
r['fwd_rel_max'] = float(((out - ref).abs().max(1).values / ref.abs().max(1).values).max())
G = torch.randn((N, 125), dtype=dt, device=dev, generator=g)
M = 1 << 16
tt = tets[:M].clone().requires_grad_(True)
(MA._quad(tt, tref, tw) * G[:M]).sum().backward(); gref = tt.grad
gt = FT.tet_moments_vjp(tets[:M], tref, tw, G[:M])
r['bwd_rel_max'] = float(((gt - gref).abs().amax((1, 2)) / gref.abs().amax((1, 2))).max())
r['bwd_rel_norm'] = float((gt - gref).norm() / gref.norm())
def torch_bwd():
    for i in range(0, N, 65536):
        x = tets[i:i + 65536].clone().requires_grad_(True)
        (MA._quad(x, tref, tw) * G[i:i + 65536]).sum().backward()
torch_bwd(); t = T(); torch_bwd(); r['fwd_bwd_torch_s'] = T() - t
FT.tet_moments_vjp(tets, tref, tw, G); t = T(); FT.tet_moments(tets, tref, tw); FT.tet_moments_vjp(tets, tref, tw, G); r['fwd_bwd_triton_s'] = T() - t
print(json.dumps(r), flush=True)
