"""Which float32 division instruction fused_hyper's Triton kernels use for w / deg (DEG = True): the PTX of _gather_t
and _scatter_t after one launch (div.rn: IEEE; div.full / div.approx: approximate). usage: python t_tdiv.py"""
import os, sys, re
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
import torch, triton
import fused_hyper as FH
FH.HK.ON = False
N, Eh, H, BF = 1000, 200, 4, 32
src = torch.randn((N, BF), device='cuda'); hn = torch.randint(0, N, (Eh, 27), device='cuda')
w = torch.randn((Eh, 27, H), device='cuda'); deg = torch.rand(N, device='cuda') + 1
FH.gather(src, hn, w, deg, H, FH.BLK, True)
z = torch.randn((H, Eh, BF), device='cuda')
FH.scatter(z, hn, w, deg, N, FH.BLK, True)
torch.cuda.synchronize()
print('triton', triton.__version__)
for name, fn in (('_gather_t', FH._gather_t), ('_scatter_t', FH._scatter_t)):
    found = []
    caches = getattr(fn, 'device_caches', None) or getattr(fn, 'cache', None)
    for dev_key, cache in caches.items():
        entries = cache[0] if isinstance(cache, tuple) else cache
        for k, ker in entries.items():
            ptx = ker.asm.get('ptx', '')
            divs = sorted(set(re.findall(r'\b(div\.[a-z0-9.]+|rcp\.[a-z0-9.]+)', ptx)))
            found.append(divs)
    print(name, found)
