import os, sys, time
sys.path.insert(0, os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'))
import torch
import fastidx as FI
dev = torch.device('cuda'); FI.ON = True
for b, dtp, n, rows in ((3, torch.float64, 400000, (11000, 81)), (2, torch.float32, 330000, (7500, 135)), (6, torch.float32, 330000, (11000, 81)), (1, torch.float64, 330000, (300000,))):
    x = torch.randn((n, b), dtype=dtp, device=dev); idx = torch.randint(0, n, rows, device=dev, dtype=torch.int32)
    res = {}
    for tri in (True, False):
        FI.TRITON = tri
        for _ in range(3): FI.rows(x, idx)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(20): FI.rows(x, idx)
            torch.cuda.synchronize()
        ka = p.key_averages()
        res[tri] = sum(e.self_device_time_total for e in ka if e.device_type == torch.autograd.DeviceType.CUDA) / 20
    print('b=%d %s rows=%s: triton %.1f us, take(cached) %.1f us (device time)' % (b, dtp, rows, res[True], res[False]))
