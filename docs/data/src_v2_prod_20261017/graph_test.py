"""Upper bound of CUDA-graph replay for one S_hat application (resident deployment cell): capture f.s_hat on a static
input, replay, compare output and time with eager; plus the kernel-only GPU time from the profiler."""
import os, json, time, sys
import models as MD  # noqa
import torch, teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
dev, dt, f32 = TE.dev, TE.dt, torch.float32
def timed(fn, rep=10):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep): out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep
FN.FUSED = True; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_TAILT_FUSED'] = '0'; FI.ON = True
body, case = sys.argv[1].split(':')
BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
r = {}
for B in (6, 12):
    qs = torch.randn((C.np_, B), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(B))
    s_e, r[f'eager_ms_{B}'] = timed(lambda: f.s_hat(qs))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        f.s_hat(qs); torch.cuda.synchronize()
    r[f'kernel_ms_{B}'] = sum(e.self_device_time_total for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA) / 1000
    r[f'kernels_{B}'] = sum(e.count for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            f.s_hat(qs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            s_out = f.s_hat(qs)
        _, r[f'graph_ms_{B}'] = timed(lambda: g.replay())
        r[f'graph_rel_{B}'] = float((s_out - s_e).norm() / s_e.norm())
    except Exception as ex:
        r[f'graph_error_{B}'] = repr(ex)[:300]
    print(json.dumps(r), flush=True)
