"""Why is the float32 element product ~1.8x slower inside S_hat than alone with a cold L2 (t_symv4 vs t_symv5)?
S_hat of a full plate cell as a CUDA graph (B = 1), replayed (a) back to back, (b) spaced by 100 ms of idle GPU (no
sustained power draw), (c) back to back after 20 s of sustained load; per mode the element-product kernels' device time
per call (torch.profiler) and the S_hat time, with nvidia-smi clocks / power sampled alongside by tst.sh. Also the
product alone back to back (no other kernels between calls) and with a 256 MB L2 flush between calls, both in the
same process (same clocks).
usage: python t_symv6.py BODY_DIR:CASE [--rep 30]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import ke_symv as KS
from torch.profiler import profile, ProfilerActivity

dev = 'cuda'
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=30)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
with torch.no_grad():
    f.s_hat(torch.zeros((C.np_, 1), dtype=torch.float64, device=dev))
if hasattr(C, 'store64'):
    C.store64(allow=True)
g = torch.Generator(device=dev); g.manual_seed(0)


def symv_profile(fn, reps, gap=0.0):
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        t = time.perf_counter()
        for _ in range(reps):
            fn()
            if gap:
                torch.cuda.synchronize(); time.sleep(gap)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) / reps
    rows = [e for e in p.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    sy = [e for e in rows if e.key.startswith('_symv')]
    tot = sum(e.self_device_time_total for e in rows) / 1e3 / reps
    sms = sum(e.self_device_time_total for e in sy) / 1e3 / reps
    n = sum(e.count for e in sy) // reps
    return dict(kernel_ms=round(tot, 3), symv_ms=round(sms, 3), symv_calls=n, us_per_symv=round(1000 * sms / max(n, 1), 1),
                wall_ms=round(1000 * wall, 2))


res = dict(case=case)
with torch.no_grad():
    q = torch.randn((C.np_, 1), dtype=torch.float64, device=dev, generator=g)
    q = q - C.Q @ (C.Q.T @ q)
    qs = torch.zeros_like(q)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            f.s_hat(qs)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    G = torch.cuda.CUDAGraph()
    with torch.cuda.graph(G):
        out = f.s_hat(qs)
    qs.copy_(q)
    rep = lambda: G.replay()
    res['back_to_back'] = symv_profile(rep, a.rep); print(json.dumps({'back_to_back': res['back_to_back']}), flush=True)
    res['spaced_100ms'] = symv_profile(rep, a.rep, gap=0.1); print(json.dumps({'spaced_100ms': res['spaced_100ms']}), flush=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 20:                                 # sustained load
        for _ in range(20):
            rep()
        torch.cuda.synchronize()
    res['after_20s_load'] = symv_profile(rep, a.rep); print(json.dumps({'after_20s_load': res['after_20s_load']}), flush=True)
    # the product alone, the same matrices as S_hat's hold() forms them
    Kc = C._Kc
    with Kc.hold():
        P = Kc._kp_tmp
        dofs = C.dofs.contiguous()
        x = torch.randn((C.nb, 1), dtype=torch.float32, device=dev, generator=g)
        y = torch.zeros_like(x)
        buf = torch.zeros(64 * 2 ** 20, dtype=torch.float32, device=dev)
        alone = lambda: KS.symv_(y, P, dofs, x)
        res['alone_back_to_back'] = symv_profile(alone, 100)
        res['alone_flush'] = symv_profile(lambda: (buf.add_(1.0), alone()), 100)
        res['alone_spaced'] = symv_profile(alone, 30, gap=0.1)
    for k in ('alone_back_to_back', 'alone_flush', 'alone_spaced'):
        print(json.dumps({k: res[k]}), flush=True)
print(json.dumps(res), flush=True)
