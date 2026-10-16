"""v3_shat.py <out.json> <body_dir>:<case> [--B 1,3,6,16] [--rep 10] [--save q_out.npz] [--ref ref.npz] [--profile]
Single-cell S_hat benchmark on the deployment path of Table 5 (Cell(deploy=True), assemble_deploy, netdata, Geo, FastNet
fused, fastidx on, OPL_TAILT_FUSED / OPL_COARSE_FP32 / OPL_COARSE_ELEM / OPL_TET_TRITON as lat_scale --coarse-tpl
--tet-triton). Records: wall ms per S_hat for each B (synchronised, mean of --rep after one warm-up), the CUDA kernel time
and launch count per S_hat (torch.profiler, B = first value), segment times (network forward / adjoint, correction,
float64 K product), allocated-memory peak of one S_hat, and the S_hat output for fixed seeded directions (--save; with
--ref: max relative difference to a saved output, per column)."""
import os, sys, json, time, argparse, gc
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3')
sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
os.environ['OPL_COARSE_ELEM'] = '1'; os.environ['OPL_TET_TRITON'] = '1'
import models as MD                                                    # noqa: F401  first (OPL_CONV_FP32)
import numpy as np
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import bench_deploy as BD
import fastidx as FI

dev, dt, f32 = TE.dev, TE.dt, torch.float32


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('spec')
    ap.add_argument('--B', default='1,3,6,16'); ap.add_argument('--rep', type=int, default=10)
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    ap.add_argument('--save', default=''); ap.add_argument('--ref', default=''); ap.add_argument('--profile', action='store_true')
    ap.add_argument('--segments', action='store_true')
    a = ap.parse_args()
    FN.FUSED = True; FI.ON = True
    body, case = a.spec.split(':')
    BD.TMP.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo); m = f.model
    with torch.no_grad():
        f.s_hat(torch.zeros((C.np_, 1), dtype=dt, device=dev))          # warm-up: correction caches
    torch.cuda.synchronize()
    r = dict(case=case, nb=int(C.nb), ni=int(C.ni), ports=int(C.np_), elements=int(len(C.M)), setup_s=time.perf_counter() - t0,
             env={k: v for k, v in os.environ.items() if k.startswith('OPL_')})
    Bs = [int(x) for x in a.B.split(',')]
    gen = torch.Generator(device=dev).manual_seed(0)
    Q = torch.randn((C.np_, max(Bs)), dtype=dt, device=dev, generator=gen)
    Q = Q - C.Q @ (C.Q.T @ Q)
    r['ms'] = {}
    outs = {}
    for B in Bs:
        q = Q[:, :B].contiguous()
        torch.cuda.reset_peak_memory_stats(); a0 = torch.cuda.memory_allocated()
        with torch.no_grad():
            y, ms = timed(lambda: f.s_hat(q), a.rep)
        r['ms'][B] = ms
        r.setdefault('peak_extra_MiB', {})[B] = (torch.cuda.max_memory_allocated() - a0) / 2 ** 20
        outs[B] = y.cpu().numpy()
    if a.save:
        np.savez(a.save, **{f'B{B}': v for B, v in outs.items()})
    if a.ref and os.path.exists(a.ref):
        ref = np.load(a.ref)
        r['vs_ref'] = {}
        for B in Bs:
            if f'B{B}' in ref:
                x, y0 = outs[B], ref[f'B{B}']
                r['vs_ref'][B] = float((np.abs(x - y0).max(0) / np.abs(y0).max(0)).max())
                r.setdefault('vs_ref_norm', {})[B] = float((np.linalg.norm(x - y0, axis=0) / np.linalg.norm(y0, axis=0)).max())
    if a.segments:
        B = Bs[0]; q = Q[:, :B].contiguous(); seg = {}
        with torch.no_grad():
            q32 = q.to(f32); cc = f.RPpinv @ q32
            _, seg['net_fwd'] = timed(lambda: f.ext(q32 - f.RP @ cc), a.rep)
            u0 = (f.ext(q32 - f.RP @ cc) + f.RA @ cc).index_copy(0, f.Pidx, q32)
            with TL._held(C):
                _, seg['wrap'] = timed(lambda: TL.wrap(C, u0, m), a.rep)
            _, seg['wrap_nohold'] = timed(lambda: TL.wrap(C, u0, m), a.rep)
            u = TL.wrap(C, u0, m).to(dt)
            y, seg['K64'] = timed(lambda: C.K @ u, a.rep)
            y32 = y.to(f32)
            with TL._held(C):
                _, seg['wrap_T'] = timed(lambda: TL.wrap_T(C, y32, m), a.rep)
            yT = TL.wrap_T(C, y32, m).to(f32)
            yI = yT.index_fill(0, f.Pidx, 0.0)
            _, seg['net_T'] = timed(lambda: f.ext_T(yI), a.rep)
            x = torch.randn((C.nb, B), dtype=dt, device=dev)
            with TL._held(C):
                _, seg['Kc_B_held'] = timed(lambda: C._Kc @ x, a.rep)
            _, seg['coarse_solve'] = timed(lambda: TL._coarse_solve(C, x[C.I]), a.rep)
        r['segments_ms'] = {k: v for k, v in seg.items() if v is not None}
    if a.profile:
        B = Bs[0]; q = Q[:, :B].contiguous()
        with torch.no_grad():
            f.s_hat(q); torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as p:
                for _ in range(3):
                    f.s_hat(q)
                torch.cuda.synchronize()
        ka = p.key_averages()
        dev_rows = [e for e in ka if e.device_type == torch.autograd.DeviceType.CUDA]
        r['profile_B'] = B
        r['kernel_ms_per_shat'] = sum(e.self_device_time_total for e in ka) / 3000
        r['launches_per_shat'] = sum(e.count for e in dev_rows) // 3
        rows = sorted(((e.key, e.self_device_time_total / 3000, e.count // 3) for e in ka if e.self_device_time_total > 0), key=lambda z: -z[1])
        r['top_kernels'] = [dict(name=k[:100], ms=round(c, 3), calls=n) for k, c, n in rows[:40]]
        ops = sorted(((e.key, e.cpu_time_total / 3000, e.count // 3) for e in ka if e.device_type == torch.autograd.DeviceType.CPU), key=lambda z: -z[2])
        r['top_ops_by_count'] = [dict(name=k[:80], cpu_ms=round(c, 3), calls=n) for k, c, n in ops[:40]]
    print(json.dumps({k: v for k, v in r.items() if k not in ('top_kernels', 'top_ops_by_count')}), flush=True)
    json.dump(r, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
