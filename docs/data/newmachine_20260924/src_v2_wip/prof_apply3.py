"""P2 solve speed: where one S_hat application of a deployment cell goes (Cell(deploy=True).assemble_deploy(), fused
network, float32 correction, fused tail transpose, float32 coarse solve), per segment and per CUDA kernel.
Usage: prof_apply3.py <out.json> <body>:<case> [--B 6] [--rep 5] [--ckpt ...]"""
import os, json, time, argparse
import models as MD                                                    # noqa: F401
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import bench_deploy as BD

dev, dt, f32 = TE.dev, TE.dt, torch.float32


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('spec')
    ap.add_argument('--B', type=int, default=6); ap.add_argument('--rep', type=int, default=5)
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    a = ap.parse_args()
    FN.FUSED = True; os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
    body, case = a.spec.split(':')
    BD.TMP.mkdir(parents=True, exist_ok=True)
    C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
    h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo); m = f.model
    q = torch.randn((C.np_, a.B), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    f.s_hat(q)
    r = dict(case=case, B=a.B, nb=C.nb, ni=C.ni, ports=C.np_, elements=int(len(C.M)), ghost_faces=C.GF64.faces,
             smooth_k=m.smooth_k, coarse=m.coarse_space, nc=int(C._cV32.shape[1]))
    x = torch.randn((C.nb, a.B), dtype=dt, device=dev)
    x2 = torch.randn((C.nb, 2 * a.B), dtype=dt, device=dev)
    seg = {}
    _, seg['total'] = timed(lambda: f.s_hat(q), a.rep)
    q32 = q.to(f32); cc = f.RPpinv @ q32
    _, seg['net_fwd'] = timed(lambda: f.ext(q32 - f.RP @ cc), a.rep)
    u0 = (f.ext(q32 - f.RP @ cc) + f.RA @ cc).index_copy(0, f.Pidx, q32)
    _, seg['wrap'] = timed(lambda: TL.wrap(C, u0, m), a.rep)
    _, seg['tail'] = timed(lambda: TL.smooth_tail(C, u0, m.smooth_k, m.smooth_alpha), a.rep)
    _, seg['coarse_correct'] = timed(lambda: TL.coarse_correct(C, x), a.rep)
    u = TL.wrap(C, u0, m).to(dt)
    y, seg['K64'] = timed(lambda: C.K @ u, a.rep)
    y32 = y.to(f32)
    _, seg['wrap_T'] = timed(lambda: TL.wrap_T(C, y32, m), a.rep)
    _, seg['tail_T'] = timed(lambda: TL.smooth_tail_T(C, y32, m.smooth_k, m.smooth_alpha), a.rep)
    yT = TL.wrap_T(C, y32, m).to(f32)
    yI = yT.index_fill(0, f.Pidx, 0.0)
    _, seg['net_T'] = timed(lambda: f.ext_T(yI), a.rep)
    _, seg['Kc_B'] = timed(lambda: C._Kc @ x, a.rep)
    _, seg['Kc_2B'] = timed(lambda: C._Kc @ x2, a.rep)
    _, seg['ghost32'] = timed(lambda: C._Kc.gf.matvec(x.to(f32), torch.zeros((C.nb, a.B), dtype=f32, device=dev)), a.rep)
    _, seg['coarse_solve'] = timed(lambda: TL._coarse_solve(C, x[C.I]), a.rep)
    r['ms'] = seg
    print(json.dumps(r), flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as p:
        for _ in range(3):
            f.s_hat(q)
        torch.cuda.synchronize()
    ka = p.key_averages()
    rows = sorted(((e.key, e.device_time_total / 3000, e.count // 3, e.cpu_time_total / 3000) for e in ka), key=lambda z: -z[1])
    r['kernels_ms'] = [dict(name=k[:90], cuda_ms=round(c, 3), calls=n, cpu_ms=round(cp, 3)) for k, c, n, cp in rows[:30]]
    r['cuda_total_ms'] = sum(e.self_device_time_total for e in ka) / 3000
    r['launches'] = sum(e.count for e in ka if e.device_type == torch.autograd.DeviceType.CUDA) // 3
    for k in r['kernels_ms'][:30]:
        print(json.dumps(k), flush=True)
    print(json.dumps(dict(cuda_total_ms=r['cuda_total_ms'], launches=r['launches'])), flush=True)
    json.dump(r, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
