"""P2 speed: where the time of one learned application S_hat q goes (network forward, correction, K, correction transpose,
network transpose), per batch width B, on a lean cell; plus the fp32 variant of the correction's stiffness products
(K_e and ghost part in float32 inside the correction only; the final K and the network unchanged).
Usage: prof_apply.py <out.json> <body>:<case> [--ckpt ...] [--B 1,6,16,64] [--rep 5]"""
import json, time, argparse
import models as MD                                                    # noqa: F401
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD

dev, dt, f32 = TE.dev, TE.dt, torch.float32


def sync():
    torch.cuda.synchronize()


def timed(fn, rep):
    fn(); sync(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    sync(); return out, (time.perf_counter() - t) / rep


class K32:
    """Correction-only float32 stiffness: K_e, G in float32, input/output float64."""

    def __init__(self, C):
        self.C, self.Ke = C, C.Ke.to(f32)
        self.G, self.Gt, self.dG = C.G.to(f32), C.Gt.to(f32), C.dG.to(f32)

    def __matmul__(self, x):
        C, x32 = self.C, x.to(f32)
        y = self.G @ x32 + self.Gt @ x32 - self.dG[:, None] * x32
        for lo in range(0, len(self.Ke), C._lean_chunk):
            de = C.dofs[lo:lo + C._lean_chunk]
            y.index_add_(0, de.reshape(-1), torch.bmm(self.Ke[lo:lo + C._lean_chunk], x32[de]).reshape(-1, x.shape[1]))
        return y.to(x.dtype)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('spec')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    ap.add_argument('--B', default='1,6,16,64'); ap.add_argument('--rep', type=int, default=5)
    a = ap.parse_args()
    body, case = a.spec.split(':')
    BD.TMP.mkdir(parents=True, exist_ok=True)
    C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
    h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
    geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    f = FN.FastNet(h.add(geo), geo); m = f.model
    C.lean()
    f.s_hat(torch.zeros((C.np_, 1), dtype=dt, device=dev))                       # correction caches
    res = dict(case=case, ports=C.np_, interior=C.ni, smooth_k=m.smooth_k, coarse=m.coarse_space, per_B={})
    for B in [int(b) for b in a.B.split(',')]:
        g = torch.Generator(device=dev).manual_seed(B)
        q = torch.randn((C.np_, B), dtype=dt, device=dev, generator=g)
        r = {}
        q32 = q.to(f32); cc = f.RPpinv @ q32
        qd = q32 - f.RP @ cc
        _, r['net_fwd'] = timed(lambda: f.ext(qd), a.rep)
        u0 = f.ext(qd) + f.RA @ cc; u0 = u0.index_copy(0, f.Pidx, q32)
        _, r['tail'] = timed(lambda: TL.smooth_tail(C, u0, m.smooth_k, m.smooth_alpha), a.rep)
        x = TL.smooth_tail(C, u0, m.smooth_k, m.smooth_alpha).to(dt)
        _, r['coarse'] = timed(lambda: TL.coarse_correct(C, x), a.rep)
        u, r['correction'] = timed(lambda: TL.wrap(C, u0, m), a.rep)
        u = u.to(dt)
        y, r['K'] = timed(lambda: C.K @ u, a.rep)
        _, r['correction_T'] = timed(lambda: TL.wrap_T(C, y, m), a.rep)
        yT = TL.wrap_T(C, y, m).to(f32)
        yI = yT.index_fill(0, f.Pidx, 0.0)
        _, r['net_T'] = timed(lambda: f.ext_T(yI), a.rep)
        s64, r['total'] = timed(lambda: f.s_hat(q), a.rep)
        # float32 stiffness inside the correction only
        K64 = C.K; k32 = K32(C)
        TL_K = TL._KMat.apply
        TL._KMat.apply = staticmethod(lambda x_, K_: (k32 @ x_) if K_ is K64 else TL_K(x_, K_))
        try:
            s32, r['total_corrK32'] = timed(lambda: f.s_hat(q), a.rep)
        finally:
            TL._KMat.apply = TL_K
        r['corrK32_rel'] = float((s32 - s64).norm() / s64.norm())
        r['energy_rel_corrK32'] = float(((q * (s32 - s64)).sum(0).abs() / (q * s64).sum(0)).max())
        r = {k: (v * 1000 if not k.endswith('rel') and not k.startswith('energy') else v) for k, v in r.items()}   # ms
        res['per_B'][B] = r
        print(json.dumps(dict(B=B, **{k: round(v, 3) if v > 1e-3 else v for k, v in r.items()})), flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
