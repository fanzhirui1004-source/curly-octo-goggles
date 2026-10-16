"""Check of teacher.Cell.lean(): K x, K_PP triplets and the learned operator S_hat q before/after, memory and K x time.
Usage: lean_test.py <out.json> <body>:<case>[,...] [--ckpt A3 best.pt]"""
import sys, json, time, argparse
import models as MD                                                    # noqa: F401
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD
import lat_multi as LM

dev, dt = TE.dev, TE.dt
G = lambda: torch.cuda.memory_allocated() / 1e9


def tK(C, x, rep=5):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        y = C.K @ x
    torch.cuda.synchronize(); return y, (time.perf_counter() - t) / rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    a = ap.parse_args()
    BD.TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in a.cases.split(','):
        body, case = spec.split(':')
        torch.cuda.empty_cache(); g0 = G()
        C = TE.Cell(case, body, log=lambda s_: None); C.assemble()
        r = dict(nb=C.nb, ports=C.np_, interior=C.ni, cell_GB=G() - g0)
        x = torch.randn((C.nb, 16), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
        y0, r['Kx16_csr_s'] = tK(C, x)
        k0 = LM.from_teacher(C).kpp()
        h = BD.ModelHolder(a.ckpt); BD.netdata(C, body, BD.TMP / case)
        geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
        g1 = G(); op = EN.FastOp(FN.FastNet(h.add(geo), geo)); r['fastnet_GB'] = G() - g1
        q = torch.randn((C.np_, 8), dtype=dt, device=dev, generator=torch.Generator(device=dev).manual_seed(1))
        s0 = op.apply(q); r['corr_cache_GB'] = G() - g1 - r['fastnet_GB']
        g2 = G(); C.lean(); r['lean_saved_GB'] = g2 - G(); r.update(C.lean_info)
        y1, r['Kx16_lean_s'] = tK(C, x)
        r['Kx_rel'] = float((y1 - y0).norm() / y0.norm())
        k1 = LM.from_teacher(C).kpp()
        r['kpp_equal'] = bool(all(torch.equal(u.cpu(), v.cpu()) for u, v in zip(k0, k1)))
        s1 = op.apply(q); r['Shat_rel_cached_corr'] = float((s1 - s0).norm() / s0.norm())
        for k in ('_cV', '_cL', '_c_space', '_tail_bounds'):
            if hasattr(C, k):
                delattr(C, k)
        torch.cuda.empty_cache()
        s2 = op.apply(q); r['Shat_rel_rebuilt_corr'] = float((s2 - s0).norm() / s0.norm())
        r['cell_after_lean_GB'] = r['cell_GB'] - r['lean_saved_GB']
        out[case] = r; print(json.dumps(dict(case=case, **r)), flush=True)
        h.model.caches.pop(case, None); del op, geo, h, C, x, y0, y1, q, s0, s1, s2; torch.cuda.empty_cache()
        open(a.out, 'w').write(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
