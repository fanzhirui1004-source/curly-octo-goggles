"""P2 step 5 check: deployment state built from scratch (teacher.Cell(..., deploy=True).assemble_deploy(): no global pattern,
no assembled matrix) against the current route (Cell + assemble + lean(deploy=True, ke_moments, ghost_faces)). Compared:
K_PP upper triplets, diag(K), node 3 x 3 diagonal blocks (network input diag3), K x, S_hat q; front-end phase times
(constructor, moments/assembly, lean, netdata, Geo, FastNet) and peak device memory of each route.
Usage: deploy_test3.py <out.json> <body>:<case>[,...] [--ckpt ...] [--B 16]"""
import os, json, time, argparse, gc
import numpy as np
import models as MD                                                    # noqa: F401
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import bench_deploy as BD
import lat_multi as LM

dev, dt = TE.dev, TE.dt


def T():
    torch.cuda.synchronize(); return time.perf_counter()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('cases')
    ap.add_argument('--ckpt', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); ap.add_argument('--B', type=int, default=16)
    a = ap.parse_args()
    FN.FUSED = True
    os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
    BD.TMP.mkdir(parents=True, exist_ok=True)
    out = {}
    for spec in a.cases.split(','):
        body, case = spec.split(':')
        rec = {}
        res = {}
        for scratch in (False, True):
            h = BD.ModelHolder(a.ckpt)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            ph = {}
            t = T()
            C = TE.Cell(case, body, log=lambda s_: None, deploy=scratch)
            ph['ctor'] = T() - t; t = T()
            if scratch:
                C.assemble_deploy()
                ph['assemble_deploy'] = T() - t; t = T()
                kpp = C._kpp_cache
            else:
                C.assemble()
                ph['assemble'] = T() - t; t = T()
            BD.netdata(C, body, BD.TMP / case)
            nd = dict(np.load(BD.TMP / case / 'NETDATA.npz'))
            ph['netdata'] = T() - t; t = T()
            if not scratch:
                C.lean(deploy=True, ke_moments=True, ghost_faces=True)
                ph['lean'] = T() - t; t = T()
                kpp = C._kpp_cache
            geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
            ph['geo'] = T() - t; t = T()
            f = FN.FastNet(h.add(geo), geo)
            ph['fastnet'] = T() - t; t = T()
            g = torch.Generator(device=dev).manual_seed(0)
            q = torch.randn((C.np_, a.B), dtype=dt, device=dev, generator=g)
            x = torch.randn((C.nb, a.B), dtype=dt, device=dev, generator=g)
            s = f.s_hat(q)
            ph['first_apply'] = T() - t
            res[scratch] = dict(kpp=[k.clone() for k in kpp], dK=C.dK.clone(), diag3=nd['diag3'], weak=nd['weak'],
                                Kx=C.K @ x, s=s)
            rec['scratch' if scratch else 'route'] = dict(phases=ph, total_s=sum(ph.values()),
                                                           peak_GB=(torch.cuda.max_memory_allocated() - base) / 1e9)
            h.model.caches.pop(case, None); del f, geo, C, h; gc.collect(); torch.cuda.empty_cache()
        r0, r1 = res[False], res[True]
        same_pattern = r0['kpp'][0].shape == r1['kpp'][0].shape and bool((r0['kpp'][0] == r1['kpp'][0]).all() and (r0['kpp'][1] == r1['kpp'][1]).all())
        rec['cmp'] = dict(kpp_nnz=[int(r0['kpp'][0].numel()), int(r1['kpp'][0].numel())], kpp_same_pattern=same_pattern,
                          kpp_rel=float((r0['kpp'][2] - r1['kpp'][2]).abs().max() / r0['kpp'][2].abs().max()) if same_pattern else None,
                          dK_rel=float(((r0['dK'] - r1['dK']).abs() / r0['dK'].abs()).max()),
                          diag3_rel=float(np.abs(r0['diag3'] - r1['diag3']).max() / np.abs(r0['diag3']).max()),
                          weak_same=bool((r0['weak'] == r1['weak']).all()),
                          Kx_rel=float((r0['Kx'] - r1['Kx']).norm() / r0['Kx'].norm()),
                          shat_rel=float((r0['s'] - r1['s']).norm() / r0['s'].norm()))
        out[case] = rec
        print(json.dumps(dict(case=case, **rec)), flush=True)
        json.dump(out, open(a.out, 'w'), indent=1)
        del res, r0, r1; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
