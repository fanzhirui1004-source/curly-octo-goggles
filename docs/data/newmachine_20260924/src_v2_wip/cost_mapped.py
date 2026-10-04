"""Direction A, step 1: per-cell cost of the mapped routes (new script; default paths unchanged; runs on the GPU server).

Same measurements as bench_deploy.per_cell, on mapped cells. Per (cell, map):
  p1         (map 'id' only, --p1base) the P1 path on the unmapped cell: teacher.Cell setup, moments, assembly
  mapped     MappedCell setup (physical ghost penalty, templates) and assembly (mapped moments x templates)
  exact      interior factor fp32 (+ fp64 refinement in apply) and fp64: seconds, device memory (cuDSS, mem_get_info);
             S q for B = 1, 16, 64 columns
  learned    NETDATA from K~ (a0_eval.write_data), Geo, model caches, nodal polar rotations + co-rotated Jacobi;
             per field (a0_budget.FIELDS, default c1, c2w): weak patch setup, first S_hat application (correction
             caches: coarse space, smoothing interval), then S_hat q (autograd adjoint, Geo.s_hat_apply) and E_hat q
             (forward only) for B = 1, 16; peak allocated memory of one B = 16 application; persistent state;
             energy ratio q^T S_hat q / q^T S~ q on 64 random port vectors (rigid part removed)
  fastnet    (map 'id' only) P1's deployed operator (fastnet.FastNet, explicit adjoint) on the same cell: S_hat q for
             B = 1, 16, 64. For the identity map V0R = V0, so its ratio to the autograd S_hat converts the autograd
             timings of the mapped fields into deployed estimates.
Usage: cost_mapped.py <out.jsonl> <ckpt> <case@body_dir>[,...] <maps.json[,...]> --maps id,strx2,twist30
       [--fields c1,c2w] [--work dir] [--p1base 1]"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import torch

import models as MD                                                       # first: applies OPL_CONV_FP32
import teacher as TE
import trainlib as TL
import a0_eval as AE
import a0_budget as AB
import mapped_cell as MC
import corot_smooth as CR
import weak_patch as WP

dev, dt = AE.dev, AE.dt


def sync():
    torch.cuda.synchronize()


def timed(fn, reps=1, warm=0):
    for _ in range(warm):
        fn()
    sync(); t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync()
    return out, (time.perf_counter() - t) / reps


def free_gb():
    gc.collect(); torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0] / 2 ** 30


def peak(fn):
    """Result and peak allocated memory (GiB) above the allocation before the call."""
    gc.collect(); torch.cuda.empty_cache(); sync()
    a0 = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    out = fn(); sync()
    return out, (torch.cuda.max_memory_allocated() - a0) / 2 ** 30


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('ckpt'); ap.add_argument('cases'); ap.add_argument('mapsjson')
    ap.add_argument('--maps', default='id,strx2,twist30'); ap.add_argument('--fields', default='c1,c2w')
    ap.add_argument('--work', default='/root/autodl-tmp/OPL/A0/work_cost'); ap.add_argument('--p1base', type=int, default=1)
    ap.add_argument('--wp_dil', type=int, default=2); ap.add_argument('--wp_seed', default='cutweakbox')
    a = ap.parse_args(argv)
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    specs = {}
    for f in a.mapsjson.split(','):
        specs.update({m['name']: m['spec'] for m in json.loads(Path(f).read_text())})
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    model = None
    CR.install()
    for item in a.cases.split(','):
        case, body = item.split('@', 1)
        for mname in a.maps.split(','):
            rec = dict(case=case, map=mname, gpu=torch.cuda.get_device_name(0))
            try:
                f0 = free_gb()
                if mname == 'id' and a.p1base:
                    P, rec['p1_setup_s'] = timed(lambda: TE.Cell(case, body, log=lambda s_: None))
                    _, rec['p1_moments_s'] = timed(lambda: P.moments(P.taus0))
                    _, rec['p1_assemble_s'] = timed(lambda: P.assemble())
                    P._free(); del P; free_gb()
                C, rec['mapped_setup_s'] = timed(lambda: MC.MappedCell(case, body, specs[mname], log=lambda s_: None))
                _, rec['mapped_assemble_s'] = timed(lambda: C.assemble())
                rec.update(dofs=int(C.nb), ports=int(C.np_), interior=int(C.ni), elements=int(len(C.cells)),
                           K_nnz_upper=int(C.vals.numel()), map_stats=C.map_stats)
                gen = torch.Generator(device=dev).manual_seed(0)
                Q = torch.randn((C.np_, 64), dtype=dt, device=dev, generator=gen)
                Q = Q - C.Q @ (C.Q.T @ Q); Q = Q / Q.norm(dim=0)
                # ------------------------------------------------ exact route
                for prec, fp32 in (('fp64', False), ('fp32', True)):
                    fb = free_gb()
                    _, rec[f'factor_{prec}_s'] = timed(lambda: C.factor(neumann=False, fp32=fp32))
                    rec[f'factor_{prec}_GB'] = fb - free_gb()
                    for B in (1, 16, 64):
                        _, rec[f'exact_{prec}_Sq_B{B}_s'] = timed(lambda: C.apply(Q[:, :B]), reps=3, warm=1)
                    if prec == 'fp64':
                        e_ex = (Q * C.apply(Q)).sum(0)
                    C._free()
                # ------------------------------------------------ learned route (V0R + co-rotated Jacobi)
                fb = free_gb()
                d = Path(a.work) / mname
                _, rec['netdata_s'] = timed(lambda: AE.write_data(C, d / case, {}, (), body))
                g, rec['geo_s'] = timed(lambda: AB.BudgetGeo(case, body, d, neumann=False, log=lambda s_: None, cell=C,
                                                            load_banks=False))
                g.case = f'{case}@{mname}'
                t = time.perf_counter()
                if model is None:
                    model = MD.build(cfg['model'], [g], **dict(cfg.get('model_args', {}))).to(dev)
                    MD.load_compat(model, ck['model']); model.eval()
                else:
                    model.add_geo(g)
                sync(); rec['model_cache_s'] = time.perf_counter() - t
                wrap = AE._Wrap(model)
                _, rec['corot_s'] = timed(lambda: CR.set_corot(C, AE.nodal_rotations(C)))
                rec['fields'] = {}
                for fname in [f for f in a.fields.split(',') if f]:
                    r = rec['fields'][fname] = {}
                    base, cyc = AB.FIELDS[fname]
                    if fname.endswith('w') and getattr(C, '_wp', 'unset') == 'unset':
                        rec['weakpatch'], rec['weakpatch_s'] = timed(lambda: WP.setup(C, g.nd, dil=a.wp_dil, seed=a.wp_seed))
                    g.set_budget(base, cyc, wrap, wp=fname.endswith('w'))
                    _, r['first_Sq_B16_s'] = timed(lambda: g.s_hat_apply(model, Q[:, :16]))
                    for B in (1, 16):
                        _, r[f'Sq_B{B}_s'] = timed(lambda: g.s_hat_apply(model, Q[:, :B]), reps=3, warm=1)
                        with torch.no_grad():
                            _, r[f'Eq_B{B}_s'] = timed(lambda: g.field(model, Q[:, :B]), reps=3, warm=1)
                    _, r['Sq_B16_peak_GB'] = peak(lambda: g.s_hat_apply(model, Q[:, :16]))
                    with torch.no_grad():
                        _, r['Eq_B16_peak_GB'] = peak(lambda: g.field(model, Q[:, :16]))
                    e_hat = torch.cat([(Q[:, j:j + 16] * g.s_hat_apply(model, Q[:, j:j + 16])).sum(0) for j in range(0, 64, 16)])
                    ratio = (e_hat / e_ex).cpu().numpy()
                    r['energy_ratio'] = dict(mean=float(ratio.mean()), min=float(ratio.min()), max=float(ratio.max()))
                    log(dict(event='FIELD', case=case, map=mname, field=fname, **r))
                rec['learned_state_GB'] = fb - free_gb()
                model.caches.pop(g.case, None)
                # ------------------------------------------------ P1's deployed operator (identity map)
                if mname == 'id':
                    try:
                        import fastnet as FN
                        CR.clear_corot(C)
                        for k in ('_cV', '_cL', '_c_space', '_tail_bounds'):
                            if hasattr(C, k):
                                delattr(C, k)
                        geo = TL.Geo(case, body, d, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
                        model.add_geo(geo)
                        fast, rec['fastnet_freeze_s'] = timed(lambda: FN.FastNet(model, geo))
                        _, rec['fastnet_first_Sq_B16_s'] = timed(lambda: fast.s_hat(Q[:, :16]))
                        for B in (1, 16, 64):
                            _, rec[f'fastnet_Sq_B{B}_s'] = timed(lambda: fast.s_hat(Q[:, :B]), reps=3, warm=1)
                        _, rec['fastnet_Sq_B16_peak_GB'] = peak(lambda: fast.s_hat(Q[:, :16]))
                        ef = torch.cat([(Q[:, j:j + 16] * fast.s_hat(Q[:, j:j + 16]).to(dt)).sum(0) for j in range(0, 64, 16)])
                        rec['fastnet_energy_ratio_mean'] = float((ef / e_ex).mean())
                        model.caches.pop(case, None); del fast, geo
                    except Exception as e:                                       # noqa: BLE001
                        import traceback
                        rec['fastnet_error'] = repr(e)[:300]; rec['fastnet_trace'] = traceback.format_exc()[-1500:]
                rec['total_GB_used'] = f0 - free_gb()
            except Exception as e:                                               # noqa: BLE001
                import traceback
                rec['error'] = repr(e)[:400]; rec['trace'] = traceback.format_exc()[-2000:]
            with open(a.out, 'a') as f:
                f.write(json.dumps(rec, default=float) + '\n')
            log(dict(event='CELL', **{k: v for k, v in rec.items() if k not in ('fields', 'map_stats', 'trace', 'fastnet_trace')}))
            for name in ('g', 'C'):
                obj = locals().get(name)
                if name == 'C' and obj is not None:
                    try:
                        WP.free(obj)
                    except Exception:                                            # noqa: BLE001
                        pass
                    try:
                        obj._free()
                    except Exception:                                            # noqa: BLE001
                        pass
            g = C = None
            gc.collect(); torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
