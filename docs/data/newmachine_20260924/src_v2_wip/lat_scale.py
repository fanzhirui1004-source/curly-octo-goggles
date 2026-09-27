"""P2 step 6: one learned lattice at scale (all cells learned, deployment state from scratch, streamed), timed by phase;
no exact reference (accuracy is established on the 8 / 18-cell lattices, lat_hetero.py).
Per design iteration:
  front end, per cell: Cell(deploy=True) (first iteration only), assemble_deploy(taus) (moments, K_PP triplets, diag,
             ghost faces), netdata, Geo, FastNet (fused hyperedge), warm-up application (correction caches), K_PP triplets to
             the host, stream wrap (stream_ops)
  lattice:   MultiLattice (first iteration), assembled K_PP, preconditioner setup (--prec), PCG to --tol with residual
             snapshots at --levels (iterations per level), warm start from the previous iteration's solution (same DOFs)
  sensitivities: per cell field u = F q and -u^T dK/dtau u for all load columns, by central differences of the moments
             ('fd': Cell.dmoments + Cell.sens, the reference route) or by reverse mode ('ad': moments_ad.cell_sens)
Iteration it > 0 moves every thickness corner by --dtau (fixed topology: the cell bodies are not regenerated, so the
timing excludes fast_prep4; the reported body time comes from the body generation logs).
Usage: lat_scale.py <out.json> <layout.json> [--body S4/body] [--model ckpt] [--prec bnn:kpp:q1r] [--tol 1e-8]
       [--levels 1e-2,3e-3,1e-3,1e-4,1e-6] [--iters 2] [--dtau 0.002] [--sens fd|ad|both] [--max-cols 16] [--limit N]"""
import os, json, time, argparse, gc
from pathlib import Path
import numpy as np
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD
import lat_multi as LM
import lat_precond as PR
import stream_ops as SO

dev, dt = TE.dev, TE.dt


def rss():
    for l in open('/proc/self/status'):
        if l.startswith('VmRSS'):
            return int(l.split()[1]) / 1e6
    return None


def T():
    torch.cuda.synchronize(); return time.perf_counter()


class _Resident:
    """A cell kept on the device (--resident-gb): the StreamedOp interface without streaming."""
    bytes, copies, nxt = 0, 0, None

    def __init__(self, op):
        self.op = op

    def active(self):
        import contextlib
        return contextlib.nullcontext()

    def apply(self, q):
        return self.op.apply(q)

    def field(self, q):
        return self.op.field(q)

    def release(self, to=None):
        self.op = None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('layout')
    ap.add_argument('--body', default='/root/autodl-tmp/OPL/S4/body')
    ap.add_argument('--model', default='/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt')
    ap.add_argument('--prec', default='bnn:kpp:q1r'); ap.add_argument('--tol', type=float, default=1e-8)
    ap.add_argument('--levels', default='1e-2,3e-3,1e-3,1e-4,1e-6'); ap.add_argument('--maxit', type=int, default=3000)
    ap.add_argument('--iters', type=int, default=2); ap.add_argument('--dtau', type=float, default=0.002)
    ap.add_argument('--sens', default='fd', choices=['fd', 'ad', 'both']); ap.add_argument('--max-cols', type=int, default=16)
    ap.add_argument('--sens-obj', default='cols', choices=['cols', 'sum'],
                    help="'sum': sensitivity of the summed compliance only (one design objective; one reverse pass with 'ad')")
    ap.add_argument('--resident-gb', type=float, default=0.0,
                    help='keep cells on the device (not streamed) while the allocated device memory stays below this')
    ap.add_argument('--sparse-coarse', action='store_true', help='OPL_COARSE_SPARSE=1: sparse coarse space (lat_precond.SparseCoarse)')
    ap.add_argument('--ad-batch', type=int, default=512, help='element rows per reverse pass (moments_ad)')
    ap.add_argument('--limit', type=int, default=0, help='first N cells of the layout only (smoke tests)')
    a = ap.parse_args(argv)
    os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'
    if a.sparse_coarse:
        os.environ['OPL_COARSE_SPARSE'] = '1'
    FN.FUSED = True
    SO.init()
    log = lambda d: print(json.dumps(d, default=float), flush=True)
    L = json.loads(Path(a.layout).read_text())
    cells = L['cells'][:a.limit] if a.limit else L['cells']
    levels = [float(x) for x in a.levels.split(',')]
    res = dict(args=vars(a), gpu=torch.cuda.get_device_name(0), layout=L['name'], cells=len(cells), iterations=[])
    BD.TMP.mkdir(parents=True, exist_ok=True)
    h = BD.ModelHolder(a.model)
    Cs, kpp_host, lat, X_prev, taus, old = {}, {}, None, None, {}, {}
    for c in cells:
        taus[c['case']] = None
    for it in range(a.iters):
        rec = dict(iteration=it, phases={}, per_cell=[])
        ph = rec['phases']
        add = lambda k, v: ph.__setitem__(k, ph.get(k, 0.0) + v)
        t_it = T()
        ops = {}
        torch.cuda.reset_peak_memory_stats()
        for c in cells:
            case, pc = c['case'], {}
            t = T()
            if case not in Cs:
                Cs[case] = TE.Cell(case, a.body, log=lambda s_: None, deploy=True)
                pc['ctor'] = T() - t; t = T()
            C = Cs[case]
            if case in old:                                                        # previous design's state back on the device
                old.pop(case).release(to=dev)
                pc['restore'] = T() - t; t = T()
            if taus[case] is None:
                taus[case] = list(C.taus0)
            else:
                taus[case] = [x + a.dtau for x in taus[case]]
            C.assemble_deploy(taus[case])
            pc['assemble_deploy'] = T() - t; t = T()
            BD.netdata(C, a.body, BD.TMP / case)
            pc['netdata'] = T() - t; t = T()
            geo = TL.Geo(case, a.body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
            if h.model is not None:
                h.model.caches.pop(case, None)
            op = EN.FastOp(FN.FastNet(h.add(geo), geo))
            pc['encode'] = T() - t; t = T()
            op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))              # correction caches (coarse factor)
            pc['warmup'] = T() - t; t = T()
            kpp_host[case] = tuple(x.cpu() for x in C._kpp_cache)
            C._kpp_cache = kpp_host[case]
            C.diag3 = None
            if torch.cuda.memory_allocated() / 1e9 < a.resident_gb:
                ops[case] = _Resident(op)
            else:
                ops[case] = SO.StreamedOp(op, C, [op.fast, geo, C])
            pc['stream_wrap'] = T() - t
            pc['stream_bytes'] = ops[case].bytes
            for k, v in pc.items():
                if k != 'stream_bytes':
                    add('fe_' + k, v)
            rec['per_cell'].append(dict(case=case, **pc))
            del geo, op
            gc.collect(); torch.cuda.empty_cache()
        add('front_end', sum(v for k, v in ph.items() if k.startswith('fe_')))
        rec['device_GB_after_front_end'] = torch.cuda.memory_allocated() / 1e9
        rec['census_after_front_end'] = SO.cuda_census()
        rec['stream_GB_total'] = sum(o.bytes for o in ops.values()) / 1e9
        rec['host_rss_GB_after_front_end'] = rss()
        log(dict(event='FRONT_END', iteration=it, host_rss_GB=rec['host_rss_GB_after_front_end'], seconds=ph['front_end'], stream_GB=rec['stream_GB_total'],
                 device_GB=rec['device_GB_after_front_end'], peak_GB=torch.cuda.max_memory_allocated() / 1e9))
        t = T()
        if lat is None:
            lay = {}
            for c in cells:
                g = LM.from_teacher(Cs[c['case']])
                g._kpp_fn = (lambda cs: (lambda: tuple(x.to(dev) for x in kpp_host[cs])))(c['case'])
                lay[tuple(c['position'])] = g
            lat = LM.MultiLattice(lay, clamp=('y', 'min'), load=('y', 'max'), loads='consistent', n_random=0, device=dev,
                                  max_cols=a.max_cols, log=lambda s_: None)
            res['lattice'] = {k: v for k, v in lat.info().items() if k not in ('positions', 'cases', 'distinct_cases')}
            ph['lattice_setup'] = T() - t; t = T()
        for g in lat.geoms:
            g._kpp = None                                                          # new design: new K_PP triplets
        kpp = lat.assemble_kpp()
        for g in lat.geoms:
            g._kpp = None
        ph['kpp_assemble'] = T() - t; t = T()
        order = [g.case for g in lat.geoms]
        olist = [ops[cs] for cs in order]
        SO.link([o for o in olist if isinstance(o, SO.StreamedOp)])
        rec['resident_cells'] = sum(isinstance(o, _Resident) for o in olist)
        fac = PR.Factory(lat, olist, shared={'kpp_triplets': kpp, 'kpp_triplets_s': 0.0}, kpp_backend='auto', log=lambda d: None)
        pcnd, st, _ = fac.build(a.prec)
        ph['prec_setup'] = T() - t; t = T()
        rec['prec'] = {k: v for k, v in st.items() if isinstance(v, (int, float, str))}
        r = PR.pcg(lat, olist, pcnd, tol=a.tol, maxit=a.maxit, X0=X_prev, snaps=levels)
        ph['solve'] = T() - t; t = T()
        X = r['X']
        rec['solve'] = dict(iterations=r['iterations'], residual=r['residual'], warm=X_prev is not None,
                            levels=[dict(level=s['level'], iterations=s['iterations']) for s in r['snaps']],
                            start_residual=r['history'][0] if r['history'] else None)
        comp = (lat.F * X).sum(0)
        rec['compliance'] = comp.cpu().tolist()
        fac.free()
        for v in fac.shared.values():                                          # K_PP factor and triplets
            if hasattr(v, 'free'):
                v.free()
        fac.shared.clear()
        del fac, pcnd, r, kpp
        gc.collect(); torch.cuda.empty_cache()
        # sensitivities
        S = {}
        for i, cs in enumerate(order):
            C, o = Cs[cs], olist[i]
            with o.active():
                u = o.field(lat.gather(X, i)).to(dt)
                if a.sens in ('fd', 'both'):
                    t1 = T()
                    C.dmoments(); s_fd = C.sens(u).cpu(); C.dM = None
                    if a.sens_obj == 'sum':
                        s_fd = s_fd.sum(1, keepdim=True)
                    add('sens_fd', T() - t1)
                    S.setdefault('fd', []).append(s_fd)
                if a.sens in ('ad', 'both'):
                    import moments_ad as MA
                    t1 = T()
                    g = C.energy_density(u)
                    if a.sens_obj == 'sum':
                        g = g.sum(2, keepdim=True)                                 # linear in g: d(sum_k c_k)/dtau
                    s_ad = MA.cell_sens(C, g, batch=a.ad_batch).cpu(); del g
                    add('sens_ad', T() - t1)
                    S.setdefault('ad', []).append(s_ad)
                del u
        ph['sens_total'] = T() - t
        del C, o
        SO.park_all()
        if 'fd' in S and 'ad' in S:
            fd, ad = torch.stack(S['fd']), torch.stack(S['ad'])
            rec['sens_ad_vs_fd_rel'] = float(((ad - fd).norm(dim=1) / fd.norm(dim=1)).max())
        ph['iteration_total'] = T() - t_it
        rec['peak_device_GB'] = torch.cuda.max_memory_allocated() / 1e9
        rec['stream_copies'] = sum(o.copies for o in olist)
        log(dict(event='ITERATION', iteration=it, phases=ph, solve=rec['solve'], peak_GB=rec['peak_device_GB']))
        res['iterations'].append(rec)
        X_prev = X
        old = ops
        del olist
        gc.collect(); torch.cuda.empty_cache()
        rec['device_GB_end'] = torch.cuda.memory_allocated() / 1e9
        rec['census_end'] = SO.cuda_census()
        log(dict(event='CENSUS', iteration=it, device_GB_end=rec['device_GB_end'], census_fe=rec['census_after_front_end'], census_end=rec['census_end']))
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
