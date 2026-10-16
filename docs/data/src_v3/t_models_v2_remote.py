"""GPU checks of the models v2 options (B1 bounded, B2 feat_v2, fringe_soft) on real step-2 geometries. Not run locally.
Usage: t_models_v2_remote.py <ckpt> <bounds.pt (calib_b1 output)> <out.json> [--orig /root/autodl-tmp/OPL/src/models.py]
                             [--cases a,b] [--slots /root/autodl-tmp/OPL/S2/slots] [--split SPLIT.json] [--batch 16]
(runs on OPL_DEV, default cuda:0; with OPL_DEV=cpu it is only a smoke test of the script: timings / memory meaningless)
Per geometry (default: the smallest training geometry fresh_train_0010_cover01_r1 and the first val geometry of the
checkpoint's split, cfg split or /root/autodl-tmp/OPL/S2/SPLIT.json):
  A flags off vs the ORIGINAL models.py (imported read-only from --orig): field rel diff <= max(3 x the original's own
    run-to-run noise, 1e-6)                                                                       [atomics on the GPU]
  B bounded=True, A = inf: same criterion; sat_stats max == 0
  C feat_v2 + load_compat on the old checkpoint: same criterion
  D calibrated bounds (<bounds.pt>): sat_stats (report; pass: finite, max <= 0.01 on training geometries), max relative
    field change and val-bank energy excess per class before / after (report)
  E all flags on (bounded + feat_v2 + fringe_soft), training path: sparse=True vs sparse=False loss and all parameter
    gradients (as t_sparse.py): loss rel <= 1e-5, global grad rel <= max(3 x einsum repeat noise, 1e-4)
  F all flags on: fastnet.FastNet vs the model path: field, s_hat, model adjoint <= 1e-5 relative; adjoint identity <= 1e-5
  G cost of a sparse training step (fwd + bwd, batch --batch) vs flags off: arm 1 (bounded + feat_v2) time <= +5% and peak
    memory <= +3%; arm 2 (+ fringe_soft: larger fringe sets) reported only
PASS = all criteria on all geometries (printed and in <out.json>)."""
import argparse, json, sys, time, gc, os, importlib.util
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trainlib as TL
import models as MD
import fastnet as FN

dev, f64 = TL.dev, torch.float64
CUDA = dev.type == 'cuda'


def sync():
    if CUDA:
        torch.cuda.synchronize()


def free():
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()


def move(obj, device, seen=None):
    """train2.move (tensors reachable through attributes / dicts / lists), so train2.py is not imported."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return obj
    seen.add(id(obj))
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        for k in list(obj):
            obj[k] = move(obj[k], device, seen)
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = move(obj[i], device, seen)
        return obj
    if hasattr(obj, '__dict__') and not isinstance(obj, (torch.nn.Module, type)):
        for k, v in list(vars(obj).items()):
            if torch.is_tensor(v) or isinstance(v, (dict, list)) or (hasattr(v, '__dict__') and type(v).__module__ in ('teacher', 'models', 'types')):
                setattr(obj, k, move(v, device, seen))
    return obj


def rel(a, b):
    a, b = a.to(f64), b.to(f64)
    return float(((a - b).norm(dim=0) / b.norm(dim=0).clamp_min(1e-300)).max())


def load_geo(slots, case):
    g = torch.load(Path(slots) / f'{case}.pt', map_location='cpu', weights_only=False)
    move(g, dev); g.C.K = g.C
    return g


def build(mod, g, cfg, sd, **over):
    args = dict(cfg.get('model_args', {})); args.update(over)
    m = mod.build(cfg['model'], [g], **args).to(dev)
    if mod is MD:
        MD.load_compat(m, sd)
    else:
        m.load_state_dict(sd, strict=False)
    return m.eval()


def field(g, m, Q):
    with torch.no_grad():
        return g.field(m, Q)


def energies(g, m, per=64, chunk=16):
    out = {}
    with torch.no_grad():
        for c in g.classes:
            Q = g.banks['val'][c][:, :per]
            e = torch.cat([TL.energy(g.field(m, Q[:, j:j + chunk]), g.C.K) - 1 for j in range(0, Q.shape[1], chunk)])
            out[c] = float(e.mean())
    return out


def step_fn(g, m, q, s0, ok):
    def step():
        u = g.field(m, q)
        loss = torch.log(TL.energy(u, g.C.K).clamp_min(1e-12)).mean()
        if ok.any():
            sh = g.sens_hat(u[:, ok])
            loss = loss + (((sh - s0[:, ok]) ** 2).sum(0) / (s0[:, ok] ** 2).sum(0)).mean()
        m.zero_grad(set_to_none=True); loss.backward()
        return float(loss)
    return step


def grads_of(g, cfg, sd, q, s0, ok, **over):
    torch.manual_seed(0)
    m = build(MD, g, cfg, sd, **over); m.train()
    loss = step_fn(g, m, q, s0, ok)()
    sync()
    gr = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    del m; free()
    return loss, gr


def flat(gr, keys):
    return torch.cat([gr[k].reshape(-1) for k in keys])


def cost(g, cfg, sd, q, s0, ok, reps=5, **over):
    torch.manual_seed(0)
    m = build(MD, g, cfg, sd, **dict(over, sparse=True)); m.train()
    step = step_fn(g, m, q, s0, ok)
    for _ in range(2):
        step()
    sync(); free()
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated() if CUDA else 0; t = time.perf_counter()
    for _ in range(reps):
        step()
    sync()
    r = ((time.perf_counter() - t) / reps, (torch.cuda.max_memory_allocated() - m0) / 2 ** 30 if CUDA else 0.0)
    del m; free()
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt'); ap.add_argument('bounds'); ap.add_argument('out')
    ap.add_argument('--orig', default='/root/autodl-tmp/OPL/src/models.py'); ap.add_argument('--cases', default='')
    ap.add_argument('--slots', default='/root/autodl-tmp/OPL/S2/slots'); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--split', default=None)
    a = ap.parse_args(argv)
    os.environ['SENS_REASSOC'] = '1'
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    cfg, sd = ck['cfg'], ck['model']
    spec = importlib.util.spec_from_file_location('models_orig', a.orig)
    MO = importlib.util.module_from_spec(spec); spec.loader.exec_module(MO)
    split = json.loads(Path(a.split or cfg.get('split', '/root/autodl-tmp/OPL/S2/SPLIT.json')).read_text())
    cases = a.cases.split(',') if a.cases else ['fresh_train_0010_cover01_r1', split['val'][0]]
    rec, allok = {}, True
    for case in cases:
        t0 = time.perf_counter()
        g = load_geo(a.slots, case)
        r = dict(dofs=g.nb, train_geometry=case in split.get('train', []))
        gen = torch.Generator(device=dev).manual_seed(0)
        Q = torch.cat([g.banks['val'][c][:, :2] for c in g.classes], 1)
        # A: flags off vs the original
        mo = build(MO, g, cfg, sd)
        uo1, uo2 = field(g, mo, Q), field(g, mo, Q)
        noise = rel(uo2, uo1); tol = max(3 * noise, 1e-6)
        del mo; free()
        m = build(MD, g, cfg, sd)
        r['A_flags_off_rel'], r['orig_repeat_noise'] = rel(field(g, m, Q), uo1), noise
        ok = r['A_flags_off_rel'] <= tol
        # B: bounded, A = inf
        m = build(MD, g, cfg, sd, bounded=True)
        r['B_bounded_inf_rel'] = rel(field(g, m, Q), uo1); r['B_sat_max'] = m.sat_stats()['max']
        ok &= r['B_bounded_inf_rel'] <= tol and r['B_sat_max'] == 0
        # C: feat_v2 zero-padded
        m = build(MD, g, cfg, sd, feat_v2=True)
        r['C_feat_v2_rel'] = rel(field(g, m, Q), uo1)
        ok &= r['C_feat_v2_rel'] <= tol
        # D: calibrated bounds
        m0 = build(MD, g, cfg, sd)
        m = build(MD, g, cfg, sd, bounded=True, bounds=a.bounds)
        u = field(g, m, Q); sat = m.sat_stats()
        r['D_field_rel_change'], r['D_sat'] = rel(u, uo1), sat
        r['D_excess_before'], r['D_excess_after'] = energies(g, m0), energies(g, m)
        ok &= bool(torch.isfinite(u).all()) and (sat['max'] <= 0.01 or not r['train_geometry'])
        del m0, m; free()
        # E: sparse vs dense training gradients, all flags on
        q, s0 = g.sample_with_sens(a.batch, np.random.default_rng(0), {c: 1.0 for c in g.classes})
        okc = ~torch.isnan(s0[0])
        on = dict(bounded=True, bounds=a.bounds, feat_v2=True, fringe_soft=True)
        l0, g0 = grads_of(g, cfg, sd, q, s0, okc, sparse=False, **on)
        la, ga = grads_of(g, cfg, sd, q, s0, okc, sparse=False, **on)
        l1, g1 = grads_of(g, cfg, sd, q, s0, okc, sparse=True, **on)
        keys = sorted(g0)
        f0 = flat(g0, keys)
        r['E_einsum_repeat_rel'] = float((flat(ga, keys) - f0).norm() / f0.norm())
        r['E_grad_rel'] = float((flat(g1, keys) - f0).norm() / f0.norm()) if sorted(g1) == keys else float('inf')
        r['E_loss_rel'] = abs(l1 - l0) / abs(l0)
        r['E_new_cols_grad'] = float(g0['node_in.0.weight'][:, -MD.NF2:].abs().max())
        ok &= r['E_loss_rel'] <= 1e-5 and r['E_grad_rel'] <= max(3 * r['E_einsum_repeat_rel'], 1e-4)
        del g0, ga, g1; free()
        # F: fastnet, all flags on (non-trivial new columns)
        m = build(MD, g, cfg, sd, **on)
        with torch.no_grad():
            m.node_in[0].weight[:, -MD.NF2:] = 0.05 * torch.randn(m.node_in[0].weight.shape[0], MD.NF2, device=dev, generator=gen)
        fast = FN.FastNet(m, g)
        Qf = torch.cat([Q.to(f64), torch.randn((g.np_, 4), dtype=f64, device=dev, generator=gen)], 1)
        r['F_field_rel'] = rel(fast.field(Qf), field(g, m, Qf))
        r['F_s_hat_rel'] = rel(fast.s_hat(Qf), g.s_hat_apply(m, Qf))
        V = torch.randn((g.nb, Qf.shape[1]), dtype=torch.float32, device=dev, generator=gen)
        lhs = (fast.field(Qf) * V).sum(0).to(f64); rhs = (Qf.float() * fast.field_T(V)).sum(0).to(f64)
        r['F_adjoint_identity_rel'] = float(((lhs - rhs).abs() / (fast.field(Qf).norm(dim=0) * V.norm(dim=0)).to(f64)).max())
        qd = torch.randn((g.np_, 4), device=dev, generator=gen); y = torch.randn((g.nb, 4), device=dev, generator=gen)
        qq = qd.clone().requires_grad_(True)
        with torch.enable_grad():
            gref = torch.autograd.grad(m(g, qq), qq, grad_outputs=y)[0]
        r['F_model_T_rel'] = rel(fast.ext_T(y), gref)
        ok &= max(r['F_field_rel'], r['F_s_hat_rel'], r['F_adjoint_identity_rel'], r['F_model_T_rel']) <= 1e-5
        c = m.caches[case]
        r['fringe_sets'] = dict(el=[int(c.el_fringe.sum()), len(c.en)], gp=[int(c.gp_fringe.sum()), len(c.fn)])
        del fast, m; free()
        # G: cost of a sparse training step
        t_off, m_off = cost(g, cfg, sd, q, s0, okc)
        t_a1, m_a1 = cost(g, cfg, sd, q, s0, okc, bounded=True, bounds=a.bounds, feat_v2=True)
        t_a2, m_a2 = cost(g, cfg, sd, q, s0, okc, **on)
        r['G_step_s'] = dict(off=t_off, arm1=t_a1, arm2=t_a2); r['G_peak_extra_GB'] = dict(off=m_off, arm1=m_a1, arm2=m_a2)
        ok &= t_a1 <= 1.05 * t_off and m_a1 <= 1.03 * m_off
        r['ok'], r['seconds'] = bool(ok), time.perf_counter() - t0
        allok &= bool(ok)
        rec[case] = r
        print(json.dumps({case: r}, default=float), flush=True)
        move(g, 'cpu'); del g; free()
    rec['PASS'] = bool(allok)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(dict(PASS=allok)), flush=True)


if __name__ == '__main__':
    main()
