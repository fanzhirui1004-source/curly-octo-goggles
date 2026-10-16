"""Remote GPU test of oh.py (B6 O_h views) on full trainlib.Geo objects and a trained checkpoint (not run locally).
For every base case and every rotated packet <base>_rot<name> (make_rot.ROTS) that has a slot (S2/slots) or prepared data
(S2/data/<case>/DONE.json):
  A. energy equivalence (PASS criterion): for bank directions q of every class (val bank, plus up to NTRAIN train
     columns), e_view = q-energy of view_R(base).field(q) with the base K, e_rot = energy of rot.field(Pi q) with the
     rotated packet's own K (Pi = port node map x R on the components); max |e_view - e_rot| / e_rot <= 1e-5.
     Reported too: field difference (rotated packet field mapped back to the base nodes and frame) and the sensitivity
     difference with corners permuted (sens_hat on the base with its dM vs on the packet with its dM; sens_ok = <= 5e-3:
     sens_hat's fp32 products alone differ from fp64 by ~8e-4 on the fixture, so this is a consistency flag, not a gate).
  B. all 48 views of each base (PASS criteria): port values exact (bitwise), e_hat >= 1 - 1e-5 on the unit-energy val
     bank (S_hat >= S); per class mean e_hat - 1 of the identity and over the 48 views (orientation sensitivity, A9 style).
  C. adversarial search through one view (fp32 Neumann factor of the base, as train2.adversarial_on_load): Ritz values
     finite and >= 1 - 1e-4 (PASS criterion).
Usage: python t_oh_remote.py <ckpt> [<base_case> ...]        (default base: fresh_train_0021_cover01_r1)
env: OH_SLOTS, OH_DATA, OH_BODY (default /root/autodl-tmp/OPL/S2/slots, /root/autodl-tmp/OPL/S2/data, /root/autodl-tmp/OPL/S0),
     OH_NTRAIN (64), OH_ADV (1 = run C). Needs a free GPU (base + packet + view caches; run beside training only if the
     memory allows). Exit code 1 on any failed criterion."""
import os, sys, json, time, gc
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trainlib as TL
import models as MD
import train2 as T2
import oh

dev, dt = TL.dev, TL.dt
SLOTS = Path(os.environ.get('OH_SLOTS', '/root/autodl-tmp/OPL/S2/slots'))
DATA = Path(os.environ.get('OH_DATA', '/root/autodl-tmp/OPL/S2/data'))
BODY = os.environ.get('OH_BODY', '/root/autodl-tmp/OPL/S0')
NTRAIN = int(os.environ.get('OH_NTRAIN', '64'))
ROTS = {'swapxy': [[0, 1, 0], [1, 0, 0], [0, 0, 1]], 'mirrorx': [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'rotz90': [[0, -1, 0], [1, 0, 0], [0, 0, 1]], 'invert': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        'general': [[0, 0, -1], [1, 0, 0], [0, -1, 0]]}                   # make_rot.ROTS
FAIL = []


def log(d):
    print(json.dumps(d, default=float), flush=True)


def check(name, ok, **info):
    log({'check': name, 'ok': bool(ok), **info})
    if not ok:
        FAIL.append(name)


def available(case):
    return (SLOTS / f'{case}.pt').exists() or (DATA / case / 'DONE.json').exists()


def load(case):
    """A geometry as train2.Slot builds it (slot cache if present, else from body + data), banks cleaned."""
    p = SLOTS / f'{case}.pt'
    if p.exists():
        g = torch.load(p, map_location='cpu', weights_only=False)
        T2.move(g, 'cuda'); g.C.K = g.C
    else:
        g = T2.build_geo(case, dict(body=BODY, data=str(DATA)))
    T2.clean_banks(g, case, log)
    return g


def build_model(ckpt, g):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    m = MD.build(cfg['model'], [g], **cfg.get('model_args', {})).to(dev)
    try:
        r = MD.load_compat(m, ck['model'])
    except (KeyError, ValueError) as e:
        r = repr(m.load_state_dict(ck['model'], strict=False)); log(dict(event='LOAD_NONSTRICT', reason=repr(e)[:200]))
    log(dict(event='MODEL', ckpt=str(ckpt), model=cfg['model'], model_args=cfg.get('model_args', {}), step=ck.get('step'), load=r))
    return m.eval()


def port_map(base, rot, R):
    """Rotated-packet position of every base port node, and of every base node."""
    n = int(base.C.n)
    img = oh.node_map(base.C.port_node_ids, R, n)
    pm = np.searchsorted(rot.C.port_node_ids, img)
    okp = bool(np.array_equal(rot.C.port_node_ids[np.minimum(pm, len(pm) - 1)], img))
    imga = oh.node_map(base.C.nodes, R, n)
    nm = np.searchsorted(rot.C.nodes, imga)
    okn = bool(np.array_equal(rot.C.nodes[np.minimum(nm, len(nm) - 1)], imga))
    return torch.as_tensor(pm, device=dev), torch.as_tensor(nm, device=dev), okp and okn


def to_rot(q, pm, np_rot, p, s):
    """Base port block (np, B) -> rotated packet port block: node image pm, components R."""
    B = q.shape[1]
    out = torch.zeros((np_rot // 3, 3, B), dtype=q.dtype, device=q.device)
    out[pm] = oh.rot_dofs(q, p, s).reshape(-1, 3, B)
    return out.reshape(-1, B)


def cols(g, c):
    Q = g.banks['val'][c]
    if NTRAIN > 0:
        Q = torch.cat([Q, g.banks['train'][c][:, :NTRAIN]], 1)
    return Q


@torch.no_grad()
def part_a(model, base, v, rot, name, R, chunk=16):
    p, s = oh.perm_sign(R)
    pt = np.argsort(p)
    pm, nm, ok_map = port_map(base, rot, R)
    worst = dict(energy=0.0, field=0.0, sens=0.0)
    per = {}
    has_sens = getattr(base, 'dM32', None) is not None and getattr(rot, 'dM32', None) is not None
    cp = torch.as_tensor(oh.corner_perm(R), device=dev)
    for c in base.classes:
        Q = cols(base, c)
        e_rel, f_rel, s_rel = [], [], []
        for j in range(0, Q.shape[1], chunk):
            q = Q[:, j:j + chunk]
            uv = v.field(model, q)
            ur = rot.field(model, to_rot(q, pm, rot.np_, p, s))
            ev, er = TL.energy(uv, base.C.K), TL.energy(ur, rot.C.K)
            e_rel.append(((ev - er).abs() / er).cpu())
            ub = oh.rot_dofs(ur.reshape(-1, 3, q.shape[1])[nm].reshape(-1, q.shape[1]), pt, s[pt])   # packet field in base frame
            f_rel.append(((uv.to(dt) - ub.to(dt)).norm(dim=0) / ub.to(dt).norm(dim=0)).cpu())
            if has_sens:
                sv, sr = v.sens_hat(uv), rot.sens_hat(ur)                  # base corner k <-> packet corner cp[k]
                s_rel.append(((sv - sr[cp]).norm(dim=0) / sr.norm(dim=0)).cpu())
        e_rel, f_rel = torch.cat(e_rel), torch.cat(f_rel)
        per[c] = dict(n=int(Q.shape[1]), energy_max=float(e_rel.max()), field_max=float(f_rel.max()))
        worst['energy'] = max(worst['energy'], float(e_rel.max())); worst['field'] = max(worst['field'], float(f_rel.max()))
        if s_rel:
            s_rel = torch.cat(s_rel); per[c]['sens_max'] = float(s_rel.max()); worst['sens'] = max(worst['sens'], float(s_rel.max()))
    check(f'A_energy[{base.case},{name}]', ok_map and worst['energy'] <= 1e-5, node_map=ok_map, worst=worst, per_class=per,
          sens_checked=has_sens, sens_ok=(not has_sens) or worst['sens'] <= 5e-3)


@torch.no_grad()
def part_b(model, base, chunk=16, ks=range(48)):
    port, ub, stats, ks = True, np.inf, {}, list(ks)
    for k in ks:
        v = oh.view(model, base, k)
        for c in base.classes:
            Q = base.banks['val'][c]
            e = []
            for j in range(0, Q.shape[1], chunk):
                u = v.field(model, Q[:, j:j + chunk])
                port &= bool(torch.equal(u[base.P], Q[:, j:j + chunk].to(torch.float32)))
                e.append(TL.energy(u, base.C.K).cpu())
            e = torch.cat(e)
            ub = min(ub, float(e.min()))
            stats.setdefault(c, []).append(float((e - 1).mean()))
        oh.drop(model, v)
        gc.collect(); torch.cuda.empty_cache()
    i0 = ks.index(0) if 0 in ks else None
    summ = {c: dict(identity=None if i0 is None else m[i0], mean=float(np.mean(m)), median=float(np.median(m)), max=float(np.max(m)),
                    worst_view=ks[int(np.argmax(m))], max_over_identity=float(np.max(m) / m[i0]) if i0 is not None and m[i0] > 0 else None)
            for c, m in stats.items()}
    check(f'B_views[{base.case}]', port and ub >= 1 - 1e-5 and not any('@oh' in c for c in model.caches), ports_exact=port,
          views=len(stats[base.classes[0]]), min_energy=ub, mean_err_by_class=summ)


def part_c(model, base, k=29):
    C = base.C
    v = oh.view(model, base, k)
    try:
        C.factor(neumann=True, interior=False, fp32_neumann=True)
        gen = torch.Generator(device=dev).manual_seed(0)
        X, ritz = v.adversarial(model, k=8, iters=3, gen=gen)
        r = ritz.cpu().numpy()
        check(f'C_adversarial_view[{base.case},{k}]', bool(np.isfinite(r).all() and r.min() >= 1 - 1e-4), ritz=r.tolist())
    finally:
        C._free() if hasattr(C, '_free') else None
        C.sol_N = None
        oh.drop(model, v)
        gc.collect(); torch.cuda.empty_cache()


def main(ckpt, bases):
    t0 = time.perf_counter()
    for bcase in bases:
        base = load(bcase)
        model = build_model(ckpt, base)
        for name, R in ROTS.items():
            case = f'{bcase}_rot{name}'
            if not available(case):
                log(dict(event='SKIP_MISSING', case=case)); continue
            R = np.asarray(R)
            rot = load(case)
            model.add_geo(rot)
            v = oh.view(model, base, oh.index_of(R))
            part_a(model, base, v, rot, name, R)
            oh.drop(model, v); model.drop_geo(rot.case)
            del rot, v; gc.collect(); torch.cuda.empty_cache()
        part_b(model, base)
        if os.environ.get('OH_ADV', '1') == '1':
            part_c(model, base)
        del base, model; gc.collect(); torch.cuda.empty_cache()
    log(dict(event='PASS' if not FAIL else 'FAIL', failed=FAIL, seconds=time.perf_counter() - t0))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1], sys.argv[2:] or ['fresh_train_0021_cover01_r1'])
