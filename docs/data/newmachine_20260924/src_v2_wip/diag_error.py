"""Where is the extension error? (block diagnosis of a trained model; exact fields from the teacher)

For validation directions of each class: err = u_hat - u (zero on the ports), total energy err^T K err (must equal
e_hat - 1 at unit exact energy: Galerkin orthogonality), split into
  - element groups: elements with weak (fictitious-fringe) nodes / elements touching a port / other interior elements
    (element energy err_e^T K_e err_e, body part; the ghost-penalty part is reported separately),
  - scales: the smooth part of err (restricted to the 17^3 vertex grid by full weighting and prolonged back, masked to the
    interior) vs the rest; energies of both parts and their cross term,
  - relative field error in the energy norm restricted to each group.
Step-2 checkpoints (cfg without 'cases'): the geometry comes from the slot cache (--slots, default the S2 cache), moments
from NETDATA, classes = the geometry's own (non-finite bank samples dropped). Shares are also reported per element
fraction (share / fraction of elements in the group: > 1 means the group carries more than its share).
Usage: diag_error.py <checkpoint.pt> <out.json> <case> [<slots_dir>]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import trainlib as TL
import models as MD

dev, dt = TL.dev, TL.dt


def main(ckpt, out, case, slots='/root/autodl-tmp/OPL/S2/slots'):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = ck['cfg']
    if case in cfg.get('cases', []):
        geo = TL.Geo(case, cfg['body'], cfg['data'], neumann=False, log=lambda s_: None)
    else:                                                                     # step-2 checkpoint: slot cache
        import train2 as T2
        geo = torch.load(Path(slots) / f'{case}.pt', map_location='cpu', weights_only=False)
        T2.move(geo, dev); geo.C.K = geo.C
        T2.clean_banks(geo, case, lambda d_: None)
    C = geo.C
    C.factor(neumann=False)
    Mom = C.M if getattr(C, 'M', None) is not None else torch.as_tensor(geo.nd['moments'], dtype=dt, device=dev)
    model = MD.build(cfg['model'], [geo], **cfg.get('model_args', {})).to(dev)
    (MD.load_compat(model, ck['model']) if hasattr(MD, 'load_compat') else model.load_state_dict(ck['model'], strict=False))
    model.eval()
    nd = geo.nd
    en = C.dofs[:, ::3] // 3
    weak = torch.as_tensor(nd['weak'], device=dev); port = torch.as_tensor(nd['is_port'], device=dev)
    el_weak = weak[en].any(1); el_port = port[en].any(1)
    groups = {'weak_elems': el_weak, 'port_elems': el_port & ~el_weak, 'other': ~el_weak & ~el_port}
    # element body stiffness energies: e_e = err_e^T (sum_m M_em Tm_m) err_e
    Tm = C.Tm

    def elem_energy(v):
        out_ = torch.empty((len(C.cells), v.shape[1]), dtype=dt, device=dev)
        for lo in range(0, len(C.cells), 96):
            ve = v[C.dofs[lo:lo + 96]]
            z = torch.einsum('mij,ejb->emib', Tm, ve)
            g = torch.einsum('emib,eib->emb', z, ve)
            out_[lo:lo + 96] = torch.einsum("em,emb->eb", Mom[lo:lo + 96], g)
        return out_
    # smooth projection via the model's grid transfers when available (MGNO), else skip
    rec = dict(case=case, ckpt=str(ckpt), step=ck.get('step'), classes={})
    for cls in geo.classes:
        Q = geo.banks['val'][cls][:, :64].to(dt)
        with torch.no_grad():
            uh = geo.field(model, Q).to(dt)
        u = C.extend(Q)
        err = uh - u
        Ke = C.K @ err
        tot = (err * Ke).sum(0)
        eh = (uh * (C.K @ uh)).sum(0) - (u * (C.K @ u)).sum(0)
        ee = elem_energy(err)
        eu = elem_energy(u)
        body = ee.sum(0)
        r = dict(total_mean=float(tot.mean()), galerkin_check=float(((tot - eh).abs() / eh.abs()).max()),
                 ghost_share=float(((tot - body) / tot).mean()))
        for gname, m in groups.items():
            r[gname + '_share'] = float((ee[m].sum(0) / body).mean())
            r[gname + '_rel_field_err'] = float(torch.sqrt(ee[m].sum(0) / eu[m].sum(0).clamp_min(1e-300)).mean())
            r[gname + '_elements'] = int(m.sum())
            r[gname + '_share_per_fraction'] = r[gname + '_share'] / max(float(m.float().mean()), 1e-12)
        # scales: smooth part of the error from the 17^3 grid (two full-weighting restrictions and prolongations)
        if hasattr(model, 'caches'):
            c = model.caches[case]
            N = c.N; B = err.shape[1]
            x = err.reshape(N, 3, B).permute(0, 2, 1).to(torch.float32)          # N x B x 3
            ts = c.trans[:2]
            y = x
            for t in ts:
                num = torch.zeros((t['n_dst'],) + y.shape[1:], device=dev).index_add_(0, t['v'], y[t['i']] * t['w'][:, None, None])
                den = torch.zeros(t['n_dst'], device=dev).index_add_(0, t['v'], t['w'])
                y = num / den[:, None, None]
            for t in reversed(ts):
                num = torch.zeros((t['n_src'],) + y.shape[1:], device=dev).index_add_(0, t['i'], y[t['v']] * t['w'][:, None, None])
                den = torch.zeros(t['n_src'], device=dev).index_add_(0, t['i'], t['w'])
                y = num / den[:, None, None]
            y = y * (~c.is_port)[:, None, None]
            sm = y.permute(0, 2, 1).reshape(-1, B).to(dt)
            rs = err - sm
            es, er = (sm * (C.K @ sm)).sum(0), (rs * (C.K @ rs)).sum(0)
            r.update(smooth_energy_share=float((es / tot).mean()), rough_energy_share=float((er / tot).mean()),
                     cross_share=float(((tot - es - er) / tot).mean()))
        rec['classes'][cls] = r
        print(json.dumps({cls: r}), flush=True)
    Path(out).write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
