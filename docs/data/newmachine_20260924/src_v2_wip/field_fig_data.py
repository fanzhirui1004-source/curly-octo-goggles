"""Element-wise field data for the mechanical field figure: one cell, consistent-traction test directions; exact field
u = E q, and the fields of B (uncorrected), B + 8 smoothing steps, B with A3's complete correction (8 / Q1(17) / 8,
weights unchanged) and A3. Per element: centroid, material volume, exact element energy u_e^T K_e u_e and error energy
d_e^T K_e d_e (d = field - u), all in float64 with the exact stiffness. Saves an npz and a JSON summary.
Usage: field_fig_data.py <case> <out_prefix> [--m 4] [--body ...] [--data ...]"""
import sys, json, argparse, copy
from pathlib import Path
import numpy as np
import torch
import models as MD
import trainlib as TL

dev, dt = TL.dev, TL.dt
V2 = '/root/autodl-tmp/OPL/S1/V2'


def load(ckpt, geo, **over):
    ck = torch.load(ckpt, map_location=dev, weights_only=False); cfg = ck['cfg']
    args = dict(cfg.get('model_args', {}), sparse=True); args.update(over)
    m = MD.build(cfg['model'], [geo], **args).to(dev)
    MD.load_compat(m, ck['model']); m.eval()                            # the selected weights, as eval_views
    return m


def elem_energy(C, X):
    """(E, B): x_e^T K_e x_e for every element and column (float64, exact K_e = M_e T)."""
    Tf = C.Tm.reshape(125, 81 * 81); out = []
    for lo in range(0, len(C.dofs), 2048):
        de = C.dofs[lo:lo + 2048]
        Ke = (C.M[lo:lo + 2048] @ Tf).reshape(-1, 81, 81)
        xe = X[de]                                                           # e x 81 x B
        out.append((xe * torch.bmm(Ke, xe)).sum(1))
    return torch.cat(out)


def main(argv):
    ap = argparse.ArgumentParser(); ap.add_argument('case'); ap.add_argument('out')
    ap.add_argument('--m', type=int, default=4); ap.add_argument('--cls', default='force_c')
    ap.add_argument('--body', default=None); ap.add_argument('--data', default=None)
    ap.add_argument('--B', default=f'{V2}/v2L1/best.pt'); ap.add_argument('--A3', default=f'{V2}/A3_2grid/best.pt')
    a = ap.parse_args(argv)
    cfg = torch.load(a.A3, map_location='cpu', weights_only=False)['cfg']
    a.body, a.data = a.body or cfg['body'], a.data or cfg['data']
    geo = TL.Geo(a.case, a.body, a.data, neumann=False, log=lambda s_: None)
    C = geo.C
    C.factor(neumann=False)
    Q = geo.banks['test'][a.cls][:, :a.m].to(dt)
    with torch.no_grad():
        u = C.extend(Q)
        eu = TL.energy(u, C.K)
        fields = {}
        mB = load(a.B, geo)
        fields['B'] = geo.field(mB, Q).to(dt)
        mB8 = copy.copy(mB); mB8.smooth_k, mB8.smooth_alpha, mB8.coarse_space = 8, 30.0, None
        fields['B+8'] = geo.field(mB8, Q).to(dt)
        mBW = copy.copy(mB); mBW.smooth_k, mBW.smooth_alpha, mBW.coarse_space = 8, 30.0, 'Q1_17'
        for k in ('_c_space', '_cL', '_cV'):
            if hasattr(C, k):
                setattr(C, k, None)
        fields['B+W'] = geo.field(mBW, Q).to(dt)
        mA = load(a.A3, geo)
        for k in ('_c_space', '_cL', '_cV'):
            if hasattr(C, k):
                setattr(C, k, None)
        fields['A3'] = geo.field(mA, Q).to(dt)
        E0 = elem_energy(C, u)
        rec = dict(case=a.case, cls=a.cls, m=a.m, interior=int(C.ni), ports=int(C.np_), elements=int(len(C.dofs)),
                   A3_args=dict(smooth_k=getattr(mA, 'smooth_k', None), coarse=getattr(mA, 'coarse_space', None)), models={})
        save = dict(cells=np.asarray(C.cells), n=C.n, vol=C.M[:, 0].cpu().numpy(), E_exact=E0.cpu().numpy(),
                    port_nodes=np.asarray(C.port_node_ids), nodes=np.asarray(C.nodes))
        for name, f in fields.items():
            d = f - u
            d[C.P] = 0
            Ee = elem_energy(C, d)
            ex = (TL.energy(f, C.K) / eu - 1).cpu().numpy()
            rec['models'][name] = dict(energy_excess=ex.tolist(), elem_sum_over_quadratic=float((Ee.sum(0) / (TL.energy(d, C.K))).mean()),
                                       disp_rel=((d.norm(dim=0) / u.norm(dim=0)).cpu().numpy()).tolist())
            save['E_' + name.replace('+', 'p')] = Ee.cpu().numpy()
        print(json.dumps(rec), flush=True)
        np.savez_compressed(a.out + '.npz', **save)
        Path(a.out + '.json').write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
