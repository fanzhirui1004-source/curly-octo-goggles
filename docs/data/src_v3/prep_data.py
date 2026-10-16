"""Step 0: per-geometry training data for the learned extension.

1. q banks (port displacements, teacher port order, rigid part removed, normalized to unit exact energy q^T S q = 1):
   force  : q = S^+ f; f = smooth multiscale random forces on the box ports (plane waves, |k| log-uniform in [0.5, 8]
            cycles per cell, amplitude |k|^-beta) or localized Gaussian patch loads (25%); for cut cells 10% of the
            samples also load the cut band (the cut-loaded regime). Force-driven directions are the soft ones.
   macro  : uniform strain (1/3), quadratic (1/3), cubic (1/3) polynomial displacement fields on the ports
   grf    : random multiscale displacement fields on the ports, |k| in [0.5, 24] cycles per cell
   Splits per class: train 2048, val 256, test 256 (float32 arrays, samples x ports), energies checked in fp64.
   Adversarial directions are generated during training.
2. Network input data (NETDATA.npz): node ids and grid coordinates, element -> 27 local nodes, element moments (125),
   ghost faces, node masks (box, cut, port, weak), 3x3 diagonal stiffness blocks, thickness corners and plane.
Usage: prep_data.py <body_dir> <out_dir> <case> [<case> ...]
"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE

dev, dt = TE.dev, TE.dt
_SP = __import__('os').environ.get('BANK_SPLITS')                              # e.g. "512,64,64" for many geometries
SPLITS = tuple(zip(('train', 'val', 'test'), map(int, _SP.split(',')))) if _SP else (('train', 2048), ('val', 256), ('test', 256))


def plane_waves(X, k, kmin, kmax, gen, m=48):
    """k random vector fields on points X (N,3): sum of m plane waves per component."""
    out = torch.empty((X.shape[0], 3, k), dtype=dt, device=dev)
    for c0 in range(0, k, 32):
        kk = min(32, k - c0)
        d = torch.randn((kk, 3, m, 3), dtype=dt, device=dev, generator=gen); d /= d.norm(dim=-1, keepdim=True)
        mag = kmin * (kmax / kmin) ** torch.rand((kk, 3, m, 1), dtype=dt, device=dev, generator=gen)
        beta = 2 * torch.rand((kk, 1, 1, 1), dtype=dt, device=dev, generator=gen)
        amp = mag.squeeze(-1) ** (-beta.squeeze(-1)) * torch.randn((kk, 3, m), dtype=dt, device=dev, generator=gen)
        ph = 2 * np.pi * torch.rand((kk, 3, m), dtype=dt, device=dev, generator=gen)
        W = (d * mag).reshape(-1, 3)                                          # (kk*3*m, 3)
        arg = 2 * np.pi * (X @ W.T) + ph.reshape(1, -1)
        f = (torch.cos(arg) * amp.reshape(1, -1)).reshape(X.shape[0], kk, 3, m).sum(-1)
        out[:, :, c0:c0 + kk] = f.permute(0, 2, 1)
    return out


def patches(X, k, gen):
    out = torch.empty((X.shape[0], 3, k), dtype=dt, device=dev)
    for j in range(k):
        c = X[torch.randint(0, X.shape[0], (1,), device=dev, generator=gen)]
        w = 0.05 + 0.25 * torch.rand((), dtype=dt, device=dev, generator=gen)
        g = torch.exp(-((X - c) ** 2).sum(1) / (2 * w * w))
        out[:, :, j] = g[:, None] * torch.randn((1, 3), dtype=dt, device=dev, generator=gen)
    return out


def polys(X, k, gen):
    Xc = X - 0.5
    mons = [torch.ones(len(X), dtype=dt, device=dev)]
    deg = [0]
    for p in range(1, 4):
        for a in range(p + 1):
            for b in range(p + 1 - a):
                cexp = p - a - b
                mons.append(Xc[:, 0] ** a * Xc[:, 1] ** b * Xc[:, 2] ** cexp); deg.append(p)
    Mn = torch.stack(mons, 1); deg = torch.as_tensor(deg, device=dev)
    out = torch.empty((X.shape[0], 3, k), dtype=dt, device=dev)
    for j in range(k):
        p = 1 + j % 3
        coef = torch.randn((Mn.shape[1], 3), dtype=dt, device=dev, generator=gen) * (deg[:, None] <= p) * (deg[:, None] >= 1)
        out[:, :, j] = Mn @ coef
    return out


def to_ports(field, C):
    """(port nodes, 3, k) -> (port dofs, k) node-major xyz."""
    return field.reshape(-1, field.shape[-1])


def make_banks(C, out, gen, log):
    X = torch.as_tensor(np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1) / (2 * C.n), dtype=dt, device=dev)
    isbox = torch.as_tensor(C.port_is_box, device=dev); iscut = torch.as_tensor(C.port_is_cut, device=dev)
    total = sum(n for _, n in SPLITS)
    stats = {}
    for cls in ('force', 'macro', 'grf'):
        t = time.perf_counter()
        if cls == 'force':
            nw = total - total // 4
            f = torch.cat([plane_waves(X, nw, 0.5, 8.0, gen), patches(X, total - nw, gen)], 2)
            f = f[:, :, torch.randperm(total, device=dev, generator=gen)]
            cutload = torch.rand(total, device=dev, generator=gen) < (0.1 if C.is_cut.any() else 0.0)
            f = f * torch.where(cutload[None, None, :], torch.ones_like(isbox, dtype=dt)[:, None, None],
                                isbox.to(dt)[:, None, None])
            F = to_ports(f, C)
            q = torch.cat([C.neumann(F[:, j:j + 64]) for j in range(0, total, 64)], 1)
            meta = dict(cut_loaded=int(cutload.sum()))
        elif cls == 'macro':
            q = to_ports(polys(X, total, gen), C); meta = {}
        else:
            q = to_ports(plane_waves(X, total, 0.5, 24.0, gen), C); meta = {}
        q = q - C.Q @ (C.Q.T @ q)
        e = torch.cat([(q[:, j:j + 64] * C.apply(q[:, j:j + 64])).sum(0) for j in range(0, total, 64)])
        q = q / torch.sqrt(e)[None, :]
        rq = (q * q).sum(0)                                                  # 1 / Rayleigh quotient (energy = 1)
        stats[cls] = dict(meta, seconds=time.perf_counter() - t,
                          rayleigh_quantiles=np.quantile((1 / rq).cpu().numpy(), [0, .1, .5, .9, 1]).tolist())
        chk = (q[:, :16] * C.apply(q[:, :16])).sum(0)
        stats[cls]['unit_energy_check'] = float((chk - 1).abs().max())
        lo = 0
        for name, n_ in SPLITS:
            np.save(out / f'{name}_{cls}.npy', q[:, lo:lo + n_].T.contiguous().to(torch.float32).cpu().numpy()); lo += n_
        log(json.dumps(dict(event='BANK', case=C.case, cls=cls, **stats[cls])))
        del q; gc.collect(); torch.cuda.empty_cache()
    return stats


def netdata(C, out):
    N = len(C.nodes)
    diag3 = torch.zeros((N, 3, 3), dtype=dt, device=dev)
    ru, cu = C.ru.long(), C.cu.long()
    same = (ru // 3) == (cu // 3)
    r, c, v = ru[same], cu[same], C.vals[same]
    diag3.index_put_((r // 3, r % 3, c % 3), v, accumulate=True)
    off = r != c
    diag3.index_put_((c[off] // 3, c[off] % 3, r[off] % 3), v[off], accumulate=True)
    nrm = diag3.reshape(N, 9).norm(dim=1)
    weak = (nrm < 0.01 * nrm.median()).cpu().numpy()
    np.savez(out / 'NETDATA.npz', node_ids=C.nodes, grid=np.stack(np.unravel_index(C.nodes, (2 * C.n + 1,) * 3), 1).astype(np.int16),
             elem_cells=C.cells, elem_nodes=(C.dofs[:, ::3] // 3).cpu().numpy().astype(np.int32),
             moments=C.M.cpu().numpy(), gp_faces=np.load(Path(sys.argv[1]) / C.case / 'GP_FACES.npy'),
             is_box=C.is_box, is_cut=C.is_cut, is_port=np.isin(C.nodes, C.port_node_ids), weak=weak,
             diag3=diag3.cpu().numpy(), taus=np.asarray(C.taus0), normal=np.asarray(C.normal if C.normal else [0, 0, 0]),
             offset=np.asarray(C.offset if C.offset is not None else 0.0), n=C.n)
    return dict(nodes=N, weak_nodes=int(weak.sum()), weak_fraction=float(weak.mean()),
                weak_ports=int((weak & np.isin(C.nodes, C.port_node_ids)).sum()))


if __name__ == '__main__':
    body, outroot = sys.argv[1], Path(sys.argv[2])
    rec = {}
    for i, case in enumerate(sys.argv[3:]):
        out = outroot / case; out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        C = TE.Cell(case, body, log=lambda s_: None)
        C.assemble(); C.factor(neumann=True)
        gen = torch.Generator(device=dev).manual_seed(1000 + i)
        r = dict(ports=C.np_, interior=C.ni, dofs=C.nb)
        r['banks'] = make_banks(C, out, gen, log=lambda s_: print(s_, flush=True))
        r['netdata'] = netdata(C, out)
        (out / 'PORTS.json').write_text(json.dumps(dict(case=case, port_node_ids=C.port_node_ids.tolist(),
                                                         port_is_box=C.port_is_box.tolist(), port_is_cut=C.port_is_cut.tolist())))
        r['seconds'] = time.perf_counter() - t0
        rec[case] = r
        print(json.dumps(dict(case=case, **r)), flush=True)
        C._free(); del C; gc.collect(); torch.cuda.empty_cache()
    (outroot / 'PREP_DATA.json').write_text(json.dumps(rec, indent=1))
    print('DONE', flush=True)
