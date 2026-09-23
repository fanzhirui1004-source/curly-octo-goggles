"""Extra bank class 'support': force-driven directions of a cell that is elastically supported on one box face (a
stand-in for a neighbour), loaded on the other ports. For each box face with ports and two spring levels
(alpha log-uniform in [0.01, 1] times the mean diagonal stiffness of that face's DOFs):
  (K + k_s P_face) u = f,   f = smooth multiscale forces / patch loads on the remaining box ports (10% also on the band),
  q = u restricted to the ports, rigid part removed, normalized to unit exact energy q^T S q = 1.
Also writes the exact sensitivities (as prep_sens). No lattice gate load is used (no leakage).
Usage: prep_support.py <body_dir> <data_dir> <case> [<case> ...]"""
import sys, json, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import prep_data as PD

dev, dt = TE.dev, TE.dt
SPLITS = PD.SPLITS

body, data = sys.argv[1], Path(sys.argv[2])
for ci, case in enumerate(sys.argv[3:]):
    t0 = time.perf_counter()
    C = TE.Cell(case, body, log=lambda s_: None)
    C.assemble(); C.factor(neumann=False)                                       # interior factor: exact energies / fields
    C.dmoments()
    gen = torch.Generator(device=dev).manual_seed(5000 + ci)
    X = torch.as_tensor(np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1) / (2 * C.n), dtype=dt, device=dev)
    isbox = torch.as_tensor(C.port_is_box, device=dev)
    g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
    faces = []
    for a in range(3):
        for side in (0, 2 * C.n):
            m = C.port_is_box & (g[:, a] == side)
            if m.sum() > 8:
                faces.append((a, side, torch.as_tensor(m, device=dev)))
    total = sum(n for _, n in SPLITS)
    per = int(np.ceil(total / (2 * len(faces))))
    Qs = []
    ru, cu = C.ru.long(), C.cu.long()
    for (a, side, fm) in faces:
        fdofs = C.P[torch.nonzero(fm.repeat_interleave(3)).squeeze(1)]            # global DOFs of that face
        dface = C.vals[C.diag[fdofs]].mean()
        for lvl in range(2):
            alpha = float(10 ** (-2 + 2 * torch.rand((), generator=gen, device=dev)))
            v = C.vals.clone(); v[C.diag[fdofs]] += alpha * dface
            s = 1 / torch.sqrt(v[C.diag])
            sol = TE.SPDSolver(C.crow.int(), C.cu, (v * s[ru] * s[cu]).contiguous(), C.nb)
            nw = per - per // 4
            f = torch.cat([PD.plane_waves(X, nw, 0.5, 8.0, gen), PD.patches(X, per - nw, gen)], 2)
            cutload = torch.rand(per, device=dev, generator=gen) < (0.1 if C.is_cut.any() else 0.0)
            keep = (isbox & ~fm).to(dt)
            f = f * torch.where(cutload[None, None, :], (~fm).to(dt)[:, None, None], keep[:, None, None])
            F = torch.zeros((C.nb, per), dtype=dt, device=dev); F[C.P] = f.reshape(-1, per)
            u = s[:, None] * sol.solve(s[:, None] * F)
            Qs.append(u[C.P])
            sol.free(); del sol, v, F, u; gc.collect(); torch.cuda.empty_cache()
    q = torch.cat(Qs, 1)[:, torch.randperm(len(faces) * 2 * per, generator=gen, device=dev)[:total]]
    q = q - C.Q @ (C.Q.T @ q)
    e = torch.cat([(q[:, j:j + 64] * C.apply(q[:, j:j + 64])).sum(0) for j in range(0, total, 64)])
    q = q / torch.sqrt(e)[None, :]
    S = torch.cat([C.sens(C.extend(q[:, j:j + 32])) for j in range(0, total, 32)], 1)
    d = data / case; lo = 0
    for name, n_ in SPLITS:
        np.save(d / f'{name}_support.npy', q[:, lo:lo + n_].T.contiguous().to(torch.float32).cpu().numpy())
        np.save(d / f'{name}_support_sens.npy', S[:, lo:lo + n_].T.contiguous().cpu().numpy())
        lo += n_
    rq = (q * q).sum(0)
    print(json.dumps(dict(case=case, faces=len(faces), seconds=time.perf_counter() - t0,
                          rayleigh_quantiles=np.quantile((1 / rq).cpu().numpy(), [0, .1, .5, .9, 1]).tolist())), flush=True)
    C._free(); del C; gc.collect(); torch.cuda.empty_cache()
