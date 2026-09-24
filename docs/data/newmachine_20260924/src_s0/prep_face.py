"""Extra bank class 'face': force-driven directions loaded on ONE box face only, all other ports free.
This is the regime of a cell that carries no load of its own and is dragged by a neighbour through a shared face:
equilibrium of the cell makes the interface forces self-equilibrated on that face. For each box face with ports:
  f = smooth multiscale forces (plane waves, |k| log-uniform in [0.5, 8] cycles per cell) or Gaussian patch loads (25%)
      on the face nodes, projected onto the self-equilibrated subspace of that face (zero resultant force and moment),
  q = S^+ f (exact Neumann solve), rigid part removed, normalized to unit exact energy q^T S q = 1,
plus the exact sensitivities (as prep_sens). No lattice gate load is used (no leakage).
Usage: prep_face.py <body_dir> <data_dir> <case> [<case> ...]"""
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
    C.assemble(); C.factor(neumann=True)                                        # interior (fields) + Neumann (S^+)
    C.dmoments()
    gen = torch.Generator(device=dev).manual_seed(7000 + ci)
    g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
    X = torch.as_tensor(g / (2 * C.n), dtype=dt, device=dev)
    faces = []
    for a in range(3):
        for side in (0, 2 * C.n):
            m = C.port_is_box & (g[:, a] == side)
            if m.sum() > 8:
                faces.append((a, side, np.nonzero(m)[0]))
    total = sum(n for _, n in SPLITS)
    per = int(np.ceil(total / len(faces)))
    Qs = []
    for (a, side, fi) in faces:
        Xf = X[fi]
        Rf = TE.rigid_basis(C.port_node_ids[fi], C.n)                          # orthonormal rigid modes on the face
        nw = per - per // 4
        ff = torch.cat([PD.plane_waves(Xf, nw, 0.5, 8.0, gen), PD.patches(Xf, per - nw, gen)], 2).reshape(-1, per)
        ff = ff - Rf @ (Rf.T @ ff)                                              # self-equilibrated on the face
        f = torch.zeros((len(C.port_node_ids), 3, per), dtype=dt, device=dev)
        f[torch.as_tensor(fi, device=dev)] = ff.reshape(len(fi), 3, per)
        F = f.reshape(-1, per)
        Qs.append(torch.cat([C.neumann(F[:, j:j + 64]) for j in range(0, per, 64)], 1))
        del f, F, ff
    q = torch.cat(Qs, 1)[:, torch.randperm(len(faces) * per, generator=gen, device=dev)[:total]]
    del Qs
    q = q - C.Q @ (C.Q.T @ q)
    e = torch.cat([(q[:, j:j + 64] * C.apply(q[:, j:j + 64])).sum(0) for j in range(0, total, 64)])
    q = q / torch.sqrt(e)[None, :]
    chk = (q[:, :16] * C.apply(q[:, :16])).sum(0)
    S = torch.cat([C.sens(C.extend(q[:, j:j + 32])) for j in range(0, total, 32)], 1)
    d = data / case; lo = 0
    for name, n_ in SPLITS:
        np.save(d / f'{name}_face.npy', q[:, lo:lo + n_].T.contiguous().to(torch.float32).cpu().numpy())
        np.save(d / f'{name}_face_sens.npy', S[:, lo:lo + n_].T.contiguous().cpu().numpy())
        lo += n_
    rq = (q * q).sum(0)
    print(json.dumps(dict(case=case, faces=len(faces), seconds=time.perf_counter() - t0,
                          unit_energy_check=float((chk - 1).abs().max()),
                          rayleigh_quantiles=np.quantile((1 / rq).cpu().numpy(), [0, .1, .5, .9, 1]).tolist())), flush=True)
    C._free(); del C; gc.collect(); torch.cuda.empty_cache()
