"""Step 2 data for many geometries: one teacher setup per geometry for all bank classes (prep_data + prep_sens +
prep_support + prep_face in one pass; same generators, same normalization, same files).

Per geometry: Cell setup, assembly, interior + Neumann factors and dM/dtau once; then
  force / macro / grf   (prep_data.make_banks)
  face                  (self-equilibrated loads on one box face, other ports free; prep_face)
  [Neumann factor freed]
  exact sensitivities of these four classes
  support               (one box face on springs, SUPPORT_LEVELS random levels per face, default 1; prep_support
                         uses 2) + sensitivities
  NETDATA.npz, PORTS.json, DONE.json (timings).
Bank sizes from BANK_SPLITS (e.g. "512,64,64"). Skips geometries whose output is complete (restartable).
Usage: prep_geo.py <body_dir> <data_dir> <case> [<case> ...]"""
import sys, json, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import prep_data as PD

dev, dt = TE.dev, TE.dt
SPLITS = PD.SPLITS
CLASSES = ('force', 'macro', 'grf', 'support', 'face')
LEVELS = int(__import__('os').environ.get('SUPPORT_LEVELS', '1'))          # spring levels per face (prep_support: 2)


def save(d, cls, q, S):
    lo = 0
    for name, n_ in SPLITS:
        np.save(d / f'{name}_{cls}.npy', q[:, lo:lo + n_].T.contiguous().to(torch.float32).cpu().numpy())
        np.save(d / f'{name}_{cls}_sens.npy', S[:, lo:lo + n_].T.contiguous().cpu().numpy())
        lo += n_


def normalize(C, q):
    q = q - C.Q @ (C.Q.T @ q)
    e = torch.cat([(q[:, j:j + 64] * C.apply(q[:, j:j + 64])).sum(0) for j in range(0, q.shape[1], 64)])
    return q / torch.sqrt(e)[None, :]


def sens(C, q):
    return torch.cat([C.sens2(C.extend(q[:, j:j + 128])) for j in range(0, q.shape[1], 128)], 1)


def box_faces(C, g):
    out = []
    for a in range(3):
        for side in (0, 2 * C.n):
            m = C.port_is_box & (g[:, a] == side)
            if m.sum() > 8:
                out.append((a, side, m))
    return out


def support_bank(C, X, g, total, gen):
    isbox = torch.as_tensor(C.port_is_box, device=dev)
    faces = box_faces(C, g)
    per = int(np.ceil(total / (LEVELS * len(faces))))
    Qs = []
    ru, cu = C.ru.long(), C.cu.long()
    for (a, side, m) in faces:
        fm = torch.as_tensor(m, device=dev)
        fdofs = C.P[torch.nonzero(fm.repeat_interleave(3)).squeeze(1)]
        dface = C.vals[C.diag[fdofs]].mean()
        for lvl in range(LEVELS):
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
    q = torch.cat(Qs, 1)
    return q[:, torch.randperm(q.shape[1], generator=gen, device=dev)[:total]]


def face_bank(C, X, g, total, gen):
    faces = box_faces(C, g)
    per = int(np.ceil(total / len(faces)))
    Qs = []
    for (a, side, m) in faces:
        fi = np.nonzero(m)[0]
        Xf = X[torch.as_tensor(fi, device=dev)]
        Rf = TE.rigid_basis(C.port_node_ids[fi], C.n)
        nw = per - per // 4
        ff = torch.cat([PD.plane_waves(Xf, nw, 0.5, 8.0, gen), PD.patches(Xf, per - nw, gen)], 2).reshape(-1, per)
        ff = ff - Rf @ (Rf.T @ ff)
        f = torch.zeros((len(C.port_node_ids), 3, per), dtype=dt, device=dev)
        f[torch.as_tensor(fi, device=dev)] = ff.reshape(len(fi), 3, per)
        F = f.reshape(-1, per)
        Qs.append(torch.cat([C.neumann(F[:, j:j + 64]) for j in range(0, per, 64)], 1))
    q = torch.cat(Qs, 1)
    return q[:, torch.randperm(q.shape[1], generator=gen, device=dev)[:total]]


def _ready(body, case, max_wait=7200):
    """Wait for the CPU stage of this case (READY or FAILED_BODY marker in the body dir)."""
    b = Path(body) / case; t0 = time.time()
    while time.time() - t0 < max_wait:
        if (b / 'FAILED_BODY').exists():
            return False
        if (b / 'READY').exists():
            return True
        time.sleep(20)
    return False


def _acquire(Q, tag):
    """Per-geometry slot in the queue's evaluation mutex (FIFO by evalwant mtime, like E() in env.sh)."""
    w = Q / f'evalwant.{tag}'; w.touch()
    while True:
        wants = sorted(Q.glob('evalwant.*'), key=lambda p_: p_.stat().st_mtime)
        if wants and wants[0].name == w.name and not list(Q.glob('evallock.*')):
            try:
                (Q / 'elock').mkdir()
                break
            except FileExistsError:
                pass
        time.sleep(15)
    (Q / f'evallock.{tag}').touch(); w.unlink()


def _release(Q, tag):
    (Q / f'evallock.{tag}').unlink(missing_ok=True)
    try:
        (Q / 'elock').rmdir()
    except FileNotFoundError:
        pass


def main(body, data, cases):
    import os, traceback
    Q = Path(os.environ['PREP_LOCK']) if os.environ.get('PREP_LOCK') else None
    for ci, case in enumerate(cases):
        d = data / case
        if (d / 'DONE.json').exists() or (d / 'FAILED.json').exists():
            continue
        if os.environ.get('PREP_WAIT_READY') and not _ready(body, case):
            d.mkdir(parents=True, exist_ok=True)
            (d / 'FAILED.json').write_text(json.dumps(dict(case=case, stage='body')))
            print(json.dumps(dict(case=case, failed='body')), flush=True)
            continue
        tag = f'prep{os.getpid()}_{ci}'
        if Q is not None:
            _acquire(Q, tag)
        t_w = time.time()
        while torch.cuda.mem_get_info()[0] < 9 * 2 ** 30 and time.time() - t_w < 3600:   # room for one teacher setup
            time.sleep(30)
        try:
            one(body, data, case)
        except Exception as e:
            d.mkdir(parents=True, exist_ok=True)
            (d / 'FAILED.json').write_text(json.dumps(dict(case=case, stage='gpu', error=repr(e), trace=traceback.format_exc()[-2000:])))
            print(json.dumps(dict(case=case, failed=repr(e)[:300])), flush=True)
            gc.collect(); torch.cuda.empty_cache()
        finally:
            if Q is not None:
                _release(Q, tag)


def one(body, data, case):
    total = sum(n for _, n in SPLITS)
    C = None
    try:
        d = data / case; d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter(); tt = {}
        C = TE.Cell(case, body, log=lambda s_: None)
        C.assemble(); C.factor(neumann=True); C.dmoments()
        tt['setup'] = time.perf_counter() - t0
        seed = int.from_bytes(case.encode(), 'little') % (2 ** 31)
        gen = torch.Generator(device=dev).manual_seed(seed)
        g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
        X = torch.as_tensor(g / (2 * C.n), dtype=dt, device=dev)
        t = time.perf_counter()
        stats = PD.make_banks(C, d, gen, log=lambda s_: None)                 # force / macro / grf (+ saved)
        tt['banks3'] = time.perf_counter() - t
        t = time.perf_counter()
        q_face = normalize(C, face_bank(C, X, g, total, gen))                 # needs the Neumann factor
        C.sol_N.free(); C.sol_N = None; gc.collect(); torch.cuda.empty_cache()   # lower the peak before the springs
        tt['face'] = time.perf_counter() - t
        t = time.perf_counter()
        for cls in ('force', 'macro', 'grf'):
            q = torch.cat([torch.as_tensor(np.load(d / f'{s_}_{cls}.npy'), device=dev).T.to(dt) for s_, _ in SPLITS], 1)
            save(d, cls, q, sens(C, q))
        save(d, 'face', q_face, sens(C, q_face)); del q_face
        tt['sens4'] = time.perf_counter() - t
        t = time.perf_counter()
        q = normalize(C, support_bank(C, X, g, total, gen)); save(d, 'support', q, sens(C, q))
        tt['support'] = time.perf_counter() - t
        nd = PD.netdata(C, d)
        (d / 'PORTS.json').write_text(json.dumps(dict(case=case, port_node_ids=C.port_node_ids.tolist(),
                                                       port_is_box=C.port_is_box.tolist(), port_is_cut=C.port_is_cut.tolist())))
        rec = dict(case=case, ports=C.np_, interior=C.ni, dofs=C.nb, elements=int(len(C.cells)), netdata=nd,
                   banks={k: v['rayleigh_quantiles'] for k, v in stats.items()}, splits=SPLITS, seconds=tt,
                   total_seconds=time.perf_counter() - t0)
        (d / 'DONE.json').write_text(json.dumps(rec))
        print(json.dumps(rec), flush=True)
    finally:
        if C is not None:
            C._free()
        del C; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
