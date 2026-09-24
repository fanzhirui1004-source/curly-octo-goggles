"""Step 2 data for many geometries: one teacher setup per geometry for all bank classes (prep_data + prep_sens +
prep_support + prep_face in one pass; same generators, same normalization, same files).

Per geometry (one fp32 factorization alive at a time): Cell setup, assembly, dM/dtau; then
  Neumann factor:  force / macro / grf directions (prep_data generators), face directions (prep_face)
  spring factors:  support directions (one box face on springs, SUPPORT_LEVELS random levels per face, default 1;
                   prep_support uses 2), one factorization at a time
  interior factor: exact normalization (refined to fp64 accuracy) and exact sensitivities of all five classes
  NETDATA.npz, PORTS.json, DONE.json (timings, peak memory).
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
            sol = spd_safe(C.crow.int(), C.cu, (v * s[ru] * s[cu]).contiguous(), C.nb)
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
        need = float(os.environ.get('PREP_MIN_FREE_GB', '9')) * 2 ** 30
        while torch.cuda.mem_get_info()[0] < need and time.time() - t_w < float(os.environ.get('PREP_WAIT_MAX', '3600')):
            time.sleep(30)                                                      # room for one teacher setup
        err = None
        torch.cuda.reset_peak_memory_stats()
        try:
            one(body, data, case)
        except Exception as e:
            err = (repr(e), traceback.format_exc()[-2000:])
        gc.collect(); torch.cuda.empty_cache()                                  # the traceback is gone: free everything
        if err is not None:
            d.mkdir(parents=True, exist_ok=True)
            (d / 'FAILED.json').write_text(json.dumps(dict(case=case, stage='gpu', error=err[0], trace=err[1])))
            print(json.dumps(dict(case=case, failed=err[0][:300])), flush=True)
        if Q is not None:
            _release(Q, tag)


def _mem_error(e):
    r = repr(e)
    return 'ALLOC_FAILED' in r or 'out of memory' in r or 'OutOfMemory' in r


def factor_safe(C, **kw):
    """fp32 factor; on a numerical failure (e.g. a non-positive fp32 pivot on an ill-conditioned K) retry in fp64.
    Memory failures propagate (the driver retries them in a second pass with more free memory)."""
    try:
        C.factor(**kw)
        return 'fp32'
    except Exception as e:
        if _mem_error(e):
            raise
        C._free(); gc.collect(); torch.cuda.empty_cache()
        C.factor(**dict(kw, fp32=False, fp32_neumann=False))
        return 'fp64'


def spd_safe(crow, col, vals, n):
    try:
        return TE.SPDSolver(crow, col, vals, n, fdt=torch.float32)
    except Exception as e:
        if _mem_error(e):
            raise
        gc.collect(); torch.cuda.empty_cache()
        return TE.SPDSolver(crow, col, vals, n)


def raw_banks3(C, X, isbox, total, gen):
    """force / macro / grf directions before normalization (the generators of prep_data.make_banks)."""
    nw = total - total // 4
    f = torch.cat([PD.plane_waves(X, nw, 0.5, 8.0, gen), PD.patches(X, total - nw, gen)], 2)
    f = f[:, :, torch.randperm(total, device=dev, generator=gen)]
    cutload = torch.rand(total, device=dev, generator=gen) < (0.1 if C.is_cut.any() else 0.0)
    f = f * torch.where(cutload[None, None, :], torch.ones_like(isbox, dtype=dt)[:, None, None], isbox.to(dt)[:, None, None])
    F = PD.to_ports(f, C)
    out = {'force': torch.cat([C.neumann(F[:, j:j + 64]) for j in range(0, total, 64)], 1)}
    out['macro'] = PD.to_ports(PD.polys(X, total, gen), C)
    out['grf'] = PD.to_ports(PD.plane_waves(X, total, 0.5, 24.0, gen), C)
    return out, int(cutload.sum())


def one(body, data, case):
    """One factorization alive at a time: Neumann (force / face directions) -> springs one by one (support directions)
    -> interior (exact normalization and sensitivities). fp32 factors; the interior one is refined to fp64 accuracy."""
    total = sum(n for _, n in SPLITS)
    C = None
    try:
        d = data / case; d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter(); tt = {}
        C = TE.Cell(case, body, log=lambda s_: None)
        C.assemble(); C.dmoments()
        tt['setup'] = time.perf_counter() - t0
        seed = int.from_bytes(case.encode(), 'little') % (2 ** 31)
        gen = torch.Generator(device=dev).manual_seed(seed)
        g = np.stack(np.unravel_index(C.port_node_ids, (2 * C.n + 1,) * 3), 1)
        X = torch.as_tensor(g / (2 * C.n), dtype=dt, device=dev)
        isbox = torch.as_tensor(C.port_is_box, device=dev)
        t = time.perf_counter()
        tt['neumann_precision'] = factor_safe(C, neumann=True, interior=False, fp32_neumann=True)
        raw, n_cut = raw_banks3(C, X, isbox, total, gen)
        raw['face'] = face_bank(C, X, g, total, gen)
        C._free()
        tt['neumann_dirs'] = time.perf_counter() - t
        t = time.perf_counter()
        raw['support'] = support_bank(C, X, g, total, gen)
        tt['support_dirs'] = time.perf_counter() - t
        t = time.perf_counter()
        tt['interior_precision'] = factor_safe(C, neumann=False, fp32=True)
        stats = {}
        for cls in ('force', 'macro', 'grf', 'face', 'support'):
            q = normalize(C, raw.pop(cls))
            rq = (q * q).sum(0)
            stats[cls] = dict(rayleigh_quantiles=np.quantile((1 / rq).cpu().numpy(), [0, .1, .5, .9, 1]).tolist(),
                              unit_energy_check=float(((q[:, :16] * C.apply(q[:, :16])).sum(0) - 1).abs().max()))
            save(d, cls, q, sens(C, q))
            del q
        stats['force']['cut_loaded'] = n_cut
        tt['normalize_sens'] = time.perf_counter() - t
        nd = PD.netdata(C, d)
        (d / 'PORTS.json').write_text(json.dumps(dict(case=case, port_node_ids=C.port_node_ids.tolist(),
                                                       port_is_box=C.port_is_box.tolist(), port_is_cut=C.port_is_cut.tolist())))
        rec = dict(case=case, ports=C.np_, interior=C.ni, dofs=C.nb, elements=int(len(C.cells)), netdata=nd,
                   banks=stats, splits=SPLITS, seconds=tt, total_seconds=time.perf_counter() - t0,
                   peak_GB=torch.cuda.max_memory_allocated() / 2 ** 30)
        (d / 'DONE.json').write_text(json.dumps(rec))
        print(json.dumps(rec), flush=True)
    finally:
        if C is not None:
            C._free()
        del C


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3:])
