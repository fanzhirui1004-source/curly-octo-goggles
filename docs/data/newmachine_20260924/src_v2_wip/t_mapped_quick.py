"""Quick regression and timing of mapped_cell.py after a code change (GPU server): U1 (identity = teacher K), U4 (affine
point-wise = constant-A route) and the assembly time and rigid residual under a 30-degree twist.
Usage: t_mapped_quick.py <body_dir> <out_json> <case> [<case> ...]"""
import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import teacher as TE
import mapped_cell as MC

dev, dt = TE.dev, TE.dt


def rel(a, b):
    return float((a - b).norm() / b.norm())


recs = []
body_dir, out = sys.argv[1], sys.argv[2]
for case in sys.argv[3:]:
    rec = dict(case=case)
    gen = torch.Generator(device=dev).manual_seed(0)
    C0 = TE.Cell(case, body_dir, log=lambda s_: None).assemble()
    X = torch.randn((C0.nb, 4), dtype=dt, device=dev, generator=gen)
    KX = C0.K @ X
    for name, spec in (('identity', {'kind': 'identity'}), ('twist30', {'kind': 'twist', 'deg': 30})):
        torch.cuda.synchronize(); t = time.perf_counter()
        Cm = MC.MappedCell(case, body_dir, spec, log=lambda s_: None)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        Cm.assemble()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        r = dict(setup_s=t1 - t, assemble_s=t2 - t1, **Cm.map_stats)
        if name == 'identity':
            r['K_rel'] = rel(Cm.K @ X, KX)
        else:
            Xr = torch.linalg.qr(torch.randn((Cm.nb, 6), dtype=dt, device=dev, generator=gen))[0]
            r['rigid_phys_over_random'] = float((Cm.K @ Cm.Qall).norm() / (Cm.K @ Xr).norm())
        rec[name] = r
        print(json.dumps(dict(case=case, map=name, **r)), flush=True)
        del Cm; gc.collect(); torch.cuda.empty_cache()
    A = np.array([[1.3, 0.25, 0.1], [0.0, 0.8, 0.2], [0.05, 0.0, 1.1]])
    Cm = MC.MappedCell(case, body_dir, {'kind': 'affine', 'A': A.tolist()}, log=lambda s_: None)
    sel = np.sort(np.random.default_rng(1).choice(len(Cm.cells), 2000, replace=False))
    Mt = MC.mapped_moments(Cm.cells[sel], Cm.n, Cm.taus0, Cm.normal, Cm.offset, Cm.s, Cm.levels, Cm.surface,
                           Cm.xe[torch.as_tensor(sel, device=dev)], Cm.xi_nodes, Cm.lam, Cm.mu).reshape(len(sel), -1)
    Ac, _ = MC.a_tensor(torch.as_tensor(A / (2 * Cm.n), dtype=dt, device=dev), Cm.lam, Cm.mu)
    pi = [p for p, _ in MC.PAIRS]; qi = [q for _, q in MC.PAIRS]
    Mg = C0.M[torch.as_tensor(sel, device=dev)] * (2 * C0.n) ** 3
    Mt_alt = (Ac[pi, qi][None, :, None] * Mg[:, None, :]).reshape(len(sel), -1)
    rec['U4_affine_element_rel'] = rel(Mt @ Cm.Tt_up, Mt_alt @ Cm.Tt_up)
    print(json.dumps(dict(case=case, U4=rec['U4_affine_element_rel'])), flush=True)
    recs.append(rec)
    Path(out).write_text(json.dumps(recs, indent=2))
    del Cm, C0; gc.collect(); torch.cuda.empty_cache()
print('DONE', flush=True)
