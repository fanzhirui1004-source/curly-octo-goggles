"""Design sensitivities on the host CPU for the exact lattice routes of Table 5 ((a) whole-lattice direct solve, (b)
conventional exact condensation): the quantity of the learned route (lat_scale.py --sens ad --sens-obj sum), i.e. per
cell c the 8 corner derivatives of the summed compliance of the face loads,
    s_c = -sum_k u_k^T (dK_c / dtau) u_k ,
with the exact full cell fields u_k of the route. As in the learned route: energy densities g = Cell.energy_density(u),
summed over the load columns (the objective is linear in g), then one reverse-mode pass through the element moments
(moments_ad.cell_sens, rows whose moments do not depend on tau skipped: their contribution is exactly zero).
Runs with torch's intra-op threads (OMP_NUM_THREADS). On the host the reverse pass keeps the quadrature intermediates
(moments_ad.CHECKPOINT = False: no recomputation in the backward; the checkpointing exists to save GPU memory) and uses
row batches of 256 (the fastest of 256 / 1024 / 4096 on a test cell; all settings give the same result to the last bit).
NEW module; nothing else changes.

light(C)        the part of a teacher.Cell the sensitivity needs (element rows, pattern operators, DOF map, design),
                taken before the route releases the cell (a few MB per cell)
cell_sens(L, u) (8,) numpy array and the seconds of the two steps"""
import time
import types
import numpy as np
import torch
import teacher as TE
import moments_ad as MA

MA.CHECKPOINT = False                                                   # host: keep the forward intermediates

KEEP = ('case', 'n', 's', 'levels', 'cells', 'Tm', 'dofs', 'taus0', 'normal', 'offset', 'nb', 'np_', 'ni')


def light(C):
    L = types.SimpleNamespace(**{k: getattr(C, k) for k in KEEP})
    L.surface = getattr(C, 'surface', 'P') or 'P'
    L.taus = list(C.taus if getattr(C, 'taus', None) is not None else C.taus0)   # the design the route assembled
    L.P = C.P.cpu().numpy().copy(); L.I = C.I.cpu().numpy().copy()
    return L


def cell_sens(L, u, batch=256, skip_full=True):
    """u: (nb, k) float64 full field of the k face loads (numpy or torch, host). Returns (s (8,), seconds dict)."""
    u = torch.as_tensor(np.ascontiguousarray(u), dtype=torch.float64)
    t0 = time.perf_counter()
    g = TE.Cell.energy_density(L, u).sum(2, keepdim=True)                    # (E, 125, 1)
    t1 = time.perf_counter()
    s = MA.cell_sens(L, g, batch=batch, skip_full=skip_full)[:, 0]
    t2 = time.perf_counter()
    return s.numpy().astype(np.float64), dict(energy_density_s=t1 - t0, reverse_s=t2 - t1)


def energy(L, Kup, u):
    """sum_k u_k^T K u_k for the cell's upper CSR (scipy, unscaled) and full fields u (nb, k): consistency check."""
    Ku = Kup @ u + Kup.T @ u - Kup.diagonal()[:, None] * u
    return float((u * Ku).sum())
