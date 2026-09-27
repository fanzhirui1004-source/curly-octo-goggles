"""Level-set functions of the sheet families (material: |f(x)| <= tau(x), x in the unit cell, period 1).
  'P'  Schwarz P:  f = cos 2 pi x + cos 2 pi y + cos 2 pi z                      (range [-3, 3]; the training family)
  'G'  gyroid:     f = sin 2 pi x cos 2 pi y + sin 2 pi y cos 2 pi z + sin 2 pi z cos 2 pi x   (range [-1.5, 1.5])
f_np / f_torch take points with the coordinate axis last. The default everywhere is 'P' (unchanged behaviour)."""
import numpy as np


def f_np(points, surface='P'):
    p = 2 * np.pi * np.asarray(points)
    if surface in (None, 'P'):
        return np.cos(p).sum(-1)
    if surface == 'G':
        s, c = np.sin(p), np.cos(p)
        return s[..., 0] * c[..., 1] + s[..., 1] * c[..., 2] + s[..., 2] * c[..., 0]
    raise ValueError(f'SURFACE:{surface}')


def f_torch(points, surface='P'):
    import torch
    if surface in (None, 'P'):
        return torch.cos(2 * torch.pi * points).sum(-1)
    if surface == 'G':
        p = 2 * torch.pi * points
        s, c = torch.sin(p), torch.cos(p)
        return s[..., 0] * c[..., 1] + s[..., 1] * c[..., 2] + s[..., 2] * c[..., 0]
    raise ValueError(f'SURFACE:{surface}')


def surface_of(ctx_case):
    return ctx_case.get('surface', 'P') or 'P'
