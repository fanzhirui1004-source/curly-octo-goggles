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


def _cos_range(a, b):
    """range of cos(2 pi x) over [a, b] (arrays), with the extrema +1 at integers and -1 at half-integers."""
    ca, cb = np.cos(2 * np.pi * a), np.cos(2 * np.pi * b)
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    hi = np.where(np.floor(b) >= np.ceil(a), 1.0, hi)
    lo = np.where(np.floor(b - 0.5) >= np.ceil(a - 0.5), -1.0, lo)
    return lo, hi


def _sin_range(a, b):
    """range of sin(2 pi x) over [a, b]: +1 at k + 1/4, -1 at k + 3/4."""
    sa, sb = np.sin(2 * np.pi * a), np.sin(2 * np.pi * b)
    lo, hi = np.minimum(sa, sb), np.maximum(sa, sb)
    hi = np.where(np.floor(b - 0.25) >= np.ceil(a - 0.25), 1.0, hi)
    lo = np.where(np.floor(b - 0.75) >= np.ceil(a - 0.75), -1.0, lo)
    return lo, hi


def f_range_box(lo, hi, surface='P'):
    """Enclosure [fmin, fmax] of f over the boxes [lo, hi] (arrays (N, 3), unit-cell coordinates). P: sum of the
    per-axis cosine ranges (the frozen support rule); G: sum of the interval products sin(x_i) cos(x_j) (a valid,
    possibly loose enclosure: loose only means fewer cells certified full)."""
    if surface in (None, 'P'):
        r = [_cos_range(lo[:, d], hi[:, d]) for d in range(3)]
        return sum(x[0] for x in r), sum(x[1] for x in r)
    if surface == 'G':
        s = [_sin_range(lo[:, d], hi[:, d]) for d in range(3)]
        c = [_cos_range(lo[:, d], hi[:, d]) for d in range(3)]
        flo, fhi = 0.0, 0.0
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p = np.stack([s[i][0] * c[j][0], s[i][0] * c[j][1], s[i][1] * c[j][0], s[i][1] * c[j][1]])
            flo = flo + p.min(0); fhi = fhi + p.max(0)
        return flo, fhi
    raise ValueError(f'SURFACE:{surface}')
