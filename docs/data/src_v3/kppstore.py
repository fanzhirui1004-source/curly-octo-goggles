"""Per-cell K_PP triplets in scratch files (src_v3, OPL_V3_KPP_DISK=1, default on with OPL_V3_SLIM).

The drivers (opt_design.analyse_fast, lat_scale) keep every cell's K_PP upper triplets (int32 rows / cols, float64 values;
10-15 MB per full plate cell) from the cell's front end until the lattice assembles K_PP (S16 frees them right after).
At the end of the front end they are the second largest host item after the streamed states (~8% of the host memory of
an analysis). Here they are written to one file per cell right after the cell's front end and read back by the lattice's
assembly (MultiLattice._assemble_kpp_lean / assemble_kpp_csr: one cell at a time); kpp_release removes the files. The
triplets are bitwise the same. Directory: OPL_V3_KPP_DIR, else <bench_deploy.TMP>/kpp_<pid>.
"""
import os
import shutil
from pathlib import Path
import numpy as np
import torch

ON = os.environ.get('OPL_V3_KPP_DISK', '1') == '1'
_DIR = []


class Stored:
    """One cell's triplets in a file: n entries, rows and columns int32, values float64 (in this order)."""
    __slots__ = ('path', 'n')

    def __init__(self, path, n):
        self.path, self.n = path, n

    @property
    def nbytes(self):
        return self.n * 16                                              # the host bytes the triplets would take

    def load(self, device=None):
        raw = np.fromfile(self.path, dtype=np.uint8)
        n = self.n
        r = torch.from_numpy(raw[:4 * n].view(np.int32))
        c = torch.from_numpy(raw[4 * n:8 * n].view(np.int32))
        v = torch.from_numpy(raw[8 * n:16 * n].view(np.float64))
        if device is None:
            return r, c, v
        return r.to(device), c.to(device), v.to(device)


def root():
    if not _DIR:
        base = os.environ.get('OPL_V3_KPP_DIR')
        if not base:
            import bench_deploy as BD
            base = str(BD.TMP)
        d = Path(base) / f'kpp_{os.getpid()}'
        d.mkdir(parents=True, exist_ok=True)
        _DIR.append(d)
    return _DIR[0]


def put(case, r, c, v):
    """Write the triplets (device or host tensors) as the drivers kept them (int32 rows / cols, float64 values)."""
    r32 = r.int().cpu().numpy()
    c32 = c.int().cpu().numpy()
    v64 = v.cpu().numpy()
    if v64.dtype != np.float64:
        raise TypeError(f'KPP_VALUES_DTYPE:{v64.dtype}')
    p = root() / f'{case}.kpp'
    with open(p, 'wb') as f:
        r32.tofile(f); c32.tofile(f); v64.tofile(f)
    return Stored(p, len(v64))


def host_entry(case, r, c, v):
    """The driver's host entry for one cell: a Stored file (ON) or the int32 / float64 host tensors (as before)."""
    if ON:
        return put(case, r, c, v)
    return r.int().cpu(), c.int().cpu(), v.cpu()


def to_device(entry, device):
    if isinstance(entry, Stored):
        return entry.load(device)
    return tuple(x.to(device) for x in entry)


def nbytes(entry):
    if entry is None:
        return 0
    if isinstance(entry, Stored):
        return entry.nbytes
    if torch.is_tensor(entry):
        return 0
    return sum(t.numel() * t.element_size() for t in entry)


def remove(entries):
    """Delete the files of Stored entries (kpp_release)."""
    for e in entries:
        if isinstance(e, Stored):
            try:
                os.unlink(e.path)
            except OSError:
                pass


def cleanup():
    for d in _DIR:
        shutil.rmtree(d, ignore_errors=True)
    _DIR.clear()


import atexit                                                           # noqa: E402
atexit.register(cleanup)
