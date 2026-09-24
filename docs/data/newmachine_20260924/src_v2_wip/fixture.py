"""Local CPU test fixture (no GPU, no cuDSS): the smallest training geometry and a few rotated-packet NETDATA files.
Usage from a test:  import os; os.environ['OPL_DEV'] = 'cpu'; import fixture as FX; geo = FX.small()
FIX/geo_small.pt   fresh_train_0010_cover01_r1 (14355 DOFs, 9594 port DOFs), a trainlib.Geo pickled from the step-2 slot
                   cache, banks cut to 16 samples per split and class; extra attributes
                   geo.fix_ustar_val[cls]  exact extension of the val bank (nb x 16, fp64, scipy on the remote CPU)
                   geo.fix_react_val[cls]  exact port reactions S q = (K u*)_P (np x 16, fp64)
                   C.U / C.Ut / C.ru were stripped for the transfer and are rebuilt here from crow / cu / vals.
FIX/<case>/        NETDATA.npz, DONE.json, FRESH_CONTEXT.json of fresh_train_0021_cover01_r1 and its rotswapxy,
                   rotgeneral, rotinvert packets (make_rot.ROTS; exact images on the fixed grid, V01), and
                   fresh_train_0010_cover01_r1
FIX/mgno2_r2_d2_best.pt   step-1 MGNO2 checkpoint (r2, arm d2); FIX/s2v1_snap_10000.pt  step-2 v1 at 10k steps
FIX/GP_TEMPLATES_n32.npz  ghost-penalty face templates
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FIX = Path(os.environ.get('OPL_FIX', HERE.parent.parent / 'fixdl' / 'FIX'))
SMALL = 'fresh_train_0010_cover01_r1'
ROT_BASE = 'fresh_train_0021_cover01_r1'
ROTS = {'swapxy': [[0, 1, 0], [1, 0, 0], [0, 0, 1]], 'general': [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
        'invert': [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]}          # make_rot.ROTS (x' = .5 + R (x - .5))


def _rebuild_K(C):
    """U = upper CSR (crow, cu, vals), Ut its transpose, ru the row index of every stored entry; C.K = C (teacher)."""
    nb = C.nb
    crow = C.crow.long()
    C.ru = torch.repeat_interleave(torch.arange(nb, dtype=torch.int32), crow[1:] - crow[:-1])
    C.U = torch.sparse_csr_tensor(crow.int(), C.cu, C.vals, size=(nb, nb))
    C.tperm = torch.argsort(C.cu.long() * nb + C.ru.long()).int()
    C.crow_t = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(torch.bincount(C.cu.long(), minlength=nb), 0)]).int()
    C.col_t = C.ru[C.tperm.long()]
    C.Ut = torch.sparse_csr_tensor(C.crow_t, C.col_t, C.vals[C.tperm.long()], size=(nb, nb))
    C.K = C


def small():
    """The small geometry as a CPU trainlib.Geo (requires OPL_DEV=cpu before importing trainlib / models)."""
    assert os.environ.get('OPL_DEV') == 'cpu', 'set OPL_DEV=cpu before importing fixture'
    sys.path.insert(0, str(HERE))
    import trainlib  # noqa: F401  (classes needed to unpickle)
    g = torch.load(FIX / 'geo_small.pt', map_location='cpu', weights_only=False)
    _rebuild_K(g.C)
    return g


def scipy_K(C):
    """Full symmetric K as scipy CSR (fp64) from the upper CSR."""
    import scipy.sparse as sp
    U = sp.csr_matrix((C.vals.double().numpy(), C.cu.long().numpy(), C.crow.long().numpy()), shape=(C.nb, C.nb))
    return (U + U.T - sp.diags(U.diagonal())).tocsr()


def netdata(case):
    return dict(np.load(FIX / case / 'NETDATA.npz'))


def checkpoint(name='mgno2_r2_d2_best.pt'):
    return torch.load(FIX / name, map_location='cpu', weights_only=False)


def model_for(geo, name='mgno2_r2_d2_best.pt', **over):
    """Build the checkpoint's model on the given geometry (CPU) and load its weights."""
    import models as MD
    ck = checkpoint(name)
    cfg = ck['cfg']
    args = dict(cfg.get('model_args', {})); args.update(over)
    m = MD.build(cfg['model'], [geo], **args)
    m.load_state_dict(ck['model'], strict=False)
    return m.eval()
