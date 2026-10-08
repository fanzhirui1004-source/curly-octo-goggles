"""Smoke-test wrapper. argv: <frac> <script> [args]. OPL_DEV=cpu: CPU environment of p1_checks.py (diag_sens patches,
interior factors by host PARDISO / SuperLU), never touching the GPU; otherwise caps this process's torch allocations."""
import sys, os, runpy
frac = float(sys.argv[1]); script = os.path.abspath(sys.argv[2])
sys.path.insert(0, os.path.dirname(script))
if os.environ.get('OPL_DEV') == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import diag_sens as DS
    import teacher as TE
    TE.Cell.factor = lambda self, neumann=True, fp32=False, **kw: DS.cpu_factor(self)
    import torch, box_encode as BX
    BX.dev = torch.device('cpu')
else:
    import torch
    torch.cuda.set_per_process_memory_fraction(frac, 0)
sys.argv = [script] + sys.argv[3:]
runpy.run_path(script, run_name='__main__')
