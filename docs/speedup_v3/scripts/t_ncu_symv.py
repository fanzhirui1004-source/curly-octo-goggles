"""Run under Nsight Compute: the float32 element product (_symv_up) alone and inside S_hat, in NVTX ranges 'alone' and
'inshat', so ncu can collect the same kernel's counters in both settings (application replay, no cache control: each
measured call sees its natural cache state).
usage: ncu --nvtx --nvtx-include alone/ --nvtx-include inshat/ -k regex:_symv_up ... python t_ncu_symv.py BODY:CASE"""
import os, sys
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import ke_symv as KS

dev = 'cuda'
FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
g = torch.Generator(device=dev); g.manual_seed(0)
with torch.no_grad():
    q = torch.randn((C.np_, 1), dtype=torch.float64, device=dev, generator=g)
    f.s_hat(q); f.s_hat(q)
    torch.cuda.synchronize()
    Kc = C._Kc
    with Kc.hold():
        P = Kc._kp_tmp
        x = torch.randn((C.nb, 1), dtype=torch.float32, device=dev, generator=g)
        y = torch.zeros_like(x)
        KS.symv_(y, P, C.dofs, x)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push('alone')
        for _ in range(3):
            KS.symv_(y, P, C.dofs, x)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push('inshat')
    f.s_hat(q)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print('done', flush=True)
