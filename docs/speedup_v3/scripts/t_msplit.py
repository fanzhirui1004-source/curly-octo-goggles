"""P12 check (OPL_V3_MSPLIT_IP): the adjoint network's mask split in place over the port rows only
(netfused.mask_split_inplace) against the copying form (netfused.mask_split), on a full plate cell's port mask:
(1) the kernels alone on random data, B = 1, 3, 16: X0b and the split array compared bitwise, and their device times;
(2) the adjoint network map ext_T with the switch off / on (random right-hand sides; the layers' atomic scatters make
reruns differ: reported against the off path's rerun difference).
usage: python t_msplit.py BODY_DIR:CASE"""
import os, sys, json
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ.setdefault('OPL_KE_SYMV', '1'); os.environ.setdefault('OPL_KE_SYMV64', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import netfused as NF

dev = 'cuda'
FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
g = torch.Generator(device=dev); g.manual_seed(0)


def ev(fn, rep=50):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(rep):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1000 * s.elapsed_time(e) / rep                              # us


res = dict(case=case, N=int(f.N), port_rows=int(len(f._pidx)), F=int(f.W_out.shape[0]))
with torch.no_grad():
    for B in (1, 3, 16):
        Fc = res['F']
        X = torch.randn((f.N, B, Fc), dtype=torch.float32, device=dev, generator=g)
        X0 = torch.randn((f.N, B, Fc), dtype=torch.float32, device=dev, generator=g)
        a0, b0 = X.clone(), X0.clone()
        A = NF.mask_split(a0, b0, f.pm8)
        a1, b1 = X.clone(), X0.clone()
        A1 = NF.mask_split_inplace(a1, b1, f._pidx)
        r = dict(split_equal=bool(torch.equal(A, A1)), x0b_equal=bool(torch.equal(b0, b1)), same_object=A1 is a1)
        Xs, X0s = X.clone(), X0.clone()
        r['us_copy'] = round(ev(lambda: NF.mask_split(Xs, X0s, f.pm8)), 1)
        r['us_inplace'] = round(ev(lambda: NF.mask_split_inplace(Xs, X0s, f._pidx)), 1)
        y = torch.randn((3 * f.N, B), dtype=torch.float32, device=dev, generator=g)
        FN.MSPLIT_IP = False
        z0 = f.ext_T(y).clone(); z0b = f.ext_T(y).clone()
        FN.MSPLIT_IP = True
        z1 = f.ext_T(y).clone()
        FN.MSPLIT_IP = False
        r.update(ext_T_max_abs_on_off=float((z1 - z0).abs().max()), ext_T_max_abs_rerun=float((z0b - z0).abs().max()),
                 scale=float(z0.abs().max()))
        res[f'B{B}'] = r
        print(json.dumps({f'B{B}': r}), flush=True)
print(json.dumps(res), flush=True)
