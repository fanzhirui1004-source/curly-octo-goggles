"""Find host synchronisations inside one S_hat application (after warm-up): torch.cuda.set_sync_debug_mode('warn') reports
every synchronising PyTorch call with its Python stack; then the CPU time to issue one S_hat behind a long GPU sleep.
usage: python t_sync.py BODY_DIR:CASE"""
import os, sys, json, time, warnings, traceback
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI

FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
C.store64()
g = torch.Generator(device='cuda'); g.manual_seed(0)
q = torch.randn((C.np_, 1), dtype=torch.float64, device='cuda', generator=g)
with torch.no_grad():
    f.s_hat(q); f.s_hat(q); torch.cuda.synchronize()
    seen = []

    def show(message, category, filename, lineno, file=None, line=None):
        st = ''.join(traceback.format_stack(limit=14)[:-2])
        seen.append(dict(msg=str(message)[:200], stack=st[-2500:]))
    warnings.showwarning = show
    warnings.simplefilter('always')
    torch.cuda.set_sync_debug_mode('warn')
    f.s_hat(q)
    torch.cuda.set_sync_debug_mode('default')
    torch.cuda.synchronize()
    for s in seen[:6]:
        print('=== SYNC:', s['msg']); print(s['stack'], flush=True)
    print(json.dumps(dict(n_sync_warnings=len(seen))), flush=True)
    ts = []
    for _ in range(3):
        torch.cuda._sleep(int(2e9)); t = time.perf_counter(); f.s_hat(q); ts.append(time.perf_counter() - t); torch.cuda.synchronize()
    print(json.dumps(dict(cpu_issue_ms=[round(1e3 * x, 1) for x in ts])), flush=True)
