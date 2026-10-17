"""Check of slotfused.clip_scale (S19) against the PyTorch path of fastnet._slot_tensors on a real cell: the three head
outputs (ab, fab, xab) bitwise (torch.equal) and the time of the recompute (fastnet._slot_tensors + _assign_slots) with
and without the fused tail. S19b (slotfused.assign, OPL_V3_SLOTASSIGN): every hyperedge layer's a, b after
_assign_slots with and without it, bitwise, and the time of _assign_slots alone.
usage: python t_slotfuse.py BODY_DIR:CASE [--rep 20]"""
import os, sys, json, time, argparse
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ['OPL_TAILT_FUSED'] = '1'; os.environ['OPL_COARSE_FP32'] = '1'; os.environ['OPL_COARSE_ELEM'] = '1'
os.environ['OPL_TET_TRITON'] = '1'; os.environ['OPL_V3_SLOTS'] = 'recompute'
import models as MD  # noqa
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI
import slotfused as SF

dev, dt = TE.dev, TE.dt
ap = argparse.ArgumentParser(); ap.add_argument('spec'); ap.add_argument('--rep', type=int, default=20)
a = ap.parse_args()
FN.FUSED = True; FI.ON = True
body, case = a.spec.split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
f = FN.FastNet(h.add(geo), geo)
m = f.model
print(json.dumps(dict(case=case, bounded=bool(m.bounded), knee=m.bound_knee, bst=dict(m._bst), H=m.H)), flush=True)


def timed(fn, rep):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize(); return out, 1000 * (time.perf_counter() - t) / rep


with torch.no_grad():
    FN.SLOTFUSE = False
    ref, t0 = timed(f._slot_tensors, a.rep)
    FN.SLOTFUSE = True
    new, t1 = timed(f._slot_tensors, a.rep)
    r = dict(ms_torch=round(t0, 3), ms_fused=round(t1, 3))
    for name, x, y in zip(('ab', 'fab', 'xab'), ref, new):
        r[name] = dict(shape=list(x.shape), equal=bool(torch.equal(x, y)),
                       max_abs_diff=float((x - y).abs().max()), n_diff=int((x != y).sum()))
    FN.SLOTFUSE = False
    _, r['ms_assign_torch'] = timed(lambda: f._assign_slots(*f._slot_tensors()), a.rep)
    FN.SLOTFUSE = True
    _, r['ms_assign_fused'] = timed(lambda: f._assign_slots(*f._slot_tensors()), a.rep)
    T = f._slot_tensors()
    hs = f._hypers()
    SF.ASSIGN = False
    _, t0 = timed(lambda: f._assign_slots(*T), a.rep)
    ref = [(h.a.clone(), h.b.clone()) for h in hs]
    SF.ASSIGN = True
    _, t1 = timed(lambda: f._assign_slots(*T), a.rep)
    eq = [bool(torch.equal(h.a, x) and torch.equal(h.b, y)) for h, (x, y) in zip(hs, ref)]
    r['assign'] = dict(layers=len(hs), equal=f'{sum(eq)}/{len(eq)}', ms_torch=round(t0, 3), ms_kernel=round(t1, 3),
                       contiguous=all(h.a.is_contiguous() and h.b.is_contiguous() for h in hs),
                       max_abs_diff=max(float(max((h.a - x).abs().max(), (h.b - y).abs().max())) for h, (x, y) in zip(hs, ref)))
    _, r['ms_recompute_all_fused'] = timed(lambda: f._assign_slots(*f._slot_tensors()), a.rep)
print(json.dumps(r), flush=True)
