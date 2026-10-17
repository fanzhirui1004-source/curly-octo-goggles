"""Probe of the deployed model's slot-weight heads (for P17): bounded / b3 / fringe_soft, clip states and knee, head
shapes, the plate cell's sizes (E, faces, N, fringe subsets), and the time of today's recompute (_slot_tensors,
_assign_slots). usage: python t_slotprobe.py BODY_DIR:CASE"""
import os, sys, json, time
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
m, c = f.model, f.c
r = dict(case=case, cls=type(m).__name__, bounded=bool(m.bounded), b3=bool(getattr(m, 'b3', False)),
         fringe_soft=bool(getattr(m, 'fringe_soft', False)), knee=float(m.bound_knee), bst=dict(m._bst),
         H=m.H, F=m.F, L=m.L_pre + m.L_post, n_fringe=getattr(m, 'n_fringe', 0),
         E=int(c.en.shape[0]), nf=int(len(c.gp_owner)) if hasattr(c, 'gp_owner') else None, N=int(c.N),
         el_fringe=int(c.el_fringe.sum()) if hasattr(c, 'el_fringe') else None,
         gp_fringe=int(c.gp_fringe.sum()) if hasattr(c, 'gp_fringe') else None,
         heads={k: [str(type(x).__name__) + (str(tuple(x.weight.shape)) if hasattr(x, 'weight') else '') for x in getattr(m, k)]
                for k in ('slot_head', 'slot_head_x', 'fslot_head', 'face_in') if hasattr(m, k)},
         slot_shape=list(m.slot.shape), ge=list(f._rc_src['ge'].shape) if f._rc_src else None,
         gn=list(f._rc_src['gn'].shape) if f._rc_src else None, slot_MiB=round(f.slot_bytes() / 2 ** 20, 1),
         gelu=str(m.slot_head[1]))
for k in ('bnd_ab', 'bnd_fab', 'bnd_xab'):
    if hasattr(m, k):
        r[k] = getattr(m, k).detach().cpu().tolist()
print(json.dumps(r), flush=True)
if f._rc_src is not None:
    f._rc = f._rc_src
    with torch.no_grad():
        for _ in range(2):
            f._assign_slots(*f._slot_tensors())
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(10):
            ab, fab, xab = f._slot_tensors()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        for _ in range(10):
            f._assign_slots(ab, fab, xab)
        torch.cuda.synchronize(); t2 = time.perf_counter()
    print(json.dumps(dict(slot_tensors_ms=round(100 * (t1 - t0), 2), assign_ms=round(100 * (t2 - t1), 2),
                          shapes=dict(ab=list(ab.shape), fab=list(fab.shape), xab=list(xab.shape)))), flush=True)
