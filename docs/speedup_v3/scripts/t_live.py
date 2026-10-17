"""Item 1 follow-up: which per-cell device tensors does an analysis still read after the setup? On a full plate cell (the
state slim_deploy leaves, moments and face arrays dropped), every CUDA tensor of the cell's objects (fastnet, geometry
wrapper, teacher Cell; >= 64 KiB; the storage once) is overwritten in turn (floating point: NaN; integers: 0, a valid
index) and S_hat (B = 1) and the sensitivity pass's field (fastnet.field, energy_density) are evaluated: a result
unchanged up to the rerun difference means the tensor is not read there. Prints per tensor: path, dtype, MiB, read by
S_hat, read by the field / energy pass.
usage: python t_live.py BODY_DIR:CASE"""
import os, sys, json
SRC = os.environ.get('SRC', '/root/autodl-tmp/OPL/src_v3'); sys.path.insert(0, SRC)
os.environ.setdefault('OPL_TAILT_FUSED', '1'); os.environ.setdefault('OPL_COARSE_FP32', '1')
os.environ.setdefault('OPL_COARSE_ELEM', '1'); os.environ.setdefault('OPL_TET_TRITON', '1')
import models as MD  # noqa: F401
import torch
import teacher as TE, trainlib as TL, fastnet as FN, bench_deploy as BD, fastidx as FI, evalnet as EN

FN.FUSED = True; FI.ON = True
body, case = sys.argv[1].split(':')
h = BD.ModelHolder('/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'); BD.TMP.mkdir(parents=True, exist_ok=True)
C = TE.Cell(case, body, log=lambda s_: None, deploy=True); C.assemble_deploy()
BD.netdata(C, body, BD.TMP / case)
geo = TL.Geo(case, body, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
model = h.add(geo)
op = EN.FastOp(FN.FastNet(model, geo))
dev = 'cuda'
f64 = torch.float64
g = torch.Generator(device=dev); g.manual_seed(0)
q = torch.randn((C.np_, 1), dtype=f64, device=dev, generator=g)
with torch.no_grad():
    op.apply(q)                                                         # warm-up (states built)
FN.slim_deploy(op, geo, C, model)
fast = op.fast


def run_shat():
    with torch.no_grad():
        return op.apply(q).clone()


def run_field():
    with torch.no_grad():
        u = fast.field(q).to(f64)
        return C.energy_density(u)[:, :, 0].clone()


ref_s, ref_s2 = run_shat(), run_shat()
ref_f, ref_f2 = run_field(), run_field()
noise_s = float((ref_s2 - ref_s).abs().max() / ref_s.abs().max())
noise_f = float((ref_f2 - ref_f).abs().max() / ref_f.abs().max())

SKIP = (torch.nn.Module,)
seen, items = set(), []


def walk(path, o, depth=0):
    if depth > 6 or id(o) in seen:
        return
    seen.add(id(o))
    if isinstance(o, dict):
        kv = [(f'{path}[{k}]', v) for k, v in o.items()]
    elif isinstance(o, (list, tuple)):
        kv = [(f'{path}[{i}]', v) for i, v in enumerate(o)]
    else:
        try:
            kv = [(f'{path}.{k}', v) for k, v in vars(o).items()]
        except TypeError:
            return
    for p, v in kv:
        if torch.is_tensor(v):
            parts = [('values', v.values())] if v.layout == torch.sparse_csr else [('', v)]
            if v.layout == torch.sparse_csr:
                parts += [('col', v.col_indices()), ('crow', v.crow_indices())]
            for tag, t in parts:
                if t.is_cuda and t.untyped_storage().nbytes() >= 64 * 1024:
                    k = t.untyped_storage().data_ptr()
                    if k not in {x[0] for x in items}:
                        items.append((k, p + (f'.{tag}' if tag else ''), t))
        elif isinstance(v, SKIP):
            continue
        elif isinstance(v, (dict, list, tuple)) or hasattr(v, '__dict__'):
            walk(p, v, depth + 1)


walk('net', fast); walk('geo', geo); walk('C', C)
out = []
for k, p, t in items:
    st = t.untyped_storage()
    full = torch.empty(0, dtype=t.dtype, device=t.device).set_(st, 0, (st.nbytes() // t.element_size(),))
    keep = full.clone()
    full.fill_(float('nan') if t.is_floating_point() else 0)
    try:
        ys = run_shat()
        rs = float(((ys - ref_s).abs().max() / ref_s.abs().max()).nan_to_num(float('inf')))
    except Exception as e:                                              # noqa: BLE001
        rs = f'error {type(e).__name__}'
    try:
        yf = run_field()
        rf = float(((yf - ref_f).abs().max() / ref_f.abs().max()).nan_to_num(float('inf')))
    except Exception as e:                                              # noqa: BLE001
        rf = f'error {type(e).__name__}'
    full.copy_(keep)
    torch.cuda.synchronize()
    rec = dict(path=p, dtype=str(t.dtype).replace('torch.', ''), MiB=round(st.nbytes() / 2 ** 20, 2),
               shat=(rs if isinstance(rs, str) else rs > 10 * noise_s + 1e-12), field=(rf if isinstance(rf, str) else rf > 10 * noise_f + 1e-12))
    out.append(rec)
    print(json.dumps(rec), flush=True)
unread = [r for r in out if r['shat'] is False and r['field'] is False]
print(json.dumps(dict(case=case, noise_shat=noise_s, noise_field=noise_f, tensors=len(out),
                      MiB_total=round(sum(r['MiB'] for r in out), 1),
                      MiB_unread=round(sum(r['MiB'] for r in unread), 1),
                      unread=[(r['path'], r['MiB']) for r in unread])), flush=True)
