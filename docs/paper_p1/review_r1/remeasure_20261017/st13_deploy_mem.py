"""ST13 re-measurement (2026-10-17): the deployed per-cell state of the four deployment cells G1-G4, on the deployment
code path of Table 5 (Cell(deploy=True) and assemble_deploy, netdata, Geo, FastNet with the fused hyperedge layers, one
warm-up application that builds the correction caches; lat_scale.py does the same per cell).

Memory after the warm-up, with the allocator cache emptied:
  by_attr     CUDA storage reachable from the cell (C.*), its geometry (geo.*) and its frozen network (net.*), each storage
              counted once, credited to the first top-level attribute under which it is found (C, then geo, then net);
  groups      by_attr summed into stiffness (element moments, float32 copy, element DOF map, ghost-face data, diagonal),
              correction (coarse factor or inverse, coarse prolongation), kpp (K_PP upper triplets, used only to assemble
              the lattice preconditioner), geometry and network;
  alloc_delta torch.cuda.memory_allocated() after minus before;  device_delta  torch.cuda.mem_get_info() free before minus
              after (whole device).
The first pass over G2 only compiles the kernels and is not reported.

Usage: python st13_deploy_mem.py <out.json>"""
import os, sys, json, time, gc, types
os.environ.setdefault('OPL_GP_CACHE', '0')
os.environ.setdefault('OPL_CONV_FP32', '1')
os.environ.setdefault('OPL_COARSE_PIVOT_FLOOR', '1e-7')
os.environ['OPL_TAILT_FUSED'] = '1'                     # as lat_scale.main()
os.environ['OPL_COARSE_FP32'] = '1'
os.environ['OPL_COARSE_ELEM'] = '1'                     # --coarse-tpl
os.environ['OPL_TET_TRITON'] = '1'                      # --tet-triton
SRC = '/root/autodl-tmp/OPL/src_v2'
sys.path.insert(0, SRC)
import models as MD                                     # noqa: F401  first: applies OPL_CONV_FP32
import torch
import teacher as TE
import trainlib as TL
import fastnet as FN
import evalnet as EN
import bench_deploy as BD
import fastidx as FI

FN.FUSED = True                                         # as lat_scale.main()
FI.ON = True                                            # --fastidx
dev, dt = TE.dev, TE.dt
BODY = '/root/autodl-tmp/OPL/S0'
MODEL = '/root/autodl-tmp/OPL/S1/V2/A3_2grid/best.pt'
CASES = [('G1', 'fresh_train_0020_cover01_r2'), ('G2', 'fresh_train_0020_cover01_r1'),
         ('G3', 'fresh_train_0020_full'), ('G4', 'fresh_train_0007_full')]
GiB = 2 ** 30
SKIP = (torch.nn.Module, types.ModuleType, type, types.FunctionType, types.MethodType, types.BuiltinFunctionType)
CORR = ('_cL', '_cL32', '_cL32_keep', '_cAi32', '_cV', '_cV32', '_cV32t')


def sync():
    torch.cuda.synchronize()


def parts(t):
    if t.layout == torch.sparse_csr:
        return [t.crow_indices(), t.col_indices(), t.values()]
    if t.layout == torch.sparse_coo:
        return [t._indices(), t._values()]
    return [t]


def walk(root, prefix, seen_obj, seen_store, out):
    stack = [(root, prefix)]
    while stack:
        o, path = stack.pop()
        if id(o) in seen_obj:
            continue
        seen_obj.add(id(o))
        if isinstance(o, dict):
            items = [(str(k), v) for k, v in o.items()]
        elif isinstance(o, (list, tuple)):
            items = [(str(i), v) for i, v in enumerate(o)]
        else:
            try:
                items = list(vars(o).items())
            except TypeError:
                continue
        for k, v in items:
            p = f'{prefix}.{k}' if path == prefix else path
            if torch.is_tensor(v):
                for t in parts(v):
                    if t.is_cuda:
                        s = t.untyped_storage()
                        if s.data_ptr() not in seen_store:
                            seen_store.add(s.data_ptr())
                            out[p] = out.get(p, 0) + s.nbytes()
            elif isinstance(v, SKIP):
                continue
            elif isinstance(v, (dict, list, tuple)) or hasattr(v, '__dict__'):
                stack.append((v, p))


def group(by_attr):
    g = dict(stiffness=0, correction=0, kpp=0, geometry=0, network=0)
    for k, b in by_attr.items():
        root, _, attr = k.partition('.')
        if root == 'net':
            g['network'] += b
        elif root == 'geo':
            g['geometry'] += b
        elif attr in CORR:
            g['correction'] += b
        elif attr == '_kpp_cache':
            g['kpp'] += b
        else:
            g['stiffness'] += b
    return g


def one(case, h, report=True):
    rec = dict(case=case)
    gc.collect(); torch.cuda.empty_cache(); sync()
    a0, f0 = torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]
    t = time.perf_counter()
    C = TE.Cell(case, BODY, log=lambda s_: None, deploy=True); sync()
    rec['ctor_s'] = time.perf_counter() - t; t = time.perf_counter()
    C.assemble_deploy(C.taus0); sync()
    rec['assemble_deploy_s'] = time.perf_counter() - t; t = time.perf_counter()
    BD.netdata(C, BODY, BD.TMP / case); sync()
    rec['netdata_s'] = time.perf_counter() - t; t = time.perf_counter()
    geo = TL.Geo(case, BODY, BD.TMP, neumann=False, log=lambda s_: None, cell=C, load_banks=False)
    if h.model is not None:
        h.model.caches.pop(case, None)
    model = h.add(geo)
    op = EN.FastOp(FN.FastNet(model, geo)); sync()
    rec['encode_s'] = time.perf_counter() - t; t = time.perf_counter()
    with torch.no_grad():
        op.apply(torch.zeros((C.np_, 1), dtype=dt, device=dev))
    sync()
    rec['warmup_s'] = time.perf_counter() - t
    gc.collect(); torch.cuda.empty_cache(); sync()
    a1, f1 = torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]
    rec['alloc_delta_GiB'] = (a1 - a0) / GiB
    rec['device_delta_GiB'] = (f0 - f1) / GiB
    out, so, ss = {}, set(), set()
    walk(C, 'C', so, ss, out)
    walk(geo, 'geo', so, ss, out)
    walk(op.fast, 'net', so, ss, out)
    rec['walked_total_GiB'] = sum(out.values()) / GiB
    rec['groups_GiB'] = {k: v / GiB for k, v in group(out).items()}
    rec['by_attr_GiB'] = {k: v / GiB for k, v in sorted(out.items(), key=lambda kv: -kv[1]) if v >= 2 ** 20}
    cf = next((getattr(C, a) for a in ('_cAi32', '_cL32', '_cL') if getattr(C, a, None) is not None), None)
    rec.update(dofs=int(C.nb), ports=int(C.np_), elements=int(len(C.cells)),
               n_coarse=int(cf.shape[0]) if cf is not None else None,
               coarse_attr=next((a for a in ('_cAi32', '_cL32', '_cL') if getattr(C, a, None) is not None), None))
    model.caches.pop(case, None)
    del op, geo, C, cf
    gc.collect(); torch.cuda.empty_cache(); sync()
    rec['alloc_after_free_GiB'] = (torch.cuda.memory_allocated() - a0) / GiB
    return rec if report else None


def main():
    out_path = sys.argv[1]
    BD.TMP.mkdir(parents=True, exist_ok=True)
    h = BD.ModelHolder(MODEL)
    res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, model=MODEL, body=BODY,
               env={k: v for k, v in os.environ.items() if k.startswith('OPL_') or k == 'PYTORCH_CUDA_ALLOC_CONF'},
               cells=[])
    one(CASES[1][1], h, report=False)                    # kernel compilation pass, not reported
    for label, case in CASES:
        r = one(case, h)
        r['label'] = label
        res['cells'].append(r)
        print(json.dumps(r), flush=True)
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
