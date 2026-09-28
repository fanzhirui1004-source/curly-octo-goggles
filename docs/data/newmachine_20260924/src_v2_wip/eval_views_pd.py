"""Per-direction variant of eval_views.py (revision round 1, E2a + E5; NEW script, eval_views.py is unchanged).
Same protocol as eval_views.py (same geometry loading: slot cache, else built from body / data; same bank cleaning; same
energy expression TL.energy(field, K) - 1 in chunks of --chunk directions; same model construction and weight loading,
with the per-model override applied exactly as eval_views applies env OPL_MODEL_ARGS_OVERRIDE), but
  - several models in ONE pass over the geometries (each geometry is loaded once and evaluated by every model in turn;
    the physics wrapper's per-cell set-up depends only on the cell and the wrapper parameters, not on the network);
  - besides the per-geometry mean (per_geo, identical format to eval_views), it stores every per-direction error
    (per_dir[case][view][class] = list) and per-geometry statistics (dir_stats: n, mean, median, p90, p95, p99, max);
  - one output JSON per model, <out_dir>/<prefix><NAME>.json, rewritten after every geometry (partial runs keep what
    finished; --resume skips the geometries already present for every model);
  - the environment actually in force is recorded ('env': precision flags, relevant env vars, torch / GPU, ckpt md5).
Usage: eval_views_pd.py <out_dir> --model NAME=CKPT[;OVERRIDE_JSON] [--model ...] --cases c1,c2,... [--views 0]
                        [--body DIR] [--data DIR] [--chunk 16] [--prefix newval3_] [--resume] [--mem-frac F]
Example (C+W): --model 'CW=/root/autodl-tmp/OPL/S1/V2/A0_ctrl/best.pt;{"smooth_k": 8, "smooth_alpha": 30.0, "coarse_space": "Q1_17"}'
"""
import os, sys, json, time, gc, argparse, hashlib, socket
from pathlib import Path
import numpy as np
import torch
import models as MD                                                    # first: applies OPL_CONV_FP32
import trainlib as TL
import train2 as T2
import oh

dev = TL.dev
ENV_KEYS = ('OPL_CONV_FP32', 'FUSED_HYPER', 'SENS_REASSOC', 'LAT_CPU', 'OPL_MODEL_ARGS_OVERRIDE', 'OPL_GP_CACHE',
            'OPL_PACKETS_EXTRA', 'OPL_DEV', 'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')


def md5(path, bs=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def env_record():
    r = dict(TL.conv_precision())
    r.update(cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
             matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
             float32_matmul_precision=torch.get_float32_matmul_precision(),
             torch=torch.__version__, cuda=torch.version.cuda, host=socket.gethostname(),
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             env={k: os.environ.get(k) for k in ENV_KEYS}, argv=sys.argv, script=os.path.abspath(__file__),
             script_md5=md5(os.path.abspath(__file__)), energy_dtype=str(TL.dt),
             note='network forward fp32 (true fp32 convolutions when conv_tf32 is False); energies and wrapper fp64')
    return r


def stats(e):
    e = np.asarray(e, float)
    f = e[np.isfinite(e)]
    if f.size == 0:
        return dict(n=int(e.size), n_finite=0)
    return dict(n=int(e.size), n_finite=int(f.size), mean=float(f.mean()), median=float(np.median(f)),
                p90=float(np.percentile(f, 90)), p95=float(np.percentile(f, 95)), p99=float(np.percentile(f, 99)),
                max=float(f.max()), min=float(f.min()), argmax=int(np.nanargmax(np.where(np.isfinite(e), e, -np.inf))))


def parse_model(s):
    name, rest = s.split('=', 1)
    ck, ov = (rest.split(';', 1) + [''])[:2]
    return name, ck, (json.loads(ov) if ov.strip() else None)


def load_geo(case, slots, body, data):
    sf = Path(slots) / f'{case}.pt'
    if sf.exists():
        g = torch.load(sf, map_location='cpu', weights_only=False)
        T2.move(g, dev); g.C.K = g.C
    else:                                                               # new geometry: build from body / data
        g = T2.build_geo(case, dict(body=body, data=data))
        T2.move(g, dev)
    T2.clean_banks(g, case, lambda d_: None)
    return g


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--model', action='append', required=True, help='NAME=CKPT[;OVERRIDE_JSON]')
    ap.add_argument('--views', default='0'); ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--slots', default='/root/autodl-tmp/OPL/S2/slots')
    ap.add_argument('--cases', required=True, help='comma list')
    ap.add_argument('--body', default=None, help='default: first ckpt cfg'); ap.add_argument('--data', default=None)
    ap.add_argument('--prefix', default='newval3_'); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--mem-frac', type=float, default=None, help='cap this process at a fraction of device memory (smoke tests)')
    a = ap.parse_args(argv)
    if a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    ks = [int(k) for k in a.views.split(',')]
    cases = [c for c in a.cases.split(',') if c]
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    env = env_record()
    M = []
    for s in a.model:
        name, ckp, ov = parse_model(s)
        ck = torch.load(ckp, map_location=dev, weights_only=False)
        cfg = ck['cfg']
        ma = dict(cfg.get('model_args', {}))
        if ov:
            ma.update(ov)
        out = out_dir / f'{a.prefix}{name}.json'
        rec = dict(name=name, ckpt=ckp, ckpt_md5=md5(ckp), override=ov, model_args_used=ma, views=ks, per_geo={}, per_dir={},
                   dir_stats={}, ckpt_step=ck.get('step'), ckpt_weights=ck.get('weights'), ckpt_score=ck.get('score'),
                   conv=dict(TL.conv_precision(), ckpt_conv_fp32=cfg.get('conv_fp32')), env=env, seconds_model=0.0)
        if a.resume and out.exists():
            old = json.loads(out.read_text())
            if old.get('ckpt_md5') == rec['ckpt_md5'] and old.get('override') == ov and old.get('views') == ks:
                for key in ('per_geo', 'per_dir', 'dir_stats'):
                    rec[key] = old.get(key, {})
                rec['seconds_model'] = old.get('seconds_model', 0.0)
                rec['resumed_from'] = sorted(rec['per_geo'])
        M.append(dict(name=name, ck=ck, cfg=cfg, ma=ma, out=out, rec=rec, model=None))
    body = a.body or M[0]['cfg']['body']; data = a.data or M[0]['cfg']['data']
    t0 = time.perf_counter(); geo_s = 0.0
    for case in cases:
        todo = [m for m in M if case not in m['rec']['per_geo']]
        if not todo:
            continue
        tg = time.perf_counter()
        g = load_geo(case, a.slots, body, data)
        geo_s += time.perf_counter() - tg
        line = dict(case=case)
        for m in todo:
            tm = time.perf_counter()
            if m['model'] is None:
                model = MD.build(m['cfg']['model'], [g], **m['ma']).to(dev)
                (MD.load_compat(model, m['ck']['model']) if hasattr(MD, 'load_compat') else model.load_state_dict(m['ck']['model'], strict=False))
                model.eval(); m['model'] = model
            else:
                m['model'].add_geo(g)
            model = m['model']
            r, rd, rs = {}, {}, {}
            with torch.no_grad():
                for k in ks:
                    v = g if k == 0 else oh.view(model, g, k)
                    kk = str(k)                                          # JSON key form (as eval_views' output)
                    r[kk], rd[kk], rs[kk] = {}, {}, {}
                    for c in g.classes:
                        Q = g.banks['val'][c]
                        e = torch.cat([TL.energy(v.field(model, Q[:, j:j + a.chunk]), g.C.K) - 1 for j in range(0, Q.shape[1], a.chunk)])
                        r[kk][c] = float(e.mean())
                        en = e.detach().double().cpu().numpy()
                        rd[kk][c] = en.tolist(); rs[kk][c] = stats(en)
                    if k != 0:
                        oh.drop(model, v)
            rec = m['rec']
            rec['per_geo'][case], rec['per_dir'][case], rec['dir_stats'][case] = r, rd, rs
            rec['seconds_model'] += time.perf_counter() - tm
            rec['max_mem_GB'] = max(rec.get('max_mem_GB', 0.0), torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0)
            model.caches.pop(case, None)
            m['out'].write_text(json.dumps(rec))
            k0 = str(ks[0])
            line[m['name']] = {c: [round(r[k0][c], 6), round(rs[k0][c].get('max', float('nan')), 6)] for c in r[k0]}
        print(json.dumps(line), flush=True)
        T2.move(g, 'cpu'); del g; gc.collect(); torch.cuda.empty_cache()
    for m in M:
        rec = m['rec']
        pg, pd_ = rec['per_geo'], rec['per_dir']
        cls = sorted({c for r in pg.values() for c in r[str(ks[0])]})
        rec['mean'] = {str(k): {c: float(np.mean([r[str(k)][c] for r in pg.values() if c in r[str(k)]])) for c in cls} for k in ks}
        rec['pooled_dir_stats'] = {str(k): {c: stats(np.concatenate([np.asarray(r[str(k)][c], float) for r in pd_.values()
                                                                     if c in r[str(k)]])) for c in cls} for k in ks}
        rec['n_geometries'] = len(pg); rec['cases_requested'] = cases
        rec['seconds_total_pass'] = time.perf_counter() - t0; rec['seconds_geometry_load'] = geo_s
        m['out'].write_text(json.dumps(rec))
        print(json.dumps(dict(event='DONE', model=m['name'], n=len(pg), mean=rec['mean'][str(ks[0])],
                              seconds_model=rec['seconds_model'], max_mem_GB=rec.get('max_mem_GB'))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
