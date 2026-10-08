"""Revision round 1 (E2b, E6): runs lat_full.py unchanged (same argv, same protocol) and adds to its output JSON what
lat_full does not record: the environment actually in force (precision flags, LAT_CPU / FUSED_HYPER / override env,
torch / GPU), the checkpoint md5, and, when lat_full raises (e.g. EMPTY_LOAD: no loaded port on the glued face, or a
lattice Cholesky failure), a failure record with the exception, so that every attempted configuration leaves a JSON.
Also adds a per-configuration 'r1_flags' block: PCG at the cap, negative energy share, non-finite compliance errors.
Usage: exactly as lat_full.py:  r1_gate.py <out.json> --pairs '' --custom CKPT:CASE:CKPT: --nb-mode explicit --sets test --configs x"""
import os, sys, json, traceback, hashlib, socket, time
from pathlib import Path
import models as MD                                                    # noqa: F401  first: applies OPL_CONV_FP32
import numpy as np
import torch
import lat_full as LF

KEYS = ('OPL_CONV_FP32', 'FUSED_HYPER', 'LAT_CPU', 'OPL_MODEL_ARGS_OVERRIDE', 'OPL_GP_CACHE', 'OPL_PACKETS_EXTRA',
        'SENS_REASSOC', 'OPL_DEV', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'LAT_SKIP_NBR_SENS')


def md5(p, bs=1 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def env_record(argv):
    ck = ''
    if '--custom' in argv:
        ck = argv[argv.index('--custom') + 1].split(':')[0]
    return dict(cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                float32_matmul_precision=torch.get_float32_matmul_precision(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, host=socket.gethostname(),
                env={k: os.environ.get(k) for k in KEYS}, argv=argv, ckpt=ck, ckpt_md5=md5(ck) if ck and os.path.exists(ck) else None,
                script_md5=md5(os.path.abspath(__file__)), lat_full_md5=md5(LF.__file__),
                note='reference and PCG in fp64 (dense lattice Cholesky on the host with LAT_CPU=1); network forward fp32')


def flags(r, maxit):
    t = r.get('test', {})
    ce = np.asarray(t.get('compliance_rel_err', []), float)
    es = np.asarray(r.get('energy_share', []), float)
    return dict(pcg_at_cap=bool(t.get('pcg_iterations') is not None and t['pcg_iterations'] >= maxit),
                negative_energy_share=bool(es.size and (es < 0).any()), min_energy_share=float(es.min()) if es.size else None,
                nonfinite_compliance=bool(ce.size and (~np.isfinite(ce)).any()),
                n_nonfinite_compliance=int((~np.isfinite(ce)).sum()) if ce.size else 0)


def main(argv):
    out = Path(argv[0])
    maxit = int(argv[argv.index('--maxit') + 1]) if '--maxit' in argv else 400
    env = env_record(argv)
    t0 = time.perf_counter()
    try:
        LF.main(argv)
        rc, err = 0, None
    except BaseException as e:                                           # record, then fail
        rc, err = 1, dict(type=type(e).__name__, message=str(e)[:2000], traceback=traceback.format_exc()[-6000:])
    rec = json.loads(out.read_text()) if out.exists() else dict(args=argv, results=[])
    rec['r1_env'] = env
    rec['r1_seconds'] = time.perf_counter() - t0
    rec['r1_status'] = 'ok' if rc == 0 else 'failed'
    if err:
        rec['r1_error'] = err
    for r in rec.get('results', []):
        r['r1_flags'] = flags(r, maxit)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(event='R1_GATE', out=str(out), status=rec['r1_status'], error=(err or {}).get('message'),
                          flags=[r['r1_flags'] for r in rec.get('results', [])])), flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
