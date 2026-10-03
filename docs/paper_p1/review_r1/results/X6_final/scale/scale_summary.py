"""Summary of the scale demonstration on the final route (runs/ copied read-only from R1/FINAL2: <run>/history.jsonl,
<run>.log, rss_<run>.tsv).  Plates of 24, 51, 88, 110 and 135 cells (layouts of R1/SCALE), clamp x = min, in-plane load on
x = max in y, four analyses each, learned substructures resident on the GPU up to 4 GiB and streamed beyond (packed).
Degrees of freedom depend on the geometry only and are taken from ../../X6_opt/scale/scale_dofs.json (same layouts and
initial design).  Writes scale_summary.json next to this script."""
import json, re, os
HERE = os.path.dirname(os.path.abspath(__file__)); H = os.path.join(HERE, 'runs')
DOFS = json.load(open(os.path.join(HERE, '..', '..', 'X6_opt', 'scale', 'scale_dofs.json')))['runs']
out = {}


def sampler(run):
    f = f'{H}/rss_{run}.tsv'
    if not os.path.exists(f):
        return {}
    R = [l.split() for l in open(f) if l.strip()]
    return dict(sampler_rss_max_gib=round(max(float(r[1]) for r in R), 2),
                sampler_gpu_used_max_gib=round(max(int(r[2]) for r in R) / 1024, 2), sampler_samples=len(R))


for run in ['plateS24', 'plateS51', 'plateS88', 'plateS110', 'plateS135']:
    hf = f'{H}/{run}/history.jsonl'
    log = open(f'{H}/{run}.log').read() if os.path.exists(f'{H}/{run}.log') else ''
    starts = [json.loads(l) for l in log.splitlines() if l.startswith('{') and '"event": "START"' in l]
    L = [json.loads(l) for l in open(hf)] if os.path.exists(hf) else []
    d = next((v for k, v in DOFS.items() if k.startswith(run)), None)
    rec = dict(cells=starts[-1]['cells'] if starts else None, analyses=len(L), dofs=d, **sampler(run))
    if L:
        rec.update(iter_s=[round(x['times']['iter_s'], 1) for x in L],
                   phases_mean={k: round(sum(x['times'].get(k, 0) for x in L) / len(L), 1) for k in L[0]['times'] if k.endswith('_s')},
                   pcg=[x['pcg'] for x in L], true_residual=[x['true_residual'] for x in L], Ut_rho_rel=[x['Ut_rho_rel'] for x in L],
                   C=[x['C'] for x in L], gpu_peak_gb=round(max(x['gpu_peak_gb'] for x in L), 2),
                   host_peak_gb=round(max(x['host_peak_gb'] for x in L), 2), body_perturb=[x['body_perturb'] for x in L])
    else:
        rec['errors'] = re.findall(r'(cuDSSError: [A-Z_]+ \(\d+\)|torch\.OutOfMemoryError[^\n]*|CUDA out of memory[^\n]{0,120}|PCG_NOT_CONVERGED)', log)
    out[run] = rec
json.dump(out, open(os.path.join(HERE, 'scale_summary.json'), 'w'), indent=1)
for k, v in out.items():
    print(k, {kk: v[kk] for kk in v if kk not in ('C', 'true_residual', 'Ut_rho_rel', 'body_perturb', 'dofs')})
