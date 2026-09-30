"""Summary of the Section 6.11 scale runs (runs/ copied read-only from the 5090 host, R1/SCALE).
Reported set (author decision 2026-10-01): 24 (plateS24r4), 51 (plateS51r4, two analyses), 88, 110 cells, all with cells
resident on the GPU up to 4 GB, exact packed storage and the sparse coarse space; 135 cells failed (cuDSS ALLOC_FAILED in
the analysis phase of the K_PP factor). plateS24 (18 GB resident budget) and plateS51 (failed at 18 GB) are kept for the record only."""
import json, re, os
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
out = {}


def sampler(run):
    """30-s samples of rss_<run>.tsv (runs/rss_sampler_scripts.txt): RSS of the opt_design process in GiB (ru of ps, KiB/2**20)
    and device memory used on the GPU in MiB (nvidia-smi, whole device, only this process running) - unlike gpu_peak_gb
    (torch.cuda.max_memory_allocated) it includes the memory of the cuDSS factor of K_PP."""
    f = f'{H}/rss_{run}.tsv'
    if not os.path.exists(f):
        return {}
    R = [l.split() for l in open(f) if l.strip()]
    return dict(sampler_rss_max_gib=round(max(float(r[1]) for r in R), 2),
                sampler_gpu_used_max_gib=round(max(int(r[2]) for r in R) / 1024, 2), sampler_samples=len(R))
for run in ['plateS24r4', 'plateS51r4', 'plateS88', 'plateS110', 'plateS24']:
    L = [json.loads(l) for l in open(f'{H}/{run}/history.jsonl')]
    starts = [json.loads(l) for l in open(f'{H}/{run}.log') if l.startswith('{') and '"event": "START"' in l]
    a = starts[-1]['args']
    out[run] = dict(cells=starts[-1]['cells'], vertices=starts[-1]['vertices'], free=starts[-1]['free'],
                    resident_gb=a['resident_gb'], analyses=len(L), env={k: v for k, v in starts[-1]['env'].items() if k.startswith('OPL_')},
                    iter_s=[round(x['times']['iter_s'], 1) for x in L],
                    phases_mean={k: round(sum(x['times'][k] for x in L) / len(L), 1) for k in L[0]['times'] if k.endswith('_s')},
                    pcg=[x['pcg'] for x in L], true_residual=[x['true_residual'] for x in L], Ut_rho_rel=[x['Ut_rho_rel'] for x in L],
                    C=[x['C'] for x in L], gpu_peak_gb=round(max(x['gpu_peak_gb'] for x in L), 2),
                    host_peak_gb=round(max(x['host_peak_gb'] for x in L), 2), body_perturb=[x['body_perturb'] for x in L])
    out[run].update(sampler(run))
for run in ['plateS135', 'plateS51']:
    s = open(f'{H}/{run}.log').read()
    m = re.findall(r'(cuDSSError: [A-Z_]+ \(\d+\)|torch\.OutOfMemoryError[^\n]*|CUDA out of memory[^\n]{0,120})', s)
    starts = [json.loads(l) for l in s.splitlines() if l.startswith('{') and '"event": "START"' in l]
    out[run + '_fail'] = dict(cells=starts[-1]['cells'] if starts else None, attempts=len(starts),
                              resident_gb=[st['args']['resident_gb'] for st in starts], errors=m, **sampler(run))
json.dump(out, open(os.path.join(os.path.dirname(H), 'scale_summary.json'), 'w'), indent=1)
for k, v in out.items():
    print(k, {kk: v[kk] for kk in v if kk not in ('env', 'C', 'true_residual', 'Ut_rho_rel', 'body_perturb')})
