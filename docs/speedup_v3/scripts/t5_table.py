"""Table 5 lattices: baseline (remeasure 2026-10-17, src_v2 on this machine, FINAL3 command: docs/paper_p1/review_r1/
results/X6_final/remeasure_20261017/learned_<lat>.json + gpu_<lat>.csv) vs src_v3 runs (docs/speedup_v3/results/<tag>_t5).
usage: python t5_table.py TAG"""
import sys, json, csv, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(H, '..', '..', 'paper_p1', 'review_r1', 'results', 'X6_final', 'remeasure_20261017')
tag = sys.argv[1]


def load(d, lat):
    it = json.load(open(os.path.join(d, f'learned_{lat}.json')))['iterations'][0]
    g = [float(r[1]) for r in csv.reader(open(os.path.join(d, f'gpu_{lat}.csv'))) if len(r) > 1 and r[1].strip().replace('.', '').isdigit()]
    return it, max(g) / 1024


print('| lattice | PCG base / v3 | compliance max rel. diff | iteration s base / v3 | front end / precond / solve / sens s, base | same, v3 | device peak GiB base / v3 | host after front end GB base / v3 |')
print('| --- | --- | --- | --- | --- | --- | --- | --- |')
for lat in ('hlat221a', 'hlat221b', 'hlat222', 'hlat331'):
    d = os.path.join(H, '..', 'results', f'{tag}_t5')
    if not os.path.exists(os.path.join(d, f'learned_{lat}.json')):
        continue
    a, ga = load(BASE, lat)
    v, gv = load(d, lat)
    c, cb = np.array(v['compliance']), np.array(a['compliance'])
    ph = lambda it: ' / '.join('%.1f' % it['phases'].get(k, float('nan')) for k in ('front_end', 'prec_setup', 'solve', 'sens_total'))
    print(f"| {lat} | {a['solve']['iterations']} / {v['solve']['iterations']} | {(np.abs(c - cb) / np.abs(cb)).max():.1e} | "
          f"{a['phases']['iteration_total']:.1f} / {v['phases']['iteration_total']:.1f} | {ph(a)} | {ph(v)} | {ga:.2f} / {gv:.2f} | "
          f"{a['host_rss_GB_after_front_end']:.2f} / {v['host_rss_GB_after_front_end']:.2f} |")
