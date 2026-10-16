"""Plate scale table: ST20 (docs/paper_p1/.../X6_final/scale/scale_summary.json, first analysis) vs src_v3 runs
(docs/speedup_v3/results/<tag>_S<N>/analysis0.json + gpu.csv + rss.tsv). usage: python scale_table.py TAG [N ...]"""
import sys, json, csv, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
ST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'paper_p1', 'review_r1', 'results', 'X6_final', 'scale',
                  'scale_summary.json')
tag = sys.argv[1]; sizes = [int(x) for x in sys.argv[2:]] or [24, 51, 88, 110, 135, 162]
st = json.load(open(ST))
print('| cells | PCG ST20 / v3 | compliance rel. diff | analysis s ST20 / v3 | GPU peak GiB ST20 / v3 | host peak GiB ST20 / v3 |')
print('| --- | --- | --- | --- | --- | --- |')
for n in sizes:
    d = os.path.join(R, f'{tag}_S{n}')
    if not os.path.exists(os.path.join(d, 'analysis0.json')):
        continue
    a = json.load(open(os.path.join(d, 'analysis0.json')))
    g = max(float(r[1]) for r in csv.reader(open(os.path.join(d, 'gpu.csv')))) / 1024
    h = max(float(r[1]) for r in csv.reader(open(os.path.join(d, 'rss.tsv')), delimiter='\t') if len(r) > 1 and r[1])
    s = st.get(f'plateS{n}')
    if s:
        print(f"| {n} | {s['pcg'][0]} / {a['pcg']} | {abs(a['C'] - s['C'][0]) / s['C'][0]:.1e} | {s['iter_s'][0]:.0f} / "
              f"{a['times']['analysis_wall_s']:.0f} | {s['sampler_gpu_used_max_gib']:.1f} / {g:.1f} | {s['sampler_rss_max_gib']:.1f} / {h:.1f} |")
    else:
        print(f"| {n} | — / {a['pcg']} | — | — / {a['times']['analysis_wall_s']:.0f} | — / {g:.1f} | — / {h:.1f} |")
