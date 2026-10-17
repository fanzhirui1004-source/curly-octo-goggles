"""Per plate run: analysis phases, PCG, compliance, and the device memory by phase (census points) and its peak.
usage: python plate_phases.py TAG [N ...]"""
import sys, json, csv, os, datetime as D
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
tag = sys.argv[1]; sizes = [int(x) for x in sys.argv[2:]] or [24, 51, 88, 110, 135, 162, 190, 222]
print('| cells | PCG | C | front end / precond / solve / sens s | analysis s | after front end | after A Z | PCG phase | peak GiB (where) | host GiB |')
print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
for n in sizes:
    p = os.path.join(R, f'{tag}_S{n}')
    if not os.path.exists(os.path.join(p, 'analysis0.json')):
        continue
    d = json.load(open(os.path.join(p, 'analysis0.json'))); c = json.load(open(os.path.join(p, 'census.json')))
    pts = {x['name']: x for x in c['points']}
    rows = [r for r in csv.reader(open(os.path.join(p, 'gpu.csv')))]
    ts = [D.datetime.strptime(r[0].strip(), '%Y/%m/%d %H:%M:%S.%f') for r in rows]; m = [float(r[1]) / 1024 for r in rows]
    i = max(range(len(m)), key=lambda k: m[k]); frac = (ts[i] - ts[0]).total_seconds() / max(1e-9, (ts[-1] - ts[0]).total_seconds())
    t = d['times']
    h = max(float(r[1]) for r in csv.reader(open(os.path.join(p, 'rss.tsv')), delimiter='\t') if len(r) > 1 and r[1])
    print(f"| {n} | {d['pcg']} | {d['C']:.8f} | {t['prep_s']:.0f} / {t['precond_s']:.0f} / {t['solve_s']:.0f} / {t['sens_s']:.0f} | "
          f"{t['analysis_wall_s']:.0f} | {pts['after_front_end']['device_used']:.1f} | {pts['after_coarse']['device_used']:.1f} | "
          f"{pts['after_pcg']['device_used']:.1f} | {m[i]:.1f} ({100 * frac:.0f}%) | {h:.1f} |")
