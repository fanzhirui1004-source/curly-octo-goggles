"""Summary of the re-measurement of 2026-10-17 (new RTX 5090 machine): the learned route of Table 5 against FINAL3
(t5retime/learned_*.json, the numbers in the paper) with the whole-device GPU memory sampled by nvidia-smi every 100 ms,
and the deployed per-cell state of the ST13 cells G1-G4 (st13_deploy_mem.json). Writes summary.json and prints a table.
Usage: python summarize.py   (from this directory)"""
import csv, json
from pathlib import Path

GiB = 2 ** 30
OLD = Path('../t5retime')
LATS = ['hlat221a', 'hlat221b', 'hlat222', 'hlat331']
PAPER_ST13 = dict(G1=dict(state=0.74, K=1.39), G2=dict(state=0.25, K=0.51), G3=dict(state=1.20, K=2.32), G4=dict(state=1.37, K=2.71))


def smi_peak(f):
    v = [float(r[1]) for r in csv.reader(open(f)) if len(r) > 1 and r[1].strip().isdigit()]
    return max(v) / 1024, min(v) / 1024, len(v)


out = dict(lattices={}, st13={})
for L in LATS:
    a = json.loads((OLD / f'learned_{L}.json').read_text())['iterations'][0]
    b = json.loads(Path(f'learned_{L}.json').read_text())['iterations'][0]
    pk, base, n = smi_peak(f'gpu_{L}.csv')
    ph = lambda r: {k: r['phases'][k] for k in ('front_end', 'prec_setup', 'solve', 'sens_total', 'iteration_total')}
    out['lattices'][L] = dict(
        final3=dict(phases=ph(a), iterations=a['solve']['iterations'], compliance=a['compliance'],
                    torch_peak_GiB=a['peak_device_GB'] * 1e9 / GiB, host_rss_GiB=a['host_rss_GB_after_front_end'] * 1e9 / GiB),
        new=dict(phases=ph(b), iterations=b['solve']['iterations'], compliance=b['compliance'],
                 torch_peak_GiB=b['peak_device_GB'] * 1e9 / GiB, host_rss_GiB=b['host_rss_GB_after_front_end'] * 1e9 / GiB,
                 device_peak_GiB=pk, device_idle_GiB=base, samples=n),
        compliance_max_rel_diff=max(abs(x / y - 1) for x, y in zip(b['compliance'], a['compliance'])))
st = json.loads(Path('st13_deploy_mem.json').read_text())
out['st13_meta'] = {k: st[k] for k in ('gpu', 'torch', 'model', 'body', 'env')}
for c in st['cells']:
    out['st13'][c['label']] = dict(case=c['case'], dofs=c['dofs'], ports=c['ports'], elements=c['elements'], n_coarse=c['n_coarse'],
                                   alloc_delta_GiB=c['alloc_delta_GiB'], device_delta_GiB=c['device_delta_GiB'],
                                   walked_total_GiB=c['walked_total_GiB'], groups_GiB=c['groups_GiB'],
                                   stage_s={k: c[k] for k in ('ctor_s', 'assemble_deploy_s', 'netdata_s', 'encode_s', 'warmup_s')},
                                   paper=PAPER_ST13[c['label']])
Path('summary.json').write_text(json.dumps(out, indent=1))
print('lattice   time FINAL3/new (s)   PCG FINAL3/new   torch peak (GiB)   whole-device peak (GiB)   host RSS (GiB)   max rel dC')
for L, r in out['lattices'].items():
    f, n = r['final3'], r['new']
    print(f"{L:9s} {f['phases']['iteration_total']:6.1f} / {n['phases']['iteration_total']:6.1f}   {f['iterations']:3d} / {n['iterations']:3d}"
          f"   {f['torch_peak_GiB']:5.2f} / {n['torch_peak_GiB']:5.2f}   {n['device_peak_GiB']:5.2f}   {n['host_rss_GiB']:5.2f}   {r['compliance_max_rel_diff']:.1e}")
print('cell  alloc / device / walked (GiB)   stiffness  correction  kpp  geometry  network   paper state / K')
for k, r in out['st13'].items():
    g = r['groups_GiB']
    print(f"{k}  {r['alloc_delta_GiB']:.2f} / {r['device_delta_GiB']:.2f} / {r['walked_total_GiB']:.2f}   {g['stiffness']:.3f}  {g['correction']:.3f}"
          f"  {g['kpp']:.3f}  {g['geometry']:.3f}  {g['network']:.3f}   {r['paper']['state']:.2f} / {r['paper']['K']:.2f}")
