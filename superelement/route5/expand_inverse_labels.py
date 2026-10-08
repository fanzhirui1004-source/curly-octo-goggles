"""Build the A^-1 factor label beside every Cholesky label in a manifest (route 5).

Same self-tests as build_inverse_label.py: L^T L A - I, G reconstruction, an exact
read_blocks round trip, and the exact label's g through the corrected spectrum route.
"""
import argparse, json, shutil, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', type=Path, required=True)
ap.add_argument('--builder', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--only-seats', type=int, nargs='*', default=None)
ap.add_argument('--min-free-gib', type=float, default=60.0)
ap.add_argument('--python', default='/root/cutfem_neural_a_20260910/env/bin/python')
a = ap.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rows = [r for r in json.loads(a.manifest.read_text()) if r['split'] == 'train']
if a.only_seats is not None:
    keep = set(a.only_seats); rows = [r for r in rows if int(r['seat']) in keep]
rows.sort(key=lambda r: int(r['q']))
log = a.output / 'INVERSE_LABELS.jsonl'
ok = []
for i, r in enumerate(rows):
    ref = Path(r['reference']); d = int(r['q']) - 6
    rec = dict(index=i, of=len(rows), seat=int(r['seat']), q=int(r['q']))
    if (ref / 'INVERSE_RESULT.json').exists() and (ref / 'G_UPPER.npy').exists():
        rec['phase'] = 'already_built'; ok.append(rec); print(json.dumps(rec), flush=True); continue
    free = shutil.disk_usage(ref).free / 2**30
    need = (d * (d + 1) // 2) * 8 / 2**30
    if free - need < a.min_free_gib:
        rec.update(phase='stopped_low_disk', free_gib=round(free, 1)); print(json.dumps(rec), flush=True)
        log.open('a').write(json.dumps(rec) + '\n'); break
    t = time.perf_counter()
    p = subprocess.run([a.python, '-u', str(a.builder), '--reference', str(ref), '--d', str(d),
                        '--seat', str(int(r['seat']))], capture_output=True, text=True)
    if p.returncode != 0:
        rec.update(phase='failed', stderr=p.stderr[-500:]); print(json.dumps(rec), flush=True)
        log.open('a').write(json.dumps(rec) + '\n'); continue
    res = json.loads((ref / 'INVERSE_RESULT.json').read_text())
    rec.update(phase='built', seconds=round(time.perf_counter() - t, 1),
               LtLA_minus_I=res['L_T_L_times_A_minus_I'], roundtrip=res['read_blocks_roundtrip'],
               label_g=res['label_g'], log_pivot_range=[res['log_pivot_min'], res['log_pivot_max']])
    if rec['roundtrip'] != 0.0 or rec['LtLA_minus_I'] > 1e-9 or abs(rec['label_g'] - 1) > 1e-8:
        rec['phase'] = 'selftest_out_of_tolerance'
    print(json.dumps(rec), flush=True); log.open('a').write(json.dumps(rec) + '\n')
    if rec['phase'] == 'built': ok.append(rec)
summary = dict(status='INVERSE_LABEL_EXPANSION_COMPLETE', planned=len(rows), built=len(ok),
               seats=sorted(x['seat'] for x in ok),
               worst_LtLA=max((x.get('LtLA_minus_I', 0.) for x in ok), default=0.),
               worst_label_g_minus_1=max((abs(x.get('label_g', 1.) - 1) for x in ok), default=0.),
               log_pivot_max=max((x.get('log_pivot_range', [0, 0])[1] for x in ok), default=0.),
               free_gib_after=round(shutil.disk_usage(a.output).free / 2**30, 1))
(a.output / 'INVERSE_LABELS.json').write_text(json.dumps(summary, indent=1))
print(json.dumps(summary), flush=True)
