"""One line per two-sided full spectrum (direct exact action): mu range, D/d, gates, times."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob('*/FULL_SPECTRUM.json')):
    r = json.loads(f.read_text())
    rows.append(r)
    print('%-42s d=%6d mu=[%.5f, %.5f] D/d=%.2e out_target=%d target=%s act=%.0fs eig=%.0fs' % (
        f.parent.name, r['physical_dimension'], r['mu_min'], r['mu_max'], r['D_over_d'], r['outside_target_gate'],
        r['target_gate'], r['action_seconds'], r['eig_seconds']))
fails = [f.parent.name for f in root.glob('*') if f.is_dir() and not (f / 'FULL_SPECTRUM.json').exists()]
print('done', len(rows), 'without result', fails)
if rows:
    print('min mu_min %.5f  max mu_max %.5f  max D/d %.2e  all target %s' % (
        min(r['mu_min'] for r in rows), max(r['mu_max'] for r in rows), max(r['D_over_d'] for r in rows),
        all(r['target_gate'] for r in rows)))
