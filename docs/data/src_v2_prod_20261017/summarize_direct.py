"""One line per direct-solver run: encode, main stages, query 1/64 columns, witness, worst original probe ratio."""
import json, sys
from pathlib import Path
for root in sys.argv[1:]:
    root = Path(root)
    print('==', root.name)
    for log in sorted(root.glob('*.log')):
        d = log.with_suffix('')
        try:
            r = json.loads((d / 'RESULT.json').read_text())
        except Exception:
            lines = log.read_text().strip().splitlines()
            print(d.name, 'FAIL', (lines[-1] if lines else '')[:160])
            continue
        s = r['stage_seconds']
        pick = {k.split('_')[0]: round(v, 2) for k, v in s.items() if k.startswith(('E1a', 'E1d', 'E4b', 'E4c'))}
        q = r['query_seconds']
        print(d.name, 'enc %.2f' % r['encode_seconds'], pick, 'q1 %.1fms q64 %.0fms' % (q['1'] * 1e3, q['64'] * 1e3),
              'wit %.4f' % r['witness'][0], 'o7max %.5f' % max(r['original7_energy_ratio']))
