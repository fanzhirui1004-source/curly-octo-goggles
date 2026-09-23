"""Private panel for the design-sensitivity check: registered second-batch cases with all eight tau corners scaled
by the exact rational 1 + eps (the 2026-09-19/20 tau-gate protocol). The registered panel is read, never written;
the new panel carries its own FILES.json / TERMINAL.json so the frozen geometry stage's seal check applies unchanged."""
import hashlib, json, sys
from fractions import Fraction
from pathlib import Path

SRC = Path('/root/autodl-tmp/CUTFEM_FRESH_GP_20260921/CUT_COVER80_20260922_CONFIG_V3/panel')
OUT = Path(sys.argv[1])
CASES = sys.argv[2].split(',')
EPS = ['-1/1000', '-1/10000', '1/10000', '1/1000']


def tag(e):
    f = Fraction(e)
    return ('m' if f < 0 else 'p') + str(abs(f.denominator // f.numerator if f.numerator else 0))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


panel = json.loads((SRC / 'PANEL.json').read_text())
rows = []
for c in CASES:
    base = [r for r in panel['cases'] if r['case_id'] == c]
    if len(base) != 1:
        raise ValueError('CASE_NOT_UNIQUE ' + c)
    for e in EPS:
        r = dict(base[0])
        r['case_id'] = c + '_tau' + tag(e)
        r['tau_corners'] = [str(Fraction(v) * (1 + Fraction(e))) for v in base[0]['tau_corners']]
        r['sensitivity_perturbation'] = dict(base_case_id=c, epsilon=e, rule='all eight tau corners times (1 + eps), exact rational')
        rows.append(r)
OUT.mkdir(parents=True, exist_ok=False)
new = dict(configuration=panel['configuration'], cases=rows,
           derived_from=dict(panel=str(SRC / 'PANEL.json'), sha256=sha(SRC / 'PANEL.json')),
           purpose='design sensitivity finite differences; diagnostic only, not a dataset')
(OUT / 'PANEL.json').write_text(json.dumps(new, indent=2))
(OUT / 'TERMINAL.json').write_text(json.dumps(dict(status='COMPLETED', note='private sensitivity panel'), indent=2))
(OUT / 'FILES.json').write_text(json.dumps({n: sha(OUT / n) for n in ('PANEL.json', 'TERMINAL.json')}, indent=2))
print(json.dumps([r['case_id'] for r in rows]))
