"""Re-score every saved spectrum in the inversion-symmetric metric.

g = max(mu_max, 1/mu_min) is the two-sided Loewner sandwich constant. eps_op =
max|mu-1| was retired because it is capped at 1 on the soft side, which is the side
compliance lives on. Both come from the same mu_min/mu_max, which these runs saved.
"""
import json
from pathlib import Path

D = Path('docs/data')


def g_of(mn, mx):
    if mn is None or mx is None or mn <= 0:
        return None
    return max(mx, 1.0 / mn)


def show(title, rows, size_key=None, size_name='params', extra=None):
    print(f'\n=== {title} ===')
    head = f'{size_name:>14} {"frac":>8} {"eps_op":>12} {"1/mu_min":>11} {"mu_max":>10} {"g":>11} {"driven":>7}'
    print(head)
    for r in rows:
        mn, mx = r.get('mu_min'), r.get('mu_max')
        g = g_of(mn, mx)
        if g is None:
            continue
        inv = 1.0 / mn
        frac = r.get('frac_of_dense') or r.get('fraction_of_dense')
        print('%14s %8s %12.5g %11.4g %10.4g %11.4g %7s'
              % (f'{r.get(size_key, r.get("params", r.get("kept", "")))}',
                 f'{frac:.4f}' if frac is not None else '',
                 r.get('eps_op', float("nan")), inv, mx, g,
                 'soft' if inv > mx else 'stiff'))


for name, title, size_key in (
        ('HMAT_FACTOR.json', 'hierarchical low rank on the TRUE Cholesky factor (seat 0328)', 'params'),
        ('HMAT_ECONOMY4.json', 'hierarchical low rank, economy sweep (seat 0328)', 'params'),
        ('HMAT_ECONOMY3.json', 'hierarchical low rank, earlier economy sweep (seat 0328)', 'params'),
        ('codex_probe_20260917/TRUNC.json', 'magnitude truncation of the TRUE factor (seat 0253)', 'kept'),
):
    p = D / name
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    rows = d.get('rows') or d.get('cases') or []
    show(f'{title}  [{name}]', rows, size_key,
         'entries' if 'TRUNC' in name else 'params')

print('\n=== the two trained arms, for scale ===')
print('%-34s %12s %11s %10s %11s' % ('', 'eps_op', '1/mu_min', 'mu_max', 'g'))
for head, mn, mx in (('chol 20k steps (line 1)', 0.0011708893684502326, 216.71676340757185),
                     ('sqrt 20k steps (line 3)', 0.00032703219928066255, 49.02848798512618)):
    print('%-34s %12.5g %11.4g %10.4g %11.4g'
          % (head, max(abs(mn - 1), abs(mx - 1)), 1 / mn, mx, g_of(mn, mx)))
print('%-34s %12s %11s %10s %11.4g' % ('gate for 3% sensitivity error', '', '', '', 31.0))
