"""Fill the exact-check placeholders (@...@) of MANUSCRIPT_EN.md and SUPPLEMENTARY_EN.md from FACTS_FINAL.json (run
facts_final.py first).  Leaves @SCALE_MIN@ to the scale demonstration.  Prints every value it writes.
Usage: python3 fill_exact.py [--dry]"""
import json, sys
from pathlib import Path
D = Path(__file__).resolve().parent
P = D.parents[2]
F = json.loads((D / 'FACTS_FINAL.json').read_text())
kN = F['NICE']['last_k']
c0, cN, cH = F['checks']['N0'], F['checks'][f'N{kN}'], F['checks']['H']
assert c0 and cN and cH, 'exact checks missing'
m = lambda x: f'{x:.4f}'.replace('-', '−')
V = {}
V['N23_err'] = f"{-100 * cN['surrogate_err']:.3f}"
V['N23_gerr'] = f"{100 * cN['grad_rel_err']:.2f}"
V['cosmin'] = f"{int(min(c0['grad_cos'], cN['grad_cos']) * 1e6) / 1e6:.6f}"
V['p95max'] = f"{100 * max(c0['comp_err_rel_to_max']['p95'], cN['comp_err_rel_to_max']['p95'], cH['comp_err_rel_to_max']['p95']):.2f}"
V['p95N'] = f"{100 * max(c0['comp_err_rel_to_max']['p95'], cN['comp_err_rel_to_max']['p95']):.2f}"
V['HvsN'] = f"{100 * (cH['C_exact'] / cN['C_exact'] - 1):.2f}"
V['HvsN_s'] = '+' + V['HvsN']
V['H_C'] = V['H_C3'] = f"{cH['C_exact']:.3f}"
V['N23_C'] = V['N23_C3'] = f"{cN['C_exact']:.3f}"
V['H_err'] = f"{-100 * cH['surrogate_err']:.3f}"
V['macroH'] = f"{100 * (1 - F['Hmacro']['C_final'] / cH['C_exact']):.1f}"
V['N23_err4'] = m(100 * cN['surrogate_err']); V['H_err4'] = m(100 * cH['surrogate_err'])
V['N23_gerr3'] = f"{100 * cN['grad_rel_err']:.2f}"; V['H_gerr3'] = f"{100 * cH['grad_rel_err']:.2f}"
V['NvsStart'] = f"{100 * (cN['C_exact'] / c0['C_exact'] - 1):.2f}"
V['ncells'] = str(sum(c['distinct_cells'] for c in (c0, cN, cH)))
def row(name, it, c):
    e = c['comp_err_rel_to_max']
    return (f"| {name} | {it} | {c['C_exact']:.3f} | {c['C_hat']:.3f} | {m(100 * c['surrogate_err'])} | {100 * c['grad_rel_err']:.3f} | "
            f"{c['grad_cos']:.8f} | {100 * e['median']:.3f} / {100 * e['p95']:.3f} / {100 * e['max']:.3f} | {c['sign_agreement']:.3f} (64) | "
            f"{c['exact_pcg']} / {c['exact_true_residual']:.1e} |")
V['ST26ROWS'] = '\n'.join([row('NICE run', '0 (uniform start)', c0), row('NICE run', f'{kN} (final)', cN),
                           row('Homogenisation design', 'final macroscale iteration', cH)])
for k, v in V.items():
    print(k, '=', v.replace('\n', ' || '))
if '--dry' not in sys.argv:
    for f in ('MANUSCRIPT_EN.md', 'SUPPLEMENTARY_EN.md'):
        p = P / f; s = p.read_text()
        for k, v in V.items():
            s = s.replace(f'@{k}@', v)
        p.write_text(s)
