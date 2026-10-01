"""Figure 11 -> figures/F06_bernstein.{svg,pdf,png}: Bernstein restriction of the box-face displacements on H1
(fresh_val_2005_d1_v0, configuration x, exact cell operators; Table ST16a, H1 rows).

  (a) number of retained DOFs after restriction against the degree r; dashed: the 32,991 DOFs of the full representation;
  (b) maximum relative compliance error, (c) maximum relative error of the target-cell thickness-sensitivity vector, each
      over the three target-face loads and over all six face loads, with the 3% reference line.
Source: evidence/piml4_all_fresh_val_2005_d1_v0.json (orders 1, 2, 3, 5, 8; 'ctrl_dofs', 'test_face_*_max' = 3 target-face
loads, 'gate_*_max' = 6 face loads, relative errors as fractions).  Usage: python3 fig11_bernstein.py
"""
import json
from pathlib import Path
import numpy as np
import figstyle as FS
plt = FS.plt

EV = Path(__file__).resolve().parent.parent / 'evidence'
rec = json.load(open(EV / 'piml4_all_fresh_val_2005_d1_v0.json'))['results'][0]
assert rec['case'] == 'fresh_val_2005_d1_v0' and rec['config'] == 'x'
R = sorted(int(o) for o in rec['orders'])
O = [rec['orders'][str(r)] for r in R]
dofs = np.array([o['ctrl_dofs'] for o in O]) / 1e3
full = rec['free'] / 1e3
comp3 = np.array([o['test_face_compliance_max'] for o in O]) * 100
comp6 = np.array([o['gate_compliance_max'] for o in O]) * 100
sens3 = np.array([o['test_face_sens_max'] for o in O]) * 100
sens6 = np.array([o['gate_sens_max'] for o in O]) * 100
print('r', R, 'retained DOFs', (dofs * 1e3).astype(int).tolist(), 'full', rec['free'])
print('compliance 3/6 loads (%)', np.round(comp3, 3).tolist(), np.round(comp6, 3).tolist())
print('sensitivity 3/6 loads (%)', np.round(sens3, 3).tolist(), np.round(sens6, 3).tolist())

T3 = dict(color=FS.C['uncorrected'], marker='o', ms=5, lw=1.2, label='Target face: 3 loads')
A6 = dict(color=FS.C['corrected'], marker='s', ms=5, lw=1.2, label='All faces: 6 loads')

fig, axs = plt.subplots(1, 3, figsize=(178 * FS.MM, 66 * FS.MM), gridspec_kw=dict(wspace=.42))
ax = axs[0]
ax.plot(R, dofs, **T3)
ax.axhline(full, ls='--', color=FS.TEXT, lw=1.0)
ax.text(1.0, full - 1.6, 'Full representation\n(32,991)', ha='left', va='top', fontsize=6.5, color=FS.TEXT)
ax.set_ylim(0, 37); ax.set_ylabel('Retained DOFs (thousands)')
FS.panel(ax, 'a', 'Retained DOFs')
for ax, y3, y6, t, l in ((axs[1], comp3, comp6, 'b', 'Compliance'), (axs[2], sens3, sens6, 'c', 'Sensitivity')):
    ax.plot(R, y3, **T3); ax.plot(R, y6, **A6)
    ax.axhline(3, ls='--', color=FS.MUTED, lw=.8)
    ax.set_yscale('log'); ax.set_ylim(.1, 500); ax.set_ylabel('Maximum relative error (%)')
    FS.panel(ax, t, l)
for ax in axs:
    ax.set_xticks(R); ax.set_xlim(.5, 8.5); ax.set_xlabel('Degree r')
    ax.grid(axis='y', color=FS.GRID, lw=.4)
h, lab = axs[1].get_legend_handles_labels()
fig.legend(h, lab, loc='lower left', bbox_to_anchor=(.08, .97), ncol=2, frameon=False, fontsize=7.5)
FS.save(fig, 'F06_bernstein')
print('wrote figures/F06_bernstein.{svg,pdf,png}')
