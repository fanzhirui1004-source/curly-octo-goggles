"""Supplementary Figure S01 (S06 before the revision-1 renumbering): verification of the CutFEM reference (evidence/ref_valid*.json, script ref_valid.py).
(a) background resolution n: largest relative change of compliance (filled) and 8-corner sensitivity (open) against the finest
    resolution (n = 40 for U1, 48 otherwise), over the three consistent face loads; (b) ghost-penalty coefficient relative to the value used
    1e-4; (c) finite-difference step of the moment derivatives, relative to h = 1e-5 tau, and direct compliance differences."""
import json
from pathlib import Path
import numpy as np
import figstyle as FS
plt = FS.plt

EV = Path(__file__).resolve().parent.parent / 'evidence'
CASES = [('U1', 'ref_valid.json', 'fresh_val_2000_full') + FS.CELL['U1'],          # cells: blue ramp, shape by stratum
         ('M1', 'ref_valid.json', 'fresh_val_2003_d1_v1') + FS.CELL['M1'],
         ('M2', 'ref_valid.json', 'fresh_val_2006_d0_v1') + FS.CELL['M2'],
         ('H1', 'ref_valid_h1.json', 'fresh_val_2005_d1_v0') + FS.CELL['H1']]


def pct(v):
    return 100 * max(abs(x) for x in v)


fig, axs = plt.subplots(1, 3, figsize=(178 * FS.MM, 60 * FS.MM), constrained_layout=True)
for lab, f, case, col, mk in CASES:
    d = json.loads((EV / f).read_text())['per_case'][case]
    n = [k for k in sorted(d['vs_finest'], key=int) if pct(d['vs_finest'][k]['compliance_rel']) > 0]
    x = [int(k) for k in n]
    axs[0].plot(x, [pct(d['vs_finest'][k]['compliance_rel']) for k in n], '-', color=col, marker=mk, ms=4, label=lab)
    axs[0].plot(x, [pct(d['vs_finest'][k]['sens_rel']) for k in n], '--', color=col, marker=mk, ms=4, mfc='white')
    g = sorted(d['gamma'], key=float)
    gx = [float(k) for k in g if float(k) != 1e-4]
    axs[1].plot(gx, [pct(d['gamma'][k]['compliance_rel']) for k in g if float(k) != 1e-4], '-', color=col, marker=mk, ms=4)
    axs[1].plot(gx, [pct(d['gamma'][k]['sens_rel']) for k in g if float(k) != 1e-4], '--', color=col, marker=mk, ms=4, mfc='white')
    h = sorted((k for k in d['fd']['rel_to_h1e_5'] if float(k) != 1e-5), key=float)
    axs[2].plot([float(k) for k in h], [pct(d['fd']['rel_to_h1e_5'][k]) for k in h], '-', color=col, marker=mk, ms=4)
    axs[2].plot([1e-5], [pct(d['fd']['direct_vs_sens'])], ls='none', color=col, marker=mk, ms=5, mfc='white')

FS.panel(axs[0], 'a', 'Background resolution')
axs[0].set_xlabel('background elements per axis, n'); axs[0].set_ylabel('largest change vs finest n (%)')
axs[0].set_yscale('log'); axs[0].set_xticks([24, 32, 40]); axs[0].axvline(32, color=FS.GRID, lw=3, zorder=0)
axs[0].text(32.5, 1.6, 'used', color=FS.MUTED, fontsize=6.5)
FS.panel(axs[1], 'b', 'Ghost penalty')
axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel(r'$\gamma$ (used: $10^{-4}$)')
axs[1].set_ylabel(r'largest change vs $\gamma=10^{-4}$ (%)')
FS.panel(axs[2], 'c', 'Derivative step')
axs[2].set_xscale('log'); axs[2].set_yscale('log'); axs[2].set_xlabel(r'step $h/\tau_c$ (used: $10^{-5}$)')
axs[2].set_ylabel(r'sensitivity change (%)')
axs[2].annotate('open: direct compliance\ndifference vs sensitivity', (1e-5, 5e-6), (1.5e-6, 1.5e-5), color=FS.MUTED, fontsize=6,
                ha='center', arrowprops=dict(arrowstyle='-', color=FS.MUTED, lw=.5))
for ax in axs:
    ax.grid(True, which='major', color=FS.GRID, lw=.5)
h1 = [plt.Line2D([], [], color=c, marker=m, ms=4, label=l) for l, _, _, c, m in CASES]
h2 = [plt.Line2D([], [], color=FS.MUTED, ls='-', marker='o', ms=4, label='compliance'),
      plt.Line2D([], [], color=FS.MUTED, ls='--', marker='o', mfc='white', ms=4, label='sensitivity')]
fig.legend(handles=h1 + h2, frameon=False, loc='outside lower center', ncol=6, handlelength=2.2)
FS.save(fig, 'S06_reference_verification')
