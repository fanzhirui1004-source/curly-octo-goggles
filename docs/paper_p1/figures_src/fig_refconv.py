"""Supplementary Figure S01 (S06 before the revision-1 renumbering): verification of the CutFEM reference (evidence/ref_valid*.json,
script ref_valid.py; review_r1/results/ref_valid_r1.json for the refinement to n = 64).
(a) background resolution n: largest relative change of compliance (filled) and 8-corner sensitivity (open) against the finest
    resolution (n = 40 for U1, 48 for M1 and M2), over the three consistent face loads; (b) ghost-penalty coefficient relative to the
    value used 1e-4; (c) finite-difference step of the moment derivatives, relative to h = 1e-5 tau, and direct compliance differences;
    (d) refinement to n = 64 of H1, H2 and the validation cells with the largest NICE error (W1) and the thinnest walls (W3), the data of
    Table ST14 (largest magnitudes over the three loads; H2 has no n = 56)."""
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


R1 = json.loads((EV.parent / 'review_r1' / 'results' / 'ref_valid_r1.json').read_text())['per_case']
CASES64 = [('H1', 'fresh_val_2005_d1_v0') + FS.CELL['H1'], ('H2', 'fresh_val_2010_d0_v0') + FS.CELL['H2'],
           ('W1', 'fresh_val_2051_d1_v1', FS.GREY[1], 'P'), ('W3', 'fresh_val_2074_d0_v0', FS.GREY[2], 'X')]

fig, axs = plt.subplots(2, 2, figsize=(178 * FS.MM, 112 * FS.MM), constrained_layout=True)
axs = axs.ravel()
for lab, f, case, col, mk in CASES:
    d = json.loads((EV / f).read_text())['per_case'][case]
    if lab != 'H1':                                                      # H1 is drawn against n = 64 in (d)
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
for lab, case, col, mk in CASES64:
    d = R1[case]
    n = [k for k in sorted(d['vs_finest'], key=int) if pct(d['vs_finest'][k]['compliance_rel']) > 0]
    x = [int(k) for k in n]
    axs[3].plot(x, [pct(d['vs_finest'][k]['compliance_rel']) for k in n], '-', color=col, marker=mk, ms=4)
    axs[3].plot(x, [pct(d['vs_finest'][k]['sens_rel']) for k in n], '--', color=col, marker=mk, ms=4, mfc='white')
    print(lab, {k: (round(pct(d['vs_finest'][k]['compliance_rel']), 3), round(pct(d['vs_finest'][k]['sens_rel']), 3)) for k in n})

FS.panel(axs[0], 'a', 'Background resolution')
axs[0].set_xlabel('background elements per axis, n'); axs[0].set_ylabel('largest change vs finest n (%)')
axs[0].set_yscale('log'); axs[0].set_xticks([24, 32, 40]); axs[0].axvline(32, color=FS.GRID, lw=3, zorder=0)
axs[0].text(32.4, .97, 'used', color=FS.MUTED, fontsize=6.5, transform=axs[0].get_xaxis_transform(), va='top')
FS.panel(axs[1], 'b', 'Ghost penalty')
axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel(r'$\gamma$ (used: $10^{-4}$)')
axs[1].set_ylabel(r'largest change vs $\gamma=10^{-4}$ (%)')
FS.panel(axs[2], 'c', 'Derivative step')
axs[2].set_xscale('log'); axs[2].set_yscale('log'); axs[2].set_xlabel(r'step $h/\tau_c$ (used: $10^{-5}$)')
axs[2].set_ylabel(r'change (%)')
axs[2].set_ylim(1e-10, 1e-4)
axs[2].text(.97, .04, 'filled: sensitivity change vs step\nopen: direct compliance difference\nvs sensitivity at the step used',
            transform=axs[2].transAxes, ha='right', va='bottom', fontsize=6.5, color=FS.MUTED, linespacing=1.15)
FS.panel(axs[3], 'd', 'Refinement to n = 64')
axs[3].set_xlabel('background elements per axis, n'); axs[3].set_ylabel('largest change vs n = 64 (%)')
axs[3].set_yscale('log'); axs[3].set_xticks([24, 32, 40, 48, 56]); axs[3].axvline(32, color=FS.GRID, lw=3, zorder=0)
axs[3].text(32.4, .97, 'used', color=FS.MUTED, fontsize=6.5, transform=axs[3].get_xaxis_transform(), va='top')
for ax in axs:
    ax.grid(True, which='major', color=FS.GRID, lw=.5)
h1 = [plt.Line2D([], [], color=c, marker=m, ms=4, label=l) for l, _, _, c, m in CASES]
h1 += [plt.Line2D([], [], color=c, marker=m, ms=4, label=l) for l, _, c, m in CASES64[1:]]
h2 = [plt.Line2D([], [], color=FS.MUTED, ls='-', marker='o', ms=4, label='compliance (a, b, d)'),
      plt.Line2D([], [], color=FS.MUTED, ls='--', marker='o', mfc='white', ms=4, label='sensitivity (a, b, d)')]
fig.legend(handles=h1 + h2, frameon=False, loc='outside lower center', ncol=9, handlelength=2.0, columnspacing=1.0, fontsize=6.5)
FS.save(fig, 'S06_reference_verification')
