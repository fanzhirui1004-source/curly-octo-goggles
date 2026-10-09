"""House style for P1 figures (after STYLE_GUIDE.md): white background, DejaVu Sans, 178 mm full width,
semantic colours, redundant marker encodings, SVG (native text) + PDF (TrueType) + 300 dpi PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEXT, MUTED, GRID = '#243447', '#657382', '#DFE5E9'
# Palette (scheme A, three layers; each colour has one meaning in the whole paper; mirrored in codex/build_*.py).
# Layer 1 - variants and references. NICE vermillion is the only warm saturated colour.
C = dict(exact='#243447',        # exact / reference / exact condensation (ink)
         base='#8A94A0',         # Base network (light grey)
         uncorrected='#0072B2', corrected='#D55E00', assembly='#009E73', smoothing='#E69F00',
         extra='#AA4499')        # homogenised model (Figures 12-13 only)
MODEL = {  # colour, marker, legend label of each of the five variants (Table 2 and the note below it); short internal keys
    'B': (C['base'], 'P', 'Base network'),
    'C': (C['uncorrected'], 'o', 'Uncorrected continuation'),
    'B+W': (C['assembly'], '^', 'Base network + correction'),
    'A2b': (C['smoothing'], 'v', 'Smoothing-trained'),
    'A3': (C['corrected'], 's', 'NICE'),
}
# Layer 2 - individual cells: blue ramp ordered by cut severity (light = uncut, dark = heavily cut), shape by stratum.
# U2 takes 'h' instead of 'o' when it appears together with U1. Cells never use layer-1 colours.
CELL = {'U1': ('#8FB1D6', 'o'), 'U2': ('#7AA2CD', 'o'), 'L1': ('#6390C2', 'D'), 'M1': ('#4A7AB0', 's'),
        'M2': ('#3A679C', 'D'), 'H1': ('#24528A', '^'), 'H2': ('#133A68', 'v'), 'H3': ('#0B2747', '<')}
# Layer 3 - any other categories: neutral greys distinguished by line style / marker / hatch; at most one accent per
# figure, NICE vermillion, and only for the complete NICE correction.
GREY = ('#243447', '#657382', '#A3ADB8')
GREY_FILL = '#CDD3DA'
MM = 1 / 25.4
OUT = Path(__file__).resolve().parent.parent / 'figures'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': 8, 'axes.labelsize': 8,
    'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'text.color': TEXT,
    'axes.labelcolor': TEXT, 'axes.edgecolor': MUTED, 'xtick.color': TEXT, 'ytick.color': TEXT, 'axes.linewidth': .6,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
})


def panel(ax, letter, title):
    ax.set_title(f'({letter}) {title}', loc='left', fontweight='bold', fontsize=8.5, pad=6)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext, kw in (('svg', {}), ('pdf', {}), ('png', dict(dpi=300))):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)
