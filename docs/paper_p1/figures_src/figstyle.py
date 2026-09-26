"""House style for P1 figures (after the Codex STYLE_GUIDE.md): white background, DejaVu Sans, 178 mm full width,
semantic colours, redundant marker encodings, SVG (native text) + PDF (TrueType) + 300 dpi PNG."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEXT, MUTED, GRID = '#243447', '#657382', '#DFE5E9'
C = dict(exact='#53616F', uncorrected='#0072B2', corrected='#D55E00', assembly='#009E73', cut='#E69F00', extra='#AA4499')
MODEL = {  # colour, marker, label
    'C': (C['uncorrected'], 'o', 'C (uncorrected)'),
    'S8': (C['extra'], 'D', 'S8 (8 smoothing steps)'),
    'B+W': (C['assembly'], '^', 'B + correction (untrained)'),
    'A2b': (C['cut'], 'v', 'A2b (smoothing-trained)'),
    'A3': (C['corrected'], 's', 'A3 (trained through correction)'),
}
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
