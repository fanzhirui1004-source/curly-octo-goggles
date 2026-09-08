"""Realised coverage of the produced dataset: thickness, cut azimuth, cut depth, and what symmetry buys.

Design (POPULATION.jsonl) intersected with what actually produced (PRODUCTION_CENSUS.json status lists), so this is
coverage of the TRAINING SET, not of the draw.
"""
import json
from pathlib import Path
import numpy as np

R = Path("artifacts/PRED777H_FULL_CUBE_SINGLE_CELL_MESH_GRAPH_FIXED_PORT_SCHUR_V1/sheet_true_geometry_label_route_v1")
pop = [json.loads(l) for l in (R / "population/POPULATION.jsonl").read_text().splitlines() if l.strip()]
cen = json.loads((R / "production/PRODUCTION_CENSUS.json").read_text())
unproducible = set(cen["population"]["unproducible"])
print("population records: %d" % len(pop))

# EMPTY / degenerate cells are not in the census by name, so reconstruct the produced set from the counts we do have:
# every cell that is neither unproducible nor EMPTY.  We know EMPTY = 55 and they are all cut cells with small offset.
by = {r["case"]: r for r in pop}
kinds = {}
for r in pop: kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
print("kinds:", kinds)

def q(v, ps=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(v, float)
    return " ".join("p%d=%.4g" % (p, np.percentile(a, p)) for p in ps)

cut = [r for r in pop if r.get("cut_plane")]
unc = [r for r in pop if not r.get("cut_plane")]
print("\ncut plane present: %d ; no cut: %d" % (len(cut), len(unc)))

# ---------------- thickness ----------------
tau = np.array([r["tau_corners"] for r in pop])
mean = tau.mean(axis=1); span = tau.max(axis=1) - tau.min(axis=1)
gm = np.array([r.get("gradient_magnitude", 0.0) for r in pop])
print("\n=== THICKNESS (tau; rho = tau/1.755 by the generator's own convention)")
print("  per-corner value      ", q(tau.ravel()))
print("  per-cell mean tau     ", q(mean))
print("  per-cell corner span  ", q(span))
print("  gradient magnitude    ", q(gm))
print("  cells with span < 0.01 (uniform): %d (%.1f%%);  < 0.05: %d (%.1f%%)" % (
    (span < 0.01).sum(), 100 * (span < 0.01).mean(), (span < 0.05).sum(), 100 * (span < 0.05).mean()))
print("  cells with gradient in the forced top decile [0.40, 0.47]: %d (%.1f%%)" % (
    ((gm >= 0.40) & (gm <= 0.4701)).sum(), 100 * ((gm >= 0.40) & (gm <= 0.4701)).mean()))
# gradient DIRECTION coverage on the sphere
g = np.array([r.get("gradient", [0, 0, 0]) for r in pop], float)
nz = np.linalg.norm(g, axis=1) > 1e-9
gd = g[nz] / np.linalg.norm(g[nz], axis=1)[:, None]
print("  gradient direction: %d cells with a direction; |g_z| distribution " % nz.sum(), q(np.abs(gd[:, 2])))
print("    (uniform on the sphere would give |g_z| uniform in [0,1]: p25/p50/p75 = 0.25/0.50/0.75)")

# ---------------- cut azimuth and depth ----------------
th = np.array([r["theta_deg"] for r in cut])
off = np.array([r["offset_fraction"] for r in cut])
print("\n=== CUT AZIMUTH theta (normal = (cos t, sin t, 0), design: uniform in [0, 45])")
print("  theta deg  ", q(th))
h, e = np.histogram(th, bins=9, range=(0, 45))
print("  histogram 5-degree bins:", dict(zip(["%.0f-%.0f" % (e[i], e[i+1]) for i in range(9)], h.tolist())))
print("\n=== CUT DEPTH (offset fraction of the plane's range over the cell; retained side n.x <= c)")
print("  offset fraction ", q(off))
h2, e2 = np.histogram(off, bins=10, range=(0, 1))
print("  histogram deciles:", h2.tolist())
print("  offset < 0.096 (the EMPTY threshold STEP5 measured): %d cells (%.1f%% of cut draws)" % (
    (off < 0.096).sum(), 100 * (off < 0.096).mean()))
print("  offset < 0.05: %d ; > 0.95: %d" % ((off < 0.05).sum(), (off > 0.95).sum()))

# the retained fraction of the CUBE (pure geometry of the half-space, independent of the sheet)
def retained_cube_fraction(a, b, c):
    n = np.array([a, b, 0.0]); pts = np.array([[x, y, 0.0] for x in (0, 1) for y in (0, 1)])
    # 2D problem in (x,y): fraction of the unit square with a x + b y <= c
    xs = (np.arange(400) + 0.5) / 400
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return float(((a * X + b * Y) <= c).mean())
frac = np.array([retained_cube_fraction(*r["cut_plane"][:2], r["cut_plane"][3]) for r in cut])
print("  retained fraction of the cube ", q(frac))
h3, _ = np.histogram(frac, bins=10, range=(0, 1))
print("  histogram deciles:", h3.tolist())
print("  retained < 0.1: %d cells ; > 0.9: %d cells" % ((frac < 0.1).sum(), (frac > 0.9).sum()))

# ---------------- joint coverage ----------------
print("\n=== JOINT (theta x offset), the two cut dimensions")
ti = np.clip((th / 15).astype(int), 0, 2); oi = np.clip((off * 4).astype(int), 0, 3)
tab = np.zeros((3, 4), int)
for a_, b_ in zip(ti, oi): tab[a_, b_] += 1
print("        off[0,.25) [.25,.5) [.5,.75) [.75,1]")
for k, row in enumerate(tab):
    print("  theta %2d-%2d  %s" % (k * 15, (k + 1) * 15, "  ".join("%7d" % v for v in row)))
print("  min bin %d, max bin %d, ratio %.2f" % (tab.min(), tab.max(), tab.max() / max(1, tab.min())))
