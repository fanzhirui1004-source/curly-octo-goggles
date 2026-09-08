"""Coverage of the TRAINING SET (produced labels), plus what the 48 cube symmetries actually buy."""
import json, itertools
from pathlib import Path
import numpy as np

R = Path("artifacts/PRED777H_FULL_CUBE_SINGLE_CELL_MESH_GRAPH_FIXED_PORT_SCHUR_V1/sheet_true_geometry_label_route_v1")
pop = {r["case"]: r for r in (json.loads(l) for l in (R / "population/POPULATION.jsonl").read_text().splitlines() if l.lstrip().startswith("{"))}
cen = {r["case"]: r for r in (json.loads(l) for l in (R / "population/CENSUS.jsonl").read_text().splitlines() if l.lstrip().startswith("{"))}
prod = json.loads((R / "production/PRODUCTION_CENSUS.json").read_text())
unprod = set(prod["population"]["unproducible"])

empty = {c for c, r in cen.items() if r.get("status") == "EMPTY"}
degen = {c for c, r in cen.items() if r.get("status") == "GEOMETRY_DEGENERATE"}
trained = [c for c in pop if c not in empty and c not in degen and c not in unprod]
print("census EMPTY %d, degenerate %d, unproducible %d  ->  usable labels %d (production run reported 1943 PASS)"
      % (len(empty), len(degen), len(unprod), len(trained)))

def q(v, ps=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(v, float)
    return " ".join("p%d=%.4g" % (p, np.percentile(a, p)) for p in ps)

# ---------- where the EMPTY cells sit, i.e. what the draw spent and the training set lost ----------
ce = [pop[c] for c in empty if pop[c].get("cut_plane")]
print("\n=== the %d cells lost to EMPTY are all cut cells:" % len(ce))
print("   their offset fraction ", q([r["offset_fraction"] for r in ce]))
print("   their theta           ", q([r["theta_deg"] for r in ce]))
print("   their mean tau        ", q([np.mean(r["tau_corners"]) for r in ce]))

T = [pop[c] for c in trained]
Tcut = [r for r in T if r.get("cut_plane")]
Tunc = [r for r in T if not r.get("cut_plane")]
print("\n=== TRAINING SET: %d cells, %d with a cut plane, %d without" % (len(T), len(Tcut), len(Tunc)))
tau = np.array([r["tau_corners"] for r in T])
span = tau.max(axis=1) - tau.min(axis=1); gm = np.array([r.get("gradient_magnitude", 0.0) for r in T])
print("  mean tau      ", q(tau.mean(axis=1)))
print("  corner span   ", q(span))
print("  |gradient|    ", q(gm))
print("  theta (cut)   ", q([r["theta_deg"] for r in Tcut]))
print("  offset (cut)  ", q([r["offset_fraction"] for r in Tcut]))

# ---------- why the top gradient decile is empty ----------
print("\n=== why the design's 'top gradient decile' never arrived")
g = np.array([r.get("gradient", [0, 0, 0]) for r in pop.values()], float)
gn = np.linalg.norm(g, axis=1); nz = gn > 1e-9
l1 = np.abs(g[nz]).sum(axis=1); l2 = gn[nz]
print("  the rejection test is 'corner span > 0.47', and corner span over the unit cube is exactly ||g||_1")
print("  ||g||_1 / ||g||_2 : ", q(l1 / l2), "  (1.0 axis aligned, 1.732 body diagonal)")
print("  so ||g||_2 <= 0.47 / (||g||_1/||g||_2): a body-diagonal gradient is capped at %.3f, an axis one at 0.470" % (0.47 / np.sqrt(3)))
print("  realised |g|: p95 = %.4f, max = %.4f, against the design maximum 0.470" % (np.percentile(gn, 95), gn.max()))
axisness = np.abs(g[nz]).max(axis=1) / l2
hi = gn[nz] > np.percentile(gn[nz], 95)
print("  direction 'axis-alignedness' max|g_i|/|g| : all cells %s" % q(axisness, (5, 50, 95)))
print("                                              top 5%% by |g| %s" % q(axisness[hi], (5, 50, 95)))
print("  -> high-gradient cells are forced to be nearly axis aligned; the design is biased, not just truncated")

# ---------- cut depth is uniform in offset, not in retained volume ----------
xs = (np.arange(500) + 0.5) / 500
X, Y = np.meshgrid(xs, xs, indexing="ij")
def frac(r):
    a, b, _, c = r["cut_plane"]; return float(((a * X + b * Y) <= c).mean())
fr = np.array([frac(r) for r in Tcut])
print("\n=== CUT DEPTH: the design samples the OFFSET uniformly, which is not uniform in retained volume")
print("  retained cube fraction ", q(fr))
h, _ = np.histogram(fr, bins=10, range=(0, 1))
print("  deciles of retained fraction:", h.tolist())
print("  below 0.1: %d (%.1f%%) ; above 0.9: %d (%.1f%%) ; in [0.4,0.6]: %d (%.1f%%)" % (
    (fr < .1).sum(), 100*(fr < .1).mean(), (fr > .9).sum(), 100*(fr > .9).mean(),
    ((fr >= .4) & (fr <= .6)).sum(), 100*((fr >= .4) & (fr <= .6)).mean()))

# ---------- the 48 cube symmetries: what is the orbit, really ----------
print("\n=== SYMMETRY AUGMENTATION: the orbit of each label under the 48 signed permutations")
corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])   # the generator's corner order
G = []
for perm in itertools.permutations(range(3)):
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                M = np.zeros((3, 3)); s = (sx, sy, sz)
                for a, p in enumerate(perm): M[a, p] = s[a]
                G.append(M)
print("  group order: %d" % len(G))

def image(r, M):
    # act on the cube about its centre; corner (i,j,k) -> which corner
    c = corners - 0.5
    cimg = (c @ M.T) + 0.5
    idx = [int(np.argmin(np.abs(corners - p).sum(axis=1))) for p in np.round(cimg).astype(int)]
    tau_img = tuple(np.round([r["tau_corners"][idx[k]] for k in range(8)], 9))
    if not r.get("cut_plane"): return (tau_img, None)
    a, b, cc, d = r["cut_plane"]; n = np.array([a, b, cc]); nimg = M @ n
    dimg = d - n @ np.array([.5, .5, .5]) + nimg @ np.array([.5, .5, .5])
    return (tau_img, tuple(np.round(np.append(nimg, dimg), 9)))

orb = [len({image(r, M) for M in G}) for r in T]
orb = np.array(orb)
print("  orbit size per label: p0=%d p50=%d p100=%d ; mean %.2f" % (orb.min(), np.median(orb), orb.max(), orb.mean()))
import collections
print("  distribution:", dict(sorted(collections.Counter(orb.tolist()).items())))
print("  effective training set after augmentation: %d distinct (geometry, operator) pairs from %d stored labels (x%.1f)"
      % (orb.sum(), len(T), orb.sum() / len(T)))

# reachable cut-plane set
print("\n  reachable cut normals: the stored normal is (cos t, sin t, 0); a signed permutation sends it to a vector")
print("  with ONE ZERO COMPONENT.  So the augmented distribution covers exactly the planes PARALLEL TO A COORDINATE")
print("  AXIS.  A general oblique plane (all three components nonzero, e.g. (1,1,1)/sqrt3) is out of distribution.")
nz3 = sum(1 for r in Tcut if all(abs(v) > 1e-12 for v in r["cut_plane"][:3]))
print("  cells in the population with a fully oblique normal: %d" % nz3)
