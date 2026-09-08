"""Quantify the two proposed sampling fixes against the current design, on the same Sobol machinery."""
import json, numpy as np
from scipy.stats import qmc
R = "artifacts/PRED777H_FULL_CUBE_SINGLE_CELL_MESH_GRAPH_FIXED_PORT_SCHUR_V1/sheet_true_geometry_label_route_v1"
TAU_MIN, TAU_MAX, DTAU = 0.1755, 0.8775, 0.47
pop = {r["case"]: r for r in (json.loads(l) for l in open(R + "/population/POPULATION.jsonl") if l.lstrip().startswith("{"))}
cen = {r["case"]: r for r in (json.loads(l) for l in open(R + "/population/CENSUS.jsonl") if l.lstrip().startswith("{"))}

def q(v, ps=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(v, float); return " ".join("p%d=%.4g" % (p, np.percentile(a, p)) for p in ps)

# ---------------------------------------------------------------- gradient
def corners_from(mean, g):
    c = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) - 0.5
    return mean + c @ g

rng = np.random.default_rng(7)
N = 200000
mean = TAU_MIN + rng.random(N) * (TAU_MAX - TAU_MIN)
u = rng.random(N)
d = rng.normal(size=(N, 3)); d /= np.linalg.norm(d, axis=1)[:, None]

print("=== GRADIENT: current design vs the span-parameterised fix")
# current: mag = DTAU * u^2 along a uniform direction, then REJECT if corner span (=||g||_1) > DTAU or corners escape
g_cur = (DTAU * u * u)[:, None] * d
span_cur = np.abs(g_cur).sum(axis=1)
C = corners_from(mean[:, None], g_cur.T).T if False else np.stack([corners_from(m, gg) for m, gg in zip(mean[:5], g_cur[:5])])
def accept(mean, g):
    c = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) - 0.5
    corners = mean[:, None] + g @ c.T
    span = corners.max(axis=1) - corners.min(axis=1)
    return (corners.min(axis=1) >= TAU_MIN) & (corners.max(axis=1) <= TAU_MAX) & (span <= DTAU), span
ok_cur, span_c = accept(mean, g_cur)
print("  CURRENT  accept rate %.3f ; accepted |g|_2 %s" % (ok_cur.mean(), q(np.linalg.norm(g_cur[ok_cur], axis=1), (50, 95, 100))))
print("           accepted corner span %s" % q(span_c[ok_cur], (5, 50, 95, 100)))
al = np.abs(d).max(axis=1)
hi = ok_cur & (np.linalg.norm(g_cur, axis=1) > np.percentile(np.linalg.norm(g_cur[ok_cur], axis=1), 95))
print("           axis-alignedness of accepted: all %s ; top 5%% by |g| %s"
      % (q(al[ok_cur], (5, 50, 95)), q(al[hi], (5, 50, 95))))

# fix: draw the TARGET CORNER SPAN S = DTAU * u^2, then set |g| = S / ||dir||_1 so the span is S whatever the direction
S = DTAU * u * u
g_fix = (S / np.abs(d).sum(axis=1))[:, None] * d
ok_fix, span_f = accept(mean, g_fix)
print("  FIXED    accept rate %.3f ; accepted |g|_2 %s" % (ok_fix.mean(), q(np.linalg.norm(g_fix[ok_fix], axis=1), (50, 95, 100))))
print("           accepted corner span %s" % q(span_f[ok_fix], (5, 50, 95, 100)))
hi2 = ok_fix & (span_f > np.percentile(span_f[ok_fix], 95))
print("           axis-alignedness of accepted: all %s ; top 5%% by span %s"
      % (q(al[ok_fix], (5, 50, 95)), q(al[hi2], (5, 50, 95))))
print("           cells reaching span >= 0.40: current %.2f%% , fixed %.2f%%"
      % (100 * (span_c[ok_cur] >= 0.40).mean(), 100 * (span_f[ok_fix] >= 0.40).mean()))


print()
print("=== CUT DEPTH: offset-uniform (current) vs retained-volume-uniform (proposed)")
# exact area of {a x + b y <= c} in the unit square, a >= b > 0, s = a + b  (piecewise quadratic, invertible)
def V(a, b, c):
    a, b = max(a, b), min(a, b); s = a + b
    c = np.clip(c, 0.0, s)
    return np.where(c <= b, c * c / (2 * a * b),
           np.where(c <= a, (c - b / 2) / a, 1.0 - (s - c) ** 2 / (2 * a * b)))
def Vinv(a, b, v):
    a, b = max(a, b), min(a, b); s = a + b
    vb = b / (2 * a); va = (a - b / 2) / a
    return np.where(v <= vb, np.sqrt(np.maximum(v * 2 * a * b, 0)),
           np.where(v <= va, v * a + b / 2, s - np.sqrt(np.maximum((1 - v) * 2 * a * b, 0))))
rng2 = np.random.default_rng(11)
M = 200000
th = rng2.random(M) * np.pi / 4
A, B = np.cos(th), np.sin(th)
v_cur = np.array([V(a, b, o * (a + b)) for a, b, o in zip(A, B, rng2.random(M))])
vt = rng2.random(M)
v_new = np.array([V(a, b, Vinv(a, b, v)) for a, b, v in zip(A, B, vt)])
for name, v in (("offset-uniform (current)", v_cur), ("volume-uniform (proposed)", v_new)):
    h, _ = np.histogram(v, bins=10, range=(0, 1))
    print("  %s deciles %s" % (name, (h / len(v) * 100).round(1).tolist()))
    print("     %s  <0.1: %4.1f%%   [0.4,0.6]: %4.1f%%   >0.9: %4.1f%%" % (" " * 22, 100*(v<.1).mean(), 100*((v>=.4)&(v<=.6)).mean(), 100*(v>.9).mean()))
emp = [pop[c] for c, r in cen.items() if r.get("status") == "EMPTY"]
ve = np.array([float(V(r["cut_plane"][0], r["cut_plane"][1], r["cut_plane"][3])) for r in emp])
print("  the 59 EMPTY cells sat at retained cube volume %s" % q(ve, (0, 50, 95, 100)))
print("  volume-uniform would put about %.1f%% of cut draws below that p95 (%.4f), against %.1f%% now"
      % (100 * np.percentile(ve, 95), np.percentile(ve, 95), 100 * (v_cur < np.percentile(ve, 95)).mean()))
