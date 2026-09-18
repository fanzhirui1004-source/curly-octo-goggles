"""The response view: c(f) = ||M_q f||^2 exactly for one cell, so the compliance error under
load f is ||M_hat f||^2 / ||M f||^2 - 1, no assembly needed.  Measure it for (i) a unit point
load at every coordinate (= the per-column relative error), (ii) smooth polynomial face
tractions, (iii) random dense loads.  Dense fp64 on the host."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/autodl-tmp/CLAUDE_EQUI_20260918/src')
from superelement.equi.context import compile_equi_inputs

E = '/root/autodl-tmp/CLAUDE_EQUI_20260918'
rows = {int(r['seat']): r for r in json.load(open('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))}


def dense(path, q):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    M = np.zeros((q, q)); off = 0
    for i in range(q):
        seg = np.asarray(p[off:off + q - i]); M[i, i:] = seg; off += q - i
    M = M + np.triu(M, 1).T
    return M


def main(seat):
    t0 = time.time()
    r = rows[seat]; ref = Path(r['reference']); q = int(r['q'])
    ev = f'{seat:04d}' if seat < 10000 else str(seat)
    T = dense(ref / 'MQ_UPPER.npy', q)
    P = dense(f'{E}/MULTI_AUGMENT/EVAL_{ev}/MQ_PRED_UPPER.npy', q)
    cache = dict(np.load(r['trace_cache'], allow_pickle=False))
    meta = json.loads((Path(r['trace_cache']).parent / 'INPUT.json').read_text())['metadata']
    ctx = compile_equi_inputs(cache, meta); pos = ctx['pos']; count = ctx['count']
    rigid = np.asarray(cache['rigid'], dtype=np.float64)
    Qr, _ = np.linalg.qr(rigid)
    out = dict(seat=seat, q=q, count=count, load_seconds=time.time() - t0)

    # (i) point loads: per-column relative error, and the compliance error per column
    tn = np.linalg.norm(T, axis=0); en = np.linalg.norm(P - T, axis=0); pn = np.linalg.norm(P, axis=0)
    rel = en / tn; cerr = (pn / tn) ** 2 - 1.0
    piv = np.diagonal(T).reshape(count, 3).mean(axis=1)
    node_rel = rel.reshape(count, 3).max(axis=1)
    out['point'] = dict(rel_median=float(np.median(rel)), rel_p90=float(np.percentile(rel, 90)),
                        rel_p99=float(np.percentile(rel, 99)), rel_max=float(rel.max()),
                        cerr_median=float(np.median(cerr)), cerr_p90=float(np.percentile(np.abs(cerr), 90)),
                        cerr_max=float(np.abs(cerr).max()),
                        frac_cerr_within_3pct=float((np.abs(cerr) <= .03).mean()),
                        colnorm_min=float(tn.min()), colnorm_med=float(np.median(tn)), colnorm_max=float(tn.max()),
                        spearman_rel_vs_pivot=float(np.corrcoef(np.argsort(np.argsort(node_rel)),
                                                                np.argsort(np.argsort(piv)))[0, 1]))
    # by pivot-scale decile: is the relative column error uniform across soft/stiff coordinates?
    dec = np.digitize(piv, np.percentile(piv, np.arange(10, 100, 10)))
    out['point_rel_median_by_pivot_decile'] = [float(np.median(node_rel[dec == k])) for k in range(10)]
    out['point_cerr_p90_by_pivot_decile'] = [float(np.percentile(np.abs(cerr.reshape(count, 3)[dec == k]), 90)) for k in range(10)]

    # (ii) smooth polynomial tractions on each face, random direction, degree <= 2
    rng = np.random.default_rng(20260919)
    smooth = []
    faces = ctx['faces']                          # count x 6 membership
    for trial in range(96):
        f_id = trial % 6; axis = f_id // 2
        members = np.flatnonzero(faces[:, f_id])
        if len(members) < 4:
            continue
        u = [k for k in range(3) if k != axis]
        s, t = pos[members, u[0]] - .5, pos[members, u[1]] - .5
        basis = np.stack([np.ones_like(s), s, t, s * s, s * t, t * t], axis=1)
        coeff = rng.standard_normal((6, 3))
        field = basis @ coeff                      # members x 3
        f = np.zeros(q); f[(members[:, None] * 3 + np.arange(3)).ravel()] = field.ravel()
        f -= Qr @ (Qr.T @ f)
        smooth.append((np.linalg.norm(P @ f) / np.linalg.norm(T @ f)) ** 2 - 1.0)
    smooth = np.array(smooth)
    out['smooth_face'] = dict(n=int(len(smooth)), cerr_median=float(np.median(np.abs(smooth))),
                              cerr_p90=float(np.percentile(np.abs(smooth), 90)), cerr_max=float(np.abs(smooth).max()),
                              signed_mean=float(smooth.mean()), frac_within_3pct=float((np.abs(smooth) <= .03).mean()))
    # (iii) random dense loads (white in coordinate space) -- the "typical arbitrary load"
    F = rng.standard_normal((q, 64)); F -= Qr @ (Qr.T @ F)
    dn = ((np.linalg.norm(P @ F, axis=0) / np.linalg.norm(T @ F, axis=0)) ** 2 - 1.0)
    out['random_dense'] = dict(cerr_median=float(np.median(np.abs(dn))), cerr_max=float(np.abs(dn).max()),
                               signed_mean=float(dn.mean()))
    # (iv) random SMOOTH global loads: low-frequency fields over the whole trace
    G = []
    for trial in range(64):
        k = rng.integers(1, 3, size=3); ph = rng.uniform(0, 2 * np.pi, size=3)
        fld = np.cos(2 * np.pi * (pos @ k) + ph[0])[:, None] * rng.standard_normal(3)[None, :]
        f = fld.ravel(); f -= Qr @ (Qr.T @ f); G.append(f)
    G = np.stack(G, axis=1)
    gn = ((np.linalg.norm(P @ G, axis=0) / np.linalg.norm(T @ G, axis=0)) ** 2 - 1.0)
    out['random_smooth_global'] = dict(cerr_median=float(np.median(np.abs(gn))), cerr_p90=float(np.percentile(np.abs(gn), 90)),
                                       cerr_max=float(np.abs(gn).max()), signed_mean=float(gn.mean()),
                                       frac_within_3pct=float((np.abs(gn) <= .03).mean()))
    out['seconds'] = time.time() - t0
    Path(f'{E}/RESPONSE_ERR').mkdir(exist_ok=True)
    Path(f'{E}/RESPONSE_ERR/SEAT_{seat}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        main(int(s))
