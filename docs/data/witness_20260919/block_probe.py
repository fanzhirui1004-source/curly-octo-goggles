"""Where does a cut cell's prediction error sit?  Block-wise and per-coordinate decomposition of
the stored dense predictions (EVAL_*/MQ_PRED_UPPER.npy) against the labels, by coordinate kind
(0 = box node value, 1 = cut-surface residual functional) and, for kind 1, by stencil size.

    python block_probe.py RUN:SEAT [RUN:SEAT ...]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

W = Path('/root/autodl-tmp/CLAUDE_EQUI_20260918')
MANIFEST = '/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'


def unpack(path, q):
    p = np.load(path, mmap_mode='r', allow_pickle=False)
    M = np.zeros((q, q)); off = 0
    for i in range(q):
        M[i, i:] = p[off:off + q - i]; off += q - i
    return M + np.triu(M, 1).T


def eval_dir(run, seat):
    for name in (f'EVAL_{seat}', f'EVAL_{seat:04d}'):
        if (W / run / name / 'MQ_PRED_UPPER.npy').exists():
            return W / run / name
    return None


def main():
    rows = {int(r['seat']): r for r in json.load(open(MANIFEST))}
    out = []
    for item in sys.argv[1:]:
        run, seat = item.split(':'); seat = int(seat)
        ed = eval_dir(run, seat)
        if ed is None:
            print(json.dumps(dict(run=run, seat=seat, error='NO_EVAL')), flush=True); continue
        t0 = time.time()
        row = rows[seat]; q = int(row['q'])
        cache = np.load(row['trace_cache'], allow_pickle=False)
        kind = np.asarray(cache['kind']).astype(int); nnz = np.diff(cache['indptr'])
        k3 = np.repeat(kind, 3); nnz3 = np.repeat(nnz, 3)
        L = unpack(Path(row['reference']) / 'MQ_UPPER.npy', q)
        P = unpack(ed / 'MQ_PRED_UPPER.npy', q)
        D = P - L
        rep = dict(run=run, seat=seat, q=q, kind1_coords=int((kind == 1).sum()), cut=bool(kind.any()),
                   total_rel=float(np.linalg.norm(D) / np.linalg.norm(L)))
        blocks = {}
        for a, b in ((0, 0), (0, 1), (1, 1)):
            ma = k3 == a; mb = k3 == b
            if ma.any() and mb.any():
                Lb = L[np.ix_(ma, mb)]; Db = D[np.ix_(ma, mb)]
                blocks[f'{a}{b}'] = dict(rel=float(np.linalg.norm(Db) / np.linalg.norm(Lb)),
                                        label_norm=float(np.linalg.norm(Lb)), error_norm=float(np.linalg.norm(Db)))
        rep['blocks'] = blocks
        # per-coordinate row error (relative), summarised by kind and by stencil size
        rown = np.linalg.norm(D, axis=1) / np.maximum(np.linalg.norm(L, axis=1), 1e-300)
        diag_rel = np.abs(np.diagonal(D)) / np.abs(np.diagonal(L))
        for a in (0, 1):
            m = k3 == a
            if m.any():
                rep[f'kind{a}_row_rel'] = dict(median=float(np.median(rown[m])), p90=float(np.quantile(rown[m], .9)), max=float(rown[m].max()))
                rep[f'kind{a}_diag_rel'] = dict(median=float(np.median(diag_rel[m])), p90=float(np.quantile(diag_rel[m], .9)), max=float(diag_rel[m].max()))
        if kind.any():
            by = {}
            for lo, hi in ((2, 3), (4, 6), (7, 10), (11, 14), (15, 30)):
                m = (k3 == 1) & (nnz3 >= lo) & (nnz3 <= hi)
                if m.any():
                    by[f'nnz{lo}-{hi}'] = dict(n=int(m.sum() // 3), row_rel_median=float(np.median(rown[m])), diag_rel_median=float(np.median(diag_rel[m])))
            rep['kind1_by_stencil'] = by
            # the diagonal itself: where the label's tiny/huge scales sit
            dg = np.diagonal(L)
            rep['label_diag_range'] = {f'kind{a}': [float(dg[k3 == a].min()), float(np.median(dg[k3 == a])), float(dg[k3 == a].max())] for a in (0, 1)}
        rep['seconds'] = round(time.time() - t0, 1)
        print(json.dumps(rep), flush=True); out.append(rep)
        del L, P, D
    Path(os.environ.get('BLOCK_PROBE_OUT', 'BLOCK_PROBE.json')).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
