"""Is the new decode path element for element the old one, and what does a larger chunk cost?

`predict_full_q` was rewritten: the mirror block is placed by broadcasting instead of an explicit
transpose, and the old completeness guard -- which formed `M - M.T` and reduced it, allocating a
second q x q matrix -- is replaced by a NaN-filled buffer plus a sampled symmetry check.  Both are
meant to be exactly equivalent, so that is measured here against the old body, reproduced verbatim
below, on a real checkpoint and a real cut seat.  The same run re-derives the claim that a larger
chunk is NOT lossless, instead of taking it from a summary.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch


@torch.no_grad()
def old_predict_full_q(T, CG, model, sample, tables, conditioning, g_view, chunk, out_device):
    """The body as it stood before the change, verbatim -- under no_grad, as the real one was.

    Without the decorator the old path builds a decoder graph it never uses, which both inflates its
    time and makes the timing comparison meaningless.
    """
    model.eval()
    ctx = tables.rotate_context(sample['ctx_t'], g_view)
    enc = model.encode(ctx)
    q = sample['q']; count = sample['count']; dev = ctx['pos'].device
    M = torch.zeros((q, q), dtype=torch.float64, device=out_device)
    total = count * (count + 1) // 2
    a = torch.arange(3, device=out_device)[None, :, None]; b = torch.arange(3, device=out_device)[None, None, :]
    g_back = int(CG.INVERSE[g_view])
    for lo in range(0, total, chunk):
        index = np.arange(lo, min(total, lo + chunk), dtype=np.int64)
        rows = np.floor((np.sqrt(8 * index + 1) - 1) / 2).astype(np.int64); cols = index - rows * (rows + 1) // 2
        r = torch.as_tensor(rows, device=dev); c = torch.as_tensor(cols, device=dev)
        blocks = model.decode(enc, ctx, r, c, conditioning).double()
        diag = r == c
        blocks[diag] = blocks[diag] + torch.tril(blocks[diag], -1).transpose(1, 2)
        blocks = tables.rotate_blocks(blocks, g_back).to(out_device)
        r = r.to(out_device); c = c.to(out_device)
        M[3 * r[:, None, None] + a, 3 * c[:, None, None] + b] = blocks
        M[3 * c[:, None, None] + a, 3 * r[:, None, None] + b] = blocks.transpose(1, 2)
    if float((M - M.T).abs().max()) != 0.0:
        raise ValueError('ASSEMBLY_NOT_SYMMETRIC')
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--seats', type=int, nargs='+', required=True)
    ap.add_argument('--manifest', type=Path, default=Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--chunks', type=int, nargs='+', default=[8192, 32768])
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.source)); sys.path.insert(0, str(a.repo))
    from superelement.equi import train_equi as T
    from superelement.equi import cubic_group as CG
    from superelement.equi.model import EquiModel, GroupTables
    from stage_cutfem_m4.factors import Conditioning
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    rows = T.selected_rows(a.manifest, list(a.seats), False)
    prep = argparse.Namespace(manifest=a.manifest, near_cells=2.0, known_328=False, seats=list(a.seats))
    samples = [T.prepare(r, prep, dev) for r in rows]
    n = samples[0]['ctx']['n']
    tables = GroupTables(n, dev, len(samples[0]['ctx']['vol_scalar']))
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    cond = Conditioning(**ck['conditioning'])
    model = EquiModel(n=n, volume_in=int(samples[0]['ctx_t']['volume'].shape[0]),
                      near_radius=float(ck['near_radius'])).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    out = dict(checkpoint=str(a.checkpoint), step=int(ck['step']), chunks=list(a.chunks), rows=[])
    for s in samples:
        record = dict(seat=s['seat'], q=s['q'], count=s['count'])
        timings = {}
        base = None
        for name, fn, chunk in ([('old', 'old', 8192), ('new', 'new', 8192)]
                                + [('new_chunk_%d' % c, 'new', c) for c in a.chunks if c != 8192]):
            best = float('inf'); M = None
            for _ in range(a.repeats):
                tick = time.perf_counter()
                if fn == 'old':
                    M = old_predict_full_q(T, CG, model, s, tables, cond, 0, chunk, dev)
                else:
                    M = T.predict_full_q(model, s, tables, cond, 0, chunk, dev,
                                         verify_symmetry_full=True, allow_unverified_chunk=True)
                best = min(best, time.perf_counter() - tick)
            timings[name] = best
            if name == 'old':
                base = M
            else:
                delta = float((M - base).abs().max())
                rel = float((M - base).norm() / base.norm())
                record[name] = dict(max_absolute_difference=delta, relative_frobenius_difference=rel,
                                    identical=bool(delta == 0.0))
                if name == 'new':
                    # the two paths must agree bit for bit, so also compare the deployed spectrum
                    record['new']['diagonal_identical'] = bool(
                        float((M.diagonal() - base.diagonal()).abs().max()) == 0.0)
                del M
        record['seconds'] = timings
        record['speedup_new_over_old'] = timings['old'] / timings['new']
        for c in a.chunks:
            if c != 8192:
                record['speedup_chunk_%d_over_old' % c] = timings['old'] / timings['new_chunk_%d' % c]
        out['rows'].append(record)
        print(json.dumps(record), flush=True)
        del base
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
