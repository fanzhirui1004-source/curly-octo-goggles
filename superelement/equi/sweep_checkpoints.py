"""Physics against training progress: the assembly-free response gate at every checkpoint.

`evaluate` runs once, at the end, so nothing says whether the physics error is still falling
when the loss is, or which way it drifts.  The load-energy diagnosis predicts a specific drift:
the compliance error is `-<log mu>_w + <log^2 mu>_w / 2`, the first term is an L2 regression
shrinkage bias (the network is 2-5 % too stiff in the energy norm) and the second is the spread.
Shrinkage decays with training, the spread need not, so the signed error should drift from
negative (too stiff) toward positive (too soft) as a run converges.  This measures that.

Per checkpoint and seat it writes the response gate on the smooth face and global families, the
signed error per load, and the per-column (point-load) relative error, which is the quantity the
--column-weight term targets.

    python -m superelement.equi.sweep_checkpoints --run <RUN_DIR> --seats 347 100051 --every 4 --output DIR
"""
from __future__ import annotations

import argparse, gc, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True, help='a training output directory with CHECKPOINT_*.pt')
    ap.add_argument('--manifest', type=Path, default=Path('/root/autodl-tmp/CLAUDE_LABELS_20260917/V2_LABELS.json'))
    ap.add_argument('--seats', type=int, nargs='+', required=True)
    ap.add_argument('--every', type=int, default=1, help='take every k-th checkpoint')
    ap.add_argument('--last', type=int, default=0, help='if set, only the last k checkpoints (after --every)')
    ap.add_argument('--near-cells', type=float, default=2.0)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--eval-device', default='cpu')
    ap.add_argument('--inference-chunk', type=int, default=8192)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--response-per-face', type=int, default=16)
    ap.add_argument('--response-seed', type=int, default=20260919)
    ap.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/CLAUDE_SQRTHEAD_20260917/src_v5'))
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(a.source)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from superelement.equi import train_equi as T
    from superelement.equi.model import EquiModel, GroupTables
    from stage_cutfem_m4.factors import Conditioning
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device); out_dev = torch.device(a.eval_device)

    checks = sorted(a.run.glob('CHECKPOINT_*.pt'))
    if not checks:
        raise ValueError(f'NO_CHECKPOINTS_IN {a.run}')
    checks = checks[::a.every]
    if a.last:
        checks = checks[-a.last:]
    prep = argparse.Namespace(manifest=a.manifest, near_cells=a.near_cells, known_328=False, seats=list(a.seats))
    rows = T.selected_rows(a.manifest, list(a.seats), False)
    samples = []
    for row in rows:
        s = T.prepare(row, prep, dev)
        s['bank'] = T.response_bank(s, a.response_per_face, a.response_seed, a.run.parent / 'RESPONSE_BANK')
        samples.append(s)
    n = samples[0]['ctx']['n']
    tables = GroupTables(n, dev, len(samples[0]['ctx']['vol_scalar']))
    print(json.dumps(dict(phase='plan', run=str(a.run), checkpoints=[p.name for p in checks],
                          seats=[s['seat'] for s in samples])), flush=True)

    rows_out = []
    for path in checks:
        ck = torch.load(path, map_location=dev, weights_only=False)
        cond = Conditioning(**ck['conditioning'])
        model = EquiModel(n=n, volume_in=int(samples[0]['ctx_t']['volume'].shape[0]),
                          near_radius=float(ck['near_radius'])).to(dev)
        model.load_state_dict(ck['model']); model.eval()
        for s in samples:
            t0 = time.time()
            g0 = T.view_frame(s, 0, False)
            M = T.predict_full_q(model, s, tables, cond, g0, a.inference_chunk, out_dev)
            gate = T.response_gate(M, s['bank'], s['q'])
            # per-column relative error: the point-load family, from the cached column norms
            label = T.unpack(s['label'], s['q'], out_dev)
            label = label + torch.triu(label, 1).T
            col = ((M - label).square().sum(dim=0).reshape(-1, 3).sum(dim=1)
                   / torch.as_tensor(s['bank']['colnorm2'], dtype=M.dtype, device=out_dev)).sqrt()
            e_factor = float((M - label).norm() / label.norm())
            del label, M; gc.collect()
            row = dict(step=int(ck['step']), checkpoint=path.name, seat=s['seat'], e_factor=e_factor,
                       column_rel_median=float(col.median()), column_rel_p90=float(col.quantile(.9)),
                       column_rel_max=float(col.max()), gate=gate, seconds=time.time() - t0)
            rows_out.append(row)
            with (a.output / 'SWEEP.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(dict(step=row['step'], seat=row['seat'], e_factor=round(e_factor, 4),
                                  col_med=round(row['column_rel_median'], 4), col_p90=round(row['column_rel_p90'], 4),
                                  face_med=round(gate['face']['median_abs'], 5), face_signed=round(gate['face']['signed_mean'], 5),
                                  glob_med=round(gate['global_smooth']['median_abs'], 5),
                                  glob_signed=round(gate['global_smooth']['signed_mean'], 5),
                                  seconds=round(row['seconds'], 1))), flush=True)
        del model, ck; gc.collect()
    (a.output / 'RESULT.json').write_text(json.dumps(dict(status='SWEEP_COMPLETE', run=str(a.run),
                                                          rows=len(rows_out), checkpoints=len(checks)), indent=1))
    print('SWEEP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
