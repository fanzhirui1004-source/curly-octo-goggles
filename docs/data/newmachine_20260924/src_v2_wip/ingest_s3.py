"""Make produced expansion cells (OPL/S3, gen_new.py --mode indep + prod_worker.sh) visible to the step-2 tools without
copying: symlinks S0/<case> -> S3/body/<case> (and its neighbour bodies <case>_nb*), S2/data/<case> -> S3/data/<case>,
S2/data_v2/<case> -> S3/data_v2/<case>, for every cell whose data dir has DONE.json. Existing names are never replaced.
Writes a split = the old split (train / val / test unchanged) + the ingested S3 train cells appended to 'train' and the S3
validation cells to 'val' (after the old val, so val_max = len(old val) keeps the old selection set); 'val_s3' lists them.
Tools reading packets need OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets.
Usage: ingest_s3.py [--split-in S2/SPLIT.json] [--split-out S2/SPLIT_V3.json]"""
import argparse, json, os
from pathlib import Path

R = Path('/root/autodl-tmp/OPL')


def link(src, dst):
    if dst.exists() or dst.is_symlink():
        return 0
    os.symlink(src, dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-in', default=str(R / 'S2/SPLIT.json'))
    ap.add_argument('--split-out', default=str(R / 'S2/SPLIT_V3.json'))
    a = ap.parse_args()
    plan = json.loads((R / 'S3/PLAN_INDEP.json').read_text())
    made, cells = 0, []
    for c in plan['train'] + plan['validation']:
        if not (R / 'S3/data' / c / 'DONE.json').exists():
            continue
        cells.append(c)
        made += link(R / 'S3/body' / c, R / 'S0' / c)
        for nb in sorted((R / 'S3/body').glob(f'{c}_nb*')):
            made += link(nb, R / 'S0' / nb.name)
        made += link(R / 'S3/data' / c, R / 'S2/data' / c)
        if (R / 'S3/data_v2' / c).exists():
            made += link(R / 'S3/data_v2' / c, R / 'S2/data_v2' / c)
    sp = json.loads(Path(a.split_in).read_text())
    have = set(cells)
    tr = [c for c in plan['train'] if c in have]
    va = [c for c in plan['validation'] if c in have]
    out = dict(sp, train=list(sp['train']) + tr, val=list(sp['val']) + va, val_s3=va, train_s3=tr,
               note='S2/SPLIT.json + ingested S3 cells (ingest_s3.py)')
    Path(a.split_out).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(ingested=len(cells), links_made=made, train_s3=len(tr), val_s3=len(va), split=a.split_out)))


if __name__ == '__main__':
    main()
