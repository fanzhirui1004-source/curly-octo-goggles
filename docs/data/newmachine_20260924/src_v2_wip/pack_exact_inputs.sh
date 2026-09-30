#!/bin/bash
# pack_exact_inputs.sh -- run on the GPU server. Packs the inputs of exact_check_cpu.py for chosen design iterations of
# opt_design.py runs into ONE tar (one design set) with a manifest. Read-only on the run data: it writes only under --out
# (the tar, its manifest, the spec record, and a temporary staging directory of symbolic links removed at the end).
#
# Usage: pack_exact_inputs.sh [--out DIR] [--name NAME] [--src DIR | --no-src] [--extra FILE]... [--no-md5] [--dry-run]
#                             <run_dir>:<iters> [<run_dir>:<iters> ...]
#   <iters>  comma list of iteration numbers, 'last' (last recorded k) and 'mid' (last // 2), as exact_check_cpu.py
#   --out    output directory (default /root/autodl-tmp/exact_pack); must not lie inside a run directory
#   --name   tar name without .tar (default exact_inputs_<date>_<time>)
#   --src    code directory packed as <NAME>/src_v2/*.py (default /root/autodl-tmp/OPL/src_v2, the code of the runs)
#   --extra  further files packed as <NAME>/extra/<basename> (e.g. exact_check_cpu.py, iparm_tuned.json)
#   --no-md5 manifest with sizes only (faster)
#   --dry-run  print the plan and the sizes, write nothing
# Contents of the tar (paths relative to <NAME>/):
#   runs/<run>/history.jsonl, meta.json, layout.json (a copy of the layout file named in meta.json),
#   runs/<run>/body/GP_TEMPLATES_n*.npz, and for every case of the chosen iterations runs/<run>/body/<case>/ and
#   runs/<run>/packets/<case>/ (no <case>_portview directories, no logs); src_v2/*.py; extra/*;
#   MANIFEST.tsv (bytes, md5, path) and SPECS.json (spec -> iterations -> cases, source paths, layout, load settings).
# Unpacked, a run directory of the tar is a valid <run_dir> for exact_check_cpu.py (it reads <run>/layout.json first).
set -euo pipefail
OUT=/root/autodl-tmp/exact_pack; NAME=exact_inputs_$(date +%Y%m%d_%H%M%S); SRC=/root/autodl-tmp/OPL/src_v2
MD5=1; DRY=0; EXTRA=(); SPECS=()
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --src) SRC=$2; shift 2 ;;
    --no-src) SRC=""; shift ;;
    --extra) EXTRA+=("$2"); shift 2 ;;
    --no-md5) MD5=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) SPECS+=("$1"); shift ;;
  esac
done
[ ${#SPECS[@]} -gt 0 ] || { echo "no <run_dir>:<iters> given" >&2; exit 2; }
PYBIN=${PY:-python3}
PLAN=-                                                  # --dry-run: no files at all
if [ "$DRY" = 0 ]; then
  PLAN=$(mktemp -d)                                     # plan files in the system temp directory, removed on exit
  trap 'rm -rf "$PLAN"' EXIT
fi

# ---------------------------------------------------------------- plan (python; reads only)
"$PYBIN" - "$PLAN" "$OUT" "$NAME" "$SRC" "$MD5" "$DRY" "${#EXTRA[@]}" "${EXTRA[@]}" "${SPECS[@]}" <<'PY'
import sys, os, json, re, hashlib
from pathlib import Path
plan, out, name, src, md5, dry, nextra = sys.argv[1:8]
rest = sys.argv[8:]
extra, specs = rest[:int(nextra)], rest[int(nextra):]
md5, dry = md5 == '1', dry == '1'
entries, seen, rec = [], set(), dict(name=name, specs=[], runs={})

def add(dest, srcp):
    srcp = Path(srcp)
    if not srcp.is_file():
        raise SystemExit(f'MISSING {srcp}')
    if dest in seen:
        return
    seen.add(dest); entries.append((dest, str(srcp.resolve()), srcp.stat().st_size))

def add_dir(dest, d):
    d = Path(d)
    if not d.is_dir():
        raise SystemExit(f'MISSING_DIR {d}')
    for p in sorted(d.iterdir()):
        if p.is_file() and not p.is_symlink():
            add(f'{dest}/{p.name}', p)

outp = Path(out).resolve()
for s in specs:
    if ':' not in s:
        raise SystemExit(f'BAD_SPEC {s}')
    run, it = s.rsplit(':', 1)
    run = Path(run).resolve()
    toks = [t for t in it.split(',') if t]
    if not toks or any(t not in ('last', 'mid') and not re.fullmatch(r'\d+', t) for t in toks):
        raise SystemExit(f'BAD_ITERS {s}')
    if outp == run or run in outp.parents:
        raise SystemExit(f'OUT_INSIDE_RUN {outp} in {run}')
    meta = json.loads((run / 'meta.json').read_text())
    hist = {}
    for line in (run / 'history.jsonl').read_text().splitlines():
        if line.strip():
            d = json.loads(line); hist[int(d['k'])] = d
    ks = sorted(hist); last = ks[-1]
    iters = []
    for t in toks:
        k = last if t == 'last' else (last // 2 if t == 'mid' else int(t))
        if k not in hist:
            raise SystemExit(f'ITERATION_NOT_RECORDED {run.name}:{k} (0..{last})')
        if k not in iters:
            iters.append(k)
    rn = run.name
    if rn in rec['runs'] and rec['runs'][rn]['source'] != str(run):
        raise SystemExit(f'RUN_NAME_CLASH {rn}')
    lay = Path(meta.get('layout') or meta['args']['layout'])
    base = f'{name}/runs/{rn}'
    add(f'{base}/history.jsonl', run / 'history.jsonl'); add(f'{base}/meta.json', run / 'meta.json')
    add(f'{base}/layout.json', lay)
    tpl = sorted((run / 'body').glob('GP_TEMPLATES_n*.npz'))
    if not tpl:
        raise SystemExit(f'NO_GP_TEMPLATES {run}/body')
    for t in tpl:
        add(f'{base}/body/{t.name}', t)
    r_ = rec['runs'].setdefault(rn, dict(source=str(run), layout=str(lay), args={k: meta.get('args', {}).get(k) for k in
                                         ('clamp', 'load', 'load_dir', 'tmin', 'tmax', 'prec', 'vstar', 'vfrac')},
                                         last_k=last, iterations={}))
    for k in iters:
        cases = hist[k]['cases']
        n0 = len(entries)
        for c in cases:
            add_dir(f'{base}/body/{c}', run / 'body' / c)
            add_dir(f'{base}/packets/{c}', run / 'packets' / c)
        r_['iterations'][str(k)] = dict(cases=cases, C_nice=hist[k]['C'], new_files=len(entries) - n0,
                                        new_bytes=sum(e[2] for e in entries[n0:]))
    rec['specs'].append(dict(spec=s, run=rn, iterations=iters))
if src:
    for p in sorted(Path(src).glob('*.py')):
        add(f'{name}/src_v2/{p.name}', p)
for e in extra:
    add(f'{name}/extra/{Path(e).name}', e)
tot = sum(e[2] for e in entries)
rec.update(files=len(entries), bytes=tot)
for rn, r_ in rec['runs'].items():
    for k, v in r_['iterations'].items():
        print(f'{rn}:{k}  cases {len(v["cases"])}  new files {v["new_files"]}  new bytes {v["new_bytes"] / 2**20:.1f} MiB'
              f'  C_nice {v["C_nice"]:.6g}')
print(f'TOTAL files {len(entries)}  bytes {tot}  ({tot / 2**30:.3f} GiB)')
if dry:
    sys.exit(0)
with open(Path(plan) / 'entries.tsv', 'w') as f:
    for dest, sp_, size in entries:
        f.write(f'{dest}\t{sp_}\t{size}\n')
with open(Path(plan) / 'MANIFEST.tsv', 'w') as f:
    f.write('bytes\tmd5\tpath\n')
    for dest, sp_, size in entries:
        h = '-'
        if md5:
            hh = hashlib.md5()
            with open(sp_, 'rb') as g:
                for chunk in iter(lambda: g.read(1 << 22), b''):
                    hh.update(chunk)
            h = hh.hexdigest()
        f.write(f'{size}\t{h}\t{dest}\n')
(Path(plan) / 'SPECS.json').write_text(json.dumps(rec, indent=1))
PY
[ "$DRY" = 1 ] && exit 0

# ---------------------------------------------------------------- stage (symbolic links under --out) and tar
mkdir -p "$OUT"
[ -e "$OUT/$NAME.tar" ] && { echo "exists: $OUT/$NAME.tar" >&2; exit 3; }
STAGE="$OUT/.stage_$NAME"
[ -e "$STAGE" ] && { echo "exists: $STAGE" >&2; exit 3; }
mkdir -p "$STAGE/$NAME"
trap 'rm -rf "$PLAN" "$STAGE"' EXIT
while IFS=$'\t' read -r dest src size; do
  mkdir -p "$STAGE/$(dirname "$dest")"
  ln -s "$src" "$STAGE/$dest"
done < "$PLAN/entries.tsv"
cp "$PLAN/MANIFEST.tsv" "$PLAN/SPECS.json" "$STAGE/$NAME/"
tar -chf "$OUT/$NAME.tar" -C "$STAGE" "$NAME"
cp "$PLAN/MANIFEST.tsv" "$OUT/$NAME.MANIFEST.tsv"; cp "$PLAN/SPECS.json" "$OUT/$NAME.SPECS.json"
( cd "$OUT" && md5sum "$NAME.tar" > "$NAME.tar.md5" )
N=$(( $(wc -l < "$PLAN/MANIFEST.tsv") - 1 ))
echo "tar $OUT/$NAME.tar  $(stat -c %s "$OUT/$NAME.tar") bytes ($(du -h "$OUT/$NAME.tar" | cut -f1))  files $N  md5 $(cut -d' ' -f1 "$OUT/$NAME.tar.md5")"
