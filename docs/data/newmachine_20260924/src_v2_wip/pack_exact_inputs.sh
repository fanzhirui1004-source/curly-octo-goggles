#!/bin/bash
# pack_exact_inputs.sh -- run on the GPU server. Packs the inputs of exact_check_cpu.py for chosen design iterations of
# opt_design.py runs into ONE tar (one design set) with a manifest. Read-only on the run data: it writes only under --out
# (the tar, its manifest, the spec record, and a temporary staging directory of symbolic links removed at the end).
#
# Usage: pack_exact_inputs.sh [--out DIR] [--name NAME] [--src DIR | --no-src] [--extra FILE]... [--no-md5] [--dry-run]
#                             [--allow-outside MOD[,MOD...]] <run_dir>:<iters> [<run_dir>:<iters> ...]
#   <iters>  comma list of iteration numbers, 'last' (last recorded k) and 'mid' (last // 2), as exact_check_cpu.py
#   --out    output directory (default /root/autodl-tmp/exact_pack); must not lie inside a run directory
#   --name   tar name without .tar (default exact_inputs_<date>_<time>)
#   --src    code directory packed as <NAME>/src_v2/*.py (default /root/autodl-tmp/OPL/src_v2, the code of the runs)
#   --extra  further files packed as <NAME>/extra/<basename> (e.g. exact_check_cpu.py, iparm_tuned.json); two extras
#            with the same basename but different contents are refused (DEST_CLASH)
#   --no-md5 manifest with sizes only (faster)
#   --dry-run  print the plan, the checks and the sizes, write nothing
#   --allow-outside  modules of the import closure that may resolve outside --src (see below) without failing
# Checks (all before anything is written, also with --dry-run): every case directory has the files teacher.Cell reads
#   (body: NODES, CELL_INDICES, dofs, GP_FACES, BOX_NODES, CUT_NODES .npy and PREP.json; packets: FRESH_CONTEXT.json,
#   SAMPLE.json), symbolic links in them are followed (a dangling one fails: DANGLING); the worker modules of
#   exact_check_cpu.py (teacher, box_encode, encode_r1, element_moments, ... see REQUIRED) are in --src; and the import
#   closure of exact_check_cpu.py and those modules (static, all import statements) resolves every module either in
#   --src, in the standard library, or in an installed distribution (site-packages or a pip --target directory) - a
#   loose module found elsewhere on sys.path would be missing on the CPU machine (IMPORT_OUTSIDE_SRC).
# Contents of the tar (paths relative to <NAME>/):
#   runs/<run>/history.jsonl, meta.json, layout.json (a copy of the layout file named in meta.json),
#   runs/<run>/body/GP_TEMPLATES_n*.npz, and for every case of the chosen iterations runs/<run>/body/<case>/ and
#   runs/<run>/packets/<case>/ (no <case>_portview directories, no logs); src_v2/*.py; extra/*;
#   MANIFEST.tsv (bytes, md5, path) and SPECS.json (spec -> iterations -> cases, source paths, layout, load settings).
# Unpacked, a run directory of the tar is a valid <run_dir> for exact_check_cpu.py (it reads <run>/layout.json first).
set -euo pipefail
OUT=/root/autodl-tmp/exact_pack; NAME=exact_inputs_$(date +%Y%m%d_%H%M%S); SRC=/root/autodl-tmp/OPL/src_v2
MD5=1; DRY=0; EXTRA=(); SPECS=(); ALLOW=
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --src) SRC=$2; shift 2 ;;
    --no-src) SRC=""; shift ;;
    --extra) EXTRA+=("$2"); shift 2 ;;
    --no-md5) MD5=0; shift ;;
    --dry-run) DRY=1; shift ;;
    --allow-outside) ALLOW=$2; shift 2 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
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
"$PYBIN" - "$PLAN" "$OUT" "$NAME" "$SRC" "$MD5" "$DRY" "$ALLOW" "${#EXTRA[@]}" ${EXTRA[@]+"${EXTRA[@]}"} "${SPECS[@]}" <<'PY'
import sys, os, json, re, hashlib, ast, importlib.util, importlib.metadata, sysconfig
from pathlib import Path
plan, out, name, src, md5, dry, allow, nextra = sys.argv[1:9]
rest = sys.argv[9:]
extra, specs = rest[:int(nextra)], rest[int(nextra):]
md5, dry = md5 == '1', dry == '1'
allow = {m for m in allow.split(',') if m}
entries, seen, rec = [], {}, dict(name=name, specs=[], runs={})
skipped_dirs = []
REQ_BODY = ('NODES.npy', 'CELL_INDICES.npy', 'dofs.npy', 'GP_FACES.npy', 'BOX_NODES.npy', 'CUT_NODES.npy', 'PREP.json')
REQ_PACKET = ('FRESH_CONTEXT.json', 'SAMPLE.json')
REQUIRED = ('teacher', 'box_encode', 'encode_r1', 'element_moments', 'element_polyref', 'polyref_torch_fast', 'surfaces',
            'lattice3', 'ops', 'fastidx', 'trainlib', 'models', 'diag_sens', 'make_T_cpu', 'bench_cpu2', 'pardiso_direct',
            'moments_ad', 'lat_multi', 'lat_precond', 't_lat_precond', 'opt_design', 'r1x3_common')

def add(dest, srcp):
    srcp = Path(srcp)
    if not srcp.is_file():
        raise SystemExit(f'MISSING {srcp}')
    real = str(srcp.resolve())
    if dest in seen:
        if seen[dest] != real:
            raise SystemExit(f'DEST_CLASH {dest}: {seen[dest]} and {real}')
        return
    seen[dest] = real; entries.append((dest, real, srcp.stat().st_size))

def add_dir(dest, d):
    d = Path(d)
    if not d.is_dir():
        raise SystemExit(f'MISSING_DIR {d}')
    for p in sorted(d.iterdir()):
        if p.is_file():                                   # symbolic links followed (resolve(); tar -h)
            add(f'{dest}/{p.name}', p)
        elif p.is_symlink():
            raise SystemExit(f'DANGLING {p}')
        elif p.is_dir():
            skipped_dirs.append(str(p))

def imports_of(path):
    """Top-level names of all absolute import statements of a file (also those inside functions)."""
    names = set()
    for n in ast.walk(ast.parse(Path(path).read_text(), str(path))):
        if isinstance(n, ast.Import):
            names |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names.add(n.module.split('.')[0])
    return names

try:
    DISTS = importlib.metadata.packages_distributions()   # top-level module -> distributions (Python >= 3.10)
except Exception:                                         # noqa: BLE001
    DISTS = {}

def where(m, srcdir):
    """Where module m resolves without --src and the current directory on sys.path."""
    if m in sys.builtin_module_names or m in getattr(sys, 'stdlib_module_names', ()):
        return 'stdlib', None
    keep = sys.path[:]
    sys.path[:] = [q for q in keep if q and Path(q).resolve() not in (srcdir, Path.cwd().resolve())]
    try:
        spec = importlib.util.find_spec(m)
    except Exception as e:                                # noqa: BLE001
        return 'not_found', repr(e)[:80]
    finally:
        sys.path[:] = keep
    if spec is None:
        return 'not_found', None
    o = spec.origin if spec.origin not in (None, 'namespace') else next(iter(spec.submodule_search_locations or []), None)
    if o in (None, 'built-in', 'frozen'):
        return 'stdlib', o
    o = Path(o).resolve()
    if 'site-packages' in o.parts or 'dist-packages' in o.parts or m in DISTS:
        return 'installed', str(o)
    std = [Path(sysconfig.get_paths()[k]).resolve() for k in ('stdlib', 'platstdlib')]
    if any(s_ == o or s_ in o.parents for s_ in std):
        return 'stdlib', str(o)
    return 'outside', str(o)

def import_check(srcdir, roots):
    """REQUIRED modules in srcdir, and the static import closure of roots (files) over srcdir."""
    srcdir = Path(srcdir).resolve()
    local = {p.stem: p for p in srcdir.glob('*.py')}
    miss = [m for m in REQUIRED if m not in local]
    if miss:
        raise SystemExit(f'MISSING_MODULES in {srcdir}: {miss}')
    done, todo, ext = set(), [Path(r) for r in roots] + [local[m] for m in REQUIRED], {}
    while todo:
        f = todo.pop()
        if str(f) in done:
            continue
        done.add(str(f))
        for m in imports_of(f):
            if m in local:
                todo.append(local[m])
            else:
                ext.setdefault(m, set()).add(f.stem)
    kinds = {}
    for m in sorted(ext):
        k_, o = where(m, srcdir)
        kinds.setdefault(k_, {})[m] = dict(origin=o, imported_by=sorted(ext[m])[:4])
    bad = {m: v for m, v in kinds.get('outside', {}).items() if m not in allow}
    r = dict(local_modules=len(done), installed=sorted(kinds.get('installed', {})), not_found=kinds.get('not_found', {}),
             outside=kinds.get('outside', {}), allowed_outside=sorted(allow))
    print(f'IMPORTS closure: {len(done)} files in --src; installed packages needed/optional: {r["installed"]}')
    if r['not_found']:
        print(f'IMPORTS not found here (optional or GPU-only imports?): '
              f'{ {m: v["imported_by"] for m, v in r["not_found"].items()} }')
    if bad:
        raise SystemExit(f'IMPORT_OUTSIDE_SRC {json.dumps(bad)} (copy them into --src, or --allow-outside after checking '
                         f'that the CPU route does not need them)')
    return r

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
            miss = [f'body/{c}/{f}' for f in REQ_BODY if f'{base}/body/{c}/{f}' not in seen]
            miss += [f'packets/{c}/{f}' for f in REQ_PACKET if f'{base}/packets/{c}/{f}' not in seen]
            if miss:
                raise SystemExit(f'INCOMPLETE {rn}:{k} {miss}')
        r_['iterations'][str(k)] = dict(cases=cases, C_nice=hist[k]['C'], new_files=len(entries) - n0,
                                        new_bytes=sum(e[2] for e in entries[n0:]))
    rec['specs'].append(dict(spec=s, run=rn, iterations=iters))
if src:
    for p in sorted(Path(src).glob('*.py')):
        add(f'{name}/src_v2/{p.name}', p)
for e in extra:
    add(f'{name}/extra/{Path(e).name}', e)
if src:
    roots = [e for e in extra if Path(e).name == 'exact_check_cpu.py'] or \
        [str(p) for p in [Path(src) / 'exact_check_cpu.py'] if p.exists()]
    if not roots:
        print('WARNING exact_check_cpu.py neither in --extra nor in --src: import closure checked from the worker modules only')
    rec['imports'] = import_check(src, roots)
else:
    print('WARNING --no-src: worker modules and imports not checked')
if skipped_dirs:
    print(f'WARNING {len(skipped_dirs)} subdirectories of case directories not packed, e.g. {skipped_dirs[:3]}')
rec['skipped_dirs'] = skipped_dirs
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
