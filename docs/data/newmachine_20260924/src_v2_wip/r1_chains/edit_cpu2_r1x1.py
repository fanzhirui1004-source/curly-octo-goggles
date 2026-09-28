import os, hashlib, subprocess, sys
p = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1] == 'dry' else '/root/autodl-tmp/OPL/S1/V2/R1/chain_r1_cpu.sh'
live = p.startswith('/root/autodl-tmp')
R = [(b'for L in $LATS; do\n  REF=$H/lat2_${L}_chol_rep1.json',
      b'for L in hlat222 hlat331; do   # hlat221a/b moved to the 60 GiB machine (in-place edit 2026-09-29)\n  REF=$H/lat2_${L}_chol_rep1.json'),
     (b'PL=hlat221b; [ "$SMOKE" = 1 ] && PL=hlatsmoke1\n',
      b'st "route (b) PARDISO-path run on hlat221b moved to the 60 GiB machine (in-place edit 2026-09-29)"\n'),
     (b'run $HC latcond_pardiso_${PL}_rep1 $PY -u lat_cond_cpu.py $HC/latcond_pardiso_${PL}_rep1.json $(lay $PL) --solver pardiso --iparm-file $H/iparm_tuned.json --margin-gib 4 --ref $H/lat2_${PL}_chol_rep1.json\n', b''),
     (b'for L in $LATS; do\n  run $H1 lat1_${L}_chol',
      b'for L in hlat222 hlat331; do   # hlat221a/b moved to the 60 GiB machine (in-place edit 2026-09-29)\n  run $H1 lat1_${L}_chol')]
b = open(p, 'rb').read()
first = min(b.index(o) for o, _ in R)
b2 = b
for o, n in R:
    assert b2.count(o) == 1, o[:40]; b2 = b2.replace(o, n)
if not live:
    open(p, 'wb').write(b2); print('mirror', hashlib.md5(b2).hexdigest()); sys.exit(0)
pid = subprocess.run(['pgrep', '-f', 'bash /root/autodl-tmp/OPL/S1/V2/R1/chain_r1_cpu.sh'], capture_output=True, text=True).stdout.split()[0]
pos = int([l for l in open(f'/proc/{pid}/fdinfo/255') if l.startswith('pos')][0].split()[1])
print('pid', pid, 'offset', pos, 'first change', first, 'inode', os.stat(p).st_ino, 'md5', hashlib.md5(b).hexdigest())
if pos >= first:
    print('NOT EDITED'); sys.exit(1)
with open(p, 'r+b') as f:
    f.seek(first); f.write(b2[first:]); f.truncate()
b3 = open(p, 'rb').read(); assert b3 == b2 and b3[:first] == b[:first]
print('edited: inode', os.stat(p).st_ino, 'md5', hashlib.md5(b3).hexdigest(), 'prefix identical')
