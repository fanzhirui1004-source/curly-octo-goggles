import os, hashlib, subprocess, sys
p = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/OPL/S1/V2/R1/chain_r1_cpu.sh'
live = p.startswith('/root/autodl-tmp')
R = [(b'  for k in $(seq 1 $NREP); do\n    run $H t6_${L}_chol_rep$k',
      b'  for k in 1; do   # single run (author decision 2026-09-29, in-place edit)\n    run $H t6_${L}_chol_rep$k'),
     (b'  for k in $(seq 1 $NREP); do\n    run $HC latcond_${L}_rep$k',
      b'  for k in 1; do   # single run (author decision 2026-09-29, in-place edit)\n    run $HC latcond_${L}_rep$k')]
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
print('text at offset:', b[max(0, pos - 120):pos + 60])
if pos >= first:
    print('NOT EDITED'); sys.exit(1)
with open(p, 'r+b') as f:
    f.seek(first); f.write(b2[first:]); f.truncate()
b3 = open(p, 'rb').read(); assert b3 == b2 and b3[:first] == b[:first]
print('edited: inode', os.stat(p).st_ino, 'md5', hashlib.md5(b3).hexdigest(), 'prefix identical')
