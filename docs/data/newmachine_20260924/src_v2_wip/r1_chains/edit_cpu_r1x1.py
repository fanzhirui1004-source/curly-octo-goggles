import os, hashlib, subprocess, sys
p = '/root/autodl-tmp/OPL/S1/V2/R1/chain_r1_cpu.sh'
b = open(p, 'rb').read()
bs = b.index(b'# ---------------- B: E7 retries')
be = b.index(b'mk R1_CPU_REF_DONE_20260928\n') + len(b'mk R1_CPU_REF_DONE_20260928\n')
newB = (b'# ---------------- B: E7 retries -- MOVED to the third CPU machine (chain_r1_cpu3.sh; in-place edit 2026-09-29)\n'
        b'st "stage B (E7 retries) moved to the third CPU machine (see chain_cpu3.status there)"\n'
        b'mk R1_CPU_REF_DONE_20260928\n')
fs = b.index(b'if [ -n "$SFB" ] && ! grep')
fe = b.index(b'\nfi\n', fs) + len(b'\nfi\n')
newF = b'st "SuperLU 2-cell fallback moved to the third CPU machine (in-place edit 2026-09-29)"\n'
b2 = b[:bs] + newB + b[be:fs] + newF + b[fe:]
pid = subprocess.run(['pgrep', '-f', 'bash /root/autodl-tmp/OPL/S1/V2/R1/chain_r1_cpu.sh'], capture_output=True, text=True).stdout.split()[0]
pos = int([l for l in open(f'/proc/{pid}/fdinfo/255') if l.startswith('pos')][0].split()[1])
print('pid', pid, 'read offset', pos, 'edit from', bs, 'inode', os.stat(p).st_ino, 'md5', hashlib.md5(b).hexdigest())
if pos >= bs or (len(sys.argv) > 1 and sys.argv[1] == 'dry'):
    print('NOT EDITED'); sys.exit(0)
with open(p, 'r+b') as f:
    f.seek(bs); f.write(b2[bs:]); f.truncate()
b3 = open(p, 'rb').read()
assert b3 == b2 and b3[:bs] == b[:bs]
print('edited: inode', os.stat(p).st_ino, 'md5', hashlib.md5(b3).hexdigest(), 'prefix identical; offset now', [l for l in open(f'/proc/{pid}/fdinfo/255') if l.startswith('pos')][0].strip())
