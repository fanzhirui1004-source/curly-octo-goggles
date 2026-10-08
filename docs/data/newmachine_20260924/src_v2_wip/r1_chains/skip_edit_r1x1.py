import os, hashlib, re, sys
R = '/root/autodl-tmp/OPL/S1/V2/R1'; ST = '/root/autodl-tmp/OPL/S1/V2/chain_p1.status'
status = open(ST).read()
jobs = [('chain_r1_host.sh', 'R1HOST', 'st "SKIPPED (moved to the dedicated CPU machine; the EXIT trap writes the completion line)"; echo "$(date +%T) R1_REF64_DONE_20260928" >> $ST; exit 0\n'),
        ('chain_r1_hostcond.sh', 'R1COND', 'st "SKIPPED (moved to the dedicated CPU machine; the EXIT trap writes the completion line)"; exit 0\n'),
        ('chain_r1_host1.sh', 'R1ONE', 'st "SKIPPED (moved to the dedicated CPU machine; the EXIT trap writes the completion line)"; exit 0\n')]
dry = len(sys.argv) > 1 and sys.argv[1] == 'dry'
for fn, tag, ins in jobs:
    p = f'{R}/{fn}'
    started = re.search(rf'^\d\d:\d\d:\d\d {tag} (START|QUIET|NOT_QUIET)', status, re.M)
    b = open(p, 'rb').read()
    anchor = b'\nst "START pid=$$"\n'
    assert b.count(anchor) == 1, fn
    pos = b.index(anchor) + 1                                  # insert at the start of the START line
    ino0, md0 = os.stat(p).st_ino, hashlib.md5(b).hexdigest()
    print(fn, 'started' if started else 'waiting', 'inode', ino0, 'md5', md0, 'insert_at', pos, 'of', len(b))
    if started or dry or ins.encode() in b:
        continue
    with open(p, 'r+b') as f:                                   # same inode; bytes before pos untouched
        f.seek(pos); f.write(ins.encode() + b[pos:]); f.truncate()
    b2 = open(p, 'rb').read()
    assert b2[:pos] == b[:pos] and b2[pos:pos + len(ins)] == ins.encode() and b2[pos + len(ins):] == b[pos:]
    print('  edited: inode', os.stat(p).st_ino, 'md5', hashlib.md5(b2).hexdigest(), 'prefix identical')
