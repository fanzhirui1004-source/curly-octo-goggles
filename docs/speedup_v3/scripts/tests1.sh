#!/bin/bash
# tests1.sh: chebfused checks (mock bitwise, plate cell), stream-concurrency probe, hyperedge kernel tiles (one GPU job each).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
B24=$W/plate/body_S24
bash $S/tst.sh cheb_mock t_cheb.py mock
bash $S/tst.sh cheb_cell t_cheb.py cell $B24:plateS24_040_o000
bash $S/tst.sh streams2 t_streams.py $B24 plateS24_040_o000,plateS24_030_o000
bash $S/tst.sh streams3 t_streams.py $B24 plateS24_040_o000,plateS24_030_o000,plateS24_020_o000
TST_TIMEOUT=3600 bash $S/tst.sh hyper t_hyper.py $B24:plateS24_040_o000
echo "$(date +%F_%T) TESTS1_DONE" >> $W/status.txt
