#!/bin/bash
# tests23.sh: where the time goes now (src_v3 defaults): kernel-level profile of S_hat (B = 1, 16), the front end per step
# (t_prep.py, five plate cells), the sensitivity phase (t_sens.py); then the PCG iteration count of hlat222 with the other
# coarse spaces / two-level forms lat_precond already has (current: bnn:kpp:q1r).
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SPEC=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh kprof t_kprof.py $SPEC
bash $S/tst.sh prep3 t_prep.py $W/plate/body_S24 plateS24_040_o000,plateS24_060_o000,plateS24_150_o000,plateS24_050_o000,plateS24_070_o000
bash $S/tst.sh sens3 t_sens.py $SPEC
for P in bnn:kpp:q1a defl:kpp:q1r defl:kpp:q1a bnn:kpp:pu12 add:kpp:q1r; do
  TMO=900 EXTRA_ARGS="--prec $P" bash $S/t5.sh pc_${P//:/_} hlat222
done
echo "$(date +%F_%T) TESTS23_DONE" >> $W/status.txt
