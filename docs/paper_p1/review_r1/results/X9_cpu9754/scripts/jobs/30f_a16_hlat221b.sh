# Table 5 route (a), 16 threads: lat_direct_cpu2.py unchanged (command of chain_r1_cpu.sh), no sensitivities
. /root/autodl-tmp/cpu9754/jobenv.sh
$PY -u lat_direct_cpu2.py $H/lat2_hlat221b_chol_rep1.json $S4/hlat221b.json --mtypes 2 --iparm-file $IPT --margin-gib 4
