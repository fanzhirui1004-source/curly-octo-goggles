# Table 5 route (b), 16 threads: lat_cond_cpu.py unchanged (block solver, command of chain_r1_cpu.sh)
. /root/autodl-tmp/cpu9754/jobenv.sh
$PY -u lat_cond_cpu.py $HC/latcond_hlat222_rep1.json $S4/hlat222.json --solver block --iparm-file $IPT --margin-gib 4 --ref $H/lat2_hlat222_chol_rep1.json
