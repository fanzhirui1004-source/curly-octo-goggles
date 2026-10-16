. /root/autodl-tmp/cpu9754/jobenv.sh
$PY -u bench_cpu2.py $H/bench_cpu2_tune.json fresh_train_0020_full --mode tune --reps 3 --iparm-file $IPT
test -s $IPT
