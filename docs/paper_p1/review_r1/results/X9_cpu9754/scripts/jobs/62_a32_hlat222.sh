# ST12b: route (a) with 32 threads (one run per lattice, as chain_r1_cpu.sh)
. /root/autodl-tmp/cpu9754/jobenv.sh
MKL_NUM_THREADS=32 OMP_NUM_THREADS=32 $PY -u lat_direct_cpu2.py $H/lat2t32_hlat222_chol.json $S4/hlat222.json --mtypes 2 --iparm-file $IPT --margin-gib 4
