# Common prologue of the campaign jobs (sourced). Results under R9754 (layout of the earlier R1/cpu tree).
. /root/autodl-tmp/cpu9754/env_cpu9754.sh
B=/root/autodl-tmp/OPL/S1/V2/R9754; H=$B/host; H32=$B/host32; H1=$B/host1; HC=$B/hostcond; IT=$B/itercmp
S4=/root/autodl-tmp/OPL/S4; SRC=/root/autodl-tmp/OPL/src_v2; IPT=$H/iparm_tuned.json; X=/root/autodl-tmp/cpu9754
G="fresh_train_0020_cover01_r2,fresh_train_0020_cover01_r1,fresh_train_0020_full,fresh_train_0007_full"
mkdir -p $H $H32 $H1 $HC $IT
cd $SRC
export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16
set -o pipefail
