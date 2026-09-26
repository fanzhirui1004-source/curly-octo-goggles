source /root/autodl-tmp/OPL/S1/env_gpu.sh
export OPL_PACKETS_EXTRA=/root/autodl-tmp/OPL/S3/packets OPL_GP_CACHE=0
cd /root/autodl-tmp/OPL/src_v2
O=/root/autodl-tmp/OPL/S1/V2
run() { n=$1; shift; echo "$(date +%T) START $n" >> $O/chain_arms.status; "$@" > $O/$n.log 2>&1; echo "$(date +%T) END $n rc=$?" >> $O/chain_arms.status; }
while ! grep -q CHAIN_TAILGATE_DONE $O/chain_arms.status; do sleep 30; done
for cc in fresh_val_2002_d0_v0:x fresh_val_2002_d0_v0:y fresh_val_2004_d0_v2:x fresh_val_2004_d0_v2:y fresh_val_2006_d0_v1:y; do
  c=${cc%:*}; conf=${cc#*:}
  OPL_CONV_FP32=1 FUSED_HYPER=1 run gate_A2_tail8_${c}_$conf $PY -u lat_full.py $O/gate_A2_tail8_${c}_$conf.json --pairs '' --custom "$O/A2_tail8/best.pt:$c:$O/A2_tail8/best.pt:" --nb-mode explicit --sets test --configs $conf
done
for cc in fresh_val_2003_d1_v1:x fresh_val_2003_d1_v1:y fresh_val_2006_d0_v1:x fresh_val_2006_d0_v1:y; do
  c=${cc%:*}; conf=${cc#*:}
  [ -f $O/decomp_A2_tail8_${c}_$conf.json ] || OPL_CONV_FP32=1 FUSED_HYPER=1 run decomp_A2_tail8_${c}_$conf $PY -u gate_decomp.py $O/decomp_A2_tail8_${c}_$conf.json $O/A2_tail8/best.pt $c --configs $conf
done
echo CHAIN_TAILGATE2_DONE >> $O/chain_arms.status
