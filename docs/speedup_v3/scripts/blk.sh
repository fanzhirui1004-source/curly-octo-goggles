#!/bin/bash
W=/root/autodl-tmp/OPL/V3W; C=/root/autodl-tmp/OPL/V3W/plate/body_S24:plateS24_040_o000
for b in 512 256 128 64 32; do
  SHAT_ENV="FUSED_BLK=$b" bash $W/scripts/shat.sh blk_$b $C --B 1,3,16 --rep 10 --save $W/shat/blk_$b.npz --ref $W/shat/blk_512.npz
done
echo "$(date +%F_%T) BLK_DONE" >> $W/status.txt
