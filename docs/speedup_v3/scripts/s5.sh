#!/bin/bash
W=/root/autodl-tmp/OPL/V3W; C=/root/autodl-tmp/OPL/S4/body:hlat222_000
bash $W/scripts/shat.sh s5_store $C --B 1,3,16 --rep 5 --save $W/shat/s5_store.npz --profile
SHAT_ENV="OPL_V3_SLOTS=recompute" bash $W/scripts/shat.sh s5_rc $C --B 1,3,16 --rep 5 --ref $W/shat/s5_store.npz --profile
echo "$(date +%F_%T) S5T_DONE" >> $W/status.txt
