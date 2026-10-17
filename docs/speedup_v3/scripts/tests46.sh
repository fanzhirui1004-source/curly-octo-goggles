#!/bin/bash
# tests46.sh: t_gk.py again (B = 1..32 with P15b gkw); 51-cell plate with all current defaults (P13-P17, P15b) and the
# resident budget of the ST20 runs (4) and 0.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts; SP=$W/plate/body_S24:plateS24_040_o000
bash $S/tst.sh gk3 t_gk.py $SP
bash $S/plate.sh v3AA 51
RGB=0 bash $S/plate.sh v3AA_rg0 51
echo "$(date +%F_%T) TESTS46_DONE" >> $W/status.txt
