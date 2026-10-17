#!/bin/bash
# chain.sh <testsNN.sh> [token]: run one test chain at a time on the machine (flock on V3W/tests/chain.lock: a second launch,
# e.g. after a lost reply, exits at once with a CHAIN_BUSY line); "START_CHAIN <name>" / "END_CHAIN <name>" in status.txt.
W=/root/autodl-tmp/OPL/V3W; S=$W/scripts
exec 9>$W/tests/chain.lock
if ! flock -n 9; then echo "$(date +%F_%T) CHAIN_BUSY $1 $2" >> $W/status.txt; exit 0; fi
echo "$(date +%F_%T) START_CHAIN $1 $2" >> $W/status.txt
bash $S/$1
echo "$(date +%F_%T) END_CHAIN $1 $2" >> $W/status.txt
