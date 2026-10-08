#!/bin/bash
# puller.sh (main machine): verify and unpack shipped cells from the transit dir into S3, ACK, delete the transit copy.
T=/autodl-fs/data/OPL_TRANSIT; S=/root/autodl-tmp/OPL/S3; Q=/autodl-fs/data/OPL_QUEUE
mkdir -p $S/status $S/incoming $T/ACK
while [ ! -f $S/PULLER_STOP ]; do
  for m in $(ls $T/*.md5 2>/dev/null); do
    c=$(basename $m .md5); [ -f $T/$c.tar ] || continue
    if cp $T/$c.tar $S/incoming/$c.tar && [ "$(md5sum < $S/incoming/$c.tar | cut -c1-32)" = "$(cat $m)" ]; then
      (cd $S && tar xf incoming/$c.tar) && cp $S/logs/$c.status.json $S/status/ 2>/dev/null
      rm -f $S/incoming/$c.tar; touch $T/ACK/$c; rm -f $T/$c.tar $m
      echo "$(date +%F_%T) PULLED $c" >> $S/puller.log
    else
      rm -f $S/incoming/$c.tar; echo "$(date +%F_%T) BAD_MD5 $c" >> $S/puller.log
    fi
  done
  sleep 30
done
