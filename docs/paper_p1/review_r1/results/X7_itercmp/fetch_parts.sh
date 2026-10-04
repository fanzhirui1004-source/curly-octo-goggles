#!/bin/bash
# fetch_parts.sh <lattice> : on the AMD host, download parts_<lattice>/ from the bjb2 Jupyter file endpoint (4 in
# parallel, each part retried until its md5 matches), join them into sys_<lattice>/ and check the whole-file md5s.
L=$1; D=/root/autodl-tmp/ITERCMP; . $D/.src_env
U=https://$SH/jupyter/files/autodl-tmp/ITERCMP/parts_$L
P=$D/parts_$L; S=$D/sys_$L; mkdir -p $P $S; cd $P
curl -sS -f -H "Authorization: token $ST" -o manifest.txt $U/manifest.txt
curl -sS -f -H "Authorization: token $ST" -o whole.md5 $U/whole.md5
get() {  # get <part> <md5>
  for t in $(seq 1 20); do
    [ -f "$1" ] && [ "$(md5sum < "$1" | cut -d' ' -f1)" = "$2" ] && return 0
    curl -sS -f --max-time 900 -H "Authorization: token $ST" -o "$1" "$U/$1" || sleep 3
  done
  echo "FAILED $1"; return 1
}
export -f get; export U ST
awk '{print $1, $2}' manifest.txt | xargs -P 4 -n 2 bash -c 'get "$0" "$1"'
for f in $(awk '{print $3}' manifest.txt | uniq); do cat $(awk -v f=$f '$3==f {print $1}' manifest.txt) > $S/$f; done
cd $S && md5sum -c $P/whole.md5 && echo "FETCH_OK $L" || echo "FETCH_BAD $L"
