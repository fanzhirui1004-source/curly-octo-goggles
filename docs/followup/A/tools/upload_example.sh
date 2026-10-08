#!/bin/bash
# Pattern used to upload a local file to the server: embed it base64 in a script that c3n.sh runs remotely.
# usage (local): bash upload_example.sh <local_file> <remote_path> > _up.sh && JRUN_NEW_KERNEL=1 bash c3n.sh _up.sh
B=$(base64 -w0 "$1"); M=$(md5sum "$1" | cut -c1-32)
cat <<EOS
set -e
mkdir -p "\$(dirname $2)"
[ -f "$2" ] && cp "$2" "$2.bak_\$(md5sum "$2" | cut -c1-8)"
echo '$B' | base64 -d > "$2"
echo "uploaded \$(md5sum "$2" | cut -c1-32) (expected $M)"
EOS
