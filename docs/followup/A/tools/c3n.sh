#!/bin/bash
# c3n.sh <script.sh>: run a bash script on the AutoDL GPU machine through its Jupyter kernel.
# Needs JHOST and JTOK in the environment (never write them into the repository), HTTPS_PROXY (set in Claude Code cloud
# sessions) and the Python package websocket-client.  Use JRUN_NEW_KERNEL=1 to avoid a stuck or busy kernel.
: "${JHOST:?set JHOST}"; : "${JTOK:?set JTOK}"
B=$(base64 -w0 "$1")
R="/root/_c3n_$$_$RANDOM.sh"
python3 "$(dirname "$0")/jrun3.py" "echo $B | base64 -d > $R; bash $R; rc=\$?; rm -f $R; exit \$rc"
