#!/bin/bash
# rr.sh <script> [timeout]: c3n.sh with retries on connection errors (output only from the successful attempt)
for i in 1 2 3 4 5; do
  out=$(timeout ${2:-100} bash "$(dirname "$0")/c3n.sh" "$1" 2>&1); rc=$?
  if ! echo "$out" | grep -q "urlopen error\|ConnectionResetError\|SSLEOFError\|ws_closed"; then echo "$out"; exit $rc; fi
  sleep $((2**i))
done
echo "$out" | tail -3; exit 1
