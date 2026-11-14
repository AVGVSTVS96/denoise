#!/bin/bash
# Runs a command and stops it while macOS still has disk to swap into: this Mac's disk is nearly full, and out of swap space it panics.
"$@" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  free=$(df -m ~ | awk 'NR==2 {print $4}')
  if [ "$free" -lt 8000 ]; then
    kill $pid; echo "guard: stopped $1 with ${free}M disk free, $(sysctl -n vm.swapusage)"; exit 3
  fi
  sleep 2
done
wait $pid
