#!/usr/bin/env bash
# Stop one run (by PID) if the concept's total spend reaches a ceiling.
#   ./watchdog.sh <run name> <spend already in other ledgers> <ceiling in dollars>
# Used by capped_run.sh so the concept's hard stop ($80) holds.
set -uo pipefail
cd "$(dirname "$0")"
run="$1"; other="$2"; ceiling="$3"
while ps -eo pid,args | grep -q "[r]un.py --run $run "; do
  total=$(python3 -c "import json,sys; print(sum((json.loads(l).get('cost') or 0) for l in open('runs/ledger.jsonl')) + float(sys.argv[1]))" "$other")
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$total" "$ceiling"; then
    for pid in $(ps -eo pid,args | grep "[r]un.py --run $run " | awk '{print $1}'); do kill "$pid"; done
    echo "watchdog: stopped $run at total spend \$$total (ceiling \$$ceiling)"
    exit 0
  fi
  sleep 15
done
echo "watchdog: $run finished; total spend \$$total"
