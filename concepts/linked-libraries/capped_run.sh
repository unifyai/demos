#!/usr/bin/env bash
# Run run.py (and then judge.py) for one run, stopping it by PID if the
# concept's total spend reaches a ceiling, so the hard stop holds.
#   ./capped_run.sh <ceiling dollars> <spend in other ledgers> <run.py args...>
# The run name is read from the --run argument. Spend is runs/ledger.jsonl plus
# the given figure for ledgers kept elsewhere (the laptop's).
set -uo pipefail
cd "$(dirname "$0")"
ceiling="$1"; other="$2"; shift 2
run=""; prev=""
for a in "$@"; do [ "$prev" = "--run" ] && run="$a"; prev="$a"; done
[ -n "$run" ] || { echo "no --run given" >&2; exit 2; }
K=../../bin/with-openrouter-key
$K uv run python run.py "$@" &
sleep 5
./watchdog.sh "$run" "$other" "$ceiling"
wait
status=$?
$K uv run python judge.py --run "$run" --concurrency "${JUDGE_CONCURRENCY:-11}"
echo "capped_run: $run finished (run.py exit $status)"
