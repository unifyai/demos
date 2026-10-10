#!/usr/bin/env bash
# One seed of an era run at size S, with the spend watchdog, then judging.
#   ./era_seed.sh <run> <model> <seed> <other-ledger spend> <ceiling> [extra run.py args...]
# Seeds run one at a time so that a budget stop never leaves a seed half done.
set -euo pipefail
cd "$(dirname "$0")"
run="$1"; model="$2"; seed="$3"; other="$4"; ceiling="$5"; shift 5
K=../../bin/with-openrouter-key
$K uv run python run.py --run "$run" --model "$model" --size S --seeds "$seed" --max-tokens 32000 "$@" &
sleep 5
./watchdog.sh "$run" "$other" "$ceiling"
wait || true
$K uv run python judge.py --run "$run"
echo "$run seed $seed done"
