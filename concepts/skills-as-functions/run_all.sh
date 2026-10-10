#!/usr/bin/env bash
# Every experiment run behind the README's results, in the order they were run.
# On the cloud machine: cd ~/demos/concepts/skills-as-functions && nohup ./run_all.sh <batch> > runs/<batch>.log 2>&1 &
# Concurrency adds up to 12 or less across the processes of a batch (eight concepts share one key).
set -euo pipefail
cd "$(dirname "$0")"
UV="${UV:-$HOME/.local/bin/uv}"
KEY=../../bin/with-openrouter-key
LEDGER="${LEDGER:-runs/ledger-vm.jsonl}"
run() { $KEY "$UV" run python run.py --label main --ledger "$LEDGER" "$@"; }
run_t() { $KEY "$UV" run python run.py --label tiers --ledger "$LEDGER" "$@"; }

case "${1:-}" in
  # today's cheap tier (run first, before the lead's amendment)
  haiku)      # every size, seeds 1-3
    run --size realistic --seeds 1,2,3 --concurrency 7 &
    run --size medium --seeds 1,2,3 --concurrency 3 &
    run --size small --seeds 1,2,3 --concurrency 2 &
    wait ;;
  haiku-notes) # the notes arm again, after its fix (fault 6)
    run --size realistic --arms notes --seeds 1,2,3 --concurrency 3 &
    run --size medium --arms notes --seeds 1,2,3 --concurrency 2 &
    run --size small --arms notes --seeds 1,2,3 --concurrency 1 &
    wait ;;
  luna)       # second family of the cheap tier, every size (after fault 7)
    run --size realistic --seeds 1,2,3 --model openai/gpt-6-luna --concurrency 4 &
    run --size medium --seeds 1,2,3 --model openai/gpt-6-luna --concurrency 2 &
    run --size small --seeds 1,2,3 --model openai/gpt-6-luna --concurrency 1 &
    wait ;;
  # amendment 2: model strength as an axis, one family (Anthropic), the medium size,
  # tool_choice "auto" (sonnet-5.5 and opus-5.5 reject a forced tool call), label "tiers"
  tier-mid)
    run_t --size medium --seeds 1,2,3 --model anthropic/claude-sonnet-5.5 --tool-choice auto --concurrency 5 ;;
  tier-top)   # first launched with seeds 1,2 at concurrency 3; restarted at 18:33 PDT with seeds 1-3 (README)
    run_t --size medium --seeds 1,2,3 --model anthropic/claude-opus-5.5 --tool-choice auto --reasoning high --extra-max-tokens 12000 --concurrency 8 ;;
  tier-cheap-auto)  # the cheap model under the same harness, as a control
    run_t --size medium --seeds 1,2,3 --model anthropic/claude-haiku-5.5 --tool-choice auto --concurrency 4 ;;
  *) echo "usage: $0 haiku|haiku-notes|luna|tier-mid|tier-top|tier-cheap-auto" >&2; exit 2 ;;
esac
