#!/usr/bin/env bash
# Every run of the experiment, in the plan's order. Each step resumes where it
# stopped. On the cloud machine pass LEDGER=runs/ledger-vm.jsonl.
#   ../../bin/with-openrouter-key bash run_all.sh            (all steps)
#   STEPS="1 2" ../../bin/with-openrouter-key bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
LEDGER=${LEDGER:-runs/ledger.jsonl}
STEPS=${STEPS:-"0 1 2 9 10 11 5"}   # as run: 0 1 2 (mid tier), then 9 10 11 (Plan amendment 4), then 5 (analysis)
ALL=relay,slow_speaks,slow_only,fast_only,fast_context,slow_filler,fast_only_think,fast_context_think
ERA=relay,slow_speaks,slow_only,fast_only,fast_context
SLOWARMS=relay,slow_speaks,slow_only,slow_filler
RUN="uv run python run.py --ledger $LEDGER --parallel 24 --concurrency 12"
for step in $STEPS; do
  case $step in
    0) $RUN --sanity --scripts 2 --seeds 0 --arms $ALL ;;                              # sanity: every arm must pass
    1) $RUN --pair anthropic --size realistic --delay 0 --scripts 16 --seeds 0,1,2 --arms $ALL
       $RUN --pair anthropic --size small --delay 0 --scripts 16 --seeds 0,1,2 --arms $ALL ;;
    2) $RUN --pair anthropic --size realistic --delay 8 --scripts 16 --seeds 0,1,2 --arms $SLOWARMS ;;
    # D = 4 was planned in step 2 and stopped as it began (Plan amendment 3); not part of the results
    12) $RUN --pair anthropic --size realistic --delay 4 --scripts 16 --seeds 0,1,2 --arms $SLOWARMS ;;
    3) $RUN --pair openai --size realistic --delay 0 --scripts 16 --seeds 0 --arms $ALL ;;
    4) uv run python judge_run.py --ledger $LEDGER runs/anthropic-*.jsonl runs/openai-*.jsonl ;;
    5) uv run python analyze.py ;;
    # Plan amendment 4: today's fast tier with cheap, mid (steps 1 and 2) and top slow tiers
    9) $RUN --pair cheap-anthropic --size realistic --delay 0 --scripts 16 --seeds 0,1,2 --arms relay,slow_speaks
       $RUN --pair cheap-openai --size realistic --delay 0 --scripts 16 --seeds 0,1,2 --arms relay,slow_speaks,fast_only,fast_context
       $RUN --pair top-anthropic --size realistic --delay 0 --scripts 16 --seeds 0 --arms relay,slow_speaks,slow_only ;;
    10) uv run python judge_run.py --ledger $LEDGER runs/anthropic-realistic-d0.jsonl runs/anthropic-realistic-d8.jsonl
        uv run python judge_run.py --ledger $LEDGER --seeds 0 runs/anthropic-small-d0.jsonl ;;
    11) uv run python judge_run.py --ledger $LEDGER runs/cheap-anthropic-realistic-d0.jsonl runs/cheap-openai-realistic-d0.jsonl \
          runs/top-anthropic-realistic-d0.jsonl ;;
    # Plan amendment 3 (superseded by amendment 4; its runs are in runs/excluded-era/)
    6) $RUN --pair feb --sanity --scripts 2 --seeds 0 --arms $ERA,slow_filler
       $RUN --pair summer --sanity --scripts 2 --seeds 0 --arms $ERA,slow_filler
       $RUN --pair feb --size realistic --delay 0 --scripts 16 --seeds 0 --arms $ERA
       $RUN --pair summer --size realistic --delay 0 --scripts 16 --seeds 0 --arms $ERA ;;
    7) uv run python judge_run.py --ledger $LEDGER runs/anthropic-realistic-d0.jsonl runs/anthropic-realistic-d8.jsonl \
         runs/anthropic-small-d0.jsonl runs/feb-realistic-d0.jsonl runs/summer-realistic-d0.jsonl ;;
    8) $RUN --pair frontier --size realistic --delay 0 --scripts 16 --seeds 0 --arms relay,slow_speaks,slow_only
       uv run python judge_run.py --ledger $LEDGER runs/frontier-realistic-d0.jsonl ;;
  esac
done
