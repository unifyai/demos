#!/usr/bin/env bash
# Every run behind the results, in order, then judging and the analysis.
# Resumable: re-running skips finished chains and tasks. Run from this folder;
# on the cloud machine: nohup ./run_all.sh > runs/run_all.log 2>&1 &
# OTHER is the spend recorded in ledgers other than runs/ledger.jsonl (the
# laptop's), so that the top tier's spend ceiling sees the whole concept.
set -euo pipefail
cd "$(dirname "$0")"
K=../../bin/with-openrouter-key
OTHER=${OTHER:-$(python3 -c "import json,os;f='runs/ledger-laptop.jsonl';print(sum((json.loads(l).get('cost') or 0) for l in open(f)) if os.path.exists(f) else 0)")}
# cheap tier: the full design, both families
for size in S M L; do
  $K uv run python run.py --run main-haiku --model anthropic/claude-haiku-5.5 --size "$size"
  $K uv run python run.py --run main-luna --model openai/gpt-6-luna --size "$size"
done
$K uv run python judge.py --run main-haiku
$K uv run python judge.py --run main-luna
# mid tier: the full design at L
$K uv run python run.py --run mid-sol --model openai/gpt-6.1-sol --size L --seeds 1 2 3
$K uv run python judge.py --run mid-sol
# top tier: one seed, in passing, the five arms of the headline comparison, 60 of 72 tasks
./capped_run.sh 78 "$OTHER" --run top-opus --model anthropic/claude-opus-5.5 --reasoning-effort high --max-tokens 16000 \
  --size L --seeds 1 --conditions passing --arms pasted rule-skills rules-file long-context linked \
  --tasks $(uv run python -m linked_libraries.generate --subset L 60) --concurrency 12
uv run python analyze.py --run main-haiku --run main-luna --run mid-sol --run top-opus \
  --ledger runs/ledger.jsonl --ledger runs/ledger-laptop.jsonl --out results.json --tables > results.md
uv run python estimate_full_size.py --run top-opus
echo "all done"
