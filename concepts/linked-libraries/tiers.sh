#!/usr/bin/env bash
# The cheap tier's remaining sizes and the mid tier, each under a spend ceiling.
set -uo pipefail
cd "$(dirname "$0")"
OTHER=4.553
for size in S M; do
  ./capped_run.sh 46 $OTHER --run main-luna --model openai/gpt-6-luna --size $size --concurrency 11
done
./capped_run.sh 49 $OTHER --run mid-sol --model openai/gpt-6.1-sol --size L --seeds 2 3 --concurrency 11
echo "tiers done"
