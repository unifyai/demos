#!/usr/bin/env bash
# Run grids in order, one process at a time (max_concurrency 12), each followed by the judge. A grid
# starts only if the ledger plus its projected cost stays under $GATE (default $76).
#   scripts/run_grids.sh part1    # plan revision 1, as first run (G1 ran; stopped by amendment 1)
#   scripts/run_grids.sh amend2   # plan amendment 2: the cheap tier, then the mid and top tiers
# On the cloud machine: cd ~/demos/concepts/conversation-context && nohup scripts/run_grids.sh part1 > runs/part1.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
K=../../bin/with-openrouter-key

spent() {
  uv run python -c "from demos_common.ledger import summarise; print(round(sum(v.get('cost', 0.0) for v in summarise(['runs/ledger.jsonl']).values()), 2))"
}

grid() {  # grid <projected dollars> <run.py arguments...>
  local projected=$1; shift
  local s; s=$(spent)
  if ! python3 -c "import sys; sys.exit(0 if $s + $projected < ${GATE:-76} else 1)"; then
    echo "=== GATE $(date -u +%H:%M): spent \$$s + projected \$$projected >= \$${GATE:-76}; not starting: $*"
    return 1
  fi
  echo "=== START $(date -u +%H:%M) spent \$$s: $*"
  $K uv run python run.py "$@" >> runs/grids.log 2>&1
  $K uv run python judge.py >> runs/grids.log 2>&1
  echo "=== DONE $(date -u +%H:%M) spent \$$(spent): $*"
}

ODD=s01,s03,s05,s07,s09,s11,s13,s15,s17,s19,s21,s23

LUNA=openai/gpt-6-luna
SONNET=anthropic/claude-sonnet-5.5
OPUS=anthropic/claude-opus-5.5
MID8=s01,s04,s07,s10,s13,s16,s19,s22

fits() {  # fits <projected dollars>: the ledger plus it stays under $76 (amendment 2)
  local s; s=$(spent)
  python3 -c "import sys; sys.exit(0 if $s + $1 < 76 else 1)"
}

era() {  # dollars spent on the older models (runs/era_models_excluded/), outside the cap since the handover
  uv run python -c "import json; print(round(sum((r.get('cost') or 0.0) for r in map(json.loads, open('runs/ledger.jsonl')) if r.get('model') in ('openai/o4-mini', 'openai/gpt-4.1', 'anthropic/claude-sonnet-4')), 2))"
}

case "${1:-}" in
  part1)
    grid 10 --grid G1 --requests model --length long --depth 3 --arms alone,brief,summary,full,single || exit 0
    grid 13 --grid G2 --requests terse --length long --depth 3 --arms alone,summary,full,full_naive || exit 0
    grid 4 --grid G2 --requests terse --length short --depth 3 --arms alone,summary,full,full_naive || exit 0
    grid 1 --grid sanity --model openai/gpt-6-luna --requests model --length short --depth 3 --kinds complete \
      --seeds 0 --present all --arms alone,brief,summary,full || exit 0
    ;;
  amend2)  # plan amendment 2: cheap tier (haiku and luna workers, haiku caller), then the mid and top tiers
    grid 9 --grid G1 --requests model --length long --depth 3 --worker-model $LUNA --arms alone,brief,summary,full \
      --prefixes-from G1__claude-haiku-5.5__model__long__d3 || exit 0
    grid 13 --grid G2 --requests terse --length long --depth 3 --arms alone,summary,full,full_naive || exit 0
    grid 4 --grid G2 --requests terse --length long --depth 3 --worker-model $LUNA --arms alone,full || exit 0
    grid 1.5 --grid G2 --requests terse --length short --depth 3 --arms alone,full || exit 0
    grid 7 --grid G2 --requests terse --length short --depth 3 --worker-model $SONNET --arms alone,full \
      --kinds settled,open --scenarios $MID8 --seeds 0 || exit 0
    # top tier: drop whole scenarios from the end of its list if money is short
    for top in s01,s04,s13,s16:8.5 s01,s04,s13:6.5 s01,s04:4.3; do
      if fits "${top##*:}"; then
        grid "${top##*:}" --grid G2 --requests terse --length short --depth 3 --worker-model $OPUS --worker-reasoning high \
          --arms alone,full --kinds settled,open --scenarios "${top%%:*}" --seeds 0
        TOP="${top%%:*}"; break
      fi
    done
    # money left: a second seed for the mid tier, then for the top tier, each only whole
    if fits 7.5; then
      grid 7.5 --grid G2 --requests terse --length short --depth 3 --worker-model $SONNET --arms alone,full \
        --kinds settled,open --scenarios $MID8 --seeds 0,1
    fi
    if [ -n "${TOP:-}" ] && fits 8.5; then
      grid 8.5 --grid G2 --requests terse --length short --depth 3 --worker-model $OPUS --worker-reasoning high \
        --arms alone,full --kinds settled,open --scenarios "$TOP" --seeds 0,1
    fi
    ;;
  amend2-extra)  # after the handover (10 Oct, 02:54 UTC): the lead ruled that the older models' $5.85 is
    # outside the $80 cap, so the same $76 gate counts today's models only (ledger minus era()). The plan's
    # last step (the top tier's seed 1, skipped by amend2's gate), then the rest of the cheap-tier design
    # in amendment 2's order, each only whole: G2-short's summary arm, then its full_naive arm.
    GATE=$(python3 -c "print(76 + $(era))"); export GATE
    grid 8.5 --grid G2 --requests terse --length short --depth 3 --worker-model $OPUS --worker-reasoning high \
      --arms alone,full --kinds settled,open --scenarios s01,s04,s13,s16 --seeds 0,1 || exit 0
    grid 1.5 --grid G2 --requests terse --length short --depth 3 --arms summary || exit 0
    grid 1.0 --grid G2 --requests terse --length short --depth 3 --arms full_naive || exit 0
    ;;
  *)
    echo "usage: $0 part1|amend2|amend2-extra"; exit 2 ;;
esac
echo "=== ALL DONE $(date -u +%H:%M) spent \$$(spent)"
