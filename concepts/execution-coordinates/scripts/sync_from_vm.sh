#!/usr/bin/env bash
# Copy runs/ back from the cloud machine without overwriting the laptop's own
# files: the laptop's ledger is runs/ledger.jsonl and its discard index is
# runs/discarded/index.jsonl; the machine's go to ledger-vm*.jsonl (written
# there under those names) and runs/discarded/index-vm.jsonl.
set -euo pipefail
VM=ubuntu@34.67.145.107
DIR=demos/concepts/execution-coordinates/runs
HERE="$(cd "$(dirname "$0")/.." && pwd)"
rsync -az --exclude 'discarded/index.jsonl' --exclude 'ledger.jsonl' "$VM:$DIR/" "$HERE/runs/"
if ssh -n -o BatchMode=yes "$VM" "test -f $DIR/discarded/index.jsonl"; then
  rsync -az "$VM:$DIR/discarded/index.jsonl" "$HERE/runs/discarded/index-vm.jsonl"
fi
