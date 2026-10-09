"""Summarise a ledger: python -m demos_common.ledger runs/ledger.jsonl [more.jsonl ...]"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def summarise(paths: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = f"{r.get('tag')} | {r.get('model')}"
            d = out[key]
            if "error" in r:
                d["errors"] += 1
                continue
            d["calls"] += 1
            d["cost"] += r.get("cost") or 0.0
            d["prompt_tokens"] += r.get("prompt_tokens") or 0
            d["completion_tokens"] += r.get("completion_tokens") or 0
    return {k: dict(v) for k, v in out.items()}


def main() -> None:
    s = summarise(sys.argv[1:])
    total = 0.0
    for k, d in sorted(s.items()):
        total += d.get("cost", 0.0)
        print(f"{k}: calls {int(d.get('calls', 0))}, errors {int(d.get('errors', 0))}, "
              f"prompt {int(d.get('prompt_tokens', 0)):,}, completion {int(d.get('completion_tokens', 0)):,}, cost ${d.get('cost', 0.0):.4f}")
    print(f"total ${total:.4f}")


if __name__ == "__main__":
    main()
