"""Print a stratified sample of judged replies, for checking the judge by hand.

    uv run python scripts/judge_sample.py [per_label] [seed]

For each judge label, prints `per_label` replies (default 6) with the question, the two candidate
answers and the reply, but not the arm. The hand check is recorded in the README.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src"))

from judge import condition_files, key, question_of  # noqa: E402
from conversation_context.grade import pattern_label  # noqa: E402
from conversation_context.scenarios import BY_ID  # noqa: E402


def main() -> None:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    judged = {}
    for line in (HERE / "runs" / "judge.jsonl").read_text().splitlines():
        j = json.loads(line)
        if j.get("label"):
            judged[j["key"]] = j["label"]
    by_label = defaultdict(list)
    for p in condition_files():
        if not p.stem.startswith(("G1", "G2", "G3")):
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if pattern_label(r["scenario"], r["final"]) in ("both", "neither") and key(r) in judged:
                by_label[judged[key(r)]].append(r)
    for lab, rs in sorted(by_label.items()):
        print(f"\n######## {lab}: {len(rs)} replies, showing {min(per, len(rs))}")
        for r in rng.sample(rs, min(per, len(rs))):
            s = BY_ID[r["scenario"]]
            print(f"\n--- {r['item']} (Maya {'there' if r['present'] else 'away'})  Q: {question_of(r)}")
            print(f"    A (target): {s['target_answer']}   B: {s['distractor_answer']}")
            print("    " + r["final"].strip().replace("\n", "\n    "))


if __name__ == "__main__":
    main()
