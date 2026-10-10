"""Add reviewed wrong answers to review.json: each wrong answer in the given
runs, with its automatic cause confirmed or overridden by hand (OVERRIDES).

    uv run python scripts/review_add.py runs/main --model claude-haiku-5.5
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from execution_coordinates.analysis import cause_of, reads_split  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
METHOD = ("For every wrong answer the cell that produced it was read, with the record of which request had set each "
          "name it read and the answers that leftover state would give; about a third were read in full (every "
          "cell and output). The cause is 'leak' when the answering cell used a value another client's request "
          "had set, 'own_whatif' when it used state the client's own what-if had changed, and 'other' otherwise (a "
          "misread request, a logic or transcription slip).")
# hand-written notes and overrides, keyed by run_id#pos
OVERRIDES = {
    "claude-haiku-5.5__per_task__main_k4_s5#7": ("other", "summed the unsigned converted column, so refunds were added "
                                                 "instead of subtracted; no state involved"),
    "gpt-6-luna__per_task__main_k8_s1#17": ("other", "copied the totals from a display in scientific notation and "
                                            "invented the decimals; no state involved"),
    "gpt-6-luna__shared__main_k8_s1#20": ("leak", "used y, Ostara's April selection (set by ostara#2), so the "
                                          "breakdown has Ostara's regions; Decimal, also another client's, is a "
                                          "harmless imported class"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    path = HERE / "review.json"
    review = json.loads(path.read_text()) if path.exists() else {"method": METHOD, "wrong_answers": {}}
    review["method"] = METHOD
    n = 0
    for p in sorted(glob.glob(f"{a.folder}/*.jsonl")):
        if a.model not in p:
            continue
        rows = [json.loads(line) for line in open(p)]
        if not rows or rows[-1].get("kind") != "run":
            continue
        for r in rows:
            if r.get("kind") != "request" or r["correct"]:
                continue
            key = f"{r['run_id']}#{r['pos']}"
            if key in OVERRIDES:
                cause, note = OVERRIDES[key]
            else:
                cause = cause_of(r, {})
                other, own = reads_split(r)
                note = (f"used {', '.join(other)} (set by another client's request) in the answering cell"
                        if cause == "leak" else f"used {', '.join(own)}" if cause == "own_whatif" else "")
            review["wrong_answers"][key] = {"cause": cause, "note": note, "client": r["client"], "request": r["req_id"],
                                            "answer": r["answer"], "truth": r["truth"]}
            n += 1
    path.write_text(json.dumps(review, indent=1))
    print(f"{n} wrong answers recorded; {len(review['wrong_answers'])} in review.json")


if __name__ == "__main__":
    main()
