"""Re-check recorded answers with the current check: python rescore.py [--labels main,tiers]

Fault 9 (README): claude-sonnet-5.5 sent some list answers as text through the
submit tool's untyped field ("['t_5010', ...]"), which the check then counted
wrong though the list was right. The check now parses such text. This script
re-checks every failed task whose only failure was its answer (no change to
the workspace was wrong), using the answer as submitted (from the transcript),
and rewrites its record with `passed` corrected and the original kept in
`passed_original`. A re-scored task's effects on the rest of its stream (the
outcome its arm was told, and what it kept) cannot be undone. Idempotent.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from skills.tasks import KINDS, answer_ok, make_stream  # noqa: E402

_streams: dict = {}


def expected(size: str, seed: int, position: int):
    if (size, seed) not in _streams:
        _streams[(size, seed)] = make_stream(size, seed)
    return _streams[(size, seed)].tasks[position - 1].expected


def submitted(transcript: dict, record: dict):
    """The answer as submitted: the submit tool's argument, else the recorded repr."""
    answer, found = None, False
    for m in transcript["messages"]:
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] == "submit":
                try:
                    answer, found = json.loads(tc["function"]["arguments"])["answer"], True
                except Exception:  # noqa: BLE001
                    pass
    if found:
        return answer
    try:
        return ast.literal_eval(record["answer"])
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="main,tiers")
    a = p.parse_args()
    changed = 0
    for label in a.labels.split(","):
        for path in sorted(glob.glob(str(HERE / "runs" / label / "*" / "*" / "*-s*.jsonl"))):
            if path.endswith(".transcripts.jsonl"):
                continue
            recs = [json.loads(line) for line in open(path) if line.strip()]
            trs = {}
            tpath = path.replace(".jsonl", ".transcripts.jsonl")
            if Path(tpath).exists():
                trs = {json.loads(line)["position"]: json.loads(line) for line in open(tpath) if line.strip()}
            dirty = False
            for r in recs:
                if r["passed"] or not r["submitted"] or r["position"] not in trs:
                    continue
                if not r["reasons"] or not all(x.startswith("answer:") for x in r["reasons"]):
                    continue
                got = submitted(trs[r["position"]], r)
                if answer_ok(KINDS[r["kind"]].answer, expected(r["size"], r["seed"], r["position"]), got):
                    r["passed_original"] = False
                    r["passed"] = True
                    r["rescored"] = "fault 9: the right answer, sent as text"
                    dirty = True
                    changed += 1
                    print(f"{label} {r['size']} {r['model']} {r['arm']} s{r['seed']} t{r['position']} {r['kind']}: now passed")
            if dirty:
                with open(path, "w") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
    print(f"{changed} records re-scored")


if __name__ == "__main__":
    main()
