"""Judge the replies that patterns can't label: both answers, or neither.

    ../../bin/with-openrouter-key uv run python judge.py [runs/<condition>.jsonl ...]

With no arguments, judges every condition file in runs/. Labels go to
runs/judge.jsonl, keyed by condition, arm, item and seed; already-judged runs are
skipped. The judge is from the other family (grade.JUDGE_FOR) and never sees the arm.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

from demos_common import LLM

from conversation_context.grade import JUDGE_VERSION, judge_family, judge_prompt, parse_judge, pattern_label
from conversation_context.scenarios import BY_ID

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "judge.jsonl"


def key(r: dict) -> str:
    """The run and a hash of its reply, so a re-run never inherits an old label."""
    h = hashlib.sha1(r["final"].encode()).hexdigest()[:10]
    return f"{r['condition']}|{r['arm']}|{r['item']}|{r['seed']}|{h}|{JUDGE_VERSION}"


def condition_files() -> list[Path]:
    return sorted(p for p in RUNS.glob("*.jsonl") if "__" in p.stem and not p.stem.endswith(("__prefixes", "__errors")))


def load_judged() -> dict[str, dict]:
    if not OUT.exists():
        return {}
    return {j["key"]: j for j in map(json.loads, OUT.read_text().splitlines()) if j.get("label")}


def question_of(r: dict) -> str:
    s = BY_ID[r["scenario"]]
    return s["question_complete"] if r["kind"] == "complete" else s["question"]


async def main(paths: list[Path]) -> None:
    judged = load_judged()
    todo = []
    for p in paths:
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if pattern_label(r["scenario"], r["final"]) in ("both", "neither") and key(r) not in judged:
                todo.append(r)
    print(f"{len(todo)} replies to judge")
    lock = asyncio.Lock()
    async with LLM(RUNS / "ledger.jsonl", "conversation-context-judge", max_concurrency=12) as llm:
        async def one(r: dict) -> None:
            model = judge_family(r.get("caller_model", r["model"]))  # the caller writes the graded reply
            reply = await llm.chat(model, judge_prompt(r["scenario"], question_of(r), r["final"]), max_tokens=2000,
                                   temperature=0, meta={"judge": key(r)})
            lab = parse_judge(reply.text)
            async with lock:
                with open(OUT, "a") as f:
                    f.write(json.dumps({"key": key(r), "judge": model, "raw": reply.text, "label": lab}) + "\n")

        await asyncio.gather(*(one(r) for r in todo))


if __name__ == "__main__":
    args = [Path(x) for x in sys.argv[1:]]
    asyncio.run(main(args or condition_files()))
