"""Judge a run's records: map unmatched reported rules to rules, then judge which
version of each changed rule a text states (task answers, the in-passing plans,
and the text that governs each procedure after the changes).

    ../../bin/with-openrouter-key uv run python judge.py --run main

Judgments are cached in runs/<run>/judgments.jsonl by rule and text, so the
same text is judged once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from demos_common import LLM

from linked_libraries.runner import Records
from linked_libraries.scoring import (JUDGE_MODEL, judgment_jobs_for_task, map_prompt, parse_json,
                                      version_jobs_for_task, version_key, version_prompt)

HERE = Path(__file__).parent


def datasets() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (HERE / "data").glob("*.json")}


def passing_task(rec: dict, ds: dict) -> dict:  # noqa: ARG001
    """An in-passing change record seen as a task: the procedure it came with,
    and the rule it changed (the one whose version we judge)."""
    return {"answer": rec.get("answer"), "rules_expected": [rec["change"]],
            "function_found": False}


async def main_async(run: str, concurrency: int, ledger: str) -> None:
    rec = Records(HERE / "runs" / run)
    dss = datasets()
    judgments = {r["key"]: r for r in rec.read("judgments")}
    tasks = rec.read("tasks")
    passing = [r for r in rec.read("changes") if r["condition"] == "passing"]
    texts = {r["hash"]: r["text"] for r in rec.read("texts")}
    state = [r for r in rec.read("state") if r["changed"]]

    async with LLM(ledger, "linked-libraries", max_concurrency=concurrency) as llm:

        async def judge(key: str, kind: str, payload: dict, ds: dict) -> None:
            if key in judgments:
                return
            if kind == "map":
                prompt = map_prompt(ds, payload["item"])
            else:
                prompt = version_prompt(ds["rules"][payload["rule"]], payload["text"])
            out, raw = {}, ""
            for attempt in range(2):
                r = await llm.chat(JUDGE_MODEL, [{"role": "user", "content": prompt}], temperature=0,
                                   max_tokens=2000, tag="judge", meta={"kind": kind, "attempt": attempt})
                raw = r.text
                out = parse_json(raw)
                if (kind == "map" and "rule" in out) or (kind == "version" and "states_new" in out and "states_old" in out):
                    break
            row = {"key": key, "kind": kind, "judge": JUDGE_MODEL, "raw": raw[:500]}
            if kind == "map":
                row["rule"] = out.get("rule") if out.get("rule") in ds["rules"] else None
                row["size"] = payload["size"]
            else:
                row["rule"] = payload["rule"]
                row["states_new"] = bool(out.get("states_new")) if "states_new" in out else None
                row["states_old"] = bool(out.get("states_old")) if "states_old" in out else None
                row["text"] = payload["text"]
            judgments[key] = row
            rec.write("judgments", row)

        # 1. map reported items whose title does not name a rule
        jobs = {}
        for t in tasks:
            ds = dss[t["size"]]
            for key, kind, payload in judgment_jobs_for_task(t, ds):
                jobs[key] = (kind, payload, ds)
        for c in passing:
            ds = dss[c["size"]]
            for key, kind, payload in judgment_jobs_for_task(passing_task(c, ds), ds):
                jobs[key] = (kind, payload, ds)
        print(f"{len([k for k in jobs if k not in judgments])} mapping judgments to make", flush=True)
        await asyncio.gather(*(judge(k, kind, p, ds) for k, (kind, p, ds) in jobs.items()))

        # 2. versions: task answers, in-passing plans, governing texts
        jobs = {}
        for t in tasks:
            ds = dss[t["size"]]
            for key, kind, payload in version_jobs_for_task(t, ds, judgments):
                jobs[key] = (kind, payload, ds)
        for c in passing:
            ds = dss[c["size"]]
            for key, kind, payload in version_jobs_for_task(passing_task(c, ds), ds, judgments):
                jobs[key] = (kind, payload, ds)
        for s in state:
            ds = dss[s["size"]]
            text = texts[s["hash"]]
            jobs[version_key(s["rule"], text)] = ("version", {"rule": s["rule"], "text": text}, ds)
        print(f"{len([k for k in jobs if k not in judgments])} version judgments to make", flush=True)
        await asyncio.gather(*(judge(k, kind, p, ds) for k, (kind, p, ds) in jobs.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--ledger", default=str(HERE / "runs" / "ledger.jsonl"))
    args = ap.parse_args()
    assert args.concurrency <= 12
    asyncio.run(main_async(args.run, args.concurrency, args.ledger))


if __name__ == "__main__":
    main()
