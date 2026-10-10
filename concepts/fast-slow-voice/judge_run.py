"""Judge every recorded call of one or more runs, without telling the judge the arm.

  ../../bin/with-openrouter-key uv run python judge_run.py runs/<condition>.jsonl [...]

Writes runs/judge/<condition>.jsonl (one line per call: the judgement and its
raw text); calls already judged are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from demos_common import LLM

from fast_slow_voice.config import PRESETS
from fast_slow_voice.judge import judge_messages, parse_judgement
from fast_slow_voice.script import Script, build_sanity

HERE = Path(__file__).parent


def scripts_by_id() -> dict[str, Script]:
    out = {s.id: s for s in (Script.from_json(d) for d in json.loads((HERE / "data" / "scripts-1.json").read_text()))}
    for k in range(4):
        s = build_sanity(f"sanity-{k}", 900 + k)
        out[s.id] = s
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--ledger", default="runs/ledger.jsonl", help="runs/ledger-vm.jsonl on the cloud machine")
    ap.add_argument("--seeds", default="", help="judge only these seeds, e.g. 0")
    a = ap.parse_args()
    cfg = PRESETS["judge"]
    scripts = scripts_by_id()
    (HERE / "runs" / "judge").mkdir(exist_ok=True)
    async with LLM(HERE / a.ledger, "fast-slow-voice/judge", max_concurrency=min(12, a.concurrency)) as llm:
        for path in a.paths:
            if ".errors." in path:
                continue
            recs = [r for r in (json.loads(l) for l in open(path) if l.strip()) if "persons" in r]
            out = HERE / "runs" / "judge" / Path(path).name
            done = set()
            if out.exists():
                for l in out.read_text().splitlines():
                    j = json.loads(l)
                    if j.get("judgement") is not None:
                        done.add((j["arm"], j["script"], j["seed"]))
            seeds = {int(x) for x in a.seeds.split(",")} if a.seeds else None
            todo = [r for r in recs if (r["arm"], r["script"], r["seed"]) not in done and (seeds is None or r["seed"] in seeds)]
            print(f"{path}: {len(todo)} to judge ({len(done)} done)", flush=True)
            lock = asyncio.Lock()

            async def one(rec: dict) -> None:
                s = scripts[rec["script"]]
                w = s.world(rec["size"])
                msgs = judge_messages(rec, s, w)
                j, text = None, ""
                for attempt in range(3):
                    r = await llm.chat(cfg.name, msgs, reasoning=cfg.reasoning, max_tokens=cfg.max_tokens,
                                       response_format={"type": "json_object"},
                                       meta={"judge": rec["condition"], "arm": rec["arm"], "script": rec["script"], "seed": rec["seed"], "attempt": attempt})
                    text = r.text
                    j = parse_judgement(text)
                    if j is not None:
                        break
                row = {"condition": rec["condition"], "arm": rec["arm"], "script": rec["script"], "seed": rec["seed"],
                       "judgement": j, "raw": text if j is None else None}
                async with lock:
                    with open(out, "a") as f:
                        f.write(json.dumps(row) + "\n")

            await asyncio.gather(*(one(r) for r in todo))


if __name__ == "__main__":
    asyncio.run(main())
