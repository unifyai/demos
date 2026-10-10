"""Run arms over sessions and append one record per run.

  ../../bin/with-openrouter-key uv run python run.py --items 0-11 --sizes 64 128 256 500 \
      --arms inline_all by_reference --seeds 0 1 2 --model anthropic/claude-haiku-5.5

Sessions come from data/sessions (make_data.py), frames from data/frames.
Records go to runs/records.jsonl (one JSON object per run, transcripts without
pixels); every model call goes to runs/ledger.jsonl through demos_common.LLM.
A run already in the records without an error is skipped, so a batch can be
re-run to fill gaps. Jobs run in a shuffled order so that no arm or size runs
at a different time of day from the others.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from demos_common import LLM

from demoref.arms import ARMS, Settings, run_arm
from demoref.session import load

HERE = Path(__file__).parent
DATA = HERE / "data"
RUNS = HERE / "runs"
CHEAP = {"anthropic": "anthropic/claude-haiku-5.5", "openai": "openai/gpt-6-luna", "google": "google/gemini-3.8-flash"}


def parse_items(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def done_keys(path: Path) -> set[tuple]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("error"):
                    keys.add((r["session"], r["arm"], r["seed"], r["model"]))
    return keys


async def main_async(a: argparse.Namespace) -> None:
    names = list(a.sessions or [])
    for item in parse_items(a.items) if a.items else []:
        names += [f"item{item:02d}-n{size}" for size in a.sizes]
        if a.sanity:
            names.append(f"sanity-{item:02d}")
    family = a.model.split("/")[0]
    provider = {"order": [family], "allow_fallbacks": True}
    st = Settings(model=a.model, caption_model=a.caption_model or CHEAP[family],
                  reasoning={"effort": a.effort}, caption_reasoning={"effort": "none"},
                  max_tokens=a.max_tokens, provider=provider,
                  caption_provider={"order": [(a.caption_model or CHEAP[family]).split("/")[0]], "allow_fallbacks": True})
    records = Path(a.records)
    records.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(records)
    jobs = [(n, arm, seed) for n in names for arm in a.arms for seed in a.seeds
            if (n, arm, seed, a.model) not in skip]
    random.Random(a.order_seed).shuffle(jobs)
    print(f"{len(jobs)} runs to do ({len(skip)} already recorded)", flush=True)
    sessions = {n: load(DATA / "sessions" / f"{n}.json") for n in {j[0] for j in jobs}}
    gate = asyncio.Semaphore(a.max_runs)
    lock = asyncio.Lock()
    t0 = time.time()
    n_done = 0

    async with LLM(RUNS / "ledger.jsonl", tag=a.tag, max_concurrency=a.concurrency) as llm:
        async def one(name: str, arm: str, seed: int) -> None:
            nonlocal n_done
            async with gate:
                res = await run_arm(arm, llm, st, sessions[name], DATA / "frames" / name, seed, name,
                                    half_root=DATA / "frames_half" / name)
            async with lock:
                with open(records, "a") as f:
                    f.write(json.dumps(res.to_json()) + "\n")
                n_done += 1
                cost = sum(c["cost"] for c in res.calls)
                print(f"[{n_done}/{len(jobs)} {time.time() - t0:.0f}s] {name} {arm} s{seed}: "
                      f"{'OK ' if res.correct else 'BAD'} {res.answer} truth {res.truth} ${cost:.4f} "
                      f"steps {res.steps} looked {len(res.looked)}{' ERROR ' + res.error if res.error else ''}"
                      f"{' OVER LIMIT ' + res.limit[:120] if res.limit else ''}", flush=True)

        await asyncio.gather(*[one(*j) for j in jobs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="")
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 500])
    ap.add_argument("--sanity", action="store_true", help="also run each item's sanity session")
    ap.add_argument("--sessions", nargs="*", help="explicit session names, e.g. sanity-00 item03-n80")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--caption-model", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--records", default=str(RUNS / "records.jsonl"))
    ap.add_argument("--tag", default="demoref")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-runs", type=int, default=16)
    ap.add_argument("--order-seed", type=int, default=0)
    a = ap.parse_args()
    if a.concurrency > 12:
        sys.exit("max_concurrency must be 12 or less (eight concepts share one key)")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
