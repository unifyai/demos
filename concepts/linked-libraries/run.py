"""Run the experiment for one model and size (resumable).

    ../../bin/with-openrouter-key uv run python run.py --run main --model anthropic/claude-haiku-5.5 --size L
    ../../bin/with-openrouter-key uv run python run.py --run smoke --size S --seeds 1 --arms pasted linked

Then: judge.py --run <run> and analyze.py --run <run>.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from demos_common import LLM

from linked_libraries.arms import TASK_ARMS
from linked_libraries.runner import CONDITIONS, Records, run

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<run>/ holds the records")
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--size", required=True, choices=["sanity", "S", "M", "L"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--arms", nargs="+", default=list(TASK_ARMS), choices=list(TASK_ARMS))
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                    help="OpenRouter's reasoning effort for every call (default: the provider's own)")
    ap.add_argument("--max-tokens", type=int, default=4000, help="output cap per call (reasoning counts against it)")
    ap.add_argument("--tasks", nargs="+", default=None, help="run only these task ids (default: every task)")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--ledger", default=str(HERE / "runs" / "ledger.jsonl"))
    args = ap.parse_args()
    assert args.concurrency <= 12, "eight concepts share one key: at most 12 calls in flight"

    ds = json.loads((HERE / "data" / f"{args.size}.json").read_text())
    records = Records(HERE / "runs" / args.run)

    async def go() -> None:
        async with LLM(args.ledger, "linked-libraries", max_concurrency=args.concurrency) as llm:
            await run(llm, records, ds, args.model, arms=args.arms, conditions=args.conditions,
                      seeds=args.seeds, lib_root=HERE / "runs" / args.run / "libs",
                      reasoning={"effort": args.reasoning_effort} if args.reasoning_effort else None,
                      max_tokens=args.max_tokens, tasks=args.tasks)

    asyncio.run(go())


if __name__ == "__main__":
    main()
