"""Run arms over task streams.

    ../../bin/with-openrouter-key uv run python run.py --label main --size realistic \
        --arms none,notes,retrieval,fullctx,library --seeds 1,2,3 --model anthropic/claude-haiku-5.5

Each (arm, seed) is one stream, run task by task in order; streams run
concurrently under one client whose concurrency is capped. Records go to
runs/<label>/<size>/<model>/<arm>-s<seed>.jsonl, with the full transcript
beside them (.transcripts.jsonl) and the arm's memory at the end
(.memory.json). Every model call is in runs/ledger.jsonl. A stream whose
records are complete is skipped; an incomplete one is moved to runs/discarded/
and run again from its first task.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from demos_common import LLM  # noqa: E402

from skills.agent import base_system, run_task  # noqa: E402
from skills.api import Api  # noqa: E402
from skills.arms import ARMS  # noqa: E402
from skills.tasks import KINDS, make_stream  # noqa: E402

TAG = "skills-as-functions"


def short(model: str) -> str:
    return model.split("/")[-1]


class Settings:
    """A condition's model settings, applied to every call it makes (agent, notes,
    save): the tool choice (a model that rejects forced tool calls gets "auto"),
    reasoning effort, and extra output room for reasoning tokens."""

    def __init__(self, llm: LLM, tool_choice: str | None = None, reasoning: str | None = None, extra_max_tokens: int = 0):
        self.llm, self.tool_choice, self.reasoning, self.extra = llm, tool_choice, reasoning, extra_max_tokens

    def describe(self) -> dict[str, Any]:
        return {"tool_choice": self.tool_choice or "required", "reasoning": self.reasoning, "extra_max_tokens": self.extra}

    async def chat(self, model: str, messages: list[dict[str, Any]], **kw: Any) -> Any:
        if self.tool_choice and "tool_choice" in kw:
            kw["tool_choice"] = self.tool_choice
        if self.reasoning:
            kw["reasoning"] = {"effort": self.reasoning}
            kw.pop("temperature", None)  # reasoning models take their own sampling
        if self.extra and kw.get("max_tokens"):
            kw["max_tokens"] = kw["max_tokens"] + self.extra
        return await self.llm.chat(model, messages, **kw)


def totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "cost", "latency_s"]
    return {k: round(sum(c[k] for c in calls), 6) for k in keys} | {"calls": len(calls)}


async def run_stream(llm: Any, model: str, size: str, arm_name: str, seed: int, out_dir: Path, label: str,
                     limit: int | None) -> str:
    stream = make_stream(size, seed)
    tasks = stream.tasks[:limit] if limit else stream.tasks
    base = out_dir / f"{arm_name}-s{seed}"
    rec_path, tr_path, mem_path = base.with_suffix(".jsonl"), Path(f"{base}.transcripts.jsonl"), Path(f"{base}.memory.json")
    if rec_path.exists():
        done = sum(1 for line in rec_path.read_text().splitlines() if line.strip())
        if done >= len(tasks) and mem_path.exists():
            return f"{arm_name} s{seed}: already complete"
        dest = HERE / "runs" / "discarded" / label / size / short(model) / time.strftime("%Y%m%d-%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        for p in (rec_path, tr_path, mem_path):
            if p.exists():
                shutil.move(str(p), dest / p.name)
        (dest / f"{arm_name}-s{seed}.REASON.txt").write_text(f"incomplete stream ({done} of {len(tasks)} tasks); rerun from the start\n")
    arm = ARMS[arm_name](size)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_version = None
    with rec_path.open("w") as rf, tr_path.open("w") as tf:
        for task in tasks:
            meta = {"label": label, "size": size, "arm": arm_name, "seed": seed, "task": task.position}
            memory_shown = arm.memory_section(task)
            run, ctx = await run_task(llm, model, arm, task, size, meta)
            mem = await arm.after_task(llm, model, task, run, ctx, meta)
            memory_calls = mem.pop("memory_calls", [])
            save_turns = mem.pop("save_turns", None)
            agent_t, mem_t = totals(run.calls), totals(memory_calls)
            all_t = totals(run.calls + memory_calls)
            k = KINDS[task.kind]
            rec = {
                "label": label, "size": size, "model": model, "arm": arm_name, "seed": seed,
                "settings": llm.describe() if hasattr(llm, "describe") else None,
                "position": task.position, "kind": task.kind, "level": task.level, "version": task.version,
                "repeat_of": task.repeat_of, "first_of_kind": task.first_of_kind, "uses": list(k.uses),
                "passed": run.passed, "submitted": run.submitted, "reasons": [r[:400] for r in run.reasons[:8]],
                "answer": repr(run.answer)[:300], "expected": repr(task.expected)[:300],
                "steps": run.steps, "cells": len(run.cells), "errors": [c["error_type"] for c in run.cells if c["error_type"]],
                "api_calls": run.api_calls, "wall_s": round(run.wall_s, 3),
                "library_calls": run.library_calls, "library_failed": run.library_failed,
                "library_failed_at": run.library_failed_at,
                **{f"agent_{k_}": v for k_, v in agent_t.items()},
                **{f"memory_{k_}": v for k_, v in mem_t.items()},
                **all_t,
                **mem,
            }
            rf.write(json.dumps(rec) + "\n")
            rf.flush()
            tr = {"position": task.position, "text": task.text, "passed": run.passed, "reasons": run.reasons,
                  "memory_shown": memory_shown, "messages": run.messages, "calls": run.calls + memory_calls}
            if task.version != last_version:
                tr["system_base"] = base_system(Api(task.before, task.version, size))
                last_version = task.version
            if arm_name == "notes":
                tr["notes_after"] = arm.snapshot()
            if save_turns:
                tr["save_turns"] = save_turns
            tf.write(json.dumps(tr) + "\n")
            tf.flush()
            print(f"{label} {size} {short(model)} {arm_name:9s} s{seed} t{task.position:3d} v{task.version} "
                  f"{task.kind:27s} {'PASS' if run.passed else 'fail'} steps={run.steps} "
                  f"cost=${all_t['cost']:.4f} lat={all_t['latency_s']:.1f}s", flush=True)
    mem_path.write_text(json.dumps(arm.snapshot(), indent=1, default=str))
    return f"{arm_name} s{seed}: done"


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--size", required=True, choices=["sanity", "small", "medium", "realistic"])
    p.add_argument("--arms", default="none,notes,retrieval,fullctx,library")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--model", default="anthropic/claude-haiku-5.5")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="only the first N tasks of each stream")
    p.add_argument("--ledger", default="runs/ledger.jsonl", help="relative to this folder (the cloud machine uses its own)")
    p.add_argument("--tool-choice", default=None, help='"auto" for a model that rejects forced tool calls')
    p.add_argument("--reasoning", default=None, help="reasoning effort, e.g. high")
    p.add_argument("--extra-max-tokens", type=int, default=0, help="output room added to every call, for reasoning tokens")
    a = p.parse_args()
    assert a.concurrency <= 12, "eight concepts share one key: keep concurrency at 12 or less"
    arms = a.arms.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    out_dir = HERE / "runs" / a.label / a.size / short(a.model)
    async with LLM(HERE / a.ledger, TAG, max_concurrency=a.concurrency) as raw:
        llm = Settings(raw, a.tool_choice, a.reasoning, a.extra_max_tokens)
        jobs = [run_stream(llm, a.model, a.size, arm, seed, out_dir, a.label, a.limit) for seed in seeds for arm in arms]
        results = await asyncio.gather(*jobs, return_exceptions=True)
    for (seed, arm), r in zip([(s, x) for s in seeds for x in arms], results):
        if isinstance(r, BaseException):
            msg = "".join(traceback.format_exception(r))
            print(f"FAILED {arm} s{seed}:\n{msg}", flush=True)
            errors = (HERE / a.ledger).with_name((HERE / a.ledger).stem.replace("ledger", "errors") + ".log")
            with errors.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {a.label} {a.size} {a.model} {arm} s{seed}\n{msg}\n")
        else:
            print(r, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
