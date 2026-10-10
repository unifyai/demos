"""Run one condition: every arm on every item and seed.

    ../../bin/with-openrouter-key uv run python run.py --grid G1 --requests model --length long --depth 3 \
        --arms alone,brief,summary,full,full_naive,single --seeds 0,1,2

Writes runs/<condition>.jsonl (one record per run), runs/<condition>__prefixes.jsonl
(the shared first turns of the top agent, when the model writes the first request) and
runs/<condition>__errors.jsonl. Re-running skips runs already recorded, so a stopped
batch resumes. Every model call goes through demos_common.LLM into runs/ledger.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import traceback
from pathlib import Path

from demos_common import LLM

from conversation_context.agents import (ARMS, Person, Run, callee_of, first_turn, fixed_prefix, run_from,
                                         run_whole, tools_for, top_prompt, totals, transcript)
from conversation_context.data import conversation, items, present
from conversation_context.grade import matches
from conversation_context.scenarios import BY_ID
from conversation_context.world import World

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SHARED = ("alone", "summary", "full", "full_naive")  # identical top agents: one first turn, shared


def condition_name(a: argparse.Namespace) -> str:
    model = a.model.split("/")[-1]
    if a.worker_model and a.worker_model != a.model:
        model += "+" + a.worker_model.split("/")[-1]
    if a.worker_reasoning:
        model += "-" + a.worker_reasoning
    return f"{a.grid}__{model}__{a.requests}__{a.length}__d{a.depth}" + (f"__{a.tag}" if a.tag else "")


def first_request(top_messages: list[dict]) -> str | None:
    for m in top_messages:
        for tc in m.get("tool_calls") or []:
            if callee_of(tc["function"]["name"]):
                try:
                    return json.loads(tc["function"]["arguments"]).get("request")
                except json.JSONDecodeError:
                    return None
    return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Runner:
    def __init__(self, a: argparse.Namespace, llm: LLM):
        self.a, self.llm = a, llm
        self.name = condition_name(a)
        self.out = RUNS / f"{self.name}.jsonl"
        self.prefix_path = RUNS / f"{self.name}__prefixes.jsonl"
        self.err_path = RUNS / f"{self.name}__errors.jsonl"
        self.world = World(0)
        self.done = {(r["arm"], r["item"], r["seed"]) for r in load_jsonl(self.out)}
        if a.prefixes_from and not self.prefix_path.exists():  # pair this condition with another on the same first turns
            src = RUNS / f"{a.prefixes_from}__prefixes.jsonl"
            self.prefix_path.write_text(src.read_text())
        self.prefixes = {(p["item"], p["seed"]): p for p in load_jsonl(self.prefix_path)}
        self.locks: dict[tuple[str, int], asyncio.Lock] = {}
        self.write_lock = asyncio.Lock()

    def item_setup(self, item: dict, seed: int) -> tuple[dict, list[dict], Person]:
        s = BY_ID[item["scenario"]]
        conv = conversation(s, item["kind"], self.a.length, seed)
        here = True if self.a.present == "all" else present(item["index"], seed)
        return s, conv, Person(here, s["note"], self.a.person_model or None)

    async def append(self, path: Path, rec: dict) -> None:
        async with self.write_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def fingerprint(self, conv: list[dict]) -> str:
        """What a shared first turn depends on: a cached one is reused only if all of it matches."""
        arm = ARMS["alone"]
        parts = [self.a.model, self.a.depth, top_prompt(self.a.depth, arm), tools_for("top", self.a.depth, arm), conv]
        if self.a.person_model:  # only when Maya is a model, so G1's cached first turns still match
            parts.append(self.a.person_model)
        blob = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    async def shared_prefix(self, item: dict, seed: int) -> dict:
        key = (item["id"], seed)
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            s, conv, person = self.item_setup(item, seed)
            fp = self.fingerprint(conv)
            if key in self.prefixes:
                if self.prefixes[key].get("fingerprint") == fp:
                    return self.prefixes[key]
                raise RuntimeError(f"stale shared first turn for {key} in {self.prefix_path.name}: move the file aside")
            run = Run(self.llm, self.a.model, ARMS["alone"], self.a.depth, self.world, person,
                      f"{self.name}|prefix|{item['id']}|{seed}", conv, worker_model=self.a.worker_model,
                      worker_reasoning=self.a.worker_reasoning or None)
            t0 = time.time()
            msgs, reply = await first_turn(run)
            p = {"item": item["id"], "seed": seed, "fingerprint": fp, "messages": msgs, "reply": reply, "calls": run.calls,
                 "person_questions": run.person_questions, "wall_s": round(time.time() - t0, 2)}
            await self.append(self.prefix_path, p)
            self.prefixes[key] = p
            return p

    async def one(self, arm_name: str, item: dict, seed: int) -> None:
        a = self.a
        s, conv, person = self.item_setup(item, seed)
        run_id = f"{self.name}|{arm_name}|{item['id']}|{seed}"
        run = Run(self.llm, a.model, ARMS[arm_name], a.depth, self.world, person, run_id, conv,
                  worker_model=a.worker_model, worker_reasoning=a.worker_reasoning or None)
        t0 = time.time()
        prefix_calls: list[dict] = []
        prefix_pq: list[dict] = []
        try:
            if a.requests == "terse":
                req = s["request_complete"] if item["kind"] == "complete" else s["request"]
                final = await run_from(run, fixed_prefix(a.depth, s["source"], req))
            elif arm_name in SHARED:
                p = await self.shared_prefix(item, seed)
                prefix_calls, prefix_pq = p["calls"], p["person_questions"]
                if p["reply"] is not None:
                    final = p["reply"]
                    run.top().messages += p["messages"]
                else:
                    final = await run_from(run, p["messages"])
            else:
                final = await run_whole(run)
        except Exception as e:  # recorded and retried on the next invocation
            await self.append(self.err_path, {"arm": arm_name, "item": item["id"], "seed": seed,
                                              "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
            print(f"ERROR {run_id}: {type(e).__name__}: {e}", flush=True)
            return
        tot = totals(run)
        for c in prefix_calls:
            tot["calls"] += 1
            tot["prompt_tokens"] += c["prompt_tokens"]
            tot["completion_tokens"] += c["completion_tokens"]
            tot["cost"] += c["cost"]
            tot["latency"] += c["latency"]
        top = run.agents[0]
        top_msgs = top.messages[len(conv):]
        req = first_request(top_msgs)
        pq = prefix_pq + run.person_questions
        rec = {
            "condition": self.name, "grid": a.grid, "model": a.model, "caller_model": a.model,
            "worker_model": a.worker_model or a.model, "worker_reasoning": a.worker_reasoning or None,
            "person_model": a.person_model or None,
            "requests": a.requests, "length": a.length,
            "depth": a.depth, "arm": arm_name, "item": item["id"], "scenario": item["scenario"], "kind": item["kind"],
            "seed": seed, "present": person.present, "conversation_messages": len(conv),
            "first_request": req, "first_request_has_detail": None if req is None else matches(req, s["detail"]),
            "final": final, "step_limits": run.step_limits,
            "q_caller": len(run.questions), "q_person": len(pq),
            "questions": run.questions, "person_questions": pq,
            "handovers": sum(1 for x in run.agents if x.level > 0),
            "max_level": max(x.level for x in run.agents),
            **tot, "prefix_shared": arm_name in SHARED and a.requests == "model",
            "wall_s": round(time.time() - t0, 2),
            "calls_detail": prefix_calls + run.calls,
            "transcript": transcript(run, len(conv)),
        }
        await self.append(self.out, rec)
        print(f"{arm_name:10s} {item['id']:14s} seed {seed} | {final[:70]!r} | ${tot['cost']:.4f} {tot['latency']:.0f}s",
              flush=True)

    async def go(self) -> None:
        a = self.a
        kinds = tuple(a.kinds.split(","))
        its = items(kinds, a.scenarios.split(",") if a.scenarios else None)
        jobs = [(arm, it, seed) for seed in a.seeds for it in its for arm in a.arms.split(",")
                if (arm, it["id"], seed) not in self.done]
        print(f"{self.name}: {len(jobs)} runs to do ({len(self.done)} already recorded)", flush=True)
        sem = asyncio.Semaphore(a.parallel_runs)

        async def guarded(job: tuple) -> None:
            async with sem:
                await self.one(*job)

        await asyncio.gather(*(guarded(j) for j in jobs))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--model", default="anthropic/claude-haiku-5.5", help="the caller's model (and the workers' unless --worker-model)")
    p.add_argument("--worker-model", default="", help="the delegated work's model, if not the caller's")
    p.add_argument("--person-model", default="", help="a model to play Maya (amendment 1); default: a fixed reply from the note")
    p.add_argument("--worker-reasoning", default="", help="reasoning effort for the workers (e.g. high)")
    p.add_argument("--prefixes-from", default="", help="reuse the shared first turns of this condition (same caller)")
    p.add_argument("--requests", choices=["model", "terse"], required=True)
    p.add_argument("--length", choices=["short", "medium", "long", "xlong"], required=True)
    p.add_argument("--depth", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--arms", default="alone,brief,summary,full,full_naive,single")
    p.add_argument("--kinds", default="settled,open,complete")
    p.add_argument("--scenarios", default="")
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    p.add_argument("--present", choices=["balanced", "all"], default="balanced")
    p.add_argument("--tag", default="")
    p.add_argument("--max-concurrency", type=int, default=12)
    p.add_argument("--parallel-runs", type=int, default=24)
    a = p.parse_args()
    assert a.max_concurrency <= 12, "eight concepts share one key"
    if a.requests == "terse":
        assert not set(a.arms.split(",")) & {"brief", "single"}, "brief and single need a model-written first request"
    return a


async def main() -> None:
    a = parse()
    RUNS.mkdir(exist_ok=True)
    async with LLM(RUNS / "ledger.jsonl", "conversation-context", max_concurrency=a.max_concurrency) as llm:
        await Runner(a, llm).go()


if __name__ == "__main__":
    asyncio.run(main())
