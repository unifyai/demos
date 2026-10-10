"""Run arms over conditions, sizes and seeds. One JSONL file per run:

    {"kind": "setup", ...}      the system prompt, tool schemas, arrival order
    {"kind": "request", ...}    one per request, in arrival order, with its transcript
    {"kind": "run", ...}        totals; its presence marks the run complete

Requests in a run are handled one at a time in arrival order; runs go in
parallel, sharing one model client (at most `max_concurrency` calls at once).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

from demos_common import LLM

from . import data as D
from .agent import handle, system_prompt
from .executor import Executor
from .scoring import correct, signature_causes
from .tools import ARMS, Ctx

HERE = Path(__file__).resolve().parents[2]
RUNS = HERE / "runs"
TAG = "execution-coordinates"


def run_id(model: str, arm: str, cond: str, k: int, seed: int) -> str:
    return f"{model.split('/')[-1]}__{arm}__{cond}_k{k}_s{seed}"


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text().strip().splitlines()
    return bool(lines) and json.loads(lines[-1]).get("kind") == "run"


def discard(path: Path, reason: str) -> None:
    dest = RUNS / "discarded" / path.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    target = dest / f"{path.stem}.{stamp}.jsonl"
    shutil.move(str(path), target)
    with open(RUNS / "discarded" / "index.jsonl", "a") as f:
        f.write(json.dumps({"file": str(target.relative_to(RUNS)), "reason": reason, "at": stamp}) + "\n")


def summarise_tool(t: dict) -> dict:
    r = t.get("record") or {}
    a = t.get("args") or {}
    return {
        "name": t["name"], "mode": r.get("mode"), "session": r.get("session"), "created": r.get("created"),
        "code": a.get("code"), "args": {k: v for k, v in a.items() if k != "code"},
        "error": (r.get("error") or "")[-600:] or None, "error_type": r.get("error_type"),
        "seconds": r.get("seconds"), "measure_s": r.get("measure_s", 0.0), "events": r.get("events", []),
        "reads": r.get("reads", {}), "bad_args": t.get("bad_args", False),
    }


async def run_one(llm: LLM, *, model: str, arm: str, cond: str, k: int, seed: int, phase: str,
                  max_calls: int = 25, reasoning: dict | None = None, max_tokens: int = 8000) -> Path:
    rid = run_id(model, arm, cond, k, seed)
    out_dir = RUNS / phase
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rid}.jsonl"
    if is_complete(path):
        return path
    if path.exists():
        discard(path, "incomplete: the process running it stopped before the run finished")
    spec = D.load_spec(cond, seed)
    clients = spec["clients"][:k]
    order = spec["orders"][str(k)]
    in_run = {c["id"] for c in clients}
    toolset_cls, design = ARMS[arm]
    scratch = Path(tempfile.gettempdir()) / "execution-coordinates" / rid
    shutil.rmtree(scratch, ignore_errors=True)
    api = {"data_dir": str(D.DATA / cond / f"seed{seed}"), "load_s": spec["load_s"], "score_s": spec["score_s"],
           "clients": sorted(in_run)}
    ex = Executor(scratch, api=api, timeout_s=300)
    tools = toolset_cls(ex)
    system = system_prompt(spec["load_s"], spec["score_s"], design)
    histories: dict[str, list] = defaultdict(list)
    t_run = time.time()
    totals = defaultdict(float)
    try:
        with open(path, "w") as f:
            f.write(json.dumps({"kind": "setup", "run_id": rid, "model": model, "arm": arm, "design": design,
                                "cond": cond, "k": k, "seed": seed, "phase": phase, "system": system,
                                "tools": tools.schemas, "order": order, "clients": sorted(in_run),
                                "reasoning": reasoning, "max_tokens": max_tokens}) + "\n")
            for pos, (ci, ri) in enumerate(order):
                c = clients[ci]
                req = c["requests"][ri]
                key = c["id"] if design == "threads" else "all"
                ctx = Ctx(client=c["id"], tag=f"{c['id']}#{ri + 1}")
                t0 = time.time()
                oc = await handle(llm, model, system, histories[key], req["text"], tools, ctx, max_calls=max_calls,
                                  meta={"run": rid, "pos": pos, "req": ctx.tag}, reasoning=reasoning,
                                  max_tokens=max_tokens)
                wall = time.time() - t0
                histories[key].extend(oc.messages)
                tl = [summarise_tool(t) for t in oc.tools]
                events = [e for t in tl for e in t["events"]]
                calls = [c_ for c_ in oc.calls if "error" not in c_]
                exec_s = sum(float(t["seconds"] or 0) - float(t["measure_s"] or 0) for t in tl)
                model_s = sum(c_["latency_s"] for c_ in calls)
                reads = [(n, tag) for t in tl for n, tag in t["reads"].items()]
                row = {
                    "kind": "request", "run_id": rid, "model": model, "arm": arm, "design": design, "cond": cond,
                    "k": k, "seed": seed, "phase": phase, "pos": pos, "client": c["id"], "client_idx": c["idx"],
                    "req_idx": ri, "req_id": req["id"], "answer_type": req["answer_type"], "text": req["text"],
                    "truth": req["truth"], "answer_raw": oc.answer_raw, "answer": oc.answer,
                    "correct": correct(req, oc.answer), "causes": signature_causes(req, oc.answer, in_run),
                    "stopped": oc.stopped, "n_calls": len(oc.calls), "llm_errors": len(oc.calls) - len(calls),
                    "prompt_tokens": sum(c_["prompt_tokens"] for c_ in calls),
                    "completion_tokens": sum(c_["completion_tokens"] for c_ in calls),
                    "reasoning_tokens": sum(c_["reasoning_tokens"] for c_ in calls),
                    "cached_tokens": sum(c_["cached_tokens"] for c_ in calls),
                    "cost": sum(c_["cost"] for c_ in calls), "model_s": round(model_s, 3), "exec_s": round(exec_s, 3),
                    "agent_s": round(model_s + exec_s, 3), "wall_s": round(wall, 3),
                    "loads": sum(1 for e in events if e["fn"] == "load_export"),
                    "loads_other_client": sum(1 for e in events if e["fn"] == "load_export" and e["client"] != c["id"]),
                    "scores": sum(1 for e in events if e["fn"] == "score_customers"),
                    "reads_other_client": sorted({f"{n}<-{tag}" for n, tag in reads if tag.split("#")[0] != c["id"]}),
                    "reads_own_earlier": sorted({f"{n}<-{tag}" for n, tag in reads if tag.split("#")[0] == c["id"]
                                                 and tag != ctx.tag}),
                    "tool_calls": tl, "messages": oc.messages,
                }
                for key_ in ("prompt_tokens", "completion_tokens", "cached_tokens", "cost", "agent_s", "loads",
                             "scores", "n_calls"):
                    totals[key_] += row[key_]
                totals["correct"] += row["correct"]
                totals["requests"] += 1
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
            sessions = [{"session": s.name, "calls": s.calls, "created_by": s.created_by}
                        for s in ex.sessions.values()]
            f.write(json.dumps({"kind": "run", "run_id": rid, "done": True, "wall_s": round(time.time() - t_run, 1),
                                "sessions": sessions, **totals}) + "\n")
    finally:
        await ex.close()
        log = scratch / "workers.log"
        if log.exists() and log.stat().st_size:
            (out_dir / "logs").mkdir(exist_ok=True)
            shutil.copy(log, out_dir / "logs" / f"{rid}.workers.log")
        shutil.rmtree(scratch, ignore_errors=True)
    return path


async def main_async(args) -> None:
    seeds = D.parse_seeds(args.seeds)
    for s in seeds:
        if not (D.DATA / args.cond / f"seed{s}" / "spec.json").exists():
            D.generate(args.cond, s)
    jobs = [(arm, k, s) for k in [int(x) for x in args.ks.split(",")] for s in seeds for arm in args.arms.split(",")]
    sem = asyncio.Semaphore(args.max_runs)
    done = 0
    ledger = Path(args.ledger) if args.ledger else RUNS / "ledger.jsonl"
    reasoning = {"effort": args.reasoning} if args.reasoning else None
    async with LLM(ledger, TAG, max_concurrency=args.max_concurrency) as llm:
        async def one(arm, k, s):
            nonlocal done
            async with sem:
                try:
                    p = await run_one(llm, model=args.model, arm=arm, cond=args.cond, k=k, seed=s, phase=args.phase,
                                      max_calls=args.max_calls, reasoning=reasoning, max_tokens=args.max_tokens)
                    done += 1
                    print(f"[{time.strftime('%H:%M:%S')}] done {done}/{len(jobs)} {p.name}", flush=True)
                except Exception:
                    print(f"[{time.strftime('%H:%M:%S')}] FAILED {arm} k{k} s{s}\n{traceback.format_exc()}", flush=True)

        # largest runs first, so the long ones are not left for last
        jobs.sort(key=lambda j: (-j[1], j[2], j[0]))
        await asyncio.gather(*(one(*j) for j in jobs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, help="a folder under runs/: sanity, smoke, main, family2, capable")
    ap.add_argument("--model", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--arms", default="fresh,shared,per_task,coordinate,long_shared,long_coordinate")
    ap.add_argument("--cond", default="main", choices=sorted(D.CONDITIONS))
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--max-runs", type=int, default=24)
    ap.add_argument("--max-concurrency", type=int, default=12)
    ap.add_argument("--max-calls", type=int, default=25)
    ap.add_argument("--ledger", default=None, help="default runs/ledger.jsonl; the cloud machine writes its own")
    ap.add_argument("--reasoning", default=None, choices=["minimal", "low", "medium", "high"],
                    help="reasoning effort (OpenRouter's unified control); default: the provider's")
    ap.add_argument("--max-tokens", type=int, default=8000, help="per model call")
    args = ap.parse_args()
    for arm in args.arms.split(","):
        assert arm in ARMS, arm
    os.environ.setdefault("PYTHONHASHSEED", "0")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
