"""Run the memory-consolidation experiment (README, "Re-running").

  uv run python run.py generate                                   # data/, no model calls
  KEY uv run python run.py embed --seeds 0 1 2 3 4                 # record embeddings (retrieval)
  KEY uv run python run.py consolidate --model haiku --seeds 0 1 2 3 4 --what hierarchy rolling mem0
  KEY uv run python run.py answer --model haiku --seeds 0 1 2 3 4 [--checkpoints 26w] [--arms ...]
  KEY uv run python run.py sanity --model haiku                    # every arm on a history that fits the budget
  KEY uv run python run.py live --model haiku                      # turn latency with consolidation in the loop
  uv run python run.py show --model haiku --qid s0-26w-T_7d --arm hierarchy

KEY is ../../bin/with-openrouter-key. Every model call goes through
demos_common.LLM (memcon.client.Client) and is written to runs/ledger.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from memcon import baselines as B
from memcon.answer import messages, parse
from memcon.client import Client
from memcon.hierarchy import BUDGET, SUMMARY_SYSTEM, Hierarchy
from memcon.hierarchy import load as load_hierarchy
from memcon.hierarchy import resume as resume_hierarchy
from memcon.hierarchy import save as save_hierarchy
from memcon.questions import Question, checkpoints, make_questions, sanity_questions, score
from memcon.world import DAYS, START, Interaction, day_of, fmt_ts, generate

HERE = Path(__file__).parent
DATA, RUNS = HERE / "data", HERE / "runs"
LEDGER = RUNS / "ledger.jsonl"
MODELS = {
    "haiku": "anthropic/claude-haiku-5.5",  # today's cheap tier
    "luna": "openai/gpt-6-luna",  # today's cheap tier, second family (stopped; see Amendment 1)
    "sol": "openai/gpt-6.1-sol",  # mid tier (Amendment 2, after Sonnet 5.5's usage-policy filter blocked most questions)
    "sonnet": "anthropic/claude-sonnet-5.5",  # its runs are excluded (runs/excluded-sonnet-content-filter/)
    "opus": "anthropic/claude-opus-5.5",  # top tier, reasoning effort high (Amendment 2)
    "o4mini": "openai/o4-mini",  # era model under Amendment 1; its runs are excluded (runs/excluded-era-models/)
    "flash": "google/gemini-2.5-flash",  # likewise
}
SLUG = {v: k for k, v in MODELS.items()}
# Native reasoning per model, the same for every call of every arm. o4-mini cannot turn it off, so it runs at low effort.
REASONING: dict[str, dict[str, Any]] = {
    "anthropic/claude-haiku-5.5": {"enabled": False}, "openai/gpt-6-luna": {"enabled": False},
    "openai/gpt-6.1-sol": {"effort": "low"},  # its endpoint cannot turn reasoning off
    "anthropic/claude-sonnet-5.5": {"effort": "low"},  # likewise
    "anthropic/claude-opus-5.5": {"effort": "high"},
    "openai/o4-mini": {"effort": "low"}, "google/gemini-2.5-flash": {"enabled": False},
}
# Reasoning tokens count against max_tokens, so a reasoning model's caps get room for them (unused tokens are not billed).
EXTRA_TOKENS = {"openai/gpt-6.1-sol": 16000, "anthropic/claude-sonnet-5.5": 16000, "anthropic/claude-opus-5.5": 16000, "openai/o4-mini": 16000}
NO_REASONING = {"enabled": False}
ARMS = ["hierarchy", "hierarchy_time", "hierarchy_count", "rolling", "recent", "retrieval", "mem0", "query_time", "full", "none"]
# Chosen by rule: of seeds 100-139 cut to 3 or 4 days, those under 30k characters with at least 35
# interactions and 3 details in the last ten; the one with the most sanity questions.
SANITY = {"seed": 113, "days": 4, "time": dt.timedelta(days=3, hours=20, minutes=30)}


# ------------------------------------------------------------------ data

def ledger_path(args: argparse.Namespace) -> Path:
    """runs/ledger.jsonl on the laptop; the cloud machine writes its own file (--ledger), so copying runs back never overwrites one."""
    return HERE / args.ledger if getattr(args, "ledger", None) else LEDGER


def history_name(seed: int | str) -> str:
    return "sanity" if seed == "sanity" else f"s{seed}"


def load_history(name: str) -> tuple[dict[str, Any], list[Interaction], list[Question], dict[str, dt.datetime]]:
    d = json.loads((DATA / f"history_{name}.json").read_text())
    its = [Interaction.from_json(x) for x in d["interactions"]]
    qs = [Question(**q) for q in d["questions"]]
    cps = {k: dt.datetime.fromisoformat(v) for k, v in d["checkpoints"].items()}
    return d, its, qs, cps


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append_jsonl(p: Path, row: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def cmd_generate(args: argparse.Namespace) -> None:
    DATA.mkdir(exist_ok=True)
    for seed in args.seeds:
        h = generate(seed)
        its = h["interactions"]
        cps = checkpoints(seed, its, h["breaks"])
        qs = [q for label, now in cps.items() for q in make_questions(seed, label, now, its)]
        _write_history(f"s{seed}", h, its, cps, qs)
        fam = Counter(q.family for q in qs)
        print(f"s{seed}: {len(its)} interactions, {sum(len(i.text) for i in its):,} chars, {len(h['projects'])} projects, "
              f"breaks {h['breaks']}, checkpoints {', '.join(f'{k} {v:%a %d %b %H:%M}' for k, v in cps.items())}, questions {dict(fam)}")
    h = generate(SANITY["seed"])
    its = [it for it in h["interactions"] if day_of(it.ts) < SANITY["days"]]
    now = START + SANITY["time"]
    qs = sanity_questions(SANITY["seed"], now, its)
    _write_history("sanity", h, its, {"sanity": now}, qs)
    print(f"sanity: {len(its)} interactions, {sum(len(i.text) for i in its):,} chars, {len(qs)} questions")


def _write_history(name: str, h: dict[str, Any], its: list[Interaction], cps: dict[str, dt.datetime], qs: list[Question]) -> None:
    out = {"name": name, "seed": h["seed"], "breaks": h["breaks"], "projects": h["projects"],
           "checkpoints": {k: v.isoformat() for k, v in cps.items()},
           "interactions": [it.to_json() for it in its], "questions": [q.to_json() for q in qs]}
    (DATA / f"history_{name}.json").write_text(json.dumps(out))


# ------------------------------------------------------------- model calls

def chat_fn(client: Client, model: str, meta: dict[str, Any]) -> B.Chat:
    async def chat(tag: str, msgs: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        r = await client.chat(model, msgs, tag=tag, meta=meta, max_tokens=max_tokens + EXTRA_TOKENS.get(model, 0), reasoning=REASONING[model])
        return {"text": r.text, "latency_s": r.latency_s, "cost": r.cost, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens, "cached_tokens": r.cached_tokens}
    return chat


def embed_fn(client: Client, meta: dict[str, Any]) -> B.Embed:
    async def embed(tag: str, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for k in range(0, len(texts), 64):
            out += (await client.embed(texts[k:k + 64], tag=tag, meta=meta)).vectors
        return out
    return embed


# ------------------------------------------------------------ write paths

async def cmd_embed(args: argparse.Namespace) -> None:
    async with Client(ledger_path(args), "memory-consolidation", max_concurrency=args.concurrency) as client:
        async def one(name: str) -> None:
            path = RUNS / "emb" / f"{name}.npy"
            if path.exists():
                print(name, "embeddings exist")
                return
            _, its, _, _ = load_history(name)
            vecs = await embed_fn(client, {"history": name})("retrieval/embed_records", [it.text for it in its])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.array(vecs, dtype=np.float32))
            print(name, "embedded", len(vecs))
        await asyncio.gather(*(one(n) for n in _names(args)))


def _names(args: argparse.Namespace) -> list[str]:
    return [history_name(s) for s in args.seeds]


async def run_hierarchy(client: Client, model: str, name: str, its: list[Interaction], out: Path, end: dt.datetime) -> None:
    path = out / "hierarchy.json"
    if path.exists():
        print(name, "hierarchy exists")
        return
    chat = chat_fn(client, model, {"history": name, "arm": "hierarchy", "slug": SLUG[model]})

    async def summarise(node: Any, prompt: str, words: int) -> dict[str, Any]:
        return await chat("hierarchy/consolidate", [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}], 10 * words + 200)

    h = Hierarchy(summarise)
    t0 = time.time()
    for it in its:
        h.observe(it)
    h.advance(end)
    await h.drain()
    path.write_text(json.dumps(save_hierarchy(h)))
    called = [n for n in h.nodes.values() if n.called]
    print(f"{name} hierarchy: {len(h.nodes)} summaries, {len(called)} model calls, ${sum(n.cost for n in called):.3f}, {time.time() - t0:.0f}s")


async def run_rolling(client: Client, model: str, name: str, its: list[Interaction], out: Path, stop: int | None = None) -> None:
    path = out / "rolling.jsonl"
    versions = read_jsonl(path)
    target = (len(its) if stop is None else stop) // B.ROLLING_EVERY * B.ROLLING_EVERY
    if versions and versions[-1]["last"] >= target:
        print(name, "rolling exists")
        return
    t0 = time.time()
    vs = await B.build_rolling(its, chat_fn(client, model, {"history": name, "arm": "rolling", "slug": SLUG[model]}),
                               on_version=lambda v: append_jsonl(path, v), resume=versions, stop=stop)
    print(f"{name} rolling: {len(vs)} versions, ${sum(v['cost'] for v in vs):.3f}, {time.time() - t0:.0f}s")


async def run_mem0(client: Client, model: str, name: str, its: list[Interaction], cps: dict[str, dt.datetime], out: Path,
                   stop: int | None = None) -> None:
    mdir = out / "mem0"
    mdir.mkdir(parents=True, exist_ok=True)
    meta = {"history": name, "arm": "mem0", "slug": SLUG[model]}
    store = B.Mem0(chat_fn(client, model, meta), embed_fn(client, meta))
    start = 0
    if (mdir / "state.json").exists():
        st = json.loads((mdir / "state.json").read_text())
        vecs = np.load(mdir / "state.npy")
        store.items = {int(m["id"]): m for m in st["items"]}
        store.next_id = st["next_id"]
        for m, v in zip(st["items"], vecs):
            store.vecs[int(m["id"])] = v
            store.toks[int(m["id"])] = Counter(B.tokens(m["text"]))
        start = st["done"]
        log = read_jsonl(mdir / "log.jsonl")
        if any(r["idx"] > start for r in log):  # exchanges logged after the last saved state are redone, so drop them
            (mdir / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log if r["idx"] <= start))
    if start >= (len(its) if stop is None else stop):
        print(name, "mem0 exists")
        return
    pending = {k: v for k, v in cps.items() if not (mdir / f"cp_{k}.json").exists()}

    def persist(done: int) -> None:
        items = store.state()
        np.save(mdir / "state.npy", np.array([store.vecs[m["id"]] for m in items], dtype=np.float32).reshape(len(items), -1))
        (mdir / "state.json").write_text(json.dumps({"items": items, "next_id": store.next_id, "done": done}))

    def on_checkpoint(label: str, s: B.Mem0) -> None:
        items = s.state()
        np.save(mdir / f"cp_{label}.npy", np.array([s.vecs[m["id"]] for m in items], dtype=np.float32).reshape(len(items), -1))
        (mdir / f"cp_{label}.json").write_text(json.dumps(items))

    def on_step(done: int, rec: dict[str, Any], s: B.Mem0) -> None:
        append_jsonl(mdir / "log.jsonl", rec)
        if done % 50 == 0 or done == len(its) or done == stop:
            persist(done)

    t0 = time.time()
    await B.build_mem0(its, store, pending, start=start, on_checkpoint=on_checkpoint, on_step=on_step, stop=stop)
    persist(len(its) if stop is None else min(stop, len(its)))
    print(f"{name} mem0: {len(store.items)} memories, {time.time() - t0:.0f}s")


async def cmd_consolidate(args: argparse.Namespace) -> None:
    model = MODELS[args.model]
    async with Client(ledger_path(args), "memory-consolidation", max_concurrency=args.concurrency) as client:
        jobs = []
        for name in _names(args):
            d, its, _, cps = load_history(name)
            out = RUNS / args.model / name
            out.mkdir(parents=True, exist_ok=True)
            end = max(cps.values()) if name == "sanity" else START + dt.timedelta(days=DAYS)
            # --upto <checkpoint>: every write path stops at the checkpoint. Rolling and mem0 resume on a later run;
            # the hierarchy is then written only up to that checkpoint (enough for every checkpoint before it).
            stop = sum(1 for it in its if it.ts < cps[args.upto]) if getattr(args, "upto", None) else None
            if "hierarchy" in args.what:
                h_its, h_end = (its[:stop], cps[args.upto]) if stop is not None else (its, end)
                jobs.append(run_hierarchy(client, model, name, h_its, out, h_end))
            if "rolling" in args.what:
                jobs.append(run_rolling(client, model, name, its, out, stop))
            if "mem0" in args.what:
                jobs.append(run_mem0(client, model, name, its, cps, out, stop))
        results = await asyncio.gather(*jobs, return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                traceback.print_exception(r)
        if any(isinstance(r, BaseException) for r in results):
            sys.exit(1)


# ------------------------------------------------------------- read paths

def load_context(slug: str, name: str, its: list[Interaction], cps: dict[str, dt.datetime], arms: list[str]) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    base = RUNS / slug / name
    if any(a.startswith("hierarchy") for a in arms):
        ctx["H"] = load_hierarchy(json.loads((base / "hierarchy.json").read_text()), its)
    if "rolling" in arms:
        ctx["rolling"] = read_jsonl(base / "rolling.jsonl")
    if "retrieval" in arms:
        ctx["vecs"] = np.load(RUNS / "emb" / f"{name}.npy")
        assert len(ctx["vecs"]) == len(its)
    if "mem0" in arms:
        ctx["mem0"] = {k: json.loads((base / "mem0" / f"cp_{k}.json").read_text()) for k in cps}
        ctx["mem0_vecs"] = {k: np.load(base / "mem0" / f"cp_{k}.npy") for k in cps}
    return ctx


VIEWS = {"hierarchy": ("time", "count"), "hierarchy_time": ("time",), "hierarchy_count": ("count",)}


async def answer_one(client: Client, model: str, q: Question, arm: str, its: list[Interaction], ctx: dict[str, Any],
                     keep_memory: bool, cold: bool | None = None) -> dict[str, Any]:
    now = dt.datetime.fromisoformat(q.now)
    n_vis = sum(1 for it in its if it.ts < now)
    meta = {"qid": q.qid, "arm": arm, "slug": SLUG[model]}
    path_calls: list[dict[str, Any]] = []
    critical = 0.0
    l0 = time.perf_counter()
    if arm in VIEWS:
        text, info = ctx["H"].snapshot(now, VIEWS[arm])
        rec = B.Recall(text, info)
    elif arm == "rolling":
        rec = B.rolling(ctx["rolling"], its, now)
    elif arm == "recent":
        rec = B.recent(its, now)
    elif arm == "full":
        rec = B.full(its, now)
    elif arm == "none":
        rec = B.none()
    elif arm in ("retrieval", "mem0"):
        e = await client.embed([q.text], tag=f"{arm}/embed_query", meta=meta)
        path_calls.append({"stage": "embed", "latency_s": e.latency_s, "cost": e.cost, "prompt_tokens": e.prompt_tokens, "completion_tokens": 0})
        critical += e.latency_s
        l0 = time.perf_counter()
        qv = np.array(e.vectors[0], dtype=np.float32)
        if arm == "retrieval":
            rec = B.retrieval(its, ctx["vecs"], qv, q.text, now)
        else:
            rec = B.mem0_recall(ctx["mem0"][q.checkpoint], ctx["mem0_vecs"][q.checkpoint], qv, q.text)
    elif arm == "query_time":
        rec = await B.query_time(its, q.text, now, chat_fn(client, model, meta))
        path_calls += rec.calls
        critical += rec.info["critical_s"]
        l0 = time.perf_counter()
    else:
        raise ValueError(arm)
    local_s = time.perf_counter() - l0
    cache = arm == "full" and model.startswith("anthropic/")
    msgs = messages(q, rec.memory, now, n_vis, cache=cache)
    r = await client.chat_stream(model, msgs, tag=f"{arm}/answer", meta=meta, max_tokens=1000 + EXTRA_TOKENS.get(model, 0), reasoning=REASONING[model])
    refused, fallback = False, False
    if not r.text.strip():
        # An empty stream is either a usage-policy refusal (Sonnet and Opus 5.5 send only a "refusal" field) or a broken
        # stream; one non-streamed call tells them apart. Its latency has no first token, so the row's ttft stays empty.
        r2 = await client.chat(model, msgs, tag=f"{arm}/answer_retry", meta=meta, max_tokens=1000 + EXTRA_TOKENS.get(model, 0),
                               reasoning=REASONING[model])
        refused = bool(r2.message.get("refusal")) and not r2.text.strip()
        fallback = True
        r = r2
    pred, parsed = parse(r.text)
    path_s = critical + local_s
    row = {
        "qid": q.qid, "seed": q.seed, "checkpoint": q.checkpoint, "family": q.family, "qtype": q.qtype, "fmt": q.fmt,
        "arm": arm, "model": model, "gold": q.gold, "pred": pred, "parsed": parsed, "score": score(q, pred),
        "memory_chars": len(rec.memory), "info": {k: v for k, v in rec.info.items() if k != "critical_s"},
        "path_s": round(path_s, 4), "local_s": round(local_s, 4),
        "ttft_s": r.first_token_s, "answer_s": round(r.latency_s, 3),
        "reply_first_token_s": None if r.first_token_s is None else round(path_s + r.first_token_s, 3),
        "reply_s": round(path_s + r.latency_s, 3),
        "answer_prompt_tokens": r.prompt_tokens, "answer_cached_tokens": r.cached_tokens, "answer_completion_tokens": r.completion_tokens,
        "answer_cost": r.cost, "path_cost": sum(c.get("cost", 0.0) for c in path_calls),
        "path_prompt_tokens": sum(c.get("prompt_tokens", 0) for c in path_calls), "path_calls": len(path_calls),
        "cold": cold, "reply": r.text, "meta": q.meta, "refused": refused, "stream_fallback": fallback,
    }
    row["cost"] = row["answer_cost"] + row["path_cost"]
    if keep_memory or arm == "query_time":
        row["memory"] = rec.memory
    return row


async def cmd_answer(args: argparse.Namespace) -> None:
    model = MODELS[args.model]
    out = RUNS / args.model / ("answers_sanity.jsonl" if args.seeds == ["sanity"] else args.out or "answers.jsonl")
    # Main answers may be split over answers.jsonl and answers-<part>.jsonl (one file per process); skip what any of them holds.
    main = out.name == "answers.jsonl" or out.name.startswith("answers-")
    files = main_answer_files(args.model) if main else [out]
    done = {(r["qid"], r["arm"]) for f in files for r in read_jsonl(f)}
    errors = RUNS / args.model / "answer_errors.jsonl"
    async with Client(ledger_path(args), "memory-consolidation", max_concurrency=args.concurrency) as client:
        for name in _names(args):
            _, its, qs, cps = load_history(name)
            qs = [q for q in qs if (not args.checkpoints or q.checkpoint in args.checkpoints)
                  and (not args.families or q.family in args.families) and (not args.qids or q.qid in args.qids)]
            ctx = load_context(args.model, name, its, {k: v for k, v in cps.items() if any(q.checkpoint == k for q in qs)}, args.arms)

            async def one(q: Question, arm: str, cold: bool | None = None) -> None:
                try:
                    row = await answer_one(client, model, q, arm, its, ctx, args.keep_memory, cold)
                    append_jsonl(out, row)
                except Exception as e:  # recorded and retried on the next run
                    append_jsonl(errors, {"qid": q.qid, "arm": arm, "error": f"{type(e).__name__}: {e}"[:500], "ts": time.time()})

            todo = [(q, a) for q in qs for a in args.arms if a != "full" and (q.qid, a) not in done]
            t0 = time.time()
            await asyncio.gather(*(one(q, a) for q, a in todo))
            if "full" in args.arms:  # one checkpoint at a time: a cold first call writes the cache, the rest read it
                for label in cps:
                    cq = [q for q in qs if q.checkpoint == label and (q.qid, "full") not in done]
                    if not cq:
                        continue
                    cold = not any((q.qid, "full") in done for q in qs if q.checkpoint == label)
                    await one(cq[0], "full", cold)
                    await asyncio.gather(*(one(q, "full", False) for q in cq[1:]))
            print(f"{name}: answered {len(todo)} + full in {time.time() - t0:.0f}s")


def main_answer_files(slug: str) -> list[Path]:
    return sorted(p for p in (RUNS / slug).glob("answers*.jsonl") if p.name == "answers.jsonl" or p.name.startswith("answers-"))


def cmd_show(args: argparse.Namespace) -> None:
    rows = [r for r in read_jsonl(RUNS / args.model / (args.file or "answers.jsonl")) if r["qid"] == args.qid and (not args.arm or r["arm"] == args.arm)]
    for r in rows:
        print("=" * 100)
        print(f"{r['arm']} | {r['qid']} | score {r['score']} | gold {r['gold']} | pred {r['pred']}")
        print(f"path {r['path_s']}s, ttft {r['ttft_s']}, answer {r['answer_s']}s, cost ${r['cost']:.5f}, info {json.dumps(r['info'])[:400]}")
        if "memory" in r:
            print("-" * 40, "memory", len(r["memory"]), "chars")
            print(r["memory"] if args.full else r["memory"][:3000] + ("\n[...]" if len(r["memory"]) > 3000 else ""))
        print("-" * 40, "reply")
        print(r["reply"])


async def cmd_sanity(args: argparse.Namespace) -> None:
    args.seeds = ["sanity"]
    await cmd_embed(args)
    args.what = ["hierarchy", "rolling", "mem0"]
    await cmd_consolidate(args)
    args.keep_memory = True
    args.checkpoints = args.families = args.qids = None
    await cmd_answer(args)
    rows = read_jsonl(RUNS / args.model / "answers_sanity.jsonl")
    by = Counter()
    tot = Counter()
    for r in rows:
        by[r["arm"]] += r["score"]
        tot[r["arm"]] += 1
    for a in args.arms:
        if tot[a]:
            print(f"{a:16} {by[a] / tot[a]:.2f} over {tot[a]}")



# ----------------------------------------------------------------- live loop

LIVE_SYSTEM = """You are the AI assistant that works for Maya Chen, who runs a small design studio in London. You handle her email, calendar, tasks, documents, quotes, purchases and travel across many projects.

The current time is {now}. The most recent interaction was #{last:04d}.

Your memory of past activity:
<memory>
{memory}
</memory>

Reply to Maya's message in one or two sentences, saying what you will do (the tools run separately)."""
LIVE_MODES = ["off", "background", "inline", "mem0_inline", "off"]


async def cmd_live(args: argparse.Namespace) -> None:
    """Back-to-back turns over three busy days of seed 0. Every turn reads the
    same snapshot (from the offline consolidation) and streams a reply; then the
    turn is logged, and depending on the mode: nothing more (off), the
    hierarchy consolidates in background tasks (background), the hierarchy's
    triggered summaries are awaited before the next turn (inline), or Mem0's
    per-exchange extraction and update are awaited (mem0_inline). A 10 ms
    heartbeat measures event-loop stalls."""
    model = MODELS[args.model]
    _, its, _, _ = load_history("s0")
    state = json.loads((RUNS / args.model / "s0" / "hierarchy.json").read_text())
    reader = load_hierarchy(state, its)
    week = max(range(0, 49, 7), key=lambda w: sum(1 for it in its if w <= day_of(it.ts) < w + 3))
    a, b = START + dt.timedelta(days=week), START + dt.timedelta(days=week + 3)
    turns = [it for it in its if a <= it.ts < b]
    out = RUNS / args.model / "live.jsonl"
    print(f"live: {len(turns)} turns, {a:%a %d %b} to {b - dt.timedelta(days=1):%a %d %b}")
    async with Client(ledger_path(args), "memory-consolidation", max_concurrency=args.concurrency) as client:
        for rep, mode in enumerate(LIVE_MODES):
            meta = {"live": mode, "rep": rep}
            chat = chat_fn(client, model, meta)

            async def summarise(node: Any, prompt: str, words: int) -> dict[str, Any]:
                return await chat("live/consolidate", [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}], 10 * words + 200)

            h = resume_hierarchy(state, its, a, summarise)
            store = B.Mem0(chat, embed_fn(client, meta))
            lags: list[float] = []
            stop = asyncio.Event()

            async def heartbeat() -> None:
                loop = asyncio.get_running_loop()
                while not stop.is_set():
                    t = loop.time()
                    await asyncio.sleep(0.01)
                    lags.append(loop.time() - t - 0.01)

            hb = asyncio.create_task(heartbeat())
            wait = 0.0
            for k, it in enumerate(turns):
                memory, info = reader.snapshot(it.ts)
                request = it.text.split("\n")[1].removeprefix("Maya: ")
                msgs = [{"role": "system", "content": LIVE_SYSTEM.format(now=fmt_ts(it.ts), last=it.idx - 1, memory=memory)},
                        {"role": "user", "content": request}]
                lags.clear()
                r = await client.chat_stream(model, msgs, tag="live/reply", meta={**meta, "turn": k}, max_tokens=150 + EXTRA_TOKENS.get(model, 0),
                                             reasoning=REASONING[model])
                row = {"mode": mode, "rep": rep, "turn": k, "idx": it.idx, "wait_s": round(wait, 3), "ttft_s": r.first_token_s,
                       "reply_s": round(r.latency_s, 3), "turn_first_token_s": None if r.first_token_s is None else round(wait + r.first_token_s, 3),
                       "max_loop_lag_s": round(max(lags), 4) if lags else 0.0, "memory_chars": len(memory)}
                t0 = time.perf_counter()
                if mode in ("background", "inline"):
                    h.observe(it)
                    h.advance(it.ts)
                    if mode == "inline":
                        await h.drain()
                elif mode == "mem0_inline":
                    await store.add(it, turns[max(0, k - B.MEM0_CONTEXT):k])
                wait = time.perf_counter() - t0
                row["log_s"] = round(wait, 4)
                append_jsonl(out, row)
            if mode == "background":
                await h.drain()
            stop.set()
            await hb
            rows = [r for r in read_jsonl(out) if r["rep"] == rep]
            ft = sorted(x["turn_first_token_s"] for x in rows if x["turn_first_token_s"] is not None)
            print(f"{mode:12} median first token {ft[len(ft) // 2]:.2f}s, p90 {ft[int(len(ft) * 0.9)]:.2f}s, "
                  f"mean wait {sum(x['wait_s'] for x in rows) / len(rows):.2f}s, max loop lag {max(x['max_loop_lag_s'] for x in rows):.3f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, model: bool = True) -> None:
        p.add_argument("--seeds", nargs="+", default=[0, 1, 2, 3, 4], type=lambda x: x if x == "sanity" else int(x))
        p.add_argument("--concurrency", type=int, default=12)
        p.add_argument("--ledger", help="ledger file under the concept folder (default runs/ledger.jsonl)")
        if model:
            p.add_argument("--model", choices=list(MODELS), default="haiku")

    p = sub.add_parser("generate")
    common(p, model=False)
    p = sub.add_parser("embed")
    common(p, model=False)
    p = sub.add_parser("consolidate")
    common(p)
    p.add_argument("--what", nargs="+", default=["hierarchy", "rolling", "mem0"])
    p.add_argument("--upto", help="stop rolling and mem0 at this checkpoint (they resume on the next run)")
    for nm in ("answer", "sanity"):
        p = sub.add_parser(nm)
        common(p)
        p.add_argument("--arms", nargs="+", default=ARMS)
        p.add_argument("--checkpoints", nargs="+")
        p.add_argument("--families", nargs="+")
        p.add_argument("--qids", nargs="+")
        p.add_argument("--keep-memory", action="store_true")
        p.add_argument("--out")
    p = sub.add_parser("live")
    common(p)
    p = sub.add_parser("show")
    p.add_argument("--model", choices=list(MODELS), default="haiku")
    p.add_argument("--qid", required=True)
    p.add_argument("--arm")
    p.add_argument("--file")
    p.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "show":
        cmd_show(args)
    else:
        asyncio.run({"embed": cmd_embed, "consolidate": cmd_consolidate, "answer": cmd_answer, "sanity": cmd_sanity,
                     "live": cmd_live}[args.cmd](args))


if __name__ == "__main__":
    main()
