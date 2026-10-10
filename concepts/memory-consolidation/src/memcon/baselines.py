"""The alternatives a competent engineer would build instead.

Fixed budget, nothing on the conversation's path but a lookup or a search:
  recent      the newest raw records that fit the budget
  rolling     one running summary, rewritten offline every 10 interactions from
              itself and the new records, plus the raw records since
  retrieval   hybrid search (BM25 and embeddings, fused by rank) over the raw
              records, top results that fit the budget, shown in time order
  mem0        Mem0-style memory: facts extracted from each exchange as it
              happens and merged into a store (add, update, delete), then
              hybrid search over the facts at question time

Work moved onto the question:
  query_time  a router call reads the question's period or count, the records
              in it are selected, and if they exceed the budget they are
              summarised for the question chunk by chunk, in parallel

References:
  full        every raw record in one long context (prompt-cached)
  none        no memory at all
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np

from .hierarchy import BUDGET, SUMMARY_SYSTEM
from .retrieve import BM25, fill, normalise, rrf_order, tokens
from .world import START, Interaction, fmt_ts

SEP = "\n\n"


@dataclass
class Recall:
    memory: str
    info: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)  # model calls made on the path before the answer
    path_s: float = 0.0  # wall time of the read path before the answer call


# A chat call: (tag, messages, max_tokens) -> {"text", "latency_s", "cost", "prompt_tokens", "completion_tokens", "cached_tokens"}
Chat = Callable[[str, list[dict[str, Any]], int], Awaitable[dict[str, Any]]]
# An embedding call: (tag, texts) -> vectors
Embed = Callable[[str, list[str]], Awaitable[list[list[float]]]]


def visible(its: list[Interaction], now: dt.datetime) -> list[Interaction]:
    return [it for it in its if it.ts < now]


# ------------------------------------------------------------------ recent

def recent(its: list[Interaction], now: dt.datetime, budget: int = BUDGET) -> Recall:
    vis = visible(its, now)
    head = "THE MOST RECENT INTERACTIONS, IN FULL (oldest first)"
    room = budget - len(head) - len(SEP)
    chosen: list[Interaction] = []
    used = 0
    for it in reversed(vis):
        need = len(it.text) + (len(SEP) if chosen else 0)
        if used + need > room:
            break
        chosen.append(it)
        used += need
    chosen.reverse()
    text = head + SEP + SEP.join(it.text for it in chosen) if chosen else "(no activity yet)"
    return Recall(text, {"chars": len(text), "records": len(chosen)})


# ----------------------------------------------------------------- rolling

ROLLING_WORDS = 3000
ROLLING_EVERY = 10
ROLLING_USER = """Current running summary ({covers}):
{summary}

New interactions #{a:04d}–#{b:04d}, raw records:
{records}

Rewrite the running summary so that it also covers the new interactions, in at most {words} words. Keep a timeline: for each project, when it was worked on and what was done, keeping the names of people, amounts, dates and deadlines that may be asked about later, and counts of repeated actions (for example "scheduled 3 meetings"). Mention one-off personal requests only briefly. Note periods with no activity. When space runs short, compress older periods before recent ones. Use only what is in the summary and the records. Reply with the summary only."""


async def build_rolling(its: list[Interaction], chat: Chat, every: int = ROLLING_EVERY, words: int = ROLLING_WORDS,
                        on_version: Callable[[dict[str, Any]], None] | None = None,
                        resume: list[dict[str, Any]] | None = None, stop: int | None = None) -> list[dict[str, Any]]:
    """Versions of the running summary, each covering interactions 1..last.
    One update per block of `every` interactions, each from the previous
    version and exactly that block's raw records."""
    versions: list[dict[str, Any]] = list(resume or [])
    summary = versions[-1]["text"] if versions else ""
    done = versions[-1]["last"] if versions else 0
    for b in range(done // every, (len(its) if stop is None else min(stop, len(its))) // every):
        block = its[b * every:(b + 1) * every]
        assert block[0].idx == b * every + 1 and block[-1].idx == (b + 1) * every
        covers = (f"covers all activity from {fmt_ts(START)} up to interaction #{done:04d}" if done else
                  "empty: this is the start of the log")
        prompt = ROLLING_USER.format(covers=covers, summary=summary or "(empty)", a=block[0].idx, b=block[-1].idx,
                                     records=SEP.join(it.text for it in block), words=words)
        out = await chat("rolling/consolidate", [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}], words * 5)
        summary = out["text"].strip()
        done = block[-1].idx
        v = {"last": done, "end": block[-1].ts.isoformat(), "text": summary, **{k: out[k] for k in ("latency_s", "cost", "prompt_tokens", "completion_tokens")}}
        versions.append(v)
        if on_version:
            on_version(v)
    return versions


def rolling(versions: list[dict[str, Any]], its: list[Interaction], now: dt.datetime, budget: int = BUDGET) -> Recall:
    vis = visible(its, now)
    usable = [v for v in versions if v["last"] <= len(vis)]
    v = usable[-1] if usable else None
    last = v["last"] if v else 0
    tail = vis[last:]
    parts = []
    if v:
        parts.append(f"RUNNING SUMMARY OF ALL ACTIVITY UP TO INTERACTION #{last:04d} ({fmt_ts(dt.datetime.fromisoformat(v['end']))})\n{v['text']}")
    dropped = 0
    while True:
        body = list(parts)
        if tail:
            body.append("INTERACTIONS SINCE THEN, IN FULL" + SEP + SEP.join(it.text for it in tail))
        text = SEP.join(body) if body else "(no activity yet)"
        if len(text) <= budget or not tail:
            break
        tail = tail[1:]
        dropped += 1
    if len(text) > budget:  # the summary alone overflows: cut its end (has not happened in runs; counted)
        text = text[:budget]
        dropped += 1
    return Recall(text, {"chars": len(text), "version_last": last, "tail": len(tail), "dropped": dropped})


# --------------------------------------------------------------- retrieval

def retrieval(its: list[Interaction], vecs: np.ndarray, q_vec: np.ndarray, question: str, now: dt.datetime,
              budget: int = BUDGET) -> Recall:
    vis = visible(its, now)
    if not vis:
        return Recall("(no activity yet)", {"chars": 0, "records": 0})
    docs = [it.text for it in vis]
    order = rrf_order([BM25(docs).scores(question), normalise(vecs[: len(vis)]) @ normalise(q_vec[None, :])[0]])
    head = "INTERACTIONS RETRIEVED AS RELEVANT TO THE QUESTION (shown in time order)"
    chosen = sorted(fill(order, [len(d) for d in docs], budget - len(head) - len(SEP), len(SEP)))
    text = head + SEP + SEP.join(docs[i] for i in chosen)
    return Recall(text, {"chars": len(text), "records": len(chosen)})


# -------------------------------------------------------------------- mem0

EXTRACT_SYSTEM = ("You extract memories from the log of an AI assistant that works for Maya Chen, who runs a small design studio in London. "
                  "The memories are kept in a store and retrieved later to answer questions about the assistant's past work.")
EXTRACT_USER = """Earlier interactions, for context only:
{context}

New interaction:
{record}

Extract the facts from the new interaction that are worth remembering: what was done and for which project, with the names of people and organisations, amounts, dates, deadlines and decisions. Write each fact as one short, self-contained sentence that starts with the date of the interaction (YYYY-MM-DD) and names the project. Skip small talk. Reply with JSON only: {{"facts": ["...", "..."]}} (an empty list if nothing is worth remembering)."""
UPDATE_SYSTEM = ("You manage the memory store of an AI assistant that works for Maya Chen. You compare newly extracted facts with related "
                 "memories already in the store and decide how to update the store.")
UPDATE_USER = """Related memories already in the store:
{memories}

New facts:
{facts}

For each new fact choose one operation:
- ADD: it is new information. Give its text.
- UPDATE: it changes or adds to an existing memory about the same thing (for example a revised amount or a moved date). Give the existing memory's id and the full new text that replaces it.
- DELETE: it shows that an existing memory is no longer true and nothing replaces it. Give the existing memory's id.
- NONE: the store already says this.
Reply with JSON only: {{"operations": [{{"op": "ADD", "text": "..."}}, {{"op": "UPDATE", "id": 3, "text": "..."}}, {{"op": "DELETE", "id": 5}}, {{"op": "NONE"}}]}}"""
MEM0_CONTEXT = 2  # earlier interactions shown to the extractor
MEM0_NEIGHBOURS = 5  # related memories fetched per new fact


def parse_json(text: str) -> Any:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object")
    return json.loads(t[a:b + 1])


class Mem0:
    """A Mem0-style store: extract facts from each exchange, then add, update or
    delete memories against the related ones already stored."""

    def __init__(self, chat: Chat, embed: Embed):
        self.chat, self.embed = chat, embed
        self.items: dict[int, dict[str, Any]] = {}  # id -> {"id", "text", "created", "updated"}
        self.vecs: dict[int, np.ndarray] = {}
        self.toks: dict[int, Counter[str]] = {}
        self.next_id = 1
        self.log: list[dict[str, Any]] = []  # per interaction: calls, latency, cost, ops

    def _related(self, fact: str, vec: np.ndarray) -> list[int]:
        ids = list(self.items)
        if not ids:
            return []
        bm = BM25(counts=[self.toks[i] for i in ids])
        emb = np.stack([self.vecs[i] for i in ids]) @ vec
        order = rrf_order([bm.scores(fact), emb])
        return [ids[j] for j in order[:MEM0_NEIGHBOURS]]

    async def add(self, it: Interaction, context: list[Interaction]) -> dict[str, Any]:
        t0 = time.time()
        calls: list[dict[str, Any]] = []
        ctx = SEP.join(c.text for c in context) or "(none)"
        out = await self.chat("mem0/extract", [{"role": "system", "content": EXTRACT_SYSTEM},
                                               {"role": "user", "content": EXTRACT_USER.format(context=ctx, record=it.text)}], 2000)
        calls.append(out)
        try:
            facts = [str(f).strip() for f in parse_json(out["text"]).get("facts", []) if str(f).strip()]
        except (ValueError, json.JSONDecodeError, AttributeError):
            facts = []
        ops_applied: list[dict[str, Any]] = []
        if facts:
            fvecs = normalise(np.array(await self.embed("mem0/embed", facts)))
            shown: list[int] = []
            for f, v in zip(facts, fvecs):
                for i in self._related(f, v):
                    if i not in shown:
                        shown.append(i)
            local = {k + 1: i for k, i in enumerate(shown)}  # small ids in the prompt, as Mem0 does
            mem_lines = "\n".join(f"{k}: {self.items[i]['text']}" for k, i in local.items()) or "(none)"
            fact_lines = "\n".join(f"{k + 1}. {f}" for k, f in enumerate(facts))
            out2 = await self.chat("mem0/update", [{"role": "system", "content": UPDATE_SYSTEM},
                                                   {"role": "user", "content": UPDATE_USER.format(memories=mem_lines, facts=fact_lines)}], 3000)
            calls.append(out2)
            try:
                ops = parse_json(out2["text"]).get("operations", [])
            except (ValueError, json.JSONDecodeError, AttributeError):
                ops = [{"op": "ADD", "text": f} for f in facts]  # an unparseable decision keeps the facts
            day = it.ts.date().isoformat()
            to_embed: list[tuple[int, str]] = []
            fact_vec = {f: v for f, v in zip(facts, fvecs)}
            for op in ops if isinstance(ops, list) else []:
                kind = str(op.get("op", "")).upper() if isinstance(op, dict) else ""
                if kind == "ADD" and op.get("text"):
                    i = self.next_id
                    self.next_id += 1
                    self.items[i] = {"id": i, "text": str(op["text"]), "created": day, "updated": day}
                    to_embed.append((i, str(op["text"])))
                elif kind == "UPDATE" and op.get("text") and _as_int(op.get("id")) in local:
                    i = local[_as_int(op.get("id"))]
                    if i in self.items:
                        self.items[i] = {**self.items[i], "text": str(op["text"]), "updated": day}
                        to_embed.append((i, str(op["text"])))
                elif kind == "DELETE" and _as_int(op.get("id")) in local:
                    i = local[_as_int(op.get("id"))]
                    self.items.pop(i, None)
                    self.vecs.pop(i, None)
                    self.toks.pop(i, None)
                else:
                    continue
                ops_applied.append({"op": kind, **({"text": op.get("text")} if op.get("text") else {})})
            need = [(i, t) for i, t in to_embed if t not in fact_vec]
            new_vecs = normalise(np.array(await self.embed("mem0/embed", [t for _, t in need]))) if need else []
            got = {i: v for (i, _), v in zip(need, new_vecs)}
            for i, t in to_embed:
                if i in self.items:
                    self.vecs[i] = got.get(i, fact_vec.get(t))
                    self.toks[i] = Counter(tokens(t))
        rec = {"idx": it.idx, "facts": len(facts), "ops": [o["op"] for o in ops_applied], "wall_s": time.time() - t0,
               "calls": [{k: c[k] for k in ("latency_s", "cost", "prompt_tokens", "completion_tokens")} for c in calls]}
        self.log.append(rec)
        return rec

    def state(self) -> list[dict[str, Any]]:
        return [dict(v) for v in sorted(self.items.values(), key=lambda x: x["id"])]


def _as_int(x: Any) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


async def build_mem0(its: list[Interaction], store: Mem0, checkpoints: dict[str, dt.datetime], start: int = 0,
                     on_checkpoint: Callable[[str, Mem0], None] | None = None,
                     on_step: Callable[[int, dict[str, Any], Mem0], None] | None = None, stop: int | None = None) -> None:
    """Replay the log through the store, one exchange at a time, from
    interaction number start + 1. Each checkpoint sees the store as it was
    before the first interaction at or after it."""
    pending = sorted((t, label) for label, t in checkpoints.items())
    end = len(its) if stop is None else min(stop, len(its))
    for k in range(start, end):
        it = its[k]
        while pending and it.ts >= pending[0][0]:
            if on_checkpoint:
                on_checkpoint(pending.pop(0)[1], store)
            else:
                pending.pop(0)
        rec = await store.add(it, its[max(0, k - MEM0_CONTEXT):k])
        if on_step:
            on_step(k + 1, rec, store)
    for t, label in pending:  # checkpoints after the last interaction processed
        if on_checkpoint and (end == len(its) or its[end].ts >= t):
            on_checkpoint(label, store)


def mem0_recall(memories: list[dict[str, Any]], vecs: np.ndarray, q_vec: np.ndarray, question: str, budget: int = BUDGET) -> Recall:
    if not memories:
        return Recall("(no stored memories)", {"chars": 0, "memories": 0})
    lines = [f"- [{m['updated']}] {m['text']}" for m in memories]
    order = rrf_order([BM25([m["text"] for m in memories]).scores(question), normalise(vecs) @ normalise(q_vec[None, :])[0]])
    head = "STORED MEMORIES RETRIEVED AS RELEVANT TO THE QUESTION (in date order)"
    chosen = fill(order, [len(x) for x in lines], budget - len(head) - 1, 1)
    chosen.sort(key=lambda i: (memories[i]["updated"], memories[i]["id"]))
    text = head + "\n" + "\n".join(lines[i] for i in chosen)
    return Recall(text, {"chars": len(text), "memories": len(chosen), "store": len(memories)})


# --------------------------------------------------------------- query_time

ROUTE_SYSTEM = "You route questions about an AI assistant's past activity to the part of its activity log that can answer them."
ROUTE_USER = """The current time is {now}. The most recent interaction was #{last:04d}. The log starts on {start}.

Question: {question}

Which part of the log is needed to answer this question? Reply with JSON only, one of:
{{"kind": "period", "start": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM"}}  for a period of time (the end is exclusive)
{{"kind": "last", "n": <number>}}  for the most recent n interactions
{{"kind": "all"}}  when the question is not limited to a period or to the most recent interactions"""
MAP_USER = """Raw records of the AI assistant's interactions with Maya, #{a:04d}–#{b:04d} ({start} to {end}):

{records}

Question: {question}

Extract from the records above everything needed to answer the question. Be exhaustive for lists and counts (for example, list every project or every matching action with its date), and keep names, dates and amounts exactly. At most 250 words. If nothing in these records is relevant, reply "Nothing relevant." """
MERGE_USER = """Notes extracted for a question from consecutive parts of an AI assistant's activity log:

{notes}

Question: {question}

Merge the notes above into one set of notes that keeps everything needed to answer the question: every listed item, count and date. At most 400 words."""
MAP_CHUNK = 200_000  # characters of raw records per map call (about 70k tokens)


def route_selection(vis: list[Interaction], route: dict[str, Any]) -> list[Interaction]:
    kind = route.get("kind")
    if kind == "last":
        n = max(1, int(route.get("n") or 1))
        return vis[-n:]
    if kind == "period":
        try:
            a = dt.datetime.fromisoformat(str(route["start"]).replace(" ", "T"))
            b = dt.datetime.fromisoformat(str(route["end"]).replace(" ", "T"))
        except (KeyError, ValueError):
            return vis
        return [it for it in vis if a <= it.ts < b]
    return vis


def chunks(its: list[Interaction], size: int = MAP_CHUNK) -> list[list[Interaction]]:
    out: list[list[Interaction]] = []
    cur: list[Interaction] = []
    used = 0
    for it in its:
        if cur and used + len(it.text) + len(SEP) > size:
            out.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += len(it.text) + len(SEP)
    if cur:
        out.append(cur)
    return out


async def query_time(its: list[Interaction], question: str, now: dt.datetime, chat: Chat, budget: int = BUDGET) -> Recall:
    """Route, select, and summarise the selection for the question if it is
    over the budget. info["critical_s"] is the path's latency with the map calls
    in parallel: route + slowest map + slowest merge of each round."""
    t0 = time.time()
    vis = visible(its, now)
    calls: list[dict[str, Any]] = []
    out = await chat("query_time/route", [{"role": "system", "content": ROUTE_SYSTEM},
                                          {"role": "user", "content": ROUTE_USER.format(now=fmt_ts(now), last=len(vis), start=fmt_ts(START), question=question)}], 300)
    calls.append({**out, "stage": "route"})
    critical = out["latency_s"]
    try:
        route = parse_json(out["text"])
    except (ValueError, json.JSONDecodeError):
        route = {"kind": "all", "unparsed": True}
    sel = route_selection(vis, route)
    info: dict[str, Any] = {"route": route, "selected": len(sel), "selected_chars": sum(len(it.text) for it in sel), "map_calls": 0}
    head_raw = "INTERACTIONS IN THE PART OF THE LOG THE QUESTION ASKS ABOUT, IN FULL"
    if not sel:
        text = "(no interactions in the part of the log the question asks about)"
    elif len(head_raw) + len(SEP) + sum(len(it.text) + len(SEP) for it in sel) <= budget:
        text = head_raw + SEP + SEP.join(it.text for it in sel)
    else:
        parts = chunks(sel)
        info["map_calls"] = len(parts)

        async def one(part: list[Interaction]) -> tuple[str, dict[str, Any]]:
            o = await chat("query_time/map", [{"role": "user", "content": MAP_USER.format(
                a=part[0].idx, b=part[-1].idx, start=fmt_ts(part[0].ts), end=fmt_ts(part[-1].ts), question=question,
                records=SEP.join(x.text for x in part))}], 2000)
            note = f"[Notes from interactions #{part[0].idx:04d}–#{part[-1].idx:04d}, {fmt_ts(part[0].ts)} to {fmt_ts(part[-1].ts)}]\n{o['text'].strip()}"
            return note, o

        done = await asyncio.gather(*(one(p) for p in parts))
        notes = [n for n, _ in done]
        calls += [{**o, "stage": "map"} for _, o in done]
        critical += max(o["latency_s"] for _, o in done)
        head = "NOTES EXTRACTED FOR THIS QUESTION FROM THE PART OF THE LOG IT ASKS ABOUT"
        rounds = 0
        while len(head) + len(SEP) + sum(len(n) + len(SEP) for n in notes) > budget and len(notes) > 1:
            pairs = [notes[k:k + 2] for k in range(0, len(notes), 2)]

            async def merge(pair: list[str]) -> tuple[str, dict[str, Any] | None]:
                if len(pair) == 1:
                    return pair[0], None
                o = await chat("query_time/merge", [{"role": "user", "content": MERGE_USER.format(question=question, notes=SEP.join(pair))}], 3000)
                return o["text"].strip(), o

            merged = await asyncio.gather(*(merge(p) for p in pairs))
            notes = [n for n, _ in merged]
            outs = [o for _, o in merged if o]
            calls += [{**o, "stage": "merge"} for o in outs]
            critical += max((o["latency_s"] for o in outs), default=0.0)
            rounds += 1
        info["merge_rounds"] = rounds
        text = head + SEP + SEP.join(notes)
        if len(text) > budget:
            text = text[:budget]
            info["cut"] = True
    info["chars"] = len(text)
    info["critical_s"] = critical
    return Recall(text, info, calls, time.time() - t0)


# -------------------------------------------------------------------- full

def full(its: list[Interaction], now: dt.datetime) -> Recall:
    vis = visible(its, now)
    text = "ALL INTERACTIONS SO FAR, IN FULL" + SEP + SEP.join(it.text for it in vis) if vis else "(no activity yet)"
    return Recall(text, {"chars": len(text), "records": len(vis)})


def none() -> Recall:
    return Recall("(none: you have no record of past activity)", {"chars": 0})
