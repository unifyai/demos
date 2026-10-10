"""The idea: consolidate an agent's activity offline into two hierarchies of
summaries, and read a fixed-size snapshot of them on every turn.

Time hierarchy (calendar blocks, aligned to the history's first Monday):
  day        from the raw records of one calendar day
  week       from 7 day summaries
  4 weeks    from 4 week summaries
  12 weeks   from 3 four-week summaries
  48 weeks   from 4 twelve-week summaries

Count hierarchy (blocks of consecutive interactions):
  10         from 10 raw records
  40         from 4 summaries of 10
  120        from 3 summaries of 40
  480        from 4 summaries of 120

Each summary reads exactly its window: the base reads the raw records in it,
and every level above reads exactly the fan-in summaries of the level below
that tile its window. A summary is scheduled the moment its last input is
scheduled (a day when the clock passes its midnight, a block of ten when its
tenth interaction is logged, any higher block when its last child is), and
written as a background task that awaits its inputs. Nothing here runs on the
conversation's path: `observe` and `advance` only schedule tasks, and
`snapshot` is a lookup.

The snapshot shows, at each level, the latest summaries (as many as make one
summary of the level above, and every summary of the top level) and the last
ten interactions in full, so every scale has a complete view of its recent
past, newest at the finest resolution. Each summary is labelled by the code
with its window, its number of interactions and their range.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .world import START, Interaction, fmt_day, fmt_ts

BUDGET = 32_000  # characters of memory in the answer prompt, for every fixed-budget arm (about 8k tokens)

# (name, how many of the level below make one, word limit)
TIME_LEVELS: list[tuple[str, int, int]] = [("day", 1, 45), ("week", 7, 75), ("4 weeks", 4, 100), ("12 weeks", 3, 130), ("48 weeks", 4, 160)]
COUNT_LEVELS: list[tuple[str, int, int]] = [("10 interactions", 10, 45), ("40 interactions", 4, 75), ("120 interactions", 3, 100), ("480 interactions", 4, 130)]
TIME_SPAN_DAYS = [1, 7, 28, 84, 336]
COUNT_SPAN = [10, 40, 120, 480]
# How many of each level the snapshot shows (None: all of them).
TIME_SHOW: list[int | None] = [7, 4, 3, 4, None]
COUNT_SHOW: list[int | None] = [4, 3, 4, None]
RAW_SHOW = 10
NO_ACTIVITY = "No activity."

SUMMARY_SYSTEM = (
    "You keep the long-term memory of an AI assistant that works for Maya Chen, who runs a small design studio in London. "
    "You write a summary of one window of the assistant's activity, which the assistant will read later when it has to answer "
    "questions about its past work."
)
SUMMARY_USER = """Window: {label}

{inputs_title}:
{inputs}

Write a summary of this window in at most {words} words. Name every project that was worked on and what was done for it, keeping the names of people, amounts, dates and deadlines that may be asked about later, and counts of repeated actions (for example "scheduled 3 meetings"). Mention one-off personal requests only briefly. If some days or periods had no activity, say so. Use only what is in the inputs. Reply with the summary only."""


@dataclass
class Node:
    axis: str  # "time" or "count"
    level: int  # 1 is the base, read from raw records
    block: int  # 0-based index of the block at this level
    start: str  # ISO; time: block start; count: time of its first interaction
    end: str  # ISO; time: block end (exclusive); count: time of its last interaction
    first: int | None  # first and last interaction numbers in the window
    last: int | None
    n: int  # interactions in the window
    inputs: list[Any]  # level 1: interaction numbers; above: ids of the child summaries
    text: str = ""
    called: bool = False  # False when the window was empty and no model was called
    latency_s: float = 0.0
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done_at: float = 0.0  # wall clock when written (for cascade timing)

    @property
    def id(self) -> str:
        return node_id(self.axis, self.level, self.block)

    @property
    def end_dt(self) -> dt.datetime:
        return dt.datetime.fromisoformat(self.end)

    def label(self) -> str:
        span = f"{self.n} interactions (#{self.first:04d}–#{self.last:04d})" if self.n else "no interactions"
        if self.axis == "time":
            a = dt.datetime.fromisoformat(self.start)
            b = self.end_dt - dt.timedelta(days=1)
            name = TIME_LEVELS[self.level - 1][0]
            if self.level == 1:
                return f"Day: {fmt_day(a)} — {span}"
            return f"{name.capitalize()}: {fmt_day(a)} to {fmt_day(b)} — {span}"
        a, b = dt.datetime.fromisoformat(self.start), self.end_dt
        return f"Interactions #{self.first:04d}–#{self.last:04d}: {fmt_ts(a)} to {fmt_ts(b)}"


def node_id(axis: str, level: int, block: int) -> str:
    return f"{axis}{level}:{block}"


Summarise = Callable[[Node, str, int], Awaitable[dict[str, Any]]]


class Hierarchy:
    """Both hierarchies over one agent's log. `summarise(node, prompt, words)`
    returns {"text", "latency_s", "cost", "prompt_tokens", "completion_tokens"}."""

    def __init__(self, summarise: Summarise | None = None, start: dt.datetime = START):
        self.summarise = summarise
        self.start = start
        self.raw: list[Interaction] = []
        self.nodes: dict[str, Node] = {}
        self.tasks: dict[str, asyncio.Future[None]] = {}
        self.days_closed = 0

    # ------------------------------------------------------------ write path
    def observe(self, it: Interaction) -> None:
        """An interaction was logged. Schedules any summaries it completes."""
        self.advance(it.ts)
        assert it.idx == len(self.raw) + 1, "interactions must arrive in order"
        self.raw.append(it)
        if len(self.raw) % COUNT_LEVELS[0][1] == 0:
            self._schedule("count", 1, len(self.raw) // COUNT_LEVELS[0][1] - 1)

    def advance(self, now: dt.datetime) -> None:
        """The clock reached `now`: schedule the summary of every day that has ended."""
        while self.start + dt.timedelta(days=self.days_closed + 1) <= now:
            self._schedule("time", 1, self.days_closed)
            self.days_closed += 1

    async def drain(self) -> None:
        """Wait for every scheduled summary (the offline replay uses this)."""
        while True:
            pending = [t for t in self.tasks.values() if not t.done()]
            if not pending:
                break
            await asyncio.gather(*pending)
        for t in self.tasks.values():
            t.result()  # surface failures

    def _schedule(self, axis: str, level: int, block: int) -> None:
        node = self._make(axis, level, block)
        assert node.id not in self.nodes, node.id
        self.nodes[node.id] = node
        self.tasks[node.id] = asyncio.ensure_future(self._write(node))
        levels = TIME_LEVELS if axis == "time" else COUNT_LEVELS
        if level < len(levels):
            fan = levels[level][1]
            if (block + 1) % fan == 0:  # this node completes a block of the level above
                self._schedule(axis, level + 1, (block + 1) // fan - 1)

    def _make(self, axis: str, level: int, block: int) -> Node:
        if axis == "time":
            span = TIME_SPAN_DAYS[level - 1]
            a = self.start + dt.timedelta(days=block * span)
            b = a + dt.timedelta(days=span)
            if level == 1:
                its = [it for it in self.raw if a <= it.ts < b]
                inputs: list[Any] = [it.idx for it in its]
                first = its[0].idx if its else None
                last = its[-1].idx if its else None
                return Node("time", 1, block, a.isoformat(), b.isoformat(), first, last, len(its), inputs)
            fan = TIME_LEVELS[level - 1][1]
            kids = [self.nodes[node_id("time", level - 1, block * fan + k)] for k in range(fan)]
            firsts = [k.first for k in kids if k.first is not None]
            lasts = [k.last for k in kids if k.last is not None]
            return Node("time", level, block, a.isoformat(), b.isoformat(), min(firsts) if firsts else None,
                        max(lasts) if lasts else None, sum(k.n for k in kids), [k.id for k in kids])
        if level == 1:
            size = COUNT_LEVELS[0][1]
            its = self.raw[block * size:(block + 1) * size]
            assert len(its) == size
            return Node("count", 1, block, its[0].ts.isoformat(), its[-1].ts.isoformat(), its[0].idx, its[-1].idx, size, [it.idx for it in its])
        fan = COUNT_LEVELS[level - 1][1]
        kids = [self.nodes[node_id("count", level - 1, block * fan + k)] for k in range(fan)]
        return Node("count", level, block, kids[0].start, kids[-1].end, kids[0].first, kids[-1].last, sum(k.n for k in kids), [k.id for k in kids])

    async def _write(self, node: Node) -> None:
        if node.level == 1:
            if not node.inputs:
                node.text = NO_ACTIVITY
                return
            title = "Raw records of the interactions in this window"
            body = "\n\n".join(self.raw[i - 1].text for i in node.inputs)
        else:
            await asyncio.gather(*(self.tasks[c] for c in node.inputs))
            kids = [self.nodes[c] for c in node.inputs]
            if node.n == 0:
                node.text = NO_ACTIVITY
                return
            title = f"Summaries of the {len(kids)} shorter windows that make up this window, in order"
            body = "\n\n".join(f"[{k.label()}]\n{k.text}" for k in kids)
        words = (TIME_LEVELS if node.axis == "time" else COUNT_LEVELS)[node.level - 1][2]
        prompt = SUMMARY_USER.format(label=node.label(), inputs_title=title, inputs=body, words=words)
        assert self.summarise is not None
        out = await self.summarise(node, prompt, words)
        node.text = out["text"].strip()
        node.called = True
        node.latency_s = out.get("latency_s", 0.0)
        node.cost = out.get("cost", 0.0)
        node.prompt_tokens = out.get("prompt_tokens", 0)
        node.completion_tokens = out.get("completion_tokens", 0)
        node.done_at = asyncio.get_running_loop().time()

    # ------------------------------------------------------------- read path
    def available(self, axis: str, level: int, now: dt.datetime) -> list[Node]:
        """Summaries at one level whose windows closed before `now`, oldest first."""
        out = [n for n in self.nodes.values() if n.axis == axis and n.level == level and n.text
               and (n.end_dt <= now if axis == "time" else n.end_dt < now)]
        return sorted(out, key=lambda n: n.block)

    def snapshot(self, now: dt.datetime, views: tuple[str, ...] = ("time", "count"), budget: int = BUDGET) -> tuple[str, dict[str, Any]]:
        """The memory block for a turn at `now`, and what it holds."""
        sections: list[tuple[str, list[Node]]] = []
        if "time" in views:
            for lv in range(len(TIME_LEVELS), 0, -1):
                k = TIME_SHOW[lv - 1]
                nodes = self.available("time", lv, now)
                sections.append(("time", nodes if k is None else nodes[-k:]))
        if "count" in views:
            for lv in range(len(COUNT_LEVELS), 0, -1):
                k = COUNT_SHOW[lv - 1]
                nodes = self.available("count", lv, now)
                sections.append(("count", nodes if k is None else nodes[-k:]))
        raw = [it for it in self.raw if it.ts < now][-RAW_SHOW:]
        dropped = 0
        while True:
            text = _render(sections, raw)
            if len(text) <= budget:
                break
            dropped += 1
            if len(raw) > 3:
                raw = raw[1:]
            elif any(nodes for _, nodes in sections):  # drop the oldest summary of the fullest level
                i = max(range(len(sections)), key=lambda j: len(sections[j][1]))
                sections[i] = (sections[i][0], sections[i][1][1:])
            elif raw:
                raw = raw[1:]
            else:
                text = text[:budget]
                break
        info = {"chars": len(text), "dropped": dropped, "summaries": sum(len(s[1]) for s in sections), "raw": len(raw)}
        return text, info


def _render(sections: list[tuple[str, list[Node]]], raw: list[Interaction]) -> str:
    parts: list[str] = []
    time_nodes = [n for axis, ns in sections if axis == "time" for n in ns]
    count_nodes = [n for axis, ns in sections if axis == "count" for n in ns]
    if time_nodes:
        parts.append("SUMMARIES BY CALENDAR PERIOD (longest periods first; oldest first within each length)")
        parts += [f"[{n.label()}]\n{n.text}" for n in time_nodes]
    if count_nodes:
        parts.append("SUMMARIES BY BLOCKS OF CONSECUTIVE INTERACTIONS (largest blocks first; oldest first within each size)")
        parts += [f"[{n.label()}]\n{n.text}" for n in count_nodes]
    if raw:
        parts.append(f"THE LAST {len(raw)} INTERACTIONS, IN FULL")
        parts += [it.text for it in raw]
    return "\n\n".join(parts) if parts else "(no activity yet)"


def save(h: Hierarchy) -> dict[str, Any]:
    return {"nodes": [asdict(n) for n in h.nodes.values()]}


def load(state: dict[str, Any], interactions: list[Interaction], start: dt.datetime = START) -> Hierarchy:
    h = Hierarchy(None, start)
    h.raw = list(interactions)
    for d in state["nodes"]:
        n = Node(**d)
        h.nodes[n.id] = n
    return h


def resume(state: dict[str, Any], interactions: list[Interaction], at: dt.datetime, summarise: Summarise,
           start: dt.datetime = START) -> Hierarchy:
    """The hierarchy as it stood at `at` (raw records before it, and every summary
    whose window had closed), ready to keep consolidating live. Call inside a
    running event loop."""
    h = Hierarchy(summarise, start)
    h.raw = [it for it in interactions if it.ts < at]
    loop = asyncio.get_running_loop()
    for d in state["nodes"]:
        n = Node(**d)
        closed = n.end_dt <= at if n.axis == "time" else (n.last is not None and n.last <= len(h.raw))
        if closed:
            h.nodes[n.id] = n
            fut: asyncio.Future[None] = loop.create_future()
            fut.set_result(None)
            h.tasks[n.id] = fut
    h.days_closed = sum(1 for n in h.nodes.values() if n.axis == "time" and n.level == 1)
    assert h.start + dt.timedelta(days=h.days_closed) <= at < h.start + dt.timedelta(days=h.days_closed + 1)
    return h
