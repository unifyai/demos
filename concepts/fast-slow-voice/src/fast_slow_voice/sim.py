"""One call, simulated in virtual time.

The simulator owns the floor (who is speaking), Sam's scripted turns and their
triggers, the queue of agent speech, tools, and the model calls. An arm decides
what to call and what to say; the simulator decides when things happen.

Model calls are real. A call started at virtual time t0 is sent at once; its
virtual first-token and finish times come from the timing model and the tokens
the call used. Events are processed in time order; before processing an event
at time t, every call started at or before t is awaited, so a call's results
can never be missed by an event that comes after them.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .keys import answer_in, is_holding
from .script import Script, Turn
from .timing import Timing, heard_prefix, speech_seconds
from .world import Tools, World, tool_latency

# Event priorities at equal times: speech ending first, then Sam ending, results,
# then the agent hearing Sam, wakes, and Sam's checks last.
P_AGENT_END, P_CUT, P_PERSON_END, P_TOOL, P_FIRST, P_DONE, P_HEARD, P_READY, P_WAKE, P_CHECK, P_DEADLINE = range(11)


@dataclass
class ModelCfg:
    name: str
    reasoning: dict | None = None
    max_tokens: int = 1000

    @property
    def anthropic(self) -> bool:
        return self.name.startswith("anthropic/")


@dataclass
class Utterance:
    id: int
    text: str
    source: str  # who wrote it: single, fast, slow, filler, replay
    kind: str  # voice: dropped if Sam takes the floor first; verbatim: waits for the floor
    t_queued: float
    t_ready: float
    call_id: int | None = None
    t_start: float | None = None
    t_end: float | None = None
    heard: str = ""
    rest: str = ""
    n_heard_tokens: int = 0
    cut: bool = False
    dropped: bool = False
    t_dropped: float | None = None
    label: str | None = None  # the fast model's label in slow_speaks
    call_ended: bool = False  # still being spoken when the call ended

    @property
    def started(self) -> bool:
        return self.t_start is not None

    @property
    def ended(self) -> bool:
        return self.t_end is not None


@dataclass
class PersonTurn:
    idx: int
    turn: Turn
    t_start: float
    t_end: float
    t_heard: float
    barged: bool = False
    cut_utt: int | None = None  # the agent utterance this turn cut into


@dataclass
class ToolCall:
    id: int
    role: str
    name: str
    args: dict
    t_call: float
    t_done: float
    call_id: int
    result: Any = None
    done: bool = False


@dataclass
class Note:
    id: int
    t: float
    text: str
    call_id: int


@dataclass
class Call:
    id: int
    role: str
    model: str
    t0: float
    purpose: str
    meta: dict = field(default_factory=dict)
    task: asyncio.Task | None = None
    resolved: bool = False
    cancelled: bool = False
    t_cancel: float | None = None
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    t_first: float | None = None
    t_done: float | None = None
    prefix_tokens: int = 0


class CallCapExceeded(RuntimeError):
    pass


_SILENT_WORDS = {"", "silent", "silence", "no text", "no reply", "no response", "no output", "nothing", "nothing to say",
                 "nothing to add", "nothing new", "no note", "none", "n/a", "na", "no text needed", "no reply needed"}
_LABEL_AT_LINE = re.compile(r"(?:^|\n)\s*(?:Ava|Assistant)\s*(?:\([^)]*\))?\s*:\s*", re.I)


_SILENT_MARK = re.compile(r"\[[^\[\]\n]{0,12}ilent\]", re.I)  # [silent], also as GPT-6 Luna garbled it: [ѕilent], [ლsilent]


def is_placeholder(text: str) -> bool:
    """Text that stands for saying nothing: [silent], "(no text)", "<br>", "...", ""."""
    t = (text or "").strip()
    if _SILENT_MARK.search(t):  # the model chose silence; text around the marker was always narration
        return True
    t = re.sub(r"<[^>]*>", " ", t)  # html-ish tags such as <br>
    t = re.sub(r"[\[\]()*_.!\-–—:;\"'`]", " ", t).strip().lower()
    t = " ".join(t.split())
    return t in _SILENT_WORDS


def clean_spoken(text: str) -> str:
    """What a speaking model's reply puts on the line: a leading speaker label is
    dropped (and any commentary before one), and a trailing [silent] marker."""
    t = (text or "").strip()
    labels = list(_LABEL_AT_LINE.finditer(t))
    if labels:
        t = t[labels[0].end():]
    t = re.sub(r"\s*\[silent\]\s*$", "", t, flags=re.I)
    return " ".join(t.split())


class Sim:
    def __init__(self, script: Script, world: World, arm: Any, timing: Timing, llm: Any, *,
                 run_seed: int, tag: str, meta: dict | None = None, max_calls: int = 160,
                 t_limit: float = 900.0, tail: float = 20.0):
        self.script, self.world, self.arm, self.timing, self.llm = script, world, arm, timing, llm
        self.tools = Tools(world)
        self.run_seed, self.tag = run_seed, tag
        self.meta = meta or {}
        self.max_calls, self.t_limit, self.tail = max_calls, t_limit, tail
        self.now = 0.0
        self._heap: list = []
        self._seq = itertools.count()
        self._ids = itertools.count(1)
        # Sam
        self.turn_idx = 0
        self.person_cur: PersonTurn | None = None
        self.persons: list[PersonTurn] = []
        self.floor_free_at = 0.0
        self._check_version = 0
        # agent speech
        self.agent_cur: Utterance | None = None
        self.queue: list[Utterance] = []
        self.utts: list[Utterance] = []
        self.last_agent_end: float | None = None
        # work
        self.calls: dict[int, Call] = {}
        self.tool_calls: list[ToolCall] = []
        self.notes: list[Note] = []
        self.events_log: list[dict] = []
        self.flags: list[str] = []
        self.finished = False

    # ------------------------------------------------------------------ events
    def push(self, t: float, prio: int, kind: str, payload: Any = None) -> None:
        heapq.heappush(self._heap, (t, prio, next(self._seq), kind, payload))

    def wake_at(self, t: float, tag: Any) -> None:
        self.push(max(t, self.now), P_WAKE, "wake", tag)

    def log(self, kind: str, **kw: Any) -> None:
        self.events_log.append({"t": round(self.now, 3), "kind": kind, **kw})

    # ------------------------------------------------------------------ calls
    def start_call(self, role: str, cfg: ModelCfg, messages: list[dict], tools: list[dict] | None,
                   purpose: str, meta: dict | None = None, prefix_tokens: int = 0) -> Call:
        if len(self.calls) >= self.max_calls:
            raise CallCapExceeded(f"more than {self.max_calls} model calls")
        c = Call(next(self._ids), role, cfg.name, self.now, purpose, dict(meta or {}), prefix_tokens=prefix_tokens)
        self.calls[c.id] = c
        c.task = asyncio.ensure_future(self._api(c, cfg, messages, tools))
        self.log("call_start", call=c.id, role=role, purpose=purpose)
        return c

    async def _api(self, c: Call, cfg: ModelCfg, messages: list[dict], tools: list[dict] | None):
        kw: dict[str, Any] = {"max_tokens": cfg.max_tokens}
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        if cfg.reasoning is not None:
            kw["reasoning"] = cfg.reasoning
        meta = {**self.meta, "role": c.role, "purpose": c.purpose, "call": c.id, "t0": round(c.t0, 3)}
        return await self.llm.chat(cfg.name, messages, tag=self.tag, meta=meta, **kw)

    async def _resolve(self, c: Call) -> None:
        reply = await c.task
        c.resolved = True
        c.text = (reply.text or "").strip()
        c.tool_calls = []
        for tc in reply.tool_calls:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {"_unparsed": raw}
            c.tool_calls.append({"name": fn.get("name", ""), "args": args if isinstance(args, dict) else {"_value": args}})
        c.prompt_tokens, c.completion_tokens = reply.prompt_tokens, reply.completion_tokens
        c.reasoning_tokens, c.cached_tokens = reply.reasoning_tokens, reply.cached_tokens
        c.cost, c.latency_s = reply.cost, reply.latency_s
        first, done = self.timing.call_times(c.model, c.prompt_tokens, c.completion_tokens, c.reasoning_tokens,
                                             c.prefix_tokens if c.text else 0)
        c.t_first = c.t0 + first if c.text else None
        c.t_done = c.t0 + done
        if c.cancelled:
            return
        if c.text:
            self.push(c.t_first, P_FIRST, "call_first", c.id)
        self.push(c.t_done, P_DONE, "call_done", c.id)

    def first_after(self, c: Call, t: float) -> bool:
        """Whether nothing of the call has come out by time t (so cancelling loses nothing said)."""
        if not c.resolved:
            return True
        return (c.t_first if c.t_first is not None else c.t_done) > t

    def cancel_call(self, c: Call) -> None:
        if c.cancelled:
            return
        c.cancelled = True
        c.t_cancel = self.now
        for u in list(self.queue):
            if u.call_id == c.id:
                self._drop(u)
        self.log("call_cancel", call=c.id, role=c.role)

    # ------------------------------------------------------------------ speech
    def say(self, text: str, source: str, kind: str, call_id: int | None = None, ready: float | None = None,
            label: str | None = None) -> Utterance | None:
        if is_placeholder(text):
            return None
        text = clean_spoken(text)
        if is_placeholder(text):
            return None
        if any(q.text == text for q in self.queue):  # the same words already waiting to be said
            self.log("duplicate_skipped", text=text[:80])
            return None
        u = Utterance(next(self._ids), text, source, kind, self.now, max(self.now, ready if ready is not None else self.now),
                      call_id, label=label)
        self.queue.append(u)
        self.utts.append(u)
        return u

    def _drop(self, u: Utterance) -> None:
        if u in self.queue:
            self.queue.remove(u)
            u.dropped = True
            u.t_dropped = self.now

    def drop_voice_queue(self) -> None:
        for u in list(self.queue):
            if u.kind == "voice":
                self._drop(u)

    def _try_speak(self) -> None:
        if self.agent_cur is not None or self.person_cur is not None or not self.queue:
            return
        if self.now < self.floor_free_at:
            self.push(self.floor_free_at, P_READY, "ready")
            return
        ready = [u for u in self.queue if u.t_ready <= self.now + 1e-9]
        if not ready:
            self.push(min(u.t_ready for u in self.queue), P_READY, "ready")
            return
        u = ready[0]
        self.queue.remove(u)
        u.t_start = self.now
        self.agent_cur = u
        self.push(self.now + speech_seconds(u.text, self.timing.agent_spw), P_AGENT_END, "agent_end", u.id)
        self.log("agent_start", utt=u.id, source=u.source)

    def _end_utt(self, u: Utterance, cut: bool) -> None:
        if self.agent_cur is not u:
            return
        if cut:
            heard, rest = heard_prefix(u.text, self.timing.agent_spw, self.now - u.t_start)
        else:
            heard, rest = u.text, ""
        u.heard, u.rest, u.cut = heard, rest, bool(rest)
        u.n_heard_tokens = len(heard.split())
        u.t_end = self.now
        self.agent_cur = None
        self.last_agent_end = self.now
        self.log("agent_end", utt=u.id, cut=u.cut)
        self.arm.on_utterance_end(self, u)

    def agent_busy(self) -> bool:
        return self.agent_cur is not None

    # ------------------------------------------------------------------ tools
    def dispatch_tool(self, role: str, name: str, args: dict, call_id: int) -> ToolCall:
        lat = tool_latency(name, args, self.script.seed, self.run_seed)
        tc = ToolCall(next(self._ids), role, name, args, self.now, self.now + lat, call_id)
        self.tool_calls.append(tc)
        self.push(tc.t_done, P_TOOL, "tool_done", tc.id)
        self.log("tool_call", tool=tc.id, name=name)
        return tc

    def add_note(self, text: str, call_id: int) -> Note:
        n = Note(next(self._ids), self.now, text, call_id)
        self.notes.append(n)
        self.log("note", note=n.id)
        return n

    # ------------------------------------------------------------------ Sam
    @property
    def prev_end(self) -> float:
        return self.persons[-1].t_end if self.persons else 0.0

    def _utts_since(self, t: float) -> list[Utterance]:
        return [u for u in self.utts if u.t_start is not None and u.t_start >= t - 1e-9]

    def last_heard(self) -> PersonTurn | None:
        heard = [p for p in self.persons if p.t_heard <= self.now + 1e-9]
        return heard[-1] if heard else None

    def answered_since(self, iid: str, t: float) -> bool:
        it = self.script.item(iid)
        return any(answer_in(u.heard, it.keys, it.key_type) for u in self._utts_since(t) if u.ended)

    def question_end(self, iid: str) -> float:
        for p in self.persons:
            if iid in p.turn.asks:
                return p.t_end
        return math.inf

    def _due(self, turn: Turn) -> float:
        tr = turn.trigger
        pe = self.prev_end
        speaking = self.agent_cur is not None
        if tr.mode == "at":
            return pe + tr.delay
        if tr.mode == "barge_answer":
            it = self.script.item(tr.items[0])
            for u in self._utts_since(pe):
                if answer_in(u.text, it.keys, it.key_type):
                    if not u.ended:
                        planned_end = u.t_start + speech_seconds(u.text, self.timing.agent_spw)
                        cut_at = u.t_start + tr.delay
                        return cut_at if cut_at < planned_end else planned_end + tr.gap
                    return u.t_end + tr.gap
            return math.inf if speaking else pe + tr.patience
        if speaking:
            return math.inf
        since = self._utts_since(pe)
        if tr.mode == "after_any":
            ok = bool(since)
        elif tr.mode == "after_substance":
            ok = any(not is_holding(u.heard if u.ended else u.text) for u in since)
        elif tr.mode == "after_answer":
            answered = all(self.answered_since(i, self.question_end(i)) for i in tr.items)
            if tr.since_prev == "any":
                ok = answered and bool(since)
            elif tr.since_prev == "answer":
                ok = all(self.answered_since(i, pe) for i in tr.items)
            else:
                ok = answered
            if ok and not since:  # answered before Sam's last turn: Sam leaves a moment for a reply, then goes on
                return pe + max(tr.gap, 2.0)
        else:
            raise ValueError(tr.mode)
        if ok and self.last_agent_end is not None:
            return self.last_agent_end + tr.gap
        return pe + tr.patience

    def _eval_person(self) -> None:
        if self.person_cur is not None or self.turn_idx >= len(self.script.turns):
            return
        due = self._due(self.script.turns[self.turn_idx])
        if due <= self.now + 1e-9:
            self._start_person()
        elif due < math.inf:
            self._check_version += 1
            self.push(due, P_CHECK, "check", self._check_version)

    def _start_person(self) -> None:
        turn = self.script.turns[self.turn_idx]
        dur = speech_seconds(turn.text, self.timing.person_spw)
        pt = PersonTurn(self.turn_idx, turn, self.now, self.now + dur, self.now + dur + self.timing.endpoint)
        self.turn_idx += 1
        self.person_cur = pt
        self.persons.append(pt)
        if self.agent_cur is not None:
            pt.barged = True
            pt.cut_utt = self.agent_cur.id
            self.push(self.now + self.timing.barge_reaction, P_CUT, "agent_cut", self.agent_cur.id)
        self.drop_voice_queue()
        self.push(pt.t_end, P_PERSON_END, "person_end", pt.idx)
        self.log("person_start", turn=turn.id, barged=pt.barged)

    # ------------------------------------------------------------------ run
    _BUSY_EVENTS = {"call_first", "call_done", "tool_done", "agent_end", "agent_cut", "person_end", "person_heard"}

    def idle(self) -> bool:
        if self.agent_cur is not None or self.queue or self.person_cur is not None:
            return False
        if any(ev[3] in self._BUSY_EVENTS for ev in self._heap):
            return False
        if any(not c.cancelled and (not c.resolved or c.t_done > self.now + 1e-9) for c in self.calls.values()):
            return False
        if any(not tc.done for tc in self.tool_calls):
            return False
        return True

    async def run(self) -> None:
        self.arm.on_start(self)
        self._eval_person()
        while True:
            t_next = self._heap[0][0] if self._heap else math.inf
            pending = [c for c in self.calls.values() if not c.resolved and c.t0 <= t_next]
            if pending:
                await asyncio.gather(*(self._resolve(c) for c in pending))
                continue
            if not self._heap:
                break
            t, prio, _, kind, payload = heapq.heappop(self._heap)
            self.now = max(self.now, t)
            if self.now > self.t_limit:
                self.flags.append("time_limit")
                break
            self._dispatch(kind, payload)
            if self.finished:
                break
            self._try_speak()
            self._eval_person()
            if self.turn_idx >= len(self.script.turns) and self.person_cur is None and self.idle():
                last = self.persons[-1]
                if self.now >= last.t_heard:
                    break
        # a line still being spoken when the call ends was heard only up to here
        if self.agent_cur is not None:
            u = self.agent_cur
            self._end_utt(u, cut=True)
            u.call_ended = True
        # cancel what is still running in the real world
        for c in self.calls.values():
            if not c.resolved and c.task is not None:
                try:
                    await c.task
                    c.resolved = True
                except Exception:  # noqa: BLE001
                    pass

    def _dispatch(self, kind: str, payload: Any) -> None:
        if kind == "call_first":
            c = self.calls[payload]
            if not c.cancelled:
                self.arm.on_call_first(self, c)
        elif kind == "call_done":
            c = self.calls[payload]
            if not c.cancelled:
                self.arm.on_call_done(self, c)
        elif kind == "tool_done":
            tc = next(x for x in self.tool_calls if x.id == payload)
            tc.result = self.tools.call(tc.name, tc.args)
            tc.done = True
            self.log("tool_done", tool=tc.id)
            self.arm.on_tool_result(self, tc)
        elif kind == "agent_end":
            if self.agent_cur is not None and self.agent_cur.id == payload:
                self._end_utt(self.agent_cur, cut=False)
        elif kind == "agent_cut":
            if self.agent_cur is not None and self.agent_cur.id == payload:
                self._end_utt(self.agent_cur, cut=True)
        elif kind == "person_end":
            pt = self.persons[payload]
            self.person_cur = None
            self.floor_free_at = self.now + self.timing.endpoint
            self.push(pt.t_heard, P_HEARD, "person_heard", payload)
            self.log("person_end", turn=pt.turn.id)
        elif kind == "person_heard":
            pt = self.persons[payload]
            self.drop_voice_queue()
            self.log("person_heard", turn=pt.turn.id)
            self.arm.on_person(self, pt)
            if pt.idx == len(self.script.turns) - 1:
                self.push(self.now + self.tail, P_DEADLINE, "deadline")
        elif kind == "wake":
            self.arm.on_wake(self, payload)
        elif kind == "check":
            if payload == self._check_version:
                pass  # _eval_person runs after every event
        elif kind == "ready":
            pass
        elif kind == "deadline":
            if not self.idle():
                self.flags.append("tail_deadline")
            self.finished = True

    # ------------------------------------------------------------------ record
    def record(self) -> dict:
        return {
            **self.meta,
            "persons": [{"idx": p.idx, "turn": p.turn.id, "kind": p.turn.kind, "text": p.turn.text,
                         "t_start": round(p.t_start, 3), "t_end": round(p.t_end, 3), "t_heard": round(p.t_heard, 3),
                         "barged": p.barged, "asks": p.turn.asks, "withdraws": p.turn.withdraws,
                         "expects_reply": p.turn.expects_reply, "episode": p.turn.episode} for p in self.persons],
            "utterances": [{"id": u.id, "text": u.text, "source": u.source, "kind": u.kind, "label": u.label,
                            "t_queued": round(u.t_queued, 3), "t_ready": round(u.t_ready, 3),
                            "t_start": None if u.t_start is None else round(u.t_start, 3),
                            "t_end": None if u.t_end is None else round(u.t_end, 3),
                            "heard": u.heard, "cut": u.cut, "dropped": u.dropped, "call": u.call_id,
                            "call_ended": u.call_ended} for u in self.utts],
            "tool_calls": [{"id": tc.id, "role": tc.role, "name": tc.name, "args": tc.args, "t_call": round(tc.t_call, 3),
                            "t_done": round(tc.t_done, 3), "done": tc.done, "result": tc.result, "call": tc.call_id}
                           for tc in self.tool_calls],
            "notes": [{"id": n.id, "t": round(n.t, 3), "text": n.text, "call": n.call_id} for n in self.notes],
            "calls": [{"id": c.id, "role": c.role, "model": c.model, "purpose": c.purpose, "t0": round(c.t0, 3),
                       "t_first": None if c.t_first is None else round(c.t_first, 3),
                       "t_done": None if c.t_done is None else round(c.t_done, 3),
                       "cancelled": c.cancelled, "t_cancel": c.t_cancel, "text": c.text, "tool_calls": c.tool_calls,
                       "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
                       "reasoning_tokens": c.reasoning_tokens, "cached_tokens": c.cached_tokens,
                       "cost": c.cost, "latency_s": round(c.latency_s, 3), "meta": c.meta} for c in self.calls.values()],
            "flags": self.flags,
            "t_end": round(self.now, 3),
            "turns_spoken": self.turn_idx,
            "turns_total": len(self.script.turns),
        }
