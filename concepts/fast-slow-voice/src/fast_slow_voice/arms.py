"""The arms. Each decides which model to call and what is said; the simulator
decides when.

Two loop shapes are shared by every arm:

- VoiceLoop, for a model whose words are spoken: one call at a time; a new
  event that arrives before the in-flight call's first token cancels it and
  starts a fresh call with everything; a later one waits for the call to end.
- Worker, for the slow model behind a fast voice (relay, slow_speaks): it never
  cancels an in-flight call; events that arrive meanwhile go into its next call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from . import prompts as P
from .render import messages, view_lines
from .sim import Call, ModelCfg, PersonTurn, Sim, ToolCall, Utterance, is_placeholder
from .timing import resume_from_sentence
from .world import ACTION_TOOLS, TOOL_SPECS


@dataclass
class Models:
    fast: ModelCfg
    slow: ModelCfg
    fast_think: ModelCfg | None = None  # the fast model with reasoning, for fast_only_think and fast_context_think


class VoiceLoop:
    def __init__(self, start: Callable[[Sim, list[str]], Call]):
        self.start_fn = start
        self.call: Call | None = None
        self.call_events: list[str] = []
        self.waiting: list[str] = []

    def poke(self, sim: Sim, event: str) -> None:
        self.waiting.append(event)
        c = self.call
        if c is None:
            self._start(sim)
        elif sim.first_after(c, sim.now):
            sim.cancel_call(c)
            self.waiting = self.call_events + self.waiting
            self._start(sim)

    def _start(self, sim: Sim) -> None:
        self.call_events, self.waiting = self.waiting, []
        self.call = self.start_fn(sim, self.call_events)

    def on_done(self, sim: Sim, call: Call) -> None:
        if call is not self.call:
            return
        self.call = None
        if self.waiting:
            self._start(sim)

    def busy(self) -> bool:
        return self.call is not None


class Worker:
    def __init__(self, start: Callable[[Sim, list[tuple]], Call]):
        self.start_fn = start
        self.call: Call | None = None
        self.triggers: list[tuple] = []

    def poke(self, sim: Sim, trigger: tuple) -> None:
        self.triggers.append(trigger)
        if self.call is None:
            self._start(sim)

    def _start(self, sim: Sim) -> None:
        trig, self.triggers = self.triggers, []
        self.call = self.start_fn(sim, trig)
        self.call.meta["triggers"] = [list(t) for t in trig]

    def on_done(self, sim: Sim, call: Call) -> None:
        if call is not self.call:
            return
        self.call = None
        if self.triggers:
            self._start(sim)

    def run_for_turn(self, sim: Sim, turn_id: str) -> Call | None:
        """The run that cancel_turn(turn_id) would cancel: the one in flight for that turn alone."""
        c = self.call
        if c is not None and c.meta.get("triggers") == [["person", turn_id]] and not (c.resolved and c.t_done <= sim.now):
            return c
        return None

    def cancel_turn(self, sim: Sim, turn_id: str) -> str:
        """Drop the run for one of Sam's turns (slow_speaks, when the voice handled it).
        Returns what was done, for the record."""
        before = len(self.triggers)
        self.triggers = [t for t in self.triggers if t != ("person", turn_id)]
        if len(self.triggers) < before:
            return "dequeued"
        c = self.run_for_turn(sim, turn_id)
        if c is not None:
            sim.cancel_call(c)
            self.call = None
            if self.triggers:
                self._start(sim)
            return "cancelled"
        return "kept"


def _describe(events: list) -> str:
    return "; ".join(str(e) for e in events) if events else "nothing new"


class Arm:
    name = "arm"

    def __init__(self, models: Models):
        self.models = models

    def on_start(self, sim: Sim) -> None: ...
    def on_person(self, sim: Sim, pt: PersonTurn) -> None: ...
    def on_call_first(self, sim: Sim, call: Call) -> None: ...
    def on_call_done(self, sim: Sim, call: Call) -> None: ...
    def on_tool_result(self, sim: Sim, tc: ToolCall) -> None: ...
    def on_utterance_end(self, sim: Sim, u: Utterance) -> None: ...
    def on_wake(self, sim: Sim, tag: Any) -> None: ...

    def dispatch_tools(self, sim: Sim, call: Call, role: str, allowed: list[dict]) -> None:
        names = {t["function"]["name"] for t in allowed}
        for tc in call.tool_calls:
            if tc["name"] in names:
                sim.dispatch_tool(role, tc["name"], tc["args"], call.id)
            else:  # a tool the model made up (seen: "noop"); nothing to run
                call.meta.setdefault("ignored_tools", []).append(tc["name"])


# ---------------------------------------------------------------------------
# One model does everything
# ---------------------------------------------------------------------------


class Single(Arm):
    """slow_only, fast_only, fast_context, slow_filler."""

    def __init__(self, models: Models, which: str, context: bool = False, filler: bool = False, name: str = ""):
        super().__init__(models)
        self.cfg = {"slow": models.slow, "fast": models.fast, "fast_think": models.fast_think}[which]
        self.context, self.filler = context, filler
        self.name = name
        self.loop: VoiceLoop | None = None
        self.system = ""
        self.n_fillers = 0

    def on_start(self, sim: Sim) -> None:
        self.loop = VoiceLoop(self._start)
        self.system = P.SINGLE_CONTEXT.replace("{records}", sim.world.as_text()) if self.context else P.SINGLE
        self.tools = ACTION_TOOLS if self.context else TOOL_SPECS

    def _start(self, sim: Sim, events: list[str]) -> Call:
        instr = (f"Time now: {sim.now:.1f}s. Since your last turn: {_describe(events)}. "
                 "Reply with exactly the words Ava says now, or [silent]; call any tools you need.")
        msgs = messages(self.system, view_lines(sim, "single"), instr, self.cfg.anthropic)
        return sim.start_call("voice", self.cfg, msgs, self.tools, "turn", {"events": events})

    def on_person(self, sim: Sim, pt: PersonTurn) -> None:
        self.loop.poke(sim, f"Sam spoke at {pt.t_start:.1f}s")
        if self.filler:
            sim.wake_at(pt.t_end + 1.2, ("filler", pt.idx))

    def on_tool_result(self, sim: Sim, tc: ToolCall) -> None:
        self.loop.poke(sim, f"tool result #{tc.id} ({tc.name}) arrived")

    def on_call_first(self, sim: Sim, call: Call) -> None:
        sim.say(call.text, "single", "voice", call.id, ready=sim.now + sim.timing.tts)

    def on_call_done(self, sim: Sim, call: Call) -> None:
        self.dispatch_tools(sim, call, "voice", self.tools)
        self.loop.on_done(sim, call)

    def on_wake(self, sim: Sim, tag: Any) -> None:
        if not (isinstance(tag, tuple) and tag[0] == "filler"):
            return
        idx = tag[1]
        last = sim.last_heard()
        if last.idx != idx or sim.person_cur is not None or sim.agent_busy():
            return
        if any(u.t_start is not None and u.t_start >= last.t_end for u in sim.utts):
            return
        if any(u.t_ready <= sim.now for u in sim.queue) or not self.loop.busy():
            return
        sim.say(P.FILLERS[self.n_fillers % len(P.FILLERS)], "filler", "voice")
        self.n_fillers += 1


# ---------------------------------------------------------------------------
# Design A: the fast model speaks every word; the slow one passes data
# ---------------------------------------------------------------------------


class Relay(Arm):
    name = "relay"

    def on_start(self, sim: Sim) -> None:
        self.fast = VoiceLoop(self._start_fast)
        self.slow = Worker(self._start_slow)

    def _start_fast(self, sim: Sim, events: list[str]) -> Call:
        cfg = self.models.fast
        instr = f"Time now: {sim.now:.1f}s. Since your last turn: {_describe(events)}. Reply with exactly the words Ava says now, or [silent]."
        msgs = messages(P.RELAY_FAST, view_lines(sim, "relay_fast"), instr, cfg.anthropic)
        return sim.start_call("fast", cfg, msgs, None, "voice", {"events": events})

    def _start_slow(self, sim: Sim, triggers: list[tuple]) -> Call:
        cfg = self.models.slow
        desc = [f"Sam spoke ({t[1]})" if t[0] == "person" else f"tool result #{t[1]} arrived" for t in triggers]
        instr = (f"Time now: {sim.now:.1f}s. New since your last look: {_describe(desc)}. Call tools as needed. "
                 "Reply with a note if there is new data for the voice, or [silent].")
        msgs = messages(P.RELAY_SLOW, view_lines(sim, "relay_slow"), instr, cfg.anthropic)
        return sim.start_call("slow", cfg, msgs, TOOL_SPECS, "work")

    def on_person(self, sim: Sim, pt: PersonTurn) -> None:
        self.fast.poke(sim, f"Sam spoke at {pt.t_start:.1f}s")
        self.slow.poke(sim, ("person", pt.turn.id))

    def on_tool_result(self, sim: Sim, tc: ToolCall) -> None:
        self.slow.poke(sim, ("tool", tc.id))

    def on_call_first(self, sim: Sim, call: Call) -> None:
        if call.role == "fast":
            sim.say(call.text, "fast", "voice", call.id, ready=sim.now + sim.timing.tts)

    def on_call_done(self, sim: Sim, call: Call) -> None:
        if call.role == "fast":
            self.fast.on_done(sim, call)
            return
        self.dispatch_tools(sim, call, "slow", TOOL_SPECS)
        if not is_placeholder(call.text):
            n = sim.add_note(call.text.strip(), call.id)
            self.fast.poke(sim, f"a note from the back office arrived at {n.t:.1f}s")
        self.slow.on_done(sim, call)


# ---------------------------------------------------------------------------
# Design B: the slow model speaks every substantive line; the fast one buys time
# ---------------------------------------------------------------------------

_LABEL = re.compile(r"(?:^|\n)\s*\**\s*(SMALL_TALK|SMALL TALK|HOLD|REPLAY|NONE)\b\s*\**\s*:?[ \t]*", re.I)


def parse_label(text: str) -> tuple[str, str]:
    """The fast model's label and its line. The line is what follows the first
    label on the same line (seen: GPT-6 Luna repeating "HOLD: Sure, one sec." on
    a second line)."""
    t = (text or "").strip()
    m = _LABEL.search(t)
    if not m:
        return ("UNLABELLED", t)
    label = m.group(1).upper().replace(" ", "_")
    line = t[m.end():].split("\n")[0]
    line = _LABEL.split(line)[0] if _LABEL.search(line) else line
    return label, line.strip().strip('"').strip()


class SlowSpeaks(Arm):
    name = "slow_speaks"
    LABEL_TOKENS = 4  # the label comes before the spoken line

    def on_start(self, sim: Sim) -> None:
        self.fast = VoiceLoop(self._start_fast)
        self.slow = Worker(self._start_slow)
        self.parse_failures = 0
        self.replay_fallbacks = 0

    def _last_slow_line(self, sim: Sim) -> Utterance | None:
        for u in reversed(sim.utts):
            if u.source in ("slow", "replay") and u.t_start is not None:
                return u
        return None

    def _start_fast(self, sim: Sim, events: list) -> Call:
        cfg = self.models.fast
        pt = sim.last_heard()
        cut = ""
        if pt.cut_utt is not None:
            u = next(x for x in sim.utts if x.id == pt.cut_utt)
            if u.source in ("slow", "replay") and u.cut:
                cut = (f' Sam cut into the main assistant\'s line "{u.text}". Heard: "{u.heard}". '
                       f'Not heard: "{u.rest}".')
        instr = f'Sam\'s latest turn: "{pt.turn.text}".{cut}\nYour label and line:'
        msgs = messages(P.SLOW_SPEAKS_FAST, view_lines(sim, "slow_speaks_fast"), instr, cfg.anthropic)
        return sim.start_call("fast", cfg, msgs, None, "voice", {"turn_id": pt.turn.id}, prefix_tokens=self.LABEL_TOKENS)

    def _start_slow(self, sim: Sim, triggers: list[tuple]) -> Call:
        cfg = self.models.slow
        desc = [f"Sam spoke ({t[1]})" if t[0] == "person" else f"tool result #{t[1]} arrived" for t in triggers]
        instr = (f"Time now: {sim.now:.1f}s. New since your last look: {_describe(desc)}. Call tools as needed. "
                 "Reply with exactly the words Ava says now if you have something substantive, or [silent].")
        msgs = messages(P.SLOW_SPEAKS_SLOW, view_lines(sim, "slow_speaks_slow"), instr, cfg.anthropic)
        return sim.start_call("slow", cfg, msgs, TOOL_SPECS, "work")

    def on_person(self, sim: Sim, pt: PersonTurn) -> None:
        self.fast.poke(sim, pt.turn.id)
        self.slow.poke(sim, ("person", pt.turn.id))

    def on_tool_result(self, sim: Sim, tc: ToolCall) -> None:
        self.slow.poke(sim, ("tool", tc.id))

    def _replay_text(self, sim: Sim, pt: PersonTurn, doomed: int | None = None) -> str | None:
        """The slow model's line to replay for Sam's turn pt: the line Sam cut
        into, else the last one that started before Sam spoke. None when the
        slow model has already spoken again (or has a line waiting), since a
        replay would then repeat it. A line waiting from the slow run for this
        very turn (call `doomed`) does not count: the REPLAY label cancels that
        run, and its waiting line with it."""
        if any(u.source == "slow" and ((u.t_start is not None and u.t_start >= pt.t_end - 1e-9)
                                       or (u in sim.queue and u.call_id != doomed))
               for u in sim.utts):
            return None
        target = None
        if pt.cut_utt is not None:
            target = next((u for u in sim.utts if u.id == pt.cut_utt and u.source in ("slow", "replay")), None)
        if target is None:
            before = [u for u in sim.utts if u.source in ("slow", "replay") and u.t_start is not None and u.t_start < pt.t_start]
            target = before[-1] if before else None
        if target is None:
            return ""
        if target.cut:
            rest = resume_from_sentence(target.text, target.n_heard_tokens)
            return (P.REPLAY_CUT_LEAD + rest) if rest else ""
        return P.REPLAY_FULL_LEAD + target.text

    def on_call_first(self, sim: Sim, call: Call) -> None:
        if call.role == "slow":
            sim.say(call.text, "slow", "verbatim", call.id, ready=sim.now + sim.timing.tts)
            return
        label, line = parse_label(call.text)
        call.meta["label"] = label
        ready = sim.now + sim.timing.tts
        if label == "UNLABELLED":
            self.parse_failures += 1
            sim.say(line, "fast", "voice", call.id, ready=ready, label=label)
        elif label in ("SMALL_TALK", "HOLD"):
            sim.say(line, "fast", "voice", call.id, ready=ready, label=label)
        elif label == "REPLAY":
            pt = next(p for p in sim.persons if p.turn.id == call.meta["turn_id"])
            doomed = self.slow.run_for_turn(sim, call.meta["turn_id"])
            text = self._replay_text(sim, pt, doomed.id if doomed is not None else None)
            if text is None:
                call.meta["replay"] = "skipped: the slow model has spoken again"
            elif text:
                sim.say(text, "replay", "voice", call.id, ready=ready, label=label)
            else:
                self.replay_fallbacks += 1
                sim.say("Sorry, one sec.", "fast", "voice", call.id, ready=ready, label="HOLD")
                label = "HOLD"
        if label in ("SMALL_TALK", "REPLAY", "NONE"):
            call.meta["slow_run"] = self.slow.cancel_turn(sim, call.meta["turn_id"])

    def on_call_done(self, sim: Sim, call: Call) -> None:
        if call.role == "fast":
            self.fast.on_done(sim, call)
            return
        self.dispatch_tools(sim, call, "slow", TOOL_SPECS)
        self.slow.on_done(sim, call)


ARMS = ["relay", "slow_speaks", "slow_only", "fast_only", "fast_context", "slow_filler", "fast_only_think",
        "fast_context_think"]


def make_arm(name: str, models: Models) -> Arm:
    if name == "relay":
        return Relay(models)
    if name == "slow_speaks":
        return SlowSpeaks(models)
    if name == "slow_only":
        return Single(models, "slow", name="slow_only")
    if name == "fast_only":
        return Single(models, "fast", name="fast_only")
    if name == "fast_context":
        return Single(models, "fast", context=True, name="fast_context")
    if name == "slow_filler":
        return Single(models, "slow", filler=True, name="slow_filler")
    if name == "fast_only_think":
        return Single(models, "fast_think", name="fast_only_think")
    if name == "fast_context_think":
        return Single(models, "fast_think", context=True, name="fast_context_think")
    raise ValueError(name)
