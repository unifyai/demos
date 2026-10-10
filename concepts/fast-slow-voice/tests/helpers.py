"""A stubbed model client and an oracle policy that answers every arm's calls
correctly, so the mechanics can be tested without the network."""

from __future__ import annotations

import json
from typing import Any, Callable

from demos_common import Reply

from fast_slow_voice.sim import ModelCfg
from fast_slow_voice.timing import ModelTiming, Timing

FAST = ModelCfg("test/fast", None, 300)
FAST_THINK = ModelCfg("test/fast", {"effort": "low"}, 300)
SLOW = ModelCfg("test/slow", None, 2000)


def make_timing(fast_a: float = 0.5, slow_a: float = 2.0) -> Timing:
    return Timing(models={"test/fast": ModelTiming(fast_a, 0.0, 100.0), "test/slow": ModelTiming(slow_a, 0.0, 50.0)})


class FakeLLM:
    """chat() returns what policy(model, messages, kw, meta) gives:
    (text, tool_calls [(name, args)], completion_tokens, reasoning_tokens)."""

    def __init__(self, policy: Callable[..., tuple]):
        self.policy = policy
        self.calls: list[dict] = []

    async def chat(self, model: str, messages: list[dict], *, tag: str | None = None, meta: dict | None = None, **kw: Any) -> Reply:
        text, tools, comp, reas = self.policy(model, messages, kw, meta or {})
        self.calls.append({"model": model, "meta": meta, "text": text, "tools": tools, "kw": kw})
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tools:
            msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                                 for i, (n, a) in enumerate(tools)]
        return Reply(message=msg, model=model, prompt_tokens=1000, completion_tokens=comp, reasoning_tokens=reas,
                     cached_tokens=0, cost=0.001, latency_s=0.1)


def instruction_of(messages: list[dict]) -> str:
    return messages[-1]["content"][-1]["text"]


class Oracle:
    """Answers like a perfect agent of each arm, reading the simulator's state.
    Stateless: every decision comes from what has been heard, said (or queued),
    called and returned so far, so cancelled calls cannot confuse it. It calls
    one tool per question (find_contacts with the item id as the query) and,
    when that result arrives, gives the gold answer."""

    def __init__(self, arm: str):
        self.arm = arm
        self.sim = None

    def bind(self, sim) -> None:
        self.sim = sim

    # ---- state
    def heard(self):
        return [p for p in self.sim.persons if p.t_heard <= self.sim.now + 1e-9]

    def said_after(self, t: float, needle: str | None = None) -> bool:
        for u in self.sim.utts:
            if u.dropped:
                continue
            when = u.t_start if u.t_start is not None else u.t_queued
            if when >= t - 1e-9 and (needle is None or needle in u.text):
                return True
        return False

    def tool_for(self, iid: str):
        return next((tc for tc in self.sim.tool_calls if tc.args.get("query") == iid), None)

    def withdrawn(self) -> set:
        return {i for p in self.heard() for i in p.turn.withdraws}

    def answer(self, iid: str) -> str:
        it = self.sim.script.item(iid)
        return f"The {it.thing} is {it.gold}."

    def last_lookup(self, before_idx: int):
        for p in reversed(self.heard()[:before_idx]):
            for iid in p.turn.asks:
                it = self.sim.script.item(iid)
                if it.kind not in ("text_late", "repeat"):
                    return iid
        return None

    # ---- decisions
    def tools_needed(self) -> list[tuple]:
        out = []
        for p in self.heard():
            for iid in p.turn.asks:
                it = self.sim.script.item(iid)
                if it.kind == "repeat":
                    continue
                if it.kind == "text_late":
                    if not any(tc.name == "send_text" for tc in self.sim.tool_calls):
                        out.append(("send_text", {"to": it.person, "text": "Running about 10 minutes late, sorry!"}))
                elif self.tool_for(iid) is None and not self.arm.startswith("fast_context"):
                    out.append(("find_contacts", {"query": iid}))
        return out

    def direct_answers(self) -> list[str]:
        """fast_context has the records in its prompt: it answers at once."""
        out = []
        gone = self.withdrawn()
        for p in self.heard():
            for iid in p.turn.asks:
                it = self.sim.script.item(iid)
                if it.kind not in ("repeat", "text_late") and iid not in gone and not self.said_after(p.t_end, it.gold):
                    out.append(self.answer(iid))
        return out

    def answers_due(self, check_said: bool = True) -> list[str]:
        out = []
        gone = self.withdrawn()
        for tc in self.sim.tool_calls:
            if not tc.done:
                continue
            if tc.name == "send_text":
                if not self.said_after(tc.t_done, "sent"):
                    out.append("Done, I've sent it.")
                continue
            iid = tc.args["query"]
            if iid in gone:
                continue
            ans = self.answer(iid)
            if not check_said or not self.said_after(tc.t_done, self.sim.script.item(iid).gold):
                out.append(ans)
        for p in self.heard():
            for iid in p.turn.asks:
                it = self.sim.script.item(iid)
                if it.kind == "repeat" and not self.said_after(p.t_end, it.gold):
                    out.append(self.answer(it.extra["repeats"]))
            if p.turn.kind in ("interrupt_repeat", "interrupt_resume"):
                iid = self.last_lookup(p.idx)
                known = self.arm.startswith("fast_context") or (self.tool_for(iid) and self.tool_for(iid).done) if iid else False
                if iid and known and not self.said_after(p.t_end, self.sim.script.item(iid).gold):
                    out.append(self.answer(iid))
        return out

    def small_talk_due(self) -> list[str]:
        out = []
        for p in self.heard():
            if p.turn.kind in ("greeting", "closing") and not self.said_after(p.t_end):
                out.append("Hi Sam, all good here." if p.turn.kind == "greeting" else "Bye, Sam!")
        return out

    def holds_due(self, line: str) -> list[str]:
        out = []
        for p in self.heard():
            if any(self.sim.script.item(i).kind not in ("repeat",) for i in p.turn.asks) and not self.said_after(p.t_end):
                out.append(line)
        return out[-1:]

    def __call__(self, model, messages, kw, meta):
        role = meta.get("role")
        texts: list[str] = []
        tools: list[tuple] = []
        if role == "voice":
            tools = self.tools_needed()
            texts = self.small_talk_due() + (["One sec, let me check."] if tools else []) + self.answers_due()
            if self.arm.startswith("fast_context"):
                texts += self.direct_answers()
        elif self.arm == "relay" and role == "fast":
            texts = self.small_talk_due() + self.holds_due("Let me check on that.")
            for n in self.sim.notes:
                if not self.said_after(n.t, n.text.split(" is ")[-1].rstrip(".")):
                    texts.append(n.text)
            for p in self.heard():
                for iid in p.turn.asks:
                    it = self.sim.script.item(iid)
                    if it.kind == "repeat" and not self.said_after(p.t_end, it.gold):
                        texts.append(self.answer(it.extra["repeats"]))
                if p.turn.kind in ("interrupt_repeat", "interrupt_resume"):
                    iid = self.last_lookup(p.idx)
                    if iid and not self.said_after(p.t_end, self.sim.script.item(iid).gold) and any(self.sim.script.item(iid).gold in n.text for n in self.sim.notes):
                        texts.append(self.answer(iid))
        elif self.arm == "relay" and role == "slow":
            tools = self.tools_needed()
            gone = self.withdrawn()
            for tc in self.sim.tool_calls:
                if tc.done and tc.name != "send_text" and tc.args["query"] not in gone:
                    ans = self.answer(tc.args["query"])
                    if not any(n.text == ans for n in self.sim.notes):
                        texts.append(ans)
                if tc.done and tc.name == "send_text" and not any("sent" in n.text for n in self.sim.notes):
                    texts.append("Text sent.")
        elif self.arm == "slow_speaks" and role == "fast":
            t = self.sim.last_heard().turn
            if t.kind in ("greeting", "closing", "ack", "aside"):
                return ("SMALL_TALK: " + ("Hi Sam!" if t.kind == "greeting" else "Bye, Sam!" if t.kind == "closing" else "Sure."), [], 8, 0)
            if t.kind == "interrupt_resume":
                return ("REPLAY", [], 2, 0)
            return ("HOLD: One sec.", [], 6, 0)
        elif self.arm == "slow_speaks" and role == "slow":
            tools = self.tools_needed()
            texts = self.answers_due()
        text = " ".join(x for x in texts if x)
        return (text, tools, 10 + len(text.split()) * 2, 0)
