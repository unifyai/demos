"""What every arm shares: the run's record, the model-call wrapper and mailboxes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from .sites import Clock


class Mailbox:
    """Items waiting for a loop, with a wake-up event."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.event = asyncio.Event()

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)
        self.event.set()

    def drain(self) -> list[dict[str, Any]]:
        items, self.items = self.items, []
        self.event.clear()
        return items

    def has(self, pred: Callable[[dict[str, Any]], bool] | None = None) -> bool:
        return any(pred(i) for i in self.items) if pred else bool(self.items)

    async def wait(self) -> None:
        while not self.items:
            self.event.clear()
            await self.event.wait()

    async def wait_event(self) -> None:
        """Wait for an item or a poke."""
        if self.items:
            return
        self.event.clear()
        await self.event.wait()

    def poke(self) -> None:
        """Wake whoever waits, without an item (used by stop)."""
        self.event.set()


def append_user(ctx: list[dict[str, Any]], text: str) -> None:
    """Add a user message, merging with a user message just before it."""
    if ctx and ctx[-1]["role"] == "user" and isinstance(ctx[-1].get("content"), str):
        ctx[-1]["content"] += "\n\n" + text
    else:
        ctx.append({"role": "user", "content": text})


def assistant_message(reply: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
    details = (getattr(reply, "message", None) or {}).get("reasoning_details")
    if details:  # reasoning models used with tools need their reasoning passed back
        msg["reasoning_details"] = details
    if reply.tool_calls:
        msg["tool_calls"] = [{"id": c.get("id") or f"call_{i}", "type": "function",
                              "function": {"name": c["function"]["name"],
                                           "arguments": c["function"].get("arguments") or "{}"}}
                             for i, c in enumerate(reply.tool_calls)]
    return msg


def tool_message(call: dict[str, Any], content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


def fn_schema(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


def anthropic_cache(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prompt-cache breakpoints for Anthropic models, which cache only where they are told to (OpenAI's
    models cache on their own): the system prompt, the end of the previous call's prompt (the message
    before the last assistant message, where that call wrote the cache) and the end of this one."""
    out = list(messages)

    def mark(i: int) -> None:
        m = dict(out[i])
        c = m.get("content")
        if isinstance(c, str) and c:
            m["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
            out[i] = m
    if out and out[0]["role"] == "system":
        mark(0)
    last_assistant = max((i for i, m in enumerate(out) if m["role"] == "assistant"), default=None)
    if last_assistant is not None and last_assistant > 1:
        mark(last_assistant - 1)
    if len(out) > 1 and out[-1]["role"] != "assistant":
        mark(len(out) - 1)
    return out


def parse_args(call: dict[str, Any]) -> dict[str, Any] | None:
    raw = call["function"].get("arguments")
    try:
        args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return args if isinstance(args, dict) else None


class Run:
    """Everything recorded about one run, and the one way its loops call a model."""

    def __init__(self, run_id: str, model: str, llm: Any, clock: Clock, model_kw: dict[str, Any] | None = None):
        self.run_id = run_id
        self.model = model
        self.llm = llm
        self.clock = clock
        self.model_kw = model_kw or {}
        self.person_msgs: list[dict[str, Any]] = []
        self.agent_msgs: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self.contexts: dict[str, list[dict[str, Any]]] = {}
        self.errors: list[str] = []
        self.inflight = 0
        self.last_activity = 0.0
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

    def touch(self) -> None:
        self.last_activity = self.clock.now()

    async def chat(self, loop: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *,
                   seen: int = -1, relays: list[str] | None = None, model: str | None = None,
                   **kw: Any) -> tuple[Any, dict[str, Any]]:
        self.inflight += 1
        self.touch()
        t0 = self.clock.now()
        name = model or self.model
        sent = anthropic_cache(messages) if name.startswith("anthropic/") else messages
        try:
            reply = await self.llm.chat(name, sent, tools=tools or None,
                                        meta={"run": self.run_id, "loop": loop}, **{**self.model_kw, **kw})
        finally:
            self.inflight -= 1
            self.touch()
        t1 = self.clock.now()
        rec = {"i": len(self.calls), "loop": loop, "model": model or self.model, "t0": t0, "t1": t1,
               "latency": round(reply.latency_s, 3), "overhead": round(t1 - t0 - reply.latency_s, 3),
               "prompt": reply.prompt_tokens, "completion": reply.completion_tokens, "cost": reply.cost,
               "cached": getattr(reply, "cached_tokens", 0),
               "tools": [c["function"]["name"] for c in reply.tool_calls], "text": reply.text,
               "discarded": False, "seen": seen, "relays": list(relays or []), "n_messages": len(messages),
               "reasoning": getattr(reply, "reasoning_tokens", 0),
               "finish": ((getattr(reply, "raw", None) or {}).get("choices") or [{}])[0].get("finish_reason")}
        self.calls.append(rec)
        return reply, rec

    def say(self, text: str, loop: str, seen: int) -> None:
        m = {"i": len(self.agent_msgs), "t": self.clock.now(), "text": text, "loop": loop, "seen": seen}
        self.agent_msgs.append(m)
        self.touch()
        for cb in self.listeners:
            cb(m)
