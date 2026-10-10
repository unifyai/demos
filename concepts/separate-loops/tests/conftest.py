"""A stub model client and helpers. Tests run the sites at 1/100 speed."""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
from collections.abc import Callable
from typing import Any

import pytest

from separate_loops.arms import ARMS
from separate_loops.core import Run
from separate_loops.scenarios import make_instance
from separate_loops.sites import Clock, Env

SPEED = 0.01
_ids = itertools.count()


class FakeReply:
    def __init__(self, text: str = "", tool_calls: list[dict[str, Any]] | None = None):
        self.message = {"role": "assistant", "content": text, "tool_calls": tool_calls or None}
        self.text = text
        self.tool_calls = tool_calls or []
        self.latency_s = 0.0
        self.prompt_tokens = self.completion_tokens = 0
        self.cost = 0.0


def tc(name: str, **args: Any) -> dict[str, Any]:
    return {"id": f"c{next(_ids)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def say(text: str = "", *calls: dict[str, Any]) -> FakeReply:
    return FakeReply(text, list(calls))


class FakeLLM:
    """Answers each call with policy(loop, messages, tools); records every call."""

    def __init__(self, policy: Callable[[str, list[dict[str, Any]], list[str]], FakeReply], delay: float = 0.005,
                 delays: dict[str, float] | None = None):
        self.policy = policy
        self.delay = delay
        self.delays = delays or {}
        self.calls: list[dict[str, Any]] = []

    async def chat(self, model: str, messages: list[dict[str, Any]], tools: Any = None, meta: Any = None, **kw: Any):
        loop = (meta or {}).get("loop", "?")
        names = [t["function"]["name"] for t in tools or []]
        snapshot = copy.deepcopy(messages)
        await asyncio.sleep(self.delays.get(loop.split(":")[0], self.delay))
        reply = self.policy(loop, snapshot, names)
        self.calls.append({"loop": loop, "messages": snapshot, "tools": names, "reply": reply})
        return reply


def last_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def make_arm(arm: str, llm: FakeLLM, family: str = "A", condition: str = "sanity"):
    inst = make_instance(condition, family, 0)
    clock = Clock()
    env = Env(inst, clock, speed=SPEED)
    run = Run("test", "fake/model", llm, clock)
    return ARMS[arm](run, env), run, env, inst


async def until(pred: Callable[[], bool], timeout: float = 5.0) -> None:
    t = 0.0
    while not pred():
        await asyncio.sleep(0.005)
        t += 0.005
        if t > timeout:
            raise AssertionError("condition not reached")


@pytest.fixture
def anyio_backend():
    return "asyncio"
