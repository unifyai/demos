"""A scripted stand-in for demos_common.LLM: no network, records every call."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest


@dataclass
class FakeReply:
    message: dict
    prompt_tokens: int = 100
    completion_tokens: int = 10
    reasoning_tokens: int = 0
    cost: float = 0.0001
    latency_s: float = 0.5

    @property
    def text(self) -> str:
        return self.message.get("content") or ""


def call(name: str, i: int = 1, **args: Any) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"call_{name}_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def say(text: str) -> dict:
    return {"role": "assistant", "content": text}


@dataclass
class FakeLLM:
    """policy(agent, system, messages, tools) -> assistant message. agent is meta['agent']."""

    policy: Callable[..., dict]
    calls: list[dict] = field(default_factory=list)

    async def chat(self, model: str, messages: list[dict], *, tools: list | None = None, max_tokens: int | None = None,
                   meta: dict | None = None, **kw: Any) -> FakeReply:
        agent = (meta or {}).get("agent", "")
        self.calls.append({"model": model, "agent": agent, "messages": messages, "tools": tools, "meta": meta})
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        msg = self.policy(agent, system, messages[1:], tools or [])
        return FakeReply(message=msg)


@pytest.fixture
def make_llm():
    return FakeLLM
