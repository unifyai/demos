"""Setup probe (not an experiment): does each candidate model answer a tool call,
how long does it take, and does it reason by default? Writes to runs/ledger.jsonl
with meta phase=probe.

    ../../bin/with-openrouter-key uv run python probe.py
"""

from __future__ import annotations

import asyncio
import json
import sys

from demos_common import LLM

MODELS = sys.argv[1:] or [
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
    "google/gemini-3.8-flash",
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_profile",
        "description": "Open a member's profile page.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
}]
MSGS = [
    {"role": "system", "content": "You are an assistant doing a task with tools. Keep replies short."},
    {"role": "user", "content": "Check the profiles of Ana Ruiz and Ben Cole, then tell me which of them is away."},
]


async def main() -> None:
    async with LLM("runs/ledger.jsonl", "separate-loops", max_concurrency=4) as llm:
        for model in MODELS:
            for label, kw in [("default", {}), ("effort-low", {"reasoning": {"effort": "low"}}),
                              ("no-reasoning", {"reasoning": {"enabled": False}})]:
                for i in range(2):
                    try:
                        r = await llm.chat(model, MSGS, tools=TOOLS, meta={"phase": "probe", "variant": label}, **kw)
                        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in r.tool_calls]
                        print(json.dumps({"model": model, "variant": label, "served": r.model, "latency_s": round(r.latency_s, 2),
                                          "prompt": r.prompt_tokens, "completion": r.completion_tokens,
                                          "reasoning": r.reasoning_tokens, "cost": r.cost, "text": r.text[:80], "calls": calls}))
                    except Exception as e:  # noqa: BLE001 - a probe reports and moves on
                        print(json.dumps({"model": model, "variant": label, "error": str(e)[:200]}))


asyncio.run(main())
