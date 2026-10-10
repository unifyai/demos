"""Check that each model preset accepts its settings, reports reasoning tokens,
caches a repeated prefix, and can say a line and call a tool in one response.

  ../../bin/with-openrouter-key uv run python probe.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from demos_common import LLM

from fast_slow_voice import prompts as P
from fast_slow_voice.config import PRESETS
from fast_slow_voice.render import messages
from fast_slow_voice.world import TOOL_SPECS

HERE = Path(__file__).parent


async def main() -> None:
    lines = ["[0.5s] Sam: Hi Ava, it's Sam. How are you doing today?", "[3.1s] Ava: Morning, Sam! I'm well, thanks.",
             "[5.2s] Sam: What's Priya Raman's mobile number?"]
    async with LLM(HERE / "runs" / "ledger.jsonl", "fast-slow-voice/probe", max_concurrency=6) as llm:
        import sys
        names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["haiku", "luna", "sonnet-low", "sol-low", "judge"]
        for name in names:
            cfg = PRESETS[name]
            msgs = messages(P.SINGLE, lines, "Time now: 7.4s. Since your last turn: Sam spoke at 5.2s. Respond with what you say now, if anything, and any tool calls.", cfg.anthropic)
            kw = {"max_tokens": cfg.max_tokens, "tools": TOOL_SPECS, "tool_choice": "auto"}
            if cfg.reasoning is not None:
                kw["reasoning"] = cfg.reasoning
            for rep in range(2):
                r = await llm.chat(cfg.name, msgs, meta={"probe": name, "rep": rep}, **kw)
                print(f"{name} rep{rep}: served={r.model} prompt={r.prompt_tokens} cached={r.cached_tokens} "
                      f"completion={r.completion_tokens} reasoning={r.reasoning_tokens} cost=${r.cost:.5f} "
                      f"latency={r.latency_s:.2f}s text={r.text[:80]!r} tools={json.dumps([t['function'] for t in r.tool_calls])[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
