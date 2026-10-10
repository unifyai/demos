"""One tool loop for every arm.

The model is called with the system prompt, the message and the arm's tools;
each tool call is executed and its result appended, until the model answers
without calling a tool or the step limit is reached. If the answer is not the
JSON object asked for, the model is asked once to give it.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .tools import Toolbox

NUDGE = "Reply with the JSON object only, in the form given in your instructions."


def parse_answer(text: str | None) -> dict | None:
    """The JSON object in a reply, or None."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("function" in obj or "rules" in obj):
            return obj
    return None


def _clean(message: dict[str, Any]) -> dict[str, Any]:
    """The assistant message to send back: role, content, tool calls (and reasoning details if any)."""
    out: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        out["tool_calls"] = [
            {"id": tc.get("id"), "type": "function",
             "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}}
            for tc in message["tool_calls"]
        ]
    if message.get("reasoning_details"):
        out["reasoning_details"] = message["reasoning_details"]
    if out["content"] is None and "tool_calls" not in out:
        out["content"] = ""
    return out


async def run_loop(llm, model: str, system: str, user: str, tools: Toolbox | None, *, max_steps: int,
                   tag: str, meta: dict[str, Any], max_tokens: int = 4000,
                   reasoning: dict[str, Any] | None = None) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls: list[dict[str, Any]] = []
    totals = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
              "cached_tokens": 0, "cost": 0.0, "model_seconds": 0.0}
    answer_text: str | None = None
    answer: dict | None = None
    nudged = False
    capped = False
    error = None
    t0 = time.time()
    try:
        steps = 0
        while True:
            if steps >= max_steps:
                capped = True
                break
            steps += 1
            kw: dict[str, Any] = {"max_tokens": max_tokens}
            if reasoning is not None:
                kw["reasoning"] = reasoning
            if tools is not None:
                kw["tools"] = tools.schemas()
            reply = await llm.chat(model, messages, tag=tag, meta={**meta, "step": steps}, **kw)
            totals["llm_calls"] += 1
            totals["prompt_tokens"] += reply.prompt_tokens
            totals["completion_tokens"] += reply.completion_tokens
            totals["reasoning_tokens"] += reply.reasoning_tokens
            totals["cached_tokens"] += reply.cached_tokens
            totals["cost"] += reply.cost
            totals["model_seconds"] += reply.latency_s
            msg = _clean(reply.message)
            messages.append(msg)
            if msg.get("tool_calls") and tools is not None:
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    raw = tc["function"]["arguments"]
                    try:
                        args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
                    except json.JSONDecodeError:
                        args = None
                    result = tools.call(name, args) if args is not None else "Error: the arguments are not valid JSON."
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    calls.append({"name": name, "args": args if args is not None else raw, "result": result})
                continue
            answer_text = reply.text
            answer = parse_answer(answer_text)
            if answer is None and not nudged:
                nudged = True
                messages.append({"role": "user", "content": NUDGE})
                continue
            break
    except Exception as e:  # noqa: BLE001 - a failed loop is recorded, not fatal to the run
        error = f"{type(e).__name__}: {e}"[:2000]
    return {
        "answer_text": answer_text, "answer": answer, "parsed": answer is not None, "nudged": nudged,
        "capped": capped, "error": error, "tool_calls": calls, "n_tool_calls": len(calls),
        "writes": list(tools.writes) if tools is not None else [],
        "wall_seconds": round(time.time() - t0, 3), **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in totals.items()},
        "system_chars": len(system),
    }
