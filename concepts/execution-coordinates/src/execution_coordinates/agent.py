"""A minimal tool-calling loop for one request, the same in every arm.

Every model call goes through demos_common.LLM (ledger, cost, retries). For
Anthropic models the loop marks two cache breakpoints (the system prompt and
the latest message), as a deployment would; OpenAI models cache by
themselves. The stored transcript never carries the breakpoints.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from demos_common import LLM, LLMError

from .tools import Ctx, Toolset

NUDGE = "Please continue: use the tools, and finish the request by calling final_answer."
EARLY = ("Not recorded: final_answer was called together with other tools. Call final_answer on its own, after "
         "you have seen their results.")


def system_prompt(load_s: float, score_s: float, design: str) -> str:
    def secs(x: float) -> str:
        return f"about {x:g} second{'s' if x != 1 else ''}"

    note = {
        "threads": "This conversation holds only this client's requests, with your earlier work on them; each "
                   "client has a conversation of its own.",
        "inbox": "This conversation holds every request from every client, with your earlier work on them, in the "
                 "order they arrived.",
    }[design]
    return (
        "You are a data analyst working for several client companies at once. Each request starts with the "
        "client's name and the person asking, and requests from different clients arrive interleaved. " + note +
        " You work by running Python code with the run_python tool; pandas and numpy are installed.\n\n"
        "Two functions are defined in every Python interpreter you run code in:\n"
        "- load_export(client_id) returns the client's payments export as a pandas DataFrame, one row per "
        "transaction from January to April 2025, with the columns txn_id, created_at, customer_id, region, type "
        "(\"charge\" or \"refund\"), method (\"card\", \"bank_transfer\" or \"wallet\"), status (\"settled\", "
        "\"failed\" or \"pending\"), currency and amount, all text, exactly as exported. It queries the client's "
        f"warehouse and takes {secs(load_s)} per call.\n"
        "- score_customers(client_id) runs the client's churn-risk model and returns a pandas DataFrame with the "
        f"columns customer_id and risk (0 to 1). It takes {secs(score_s)} per call.\n\n"
        "Printed output longer than 6,000 characters is cut. When you have the answer, call final_answer with "
        "it, in the form the request asks for."
    )


def parse_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace(" ", "")
        s = re.sub(r"^[A-Z]{3}", "", s)
        s = re.sub(r"[A-Z]{3}$", "", s)
        try:
            return float(s)
        except ValueError:
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            return float(m.group()) if m else None
    return None


def parse_answer(raw: Any) -> Any:
    """final_answer's argument as a number or a {name: number} dict, or None."""
    if raw is None:
        return None
    val = raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            n = parse_number(v)
            if n is None:
                return None
            out[str(k)] = n
        return out
    return parse_number(val)


def with_cache(model: str, messages: list[dict]) -> list[dict]:
    if not model.startswith("anthropic/"):
        return messages
    msgs = copy.deepcopy(messages)

    def mark(m: dict) -> bool:
        c = m.get("content")
        if isinstance(c, str) and c:
            m["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
            return True
        return False

    mark(msgs[0])
    for m in reversed(msgs[1:]):
        if mark(m):
            break
    return msgs


def clean_assistant(msg: dict) -> dict:
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = [
            {"id": tc.get("id"), "type": "function",
             "function": {"name": tc["function"].get("name"), "arguments": tc["function"].get("arguments") or "{}"}}
            for tc in msg["tool_calls"]
        ]
    if msg.get("reasoning_details"):
        out["reasoning_details"] = msg["reasoning_details"]
    return out


@dataclass
class Outcome:
    answer_raw: Any = None
    answer: Any = None
    stopped: str = "no_answer"
    messages: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)


async def handle(llm: LLM, model: str, system: str, history: list[dict], request: str, toolset: Toolset, ctx: Ctx,
                 *, max_calls: int = 25, meta: dict | None = None, max_tokens: int = 8000,
                 reasoning: dict | None = None) -> Outcome:
    out = Outcome()
    new: list[dict] = [{"role": "user", "content": request}]
    nudges = 0
    for i in range(max_calls):
        msgs = [{"role": "system", "content": system}] + history + new
        try:
            reply = await llm.chat(model, with_cache(model, msgs), tools=toolset.schemas, tool_choice="auto",
                                   max_tokens=max_tokens, meta={**(meta or {}), "call": i}, reasoning=reasoning)
        except LLMError as e:
            out.stopped = "error"
            out.calls.append({"error": str(e)[:300]})
            break
        out.calls.append({"latency_s": reply.latency_s, "prompt_tokens": reply.prompt_tokens,
                          "completion_tokens": reply.completion_tokens, "reasoning_tokens": reply.reasoning_tokens,
                          "cached_tokens": reply.cached_tokens, "cost": reply.cost, "served": reply.model})
        msg = clean_assistant(reply.message)
        new.append(msg)
        if not msg.get("tool_calls"):
            if nudges >= 2:
                out.stopped = "no_answer"
                break
            nudges += 1
            new.append({"role": "user", "content": NUDGE})
            continue
        answered = False
        # final_answer finishes the request, so it only counts when it is the
        # message's only kind of call: an answer sent alongside code was
        # written before the code's result was seen (found with Sonnet 5.5 in
        # its smoke run; Haiku and Luna never did it).
        alone = all(tc["function"]["name"] == "final_answer" for tc in msg["tool_calls"])
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            if name == "final_answer" and not alone:
                new.append({"role": "tool", "tool_call_id": tc["id"], "content": EARLY})
                out.tools.append({"name": name, "args_raw": tc["function"]["arguments"], "early": True})
                continue
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                text = json.dumps({"error": f"The tool arguments were not a valid JSON object ({e})."})
                new.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
                out.tools.append({"name": name, "args_raw": tc["function"]["arguments"], "bad_args": True})
                continue
            if name == "final_answer":
                out.answer_raw = args.get("answer")
                out.answer = parse_answer(out.answer_raw)
                answered = True
                new.append({"role": "tool", "tool_call_id": tc["id"], "content": "Answer recorded."})
                out.tools.append({"name": name, "args": args})
                continue
            text, record = await toolset.call(name, args, ctx)
            new.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
            out.tools.append({"name": name, "args": args, "record": record})
        if answered:
            out.stopped = "answered"
            break
    else:
        out.stopped = "max_calls"
    out.messages = new
    return out
