"""The answer step, identical for every arm apart from the memory block."""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from .questions import Question, format_instruction
from .world import fmt_ts

ANSWER_SYSTEM = """You are the AI assistant that works for Maya Chen, who runs a small design studio in London. You handle her email, calendar, tasks, documents, quotes, purchases and travel across many projects, and she sometimes asks you about your past work.

The current time is {now}. The most recent interaction was #{last:04d}.

Your memory of past activity:
<memory>
{memory}
</memory>

Answer Maya's question from your memory. If your memory does not settle the answer, give your best estimate rather than refusing."""

ANSWER_USER = """{question}

{fmt}
Reply with JSON only, in this form: {{"working": "<brief notes if useful, at most 60 words>", "answer": <the answer>}}"""


def messages(q: Question, memory: str, now: dt.datetime, last: int, cache: bool = False) -> list[dict[str, Any]]:
    system = ANSWER_SYSTEM.format(now=fmt_ts(now), last=last, memory=memory)
    content: Any = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}] if cache else system
    return [{"role": "system", "content": content},
            {"role": "user", "content": ANSWER_USER.format(question=q.text, fmt=format_instruction(q))}]


def parse(text: str) -> tuple[Any, bool]:
    """The answer field, and whether the reply parsed as JSON."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        try:
            d = json.loads(t[a:b + 1])
            if isinstance(d, dict) and "answer" in d:
                return d["answer"], True
        except json.JSONDecodeError:
            pass
    m = re.search(r'"answer"\s*:\s*(\[[^\]]*\]|"[^"]*"|[-\d.,]+)', t)
    if m:
        raw = m.group(1)
        try:
            return json.loads(raw), False
        except json.JSONDecodeError:
            return raw.strip('"'), False
    return None, False
