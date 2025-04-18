"""
Small wrapper around OpenAI ChatCompletion that converts free‑form English
into a structured action object *matching* CommandRunner’s capabilities.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal, Optional, TypedDict

import openai
from pydantic import BaseModel, Field, ValidationError


# ---------- pydantic output schema -----------------------------------------
class ActionName(str, Enum):
    click_button = "click_button"
    scroll = "scroll"          # one‑off smooth scroll
    start_scroll = "start_scroll"
    stop_scroll = "stop_scroll"
    new_tab = "new_tab"
    close_tab = "close_tab"
    switch_tab = "switch_tab"


class Action(BaseModel):
    """Structured command returned by the LLM."""
    action: ActionName
    direction: Optional[Literal["up", "down"]] = None
    pixels: Optional[int] = Field(
        None, ge=1, description="number of pixels for a single scroll"
    )
    tab_text: Optional[str] = None
    button_text: Optional[str] = None


# ---------- prompt pieces ---------------------------------------------------
_SYSTEM_PROMPT = """\
You are a controller that maps plain‑English user requests to structured\
 browser actions.

Available *actions* and their arguments:

• click_button      button_text(str) – click the first visible element
                    whose (innerText | aria‑label | title | alt | href)
                    contains the substring, case‑insensitive.
                    example: {"action":"click_button","button_text":"accept"}

• scroll            direction(up|down)  pixels(int>0)
                    example: {"action":"scroll","direction":"down","pixels":300}

• start_scroll      direction(up|down)
                    example: {"action":"start_scroll","direction":"down"}

• stop_scroll       (no args)
                    example: {"action":"stop_scroll"}

• new_tab           (no args)

• close_tab         tab_text(optional str) – substring of tab title. If omitted,
                    close the active tab.

• switch_tab        tab_text(str) – substring of desired tab title.

Rules:
1. Choose the single best action for the request.
2. Return ONLY valid JSON matching the schema with double quotes.
3. Do NOT wrap JSON in markdown.
"""

_MODEL = "o3-mini"
_TOOL_ID = "action_schema"  # OpenAI tool calling (optional but nice)

_TOOL_SPEC: list[dict[str, str]] = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_ID,
            "description": "Structured browser action",
            "parameters": Action.schema(),
        },
    }
]


# ---------- public helper ---------------------------------------------------
def parse_instruction(text: str) -> Action | None:
    """
    Ask the model to convert `text` into an `Action`.
    Returns None on failure.
    """
    try:
        chat = openai.ChatCompletion.create(
            model=_MODEL,
            temperature=0,
            tools=_TOOL_SPEC,
            tool_choice={"type": "function", "function": {"name": _TOOL_ID}},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        func_call = chat.choices[0].message.tool_calls[0]
        payload = json.loads(func_call.function.arguments)
        return Action.model_validate(payload)
    except (KeyError, IndexError, ValidationError, openai.OpenAIError) as e:
        # optional: log/print e for debugging
        return None


# quick manual test
if __name__ == "__main__":
    import os, sys

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first")
    print(parse_instruction("could you scroll down a bit more?"))
