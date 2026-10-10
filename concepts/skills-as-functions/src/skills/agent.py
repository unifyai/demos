"""The agent loop every arm shares, and the system prompt.

The model has one tool, run_python. Its code runs in the task's namespace;
the task ends when the code calls submit(answer), or after MAX_STEPS model
calls. The arms differ only in the memory section of the system prompt, in
the conversation before the task (the long-context arm), in what the
namespace holds before the first cell (the library arm), and in what they keep
afterwards.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .api import OPS_BY_SIZE, Api
from .sandbox import Namespace
from .tasks import Task, check
from .world import clone, observable

MAX_STEPS = 10
MAX_TOKENS = 4096
CACHE = {"type": "ephemeral"}

RUN_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Run Python code in this task's namespace. Returns what the code printed, and the error if it raised one.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to run."}},
            "required": ["code"],
        },
    },
}
SUBMIT = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit the task's answer and end the task: the value the task asks for (a string, a number or a list), or null for a task that only asks you to do something.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"description": "The answer, or null."}},
            "required": ["answer"],
        },
    },
}
TOOLS = [RUN_PYTHON, SUBMIT]


def areas(size: str) -> str:
    ops = set(OPS_BY_SIZE[size])
    parts = ["contacts", "companies", "users"]
    if "search_tickets" in ops:
        parts.append("support tickets")
    if "list_orders" in ops:
        parts.append("orders")
    if "send_email" in ops:
        parts.append("email")
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def base_system(api: Api) -> str:
    """The part of the system prompt every arm sees, word for word."""
    return f"""You carry out tasks in Brightdesk, a company's customer workspace ({areas(api.size)}), by writing Python that calls its API. The tasks come from the company's staff, one at a time.

How you work:
- Run code with the run_python tool. Each task starts with a fresh Python namespace in which every API operation below is already defined as a function, and the modules re, math, json, datetime, collections, itertools, functools and statistics are imported. Names you define stay defined between run_python calls within a task.
- print() what you need to see. Output over 4,000 characters is cut.
- A failed API call raises ApiError, with a message saying why.
- When the task is done, submit the answer the task asks for (a string, a number or a list, as the task says), or None for a task that only asks you to do something: with the submit tool, or by calling submit(answer) in your code. Submitting ends the task, so do it only once you are sure.
- You have at most {MAX_STEPS} tool calls per task. Do only what the task asks: the answer and every change you make in the workspace are checked.
- Write clear code, with steps in functions that have docstrings where that helps.

API reference (version {api.version}):
{api.reference()}"""


@dataclass
class TaskRun:
    position: int
    kind: str
    calls: list[dict[str, Any]] = field(default_factory=list)  # one per model call
    messages: list[dict[str, Any]] = field(default_factory=list)  # this task's turns, after the prefix
    cells: list[dict[str, Any]] = field(default_factory=list)  # code, what the model saw, error type
    submitted: bool = False
    answer: Any = None
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    steps: int = 0
    api_calls: int = 0
    library_calls: list[str] = field(default_factory=list)
    library_failed: list[str] = field(default_factory=list)  # library functions seen in a traceback
    library_failed_at: list[str] = field(default_factory=list)  # the innermost one of each: where the error surfaced
    wall_s: float = 0.0


@dataclass
class TaskContext:
    """What an arm's memory update may use after the task: the world the agent
    left and its observable state at the check, the namespace, and the
    conversation so far (to continue it, as the library's save step does)."""

    world: dict[str, Any]
    checked_state: dict[str, Any]
    ns: Namespace
    request: Any  # () -> the messages of the task's conversation so far


def _with_cache(msg: dict[str, Any]) -> dict[str, Any]:
    """A copy of a user or system message whose text carries a cache breakpoint."""
    content = msg["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    content = [dict(p) for p in content]
    content[-1]["cache_control"] = CACHE
    return {**msg, "content": content}


def call_record(reply: Any, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "reasoning_tokens": reply.reasoning_tokens,
        "cached_tokens": reply.cached_tokens,
        "cost": reply.cost,
        "latency_s": round(reply.latency_s, 3),
    }


async def run_task(llm: Any, model: str, arm: Any, task: Task, size: str, meta: dict[str, Any]) -> tuple[TaskRun, TaskContext]:
    """One task, start to finish, for one arm."""
    t0 = time.time()
    world = clone(task.before)
    api = Api(world, task.version, size)
    ns = Namespace(api.functions(), extra=arm.namespace_extra())
    ns.load_library(arm.library_sources())
    system_parts = [{"type": "text", "text": base_system(api), "cache_control": CACHE}]
    memory = arm.memory_section(task)
    if memory:
        system_parts.append({"type": "text", "text": memory, "cache_control": CACHE})
    system = {"role": "system", "content": system_parts}
    prefix = arm.history()  # earlier tasks' turns (the long-context arm only)
    user = {"role": "user", "content": arm.user_message(task)}
    run = TaskRun(task.position, task.kind)

    def request() -> list[dict[str, Any]]:
        msgs = [system]
        marks = arm.history_cache_marks()  # indices into prefix to mark (long context)
        for i, m in enumerate(prefix):
            msgs.append(_with_cache(m) if i in marks else m)
        msgs.append(_with_cache(user))
        msgs.extend(run.messages)
        return msgs

    for _ in range(MAX_STEPS):
        reply = await llm.chat(model, request(), tools=TOOLS, tool_choice="required", temperature=0,
                               max_tokens=MAX_TOKENS, meta={**meta, "role": "agent", "step": run.steps})
        run.steps += 1
        run.calls.append(call_record(reply, "agent"))
        tool_calls = [{"id": tc.get("id"), "type": "function",
                       "function": {"name": (tc.get("function") or {}).get("name", "run_python"),
                                    "arguments": (tc.get("function") or {}).get("arguments") or ""}}
                      for tc in reply.tool_calls]
        if not tool_calls:
            run.messages.append({"role": "assistant", "content": reply.text or "(no reply)"})
            run.messages.append({"role": "user", "content": "Use the run_python tool, or the submit tool once the task is done."})
            continue
        run.messages.append({"role": "assistant", "content": reply.text or None, "tool_calls": tool_calls})
        for tc in tool_calls:
            if run.submitted:
                run.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Not run: the task was already submitted."})
                continue
            if tc["function"]["name"] == "submit":
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    if not isinstance(args, dict) or "answer" not in args:
                        raise ValueError
                except Exception:  # noqa: BLE001
                    run.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                         "content": 'Error: the arguments must be a JSON object with a field "answer".'})
                    run.cells.append({"code": None, "seen": "bad submit arguments", "error_type": "BadArguments"})
                    continue
                run.submitted, run.answer = True, args["answer"]
                run.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Submitted."})
                run.cells.append({"code": None, "seen": "submit tool", "error_type": None, "submitted": True,
                                  "cell": None, "tool_submit": True})
                continue
            try:
                code = json.loads(tc["function"]["arguments"])["code"]
                if not isinstance(code, str):
                    raise TypeError
            except Exception:  # noqa: BLE001
                run.messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "content": 'Error: the arguments must be a JSON object with a string field "code".'})
                run.cells.append({"code": None, "seen": "bad arguments", "error_type": "BadArguments"})
                continue
            r = ns.run(code)
            seen = ns.format(r)
            if r.library_in_traceback:
                run.library_failed.extend(n for n in r.library_in_traceback if n not in run.library_failed)
                if r.library_in_traceback[-1] not in run.library_failed_at:
                    run.library_failed_at.append(r.library_in_traceback[-1])
                seen += arm.traceback_note(r.library_in_traceback, ns)
            run.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": seen})
            run.cells.append({"code": code, "seen": seen, "error_type": r.error_type, "submitted": r.submitted, "cell": ns.cells})
            if r.submitted:
                run.submitted, run.answer = True, r.answer
        if run.submitted:
            break
        # tell the model its remaining budget, on the last tool result of the turn
        left = MAX_STEPS - run.steps
        if left > 0 and run.messages and run.messages[-1]["role"] == "tool":
            note = (f"[{left} tool calls left in this task]" if left > 1
                    else "[1 tool call left in this task: submit in it]")
            run.messages[-1]["content"] += "\n" + note

    run.api_calls = api.calls
    run.library_calls = sorted(ns.library_calls)
    ok, reasons = check(task, run.answer, world)
    run.passed = run.submitted and ok
    run.reasons = reasons if run.submitted else ["never submitted"] + reasons
    run.wall_s = time.time() - t0
    arm.record_turns(user, run.messages)
    return run, TaskContext(world, observable(world), ns, request)
