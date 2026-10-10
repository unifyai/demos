"""The judge (google/gemini-3.8-flash, a third family): which of the agent's
messages first answers a question about progress accurately, given the sites'
log at the time, and whether the agent's last messages report the outcome
accurately. Unrelated questions are checked without a judge (checks.py).
"""

from __future__ import annotations

import json
import re
from typing import Any

JUDGE_MODEL = "google/gemini-3.8-flash"
JUDGE_KW = {"max_tokens": 6000}
STATUS_KINDS = ("status", "status2", "status_both")

SYSTEM = "You check an AI assistant's messages against the record of what actually happened. You answer only as JSON."

STATUS_PROMPT = """A person asked their assistant a question about the progress of some work the assistant was doing for them.

The person's request at the start:
{request}

{changes}The question, at {qt}:
{question}

What the work actually did, from the websites' own log (times in mm:ss; an action is listed when it started, with its result when it finished):
{log}

The assistant's messages to the person after the question:
{msgs}

For each message, decide whether it answers the question accurately. It must address what was asked, and what it says about progress must agree with the log as of the time the message was sent: actions that had finished by then are done, an action that had started but not finished may be described either way, and actions that had not started are not done. Small slips (one item more or less while an action was finishing) are acceptable; claiming things were done that had not been, or a wrong count, is not. A message that only acknowledges the question ("let me check", "working on it") or gives no specifics does not answer it.

Answer with JSON only: {{"first_answer": <number of the first message that answers the question at all, or null>, "first_accurate": <number of the first message that answers it accurately, or null>, "notes": "<one sentence>"}}"""

OUTCOME_PROMPT = """A person asked their assistant to do some work, and sent more messages while it ran.

The person's messages:
{person}

The record of what was actually done, from the websites' state at the end:
{state}

The assistant's last messages to the person:
{msgs}

Did these messages tell the person the outcome of the work, accurately? "reported" is true if they state what was done with the key details (for invitations: who was invited and who was skipped; for a table booking: which restaurant was held, for how many people and when; for an order: the order number and total), or state clearly that the work was not finished and what is missing. "accurate" is true if nothing in them contradicts the record (for example saying something was done that was not, or the wrong restaurant, count, time or total).

Answer with JSON only: {{"reported": true or false, "accurate": true or false, "notes": "<one sentence>"}}"""


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _parse(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _short(v: Any, n: int) -> str:
    s = v if isinstance(v, str) else json.dumps(v, separators=(",", ":"))
    return s if len(s) <= n else s[: n - 3] + "..."


def action_log(rec: dict[str, Any], sites: set[str], until: float) -> str:
    lines = []
    for a in rec["actions"]:
        if a["site"] not in sites or a["t_start"] is None or a["t_start"] > until:
            continue
        args = ", ".join(f"{k}={_short(v, 90)}" for k, v in a["args"].items())
        if a["t_end"] is not None and a["t_end"] <= until:
            res = f" -> finished {_mmss(a['t_end'])}: {_short(a['result'], 160)}"
        else:
            res = " -> (not finished at the time of the last message)"
        lines.append(f"[{_mmss(a['t_start'])}] {a['tool']}({args}){res}")
    return "\n".join(lines) or "(nothing yet)"


def state_summary(rec: dict[str, Any], inst: dict[str, Any]) -> str:
    st = rec["state"]
    out = []
    sites = {inst["main"]["site"]} | ({inst["second"]["site"]} if inst.get("second") else set())
    if "messaging" in sites:
        by: dict[str, list[str]] = {}
        for m in st["messaging"]["sent"]:
            by.setdefault(m["to"], []).append(m["text"])
        out.append("Messages sent, by recipient:\n" + ("\n".join(
            f"- {k}: " + " | ".join(_short(x, 220) for x in v) for k, v in by.items()) or "(none)"))
        e = inst["main"]["expect"] if inst["main"]["site"] == "messaging" else inst["second"]["expect"]
        out.append(f"Group members marked away (not to be invited): {', '.join(e['away']) or 'none'}")
    if "booking" in sites:
        holds = [h for h in st["booking"]["holds"]]
        out.append("Table holds: " + (json.dumps(holds) if holds else "(none)"))
        out.append("Bookings shared: " + (json.dumps(st["booking"]["shares"]) if st["booking"]["shares"] else "(none)"))
    if "shop" in sites:
        out.append("Orders placed: " + (json.dumps(st["shop"]["orders"]) if st["shop"]["orders"] else "(none)"))
        out.append("Cart left at the end: " + json.dumps(st["shop"]["cart"]))
    return "\n".join(out)


async def judge_run(rec: dict[str, Any], inst: dict[str, Any], llm: Any, model: str = JUDGE_MODEL) -> dict[str, Any]:
    out: dict[str, Any] = {"run_id": rec["run_id"], "questions": {}, "outcome": None}
    agent = rec["agent"]
    sites = {inst["main"]["site"]} | ({inst["second"]["site"]} if inst.get("second") else set())
    person = rec["person"]
    meta = {"phase": "judge", "run": rec["run_id"]}
    for q in person:
        if q["kind"] not in STATUS_KINDS:
            continue
        horizon = q["t"] + 900  # late answers (blocking's come when the work returns) still count, as late
        after = [m for m in agent if m["t"] >= q["t"] and m["t"] <= horizon][:6]
        if not after:
            out["questions"][str(q["id"])] = {"first_answer": None, "first_accurate": None, "notes": "no messages",
                                              "t_first_accurate": None, "t_first_answer": None, "n_msgs": 0}
            continue
        changes = [m for m in person if m["t"] < q["t"] and m["kind"] not in ("request",)]
        ch = ("The person's later messages before the question:\n" + "\n".join(
            f"- [{_mmss(m['t'])}] {m['text']}" for m in changes) + "\n\n") if changes else ""
        msgs = "\n".join(f"{i + 1}. [{_mmss(m['t'])}] {m['text']}" for i, m in enumerate(after))
        prompt = STATUS_PROMPT.format(request=person[0]["text"], changes=ch, qt=_mmss(q["t"]), question=q["text"],
                                      log=action_log(rec, sites, after[-1]["t"]), msgs=msgs)
        r = await llm.chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                           meta={**meta, "item": f"q{q['id']}"}, **JUDGE_KW)
        v = _parse(r.text) or {}

        def at(k: str) -> float | None:
            i = v.get(k)
            if isinstance(i, int) and 1 <= i <= len(after):
                return round(after[i - 1]["t"] - q["t"], 2)
            return None
        out["questions"][str(q["id"])] = {"first_answer": v.get("first_answer"), "first_accurate": v.get("first_accurate"),
                                          "notes": v.get("notes"), "t_first_answer": at("first_answer"),
                                          "t_first_accurate": at("first_accurate"), "n_msgs": len(after),
                                          "parsed": bool(v)}
    # The outcome: the agent's messages after the last action on the work's sites (or its last three).
    acts = [a for a in rec["actions"] if a["site"] in sites and a["t_start"] is not None]
    last = max((a["t_end"] or a["t_start"] for a in acts), default=0.0)
    tail = [m for m in agent if m["t"] >= last] or agent[-3:]
    if tail:
        prompt = OUTCOME_PROMPT.format(
            person="\n".join(f"- [{_mmss(m['t'])}] {m['text']}" for m in person),
            state=state_summary(rec, inst),
            msgs="\n".join(f"- [{_mmss(m['t'])}] {m['text']}" for m in tail))
        r = await llm.chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                           meta={**meta, "item": "outcome"}, **JUDGE_KW)
        v = _parse(r.text) or {}
        out["outcome"] = {"reported": v.get("reported"), "accurate": v.get("accurate"), "notes": v.get("notes"),
                          "n_msgs": len(tail), "parsed": bool(v)}
    else:
        out["outcome"] = {"reported": False, "accurate": False, "notes": "no messages", "n_msgs": 0, "parsed": True}
    return out
