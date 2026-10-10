"""The judge: reads a call as Sam heard it, with its timing, without knowing the
arm, and labels naturalness and the failures a model is needed for."""

from __future__ import annotations

import json
import re
from typing import Any

from .render import judge_transcript
from .script import Script
from .world import World

JUDGE_SYSTEM = """You review phone calls between Sam and Ava, Sam's AI personal assistant, for quality. You read the call as Sam heard it, with timings. Lines marked [cut off by Sam] stopped where Sam started talking; only the words shown were heard. Silences of a second or more are marked. Sam's lines are scripted: Sam may cut in, step away, change the subject or say the line cut out; none of that is Ava's fault. Judge only Ava: what she said and when.

You are also given the questions Sam asked with the correct answers from Sam's records, which of them Sam withdrew, and the records that matter. Ava may also mention other true details from Sam's records that are not listed.

Return one JSON object with these fields:
- "naturalness": an integer from 1 to 5 for how natural and smooth the call sounds as a phone conversation, judging timing, flow, coherence and redundancy, not whether the facts were right (that is scored separately):
  5 = like a capable human assistant on the phone: prompt replies, no awkward silences beyond brief pauses while checking, nothing said twice, answers delivered clearly when ready;
  4 = minor awkwardness (one redundant filler, one rather long pause, one slightly odd transition);
  3 = noticeable problems (several long silences, repeated fillers, a double reply, an answer arriving out of context, talking past Sam);
  2 = frequent problems (many long silences or double replies, confusing order, contradictions);
  1 = broken (Sam left waiting most of the time, answers missing or garbled, Ava talks over itself or ignores Sam).
- "naturalness_reason": one or two sentences.
- "double_replies": a list, one entry per Sam turn that got two separate substantive replies from Ava (two greetings, two answers to the same thing, a reply then a second reply saying the same or a different thing). A brief holding line followed by the answer is NOT a double reply, and neither is an answer to an earlier question that arrives after Sam's later turn. Each entry: {"time": <Sam's turn start, seconds>, "what": "..."}.
- "contradictions": a list of places where Ava said something that contradicts something Ava said earlier in the call. Each: {"time": <seconds>, "what": "..."}.
- "wrong_or_invented": a list of facts Ava stated that are wrong (contradict the correct answers or records given) or look made up (a number, time, amount, code, email, name, place or status that matches nothing given), and claims that something was done before it could have been. Each: {"time": <seconds>, "claim": "...", "why": "..."}.
- "stale": a list of answers Ava gave to a question after Sam had withdrawn it. Each: {"time": <seconds>, "what": "..."}.
- "redundant_fillers": the number of holding lines ("one sec", "let me check" and the like) beyond the first for the same request.

Return only the JSON object."""


def item_brief(script: Script, world: World, rec: dict) -> str:
    asked = {i for p in rec["persons"] for i in p["asks"]}
    withdrawn = {i for p in rec["persons"] for i in p["withdraws"]}
    lines = []
    for it in script.items:
        if it.id not in asked:
            continue
        q = it.text if it.kind != "repeat" else f"{it.text} (a request to repeat {script.item(it.extra['repeats']).text!r})"
        if it.kind == "text_late":
            lines.append(f"- Sam asked: {it.text!r} -> correct: a text to {it.person} saying Sam is running late.")
            continue
        extra = ""
        if it.kind == "code_latest":
            extra = f" (an older code, {it.extra['old']}#, no longer works)"
        if it.kind == "phone_ambig":
            extra = f" (not {it.extra['distractor']}, who has the same first name)"
        w = " Sam then WITHDREW this question." if it.id in withdrawn else ""
        lines.append(f"- Sam asked: {q!r} -> correct answer: {it.gold}{extra}.{w}")
    people = {it.person for it in script.items if it.id in asked and it.person}
    recs = []
    for c in world.contacts:
        if c.name in people:
            recs.append(f"contact: {c.name}, {c.company}, mobile {c.mobile}, email {c.email}")
    return "Questions and correct answers:\n" + "\n".join(lines) + ("\n\nRelevant records:\n" + "\n".join(recs) if recs else "")


def judge_messages(rec: dict, script: Script, world: World) -> list[dict]:
    user = f"{item_brief(script, world, rec)}\n\nThe call:\n{judge_transcript(rec)}"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_judgement(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    try:
        d["naturalness"] = int(d.get("naturalness"))
    except (TypeError, ValueError):
        return None
    for k in ("double_replies", "contradictions", "wrong_or_invented", "stale"):
        if not isinstance(d.get(k), list):
            d[k] = []
    try:
        d["redundant_fillers"] = int(d.get("redundant_fillers") or 0)
    except (TypeError, ValueError):
        d["redundant_fillers"] = 0
    return d
