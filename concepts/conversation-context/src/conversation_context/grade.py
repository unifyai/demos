"""Grading the top agent's final reply to Maya.

Patterns first: a reply that holds exactly one candidate's answer is labelled
by it. A reply that holds both, or neither, goes to a judge from the other
model family, which sees the question, the two candidate answers and the reply,
never the arm, and picks one of A, B, BOTH, ASKS, NONE.

Labels: correct (gives the target), wrong (gives the distractor), hedged (gives
both without settling), flagged (asks which, or says it can't tell, without
values), none (no answer, nothing found, another value).

Right outcome: settled, complete, and open items with Maya there: correct.
Open items with Maya away: hedged or flagged (the information is nobody's to
give, so the right reply says so rather than committing to one).
"""

from __future__ import annotations

import re
from typing import Any

from .scenarios import BY_ID

JUDGE_FOR = {"anthropic": "openai/gpt-6.1-sol", "openai": "anthropic/claude-sonnet-5.5"}
JUDGE_VERSION = "v2"  # bump when the judge prompt changes: cached labels carry it
JUDGE_SYSTEM = "You grade an assistant's reply to a person. Answer with one word."
JUDGE_USER = """The person asked: "{question}"

Two of their records could answer it:
A: {a}
B: {b}

The assistant's final reply:
\"\"\"{reply}\"\"\"

Which describes the reply?
A: it answers the question with A (it may also mention B as a different thing)
B: it answers the question with B (it may also mention A as a different thing)
BOTH: it gives A and B as alternatives and leaves the person to choose which one answers the question
ASKS: it gives neither, but asks which one is meant or says it can't tell which
NONE: anything else, such as no answer, nothing found, another value or an error

Answer with one word: A, B, BOTH, ASKS or NONE."""
JUDGE_LABEL = {"A": "correct", "B": "wrong", "BOTH": "hedged", "ASKS": "flagged", "NONE": "none"}


def matches(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text or "", re.I) for p in patterns)


def pattern_label(scenario_id: str, reply: str) -> str:
    s = BY_ID[scenario_id]
    t, d = matches(reply, s["target_value"]), matches(reply, s["distractor_value"])
    if t and not d:
        return "correct"
    if d and not t:
        return "wrong"
    return "both" if t and d else "neither"


def judge_family(model: str) -> str:
    """The judge is from the other family: a model under test from Anthropic is judged by OpenAI's, and vice versa."""
    return JUDGE_FOR["anthropic" if model.startswith("anthropic/") else "openai"]


def judge_prompt(scenario_id: str, question: str, reply: str) -> list[dict]:
    s = BY_ID[scenario_id]
    a = f"{s['target_answer']} (from: {s['target']['subject']}: {s['target']['body'][:160]})"
    b = f"{s['distractor_answer']} (from: {s['distractor']['subject']}: {s['distractor']['body'][:160]})"
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(question=question, a=a, b=b, reply=reply.strip())}]


def parse_judge(text: str) -> str | None:
    m = re.search(r"\b(BOTH|ASKS|NONE|A|B)\b", (text or "").strip().upper())
    return JUDGE_LABEL[m.group(1)] if m else None


def label(scenario_id: str, reply: str, judged: str | None) -> str:
    p = pattern_label(scenario_id, reply)
    if p in ("correct", "wrong"):
        return p
    return judged or "unjudged"


def right(kind: str, present: bool, lab: str) -> bool:
    if kind == "open" and not present:
        return lab in ("hedged", "flagged")
    return lab == "correct"


def outcome(record: dict[str, Any], judged: str | None) -> dict[str, Any]:
    """Per-run measures from a run record and its judge label (if it needed one)."""
    lab = label(record["scenario"], record["final"], judged)
    asked = record["q_caller"] + record["q_person"] > 0
    informed = any(q["answered"] for q in record["person_questions"])
    committed = lab in ("correct", "wrong")
    return {
        "label": lab,
        "right": right(record["kind"], record["present"], lab),
        "correct": lab == "correct",
        "asked": asked,
        "wrong_unasked": lab == "wrong" and not asked,
        "guessed": record["kind"] == "open" and committed and not informed,
    }
