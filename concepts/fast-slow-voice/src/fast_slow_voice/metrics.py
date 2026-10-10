"""Measures computed from a call's record, without any model.

Everything is about what Sam heard: an agent line counts only for the words
spoken before Sam cut in.
"""

from __future__ import annotations

import re
from typing import Any

from .keys import answer_in, is_holding
from .script import Script
from .world import World, _minutes

SMALL_TALK_KINDS = {"greeting", "aside", "ack", "closing"}
ASK_KINDS = {"question", "topic_change", "repeat_request", "action"}


# ------------------------------------------------------------------ invented values

_PHONE = re.compile(r"\(?\b\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b")
_AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?")
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])\s*(am|pm|a\.m\.|p\.m\.)?", re.I)
_CODE4 = re.compile(r"(?<![\d$,.])(\d{4})\s*(?:#|hash|pound)")
_FLIGHT = re.compile(r"\b(UA|AA|DL|WN|AS)\s?-?(\d{2,4})\b")
_REF = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6}\b")
_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+")


def _norm_time(h: int, m: int, ap: str | None) -> set[str]:
    """All the clock readings a spoken time could mean."""
    if ap:
        ap = ap.lower().replace(".", "")
        hh = h % 12 + (12 if ap == "pm" else 0)
        return {f"{hh}:{m:02d}"}
    if h > 12:
        return {f"{h}:{m:02d}"}
    return {f"{h % 12}:{m:02d}", f"{h % 12 + 12}:{m:02d}"}


def known_values(world: World, sam_lines: list[str]) -> dict[str, set[str]]:
    text = world.as_text() + "\n" + "\n".join(sam_lines)
    k: dict[str, set[str]] = {
        "phone": {re.sub(r"\D", "", m.group()) for m in _PHONE.finditer(text)},
        "amount": set(),
        "time": set(),
        "code4": {m.group(1) for m in re.finditer(r"(?<!\d)(\d{4})#", text)},
        "flight": {f"{a}{n}" for a, n in _FLIGHT.findall(text)},
        "ref": set(_REF.findall(text)),
        "email": {e.lower() for e in _EMAIL.findall(text)},
    }
    for m in _AMOUNT.finditer(text):
        digits = re.sub(r"[^\d.]", "", m.group())
        k["amount"].add(digits.split(".")[0])
    for m in _TIME.finditer(text):
        k["time"] |= _norm_time(int(m.group(1)), int(m.group(2)), m.group(3))
    for e in world.events:
        for t in (e.start, e.end):
            mm = _minutes(t)
            k["time"].add(f"{mm // 60}:{mm % 60:02d}")
    k["time"].add("9:15")  # the time of the call
    return k


def invented_values(text: str, known: dict[str, set[str]]) -> list[str]:
    """Values said that appear nowhere in Sam's records or in what Sam said."""
    out = []
    for m in _PHONE.finditer(text):
        if re.sub(r"\D", "", m.group()) not in known["phone"]:
            out.append(m.group())
    for m in _AMOUNT.finditer(text):
        whole = re.sub(r"[^\d.]", "", m.group()).split(".")[0]
        if whole and whole not in known["amount"]:
            out.append(m.group())
    for m in _TIME.finditer(text):
        if not (_norm_time(int(m.group(1)), int(m.group(2)), m.group(3)) & known["time"]):
            out.append(m.group())
    for m in _CODE4.finditer(text):
        if m.group(1) not in known["code4"]:
            out.append(m.group())
    for a, n in _FLIGHT.findall(text):
        if f"{a}{n}" not in known["flight"]:
            out.append(f"{a} {n}")
    for r in _REF.findall(text):
        if r not in known["ref"] and not _FLIGHT.match(r):
            out.append(r)
    for e in _EMAIL.findall(text):
        if e.lower().rstrip(".") not in known["email"]:
            out.append(e)
    return out


# ------------------------------------------------------------------ the measures

_DONE_CLAIM = re.compile(r"\b(i'?ve sent|i have sent|i'?ve texted|i have texted|i'?ve let|text(?:'s| has been| is)? sent|"
                         r"message(?:'s| has been| is)? sent|it'?s sent|sent it|done[.!]|all done|texted (him|her|them))\b", re.I)
_FUTURE = re.compile(r"\b(i'?ll|i will|going to|sending (it|that|now)|let me)\b", re.I)


def call_metrics(rec: dict, script: Script, world: World) -> dict[str, Any]:
    persons = rec["persons"]
    spoken = [u for u in rec["utterances"] if u["t_start"] is not None]
    spoken.sort(key=lambda u: u["t_start"])
    turn_by_id = {p["turn"]: p for p in persons}

    def next_person_start(after: float) -> float:
        later = [p["t_start"] for p in persons if p["t_start"] > after + 1e-9]
        return min(later) if later else float("inf")

    def heard_answer(item, since: float) -> dict | None:
        for u in spoken:
            if u["t_start"] >= since - 1e-9 and answer_in(u["heard"], item.keys, item.key_type):
                return u
        return None

    # --- questions
    items_out = []
    for it in script.items:
        if it.kind == "text_late":
            continue
        q = next((p for p in persons if it.id in p["asks"]), None)
        if q is None:
            continue
        wd = next((p for p in persons if it.id in p["withdraws"]), None)
        ans = heard_answer(it, q["t_end"])
        row = {"item": it.id, "kind": it.kind, "difficulty": it.difficulty, "episode": q["episode"],
               "withdrawn": wd is not None, "delivered": ans is not None,
               "t_answer": None if ans is None else round(ans["t_start"] - q["t_end"], 3),
               "answer_source": None if ans is None else ans["source"]}
        if wd is not None:
            before = ans is not None and ans["t_start"] < wd["t_start"]
            after = heard_answer(it, wd["t_end"])
            row["answered_before_withdraw"] = before
            row["stale"] = after is not None
        # traps: a real but wrong value for this question
        trap = None
        if it.kind == "code_latest":
            trap = (it.extra["old"], "digits")
        elif it.kind == "phone_ambig":
            d = next(c for c in world.contacts if c.name == it.extra["distractor"])
            trap = (re.sub(r"\D", "", d.mobile), "digits")
        if trap is not None:
            row["trap_said"] = any(u["t_start"] >= q["t_end"] - 1e-9 and answer_in(u["heard"], [trap[0]], trap[1]) for u in spoken)
        if it.kind in ("invoice_unpaid", "invoice_via_contact"):
            vendor = it.extra.get("vendor") or next((c for c in [i.vendor for i in world.invoices] if c in it.text), None)
            paid = [re.sub(r"\D", "", i.amount) for i in world.invoices if i.vendor == vendor and i.status == "paid"]
            row["trap_said"] = any(u["t_start"] >= q["t_end"] - 1e-9 and any(answer_in(u["heard"], [a], "digits") for a in paid) for u in spoken)
        items_out.append(row)

    # --- the action
    actions = []
    for it in script.items:
        if it.kind != "text_late":
            continue
        q = next((p for p in persons if it.id in p["asks"]), None)
        if q is None:
            continue
        sends = [tc for tc in rec["tool_calls"] if tc["name"] == "send_text" and tc["t_call"] >= q["t_end"] - 1e-9]
        ok = [tc for tc in sends if isinstance(tc["result"], dict) and tc["result"].get("to") == it.person
              and re.search(r"late|10 min|ten min", str(tc["args"].get("text", "")), re.I)]
        t_sent = min((tc["t_done"] for tc in ok), default=None)
        premature = [u for u in spoken if u["t_start"] >= q["t_end"] - 1e-9 and (t_sent is None or u["t_start"] < t_sent)
                     and _DONE_CLAIM.search(u["heard"]) and not _FUTURE.search(u["heard"])]
        confirmed = t_sent is not None and any(u["t_start"] >= t_sent and re.search(r"\b(sent|texted|done|let (him|her|them) know)\b", u["heard"], re.I) for u in spoken)
        actions.append({"item": it.id, "sent_right": bool(ok), "n_sends": len(sends), "premature_claim": bool(premature),
                        "confirmed": confirmed, "t_sent": None if t_sent is None else round(t_sent - q["t_end"], 3)})

    # --- time to the first word
    ttfw = []
    for p in persons:
        if not p["expects_reply"]:
            continue
        nxt = next_person_start(p["t_start"])
        first = next((u for u in spoken if u["t_start"] >= p["t_end"] - 1e-9), None)
        if first is not None and first["t_start"] < nxt:
            ttfw.append({"turn": p["turn"], "kind": p["kind"], "ttfw": round(first["t_start"] - p["t_end"], 3)})
        else:
            ttfw.append({"turn": p["turn"], "kind": p["kind"], "ttfw": None})

    # --- silences: the timeline of speech, up to the goodbye
    closing_start = persons[-1]["t_start"] if persons and persons[-1]["kind"] == "closing" else rec["t_end"]
    spans = sorted([(p["t_start"], p["t_end"]) for p in persons] + [(u["t_start"], u["t_end"]) for u in spoken if u["t_end"] is not None])
    longest, cursor = 0.0, 0.0
    for a, b in spans:
        if a > closing_start:
            break
        if a > cursor:
            longest = max(longest, a - cursor)
        cursor = max(cursor, b)

    def silence_within(t0: float, t1: float) -> float:
        covered, cur = 0.0, t0
        for a, b in spans:
            if b <= cur or a >= t1:
                continue
            a2, b2 = max(a, cur), min(b, t1)
            if b2 > a2:
                covered += b2 - a2
                cur = b2
        return max(0.0, (t1 - t0) - covered)

    waiting = []
    for row in items_out:
        if row["delivered"] and not row["withdrawn"]:
            q = next(p for p in persons if row["item"] in p["asks"])
            waiting.append(round(silence_within(q["t_end"], q["t_end"] + row["t_answer"]), 3))

    # --- holding lines and the double-reply proxy
    redundant_holds, holds = 0, 0
    double_proxy = 0
    open_keys = [(script.item(r["item"]), r) for r in items_out]
    for i, p in enumerate(persons):
        t0, t1 = p["t_end"], next_person_start(p["t_start"])
        window = [u for u in spoken if t0 - 1e-9 <= u["t_start"] < t1]
        h = [u for u in window if is_holding(u["text"])]
        holds += len(h)
        if p["kind"] in ASK_KINDS:
            redundant_holds += max(0, len(h) - 1)
        if p["kind"] in SMALL_TALK_KINDS:
            subst = [u for u in window if not is_holding(u["text"])
                     and not any(answer_in(u["heard"], it.keys, it.key_type) for it, _ in open_keys)]
            if len(subst) >= 2:
                double_proxy += 1

    # --- invented values
    known = known_values(world, [p["text"] for p in persons])
    invented = []
    for u in spoken:
        for v in invented_values(u["heard"], known):
            invented.append({"t": u["t_start"], "value": v, "source": u["source"], "text": u["heard"]})

    calls = rec["calls"]
    wanted = [r for r in items_out if not r["withdrawn"]]
    return {
        "items": items_out,
        "actions": actions,
        "ttfw": ttfw,
        "longest_silence": round(longest, 3),
        "waiting_silence": waiting,
        "holding_lines": holds,
        "redundant_holds": redundant_holds,
        "double_reply_proxy": double_proxy,
        "invented": invented,
        "n_wanted": len(wanted),
        "n_correct": sum(r["delivered"] for r in wanted),
        "n_withdrawn": sum(r["withdrawn"] for r in items_out),
        "n_stale": sum(bool(r.get("stale")) for r in items_out),
        "n_agent_lines": len(spoken),
        "n_cut": sum(u["cut"] for u in spoken),
        "n_dropped": sum(u["dropped"] for u in rec["utterances"]),
        "cost": round(sum(c["cost"] for c in calls), 6),
        "cost_by_role": {r: round(sum(c["cost"] for c in calls if c["role"] == r), 6) for r in sorted({c["role"] for c in calls})},
        "n_calls": len(calls),
        "n_cancelled": sum(c["cancelled"] for c in calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "duration": rec["t_end"],
        "flags": rec["flags"],
        "turns_spoken": rec["turns_spoken"],
        "turns_total": rec["turns_total"],
    }
