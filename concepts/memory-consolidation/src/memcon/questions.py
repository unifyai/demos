"""Checkpoints, questions with exact answers, and scoring.

At each checkpoint the agent is asked questions whose answers the generator
knows exactly, from the structured facts of the interactions before that
moment:

  T  which projects were worked on in a period: today, the past 7, 28 and 84
     days, ever, and one past calendar week (set F1)
  C  which projects the last 10, 40, 120 and 480 interactions involved (set F1)
  D  a detail: the current quote for an item, the current date of a meeting,
     who a document was shared with, the price of a purchase; drawn by age
     (up to 7 days old, 8 to 28 days, older) (exact)
  N  how many of something happened in the past 7 or 28 days (exact)
  G  gaps: when a dormant project was last touched; the last project worked
     on before the most recent stretch of five or more days with no
     interactions, and how long that stretch was (exact)
"""

from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .world import DAYS, PROJECT_NAMES, START, Interaction, day_of

CHECKPOINT_DAYS = {"2w": 14, "8w": 56, "16w": 112, "26w": 182}
COUNT_WINDOWS = (10, 40, 120, 480)
BREAK_MIN_DAYS = 5


@dataclass
class Question:
    qid: str
    seed: int
    checkpoint: str
    now: str  # ISO datetime; only interactions strictly before it are visible
    family: str  # T, C, D, N, G
    qtype: str  # finer type, e.g. T_7d, C_40, D_quote, N_meeting_7d, G_last_touch
    text: str
    fmt: str  # projects, project, amount, date, person, count
    gold: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def checkpoints(seed: int, interactions: list[Interaction], breaks: list[tuple[int, int]], days: int = DAYS) -> dict[str, dt.datetime]:
    """Four checkpoints at about 2, 8, 16 and 26 weeks (a weekday morning to
    evening, Monday to Thursday), and one on the first active day after the
    longest break, just after its third interaction."""
    rng = random.Random(31 * seed + 5)
    out: dict[str, dt.datetime] = {}
    for label, d0 in CHECKPOINT_DAYS.items():
        if d0 + 4 > days:
            continue
        d = d0 + rng.randint(0, 3)
        out[label] = START + dt.timedelta(days=d, hours=rng.randint(10, 18), minutes=rng.choice([0, 15, 30, 45]))
    if breaks:
        a, b = max(breaks, key=lambda x: x[1] - x[0])
        after = [it for it in interactions if day_of(it.ts) >= b]
        if after:
            first_day = day_of(after[0].ts)
            today = [it for it in after if day_of(it.ts) == first_day]
            pick = today[min(2, len(today) - 1)]
            out["break"] = pick.ts + dt.timedelta(minutes=5)
    return out


def visible(interactions: list[Interaction], now: dt.datetime) -> list[Interaction]:
    return [it for it in interactions if it.ts < now]


def projects_in(its: list[Interaction]) -> list[str]:
    return sorted({it.project for it in its if it.project})


def _long_date(d: dt.date) -> str:
    return f"{d:%A} {d.day} {d:%B} {d.year}"


def _short_date(d: dt.date) -> str:
    return f"{d:%A} {d.day} {d:%B}"


def make_questions(seed: int, label: str, now: dt.datetime, interactions: list[Interaction]) -> list[Question]:
    rng = random.Random(f"{seed}-{label}")
    vis = visible(interactions, now)
    qs: list[Question] = []

    def add(family: str, qtype: str, text: str, fmt: str, gold: Any, **meta: Any) -> None:
        qs.append(Question(f"s{seed}-{label}-{qtype}", seed, label, now.isoformat(), family, qtype, text, fmt, gold, meta))

    # T: projects by period. Periods are whole calendar days ending today, and
    # the question names them, so "the past 7 days" is today and the 6 days before.
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def since(days: int) -> tuple[str, dt.datetime]:
        a = today - dt.timedelta(days=days - 1)
        return f"({_short_date(a)} to today)", a

    periods = [("T_today", f"today ({_short_date(today)})", today)]
    for qtype, days, phrase in (("T_7d", 7, "in the past 7 days"), ("T_28d", 28, "in the past 28 days"), ("T_84d", 84, "in the past 12 weeks")):
        dates, a = since(days)
        periods.append((qtype, f"{phrase} {dates}", a))
    periods.append(("T_all", "since you started working for her", START))
    for qtype, phrase, start in periods:
        gold = projects_in([it for it in vis if it.ts >= start])
        add("T", qtype, f"Which projects did you do any work on for Maya {phrase}?", "projects", gold, start=start.isoformat())
    weeks = [w for w in range(DAYS // 7) if START + dt.timedelta(days=7 * w + 7) <= now - dt.timedelta(days=7)]
    if weeks:
        w = rng.choice(weeks)
        a = START + dt.timedelta(days=7 * w)
        b = a + dt.timedelta(days=7)
        gold = projects_in([it for it in vis if a <= it.ts < b])
        add("T", "T_week", f"Which projects did you do any work on for Maya in the week of {_long_date(a.date())} (Monday to Sunday)?",
            "projects", gold, start=a.isoformat(), end=b.isoformat(), weeks_ago=(now - b).days / 7)

    # C: projects by count
    for n in COUNT_WINDOWS:
        if n <= len(vis):
            gold = projects_in(vis[-n:])
            add("C", f"C_{n}", f"Which projects did your last {n} interactions with Maya involve?", "projects", gold,
                n=n, span_days=(now - vis[-n].ts).total_seconds() / 86400)

    # D: details, two per age bucket (up to 7 days old, 8 to 28, older)
    cands = _detail_candidates(vis, now)
    buckets = {"recent": [], "medium": [], "old": []}
    for c in cands:
        age = c["age_days"]
        buckets["recent" if age <= 7 else "medium" if age <= 28 else "old"].append(c)
    chosen: list[dict[str, Any]] = []
    for name in ("recent", "medium", "old"):
        chosen += _pick_details(rng, buckets[name], 2, chosen)
    while len(chosen) < 6 and len(chosen) < len(cands):  # fill from any bucket when one is short
        extra = _pick_details(rng, cands, 1, chosen)
        if not extra:
            break
        chosen += extra
    for k, c in enumerate(chosen):
        age = c["age_days"]
        bucket = "recent" if age <= 7 else "medium" if age <= 28 else "old"
        add("D", f"D_{c['kind']}_{k}", c["text"], c["fmt"], c["gold"], age_days=round(age, 2), bucket=bucket, kind=c["kind"])

    # N: counts in the past 7 and 28 days
    kinds = [("meeting", "meetings did you newly book", " Count every meeting you booked in that time, including any you later moved; moving an existing meeting is not a new booking."),
             ("task_done", "tasks did you mark as done", ""),
             ("quote", "quotes did you log", " Count every quote logged for the first time in that time, including any later revised; a revision of an existing quote does not count."),
             ("share", "documents did you share", "")]
    picks = rng.sample(kinds, 2)
    for (action, phrase, note), days_ in zip(picks, (7, 28)):
        dates, start = since(days_)
        gold = sum(1 for it in vis if it.ts >= start and it.action == action)
        text = f"How many {phrase} in the past {days_} days {dates}?{note}"
        add("N", f"N_{action}_{days_}d", text, "count", gold, days=days_)

    # G: gaps
    dormant = []
    for p in projects_in(vis):
        last = max(it.ts for it in vis if it.project == p)
        if (now - last).days > 7:
            dormant.append((p, last))
    if dormant:
        p, last = rng.choice(dormant)
        add("G", "G_last_touch", f"On what date did you last do anything for the {p}?", "date", last.date().isoformat(),
            days_since=(now - last).days)
    gap = last_gap(vis, now)
    if gap:
        g0, g1 = gap  # day indices [g0, g1) with no interactions
        before = [it for it in vis if day_of(it.ts) < g0 and it.project]
        if before:
            add("G", "G_before_gap",
                f"Before the most recent stretch of {BREAK_MIN_DAYS} or more days with no interactions at all, what was the last project you worked on?",
                "project", before[-1].project, gap_days=g1 - g0, days_since=(now - (START + dt.timedelta(days=g1))).days)
        add("G", "G_gap_length",
            f"How many days long was the most recent stretch of {BREAK_MIN_DAYS} or more consecutive days with no interactions at all?",
            "count", g1 - g0, days_since=(now - (START + dt.timedelta(days=g1))).days)
    return qs


def last_gap(vis: list[Interaction], now: dt.datetime) -> tuple[int, int] | None:
    """The most recent run of at least BREAK_MIN_DAYS whole days with no
    interactions, among the days before today, that activity has followed."""
    today = day_of(now)
    active = {day_of(it.ts) for it in vis if it.ts < now}
    best = None
    d = 0
    while d < today:
        if d not in active:
            e = d
            while e < today and e not in active:
                e += 1
            if e - d >= BREAK_MIN_DAYS and e < today + 1 and any(x >= e for x in active):
                best = (d, e)
            d = e
        else:
            d += 1
    return best


def _detail_candidates(vis: list[Interaction], now: dt.datetime) -> list[dict[str, Any]]:
    quotes: dict[tuple[str, str], Interaction] = {}
    meetings: dict[tuple[str, str, str], Interaction] = {}
    shares: dict[tuple[str, str], Interaction] = {}
    purchases: dict[tuple[str, str], Interaction] = {}
    for it in vis:
        f = it.facts
        if it.action in ("quote", "revise_quote"):
            quotes[(it.project, f["item"])] = it
        elif it.action in ("meeting", "move_meeting"):
            meetings[(it.project, f["topic"], f["person"])] = it
        elif it.action == "share":
            shares[(it.project, f["document"])] = it
        elif it.action == "purchase":
            purchases[(it.project, f["item"])] = it
    out = []
    for (p, item), it in quotes.items():
        f = it.facts
        out.append(dict(kind="quote", idx=it.idx, key=f"q|{p}|{item}", age_days=(now - it.ts).total_seconds() / 86400, fmt="amount", gold=f["amount"],
                        text=f"What is the current quote from {f['org']} for the {item} for the {p}? Give the amount in {f['currency']}."))
    for (p, topic, person), it in meetings.items():
        out.append(dict(kind="meeting", idx=it.idx, key=f"m|{p}|{topic}|{person}", age_days=(now - it.ts).total_seconds() / 86400, fmt="date", gold=it.facts["date"],
                        text=f"On what date is the {topic} with {person} for the {p} scheduled?"))
    for (p, doc), it in shares.items():
        out.append(dict(kind="share", idx=it.idx, key=f"s|{p}|{doc}", age_days=(now - it.ts).total_seconds() / 86400, fmt="person", gold=it.facts["person"],
                        text=f"Who did you share the {doc} for the {p} with?"))
    for (p, item), it in purchases.items():
        f = it.facts
        out.append(dict(kind="purchase", idx=it.idx, key=f"p|{p}|{item}", age_days=(now - it.ts).total_seconds() / 86400, fmt="amount", gold=f["amount"],
                        text=f"How much did the {item} for the {p} cost? Give the amount in {f['currency']}."))
    out.sort(key=lambda c: c["key"])
    return out


def _pick_details(rng: random.Random, pool: list[dict[str, Any]], k: int, taken: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {c["key"] for c in taken}
    pool = [c for c in pool if c["key"] not in keys]
    out: list[dict[str, Any]] = []
    for _ in range(k):
        if not pool:
            break
        kinds = sorted({c["kind"] for c in pool})
        kind = rng.choice(kinds)  # spread across kinds
        c = rng.choice([c for c in pool if c["kind"] == kind])
        out.append(c)
        pool = [x for x in pool if x["key"] != c["key"]]
    return out


# ------------------------------------------------------------------ prompts

def format_instruction(q: Question) -> str:
    names = "; ".join(PROJECT_NAMES)
    return {
        "projects": f"The answer is a JSON list of project names, each written exactly as in this list: {names}. Use [] if there were none.",
        "project": f"The answer is one project name, written exactly as in this list: {names}.",
        "amount": "The answer is a number (no currency symbol, no thousands separators).",
        "date": "The answer is a date written as YYYY-MM-DD.",
        "person": "The answer is the person's full name.",
        "count": "The answer is a whole number.",
    }[q.fmt]


# ------------------------------------------------------------------ scoring

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(x))
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def score(q: Question, pred: Any) -> float:
    """1 for a right answer, 0 for a wrong or missing one; F1 for sets."""
    if pred is None:
        return 0.0
    if q.fmt == "projects":
        if isinstance(pred, str):
            pred = [pred] if pred.strip() else []
        if not isinstance(pred, list):
            return 0.0
        canon = {_norm(n): n for n in PROJECT_NAMES}
        p = {canon.get(_norm(x), "?" + _norm(x)) for x in pred if str(x).strip()}
        g = set(q.gold)
        if not p and not g:
            return 1.0
        tp = len(p & g)
        if tp == 0:
            return 0.0
        prec, rec = tp / len(p), tp / len(g)
        return 2 * prec * rec / (prec + rec)
    if q.fmt == "project":
        return float(_norm(pred) == _norm(q.gold))
    if q.fmt in ("amount", "count"):
        v = _number(pred)
        return float(v is not None and abs(v - float(q.gold)) < 0.5)
    if q.fmt == "date":
        m = re.search(r"\d{4}-\d{2}-\d{2}", str(pred))
        return float(bool(m) and m.group(0) == q.gold)
    if q.fmt == "person":
        return float(_norm(q.gold) in _norm(pred))
    raise ValueError(q.fmt)


SANITY_TYPES = ("T_today", "T_7d", "T_all", "C_10", "C_40")


def sanity_questions(seed: int, now: dt.datetime, interactions: list[Interaction]) -> list[Question]:
    """Questions every arm should answer on a history that fits the budget:
    projects today, this week and ever, the last 10 and 40 interactions, the
    two counts, and details from the last ten interactions."""
    qs = [q for q in make_questions(seed, "sanity", now, interactions) if q.qtype in SANITY_TYPES or q.family == "N"]
    vis = visible(interactions, now)
    recent_idx = {it.idx for it in vis[-10:]}
    for k, c in enumerate(c for c in _detail_candidates(vis, now) if c["idx"] in recent_idx):
        qs.append(Question(f"s{seed}-sanity-D_{c['kind']}_{k}", seed, "sanity", now.isoformat(), "D", f"D_{c['kind']}_{k}",
                           c["text"], c["fmt"], c["gold"], {"age_days": round(c["age_days"], 2), "bucket": "recent", "kind": c["kind"]}))
    return qs
