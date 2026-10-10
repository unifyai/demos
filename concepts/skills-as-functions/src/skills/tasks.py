"""Task kinds, the stream schedule of each size, and the exact check.

A stream is a fixed schedule of kinds per size (the same for every seed) whose
arguments and data come from the seed. Each task's expected answer and the
workspace state it should leave are computed by a reference solution that
works on the world's data directly; the next task starts from that state, so
one arm's mistake never changes what a later task expects.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .world import TODAY, clone, diff, generate, observable

SUBJECTS = ["Quick check-in", "Your account review", "Service update", "Following up on your request",
            "Thank you", "Holiday opening hours", "Renewal reminder", "New feature: shared reports"]
BODIES = [
    "Thanks for working with us this year. Reply to this email if there is anything we can help with.",
    "Our support line will be closed on Friday. Urgent requests will be answered on Monday.",
    "We have upgraded our reporting tools. Your team can now share reports with one click.",
    "Your account review is due next month. We will be in touch to arrange a time.",
    "We noticed you had trouble recently and wanted to check that everything is working now.",
    "Your renewal date is coming up. Let us know if your plan should change.",
]
APOLOGIES = [
    "We are sorry about your recent order. Your refund is on its way.",
    "Apologies for the trouble with your order. We have refunded it in full.",
    "We are sorry this order did not meet your expectations. A full refund has been issued.",
]
REASONS = ["Arrived damaged", "Late delivery", "Wrong item sent", "Customer changed their mind", "Duplicate order"]
TAGS = ["q3-review", "renewal-due", "event-invite", "priority-support", "survey-2026"]
REPLIES = [
    "We are looking into this now and will update you today.",
    "This has been escalated to our senior team, who will contact you shortly.",
    "Thanks for your patience. A specialist is now handling this ticket.",
]
DIGEST_SUBJECTS = ["Open tickets this week", "Open ticket digest", "Tickets still open"]
SINCE_DAYS = [30, 60, 90, 120]


# -- helpers over the world's data -------------------------------------------------
def _contacts_at(w: dict, company_id: str) -> list[dict]:
    return [c for c in w["contacts"].values() if c["company_id"] == company_id]


def _open_tickets(w: dict, contact_id: str) -> list[dict]:
    return [t for t in w["tickets"].values() if t["contact_id"] == contact_id and t["status"] == "open"]


def _orders(w: dict, contact_id: str, status: str | None = None) -> list[dict]:
    os_ = [o for o in w["orders"].values() if o["contact_id"] == contact_id and (status is None or o["status"] == status)]
    return sorted(os_, key=lambda o: (o["placed_on"], o["id"]), reverse=True)


def _spend(w: dict, contact_id: str, since: str) -> int:
    return sum(o["total_cents"] for o in _orders(w, contact_id, "delivered") if o["placed_on"] >= since)


def _since(days: int) -> str:
    import datetime as dt
    return (dt.date.fromisoformat(TODAY) - dt.timedelta(days=days)).isoformat()


def _agents(w: dict) -> list[dict]:
    return [u for u in w["users"].values() if u["role"] == "support agent"]


def _send(w: dict, to: str, subject: str, body: str) -> None:
    w["outbox"].append({"to": to, "subject": subject, "body": body})


def _reply(w: dict, t: dict, body: str) -> None:
    w["replies"].append({"ticket_id": t["id"], "body": body, "on": w["today"]})
    t["last_reply_on"] = w["today"]
    t["version"] += 1


def _refund_latest(w: dict, contact_id: str, reason: str) -> None:
    o = _orders(w, contact_id, "delivered")[0]
    o["status"] = "refunded"
    w["refunds"].append({"order_id": o["id"], "amount_cents": o["total_cents"], "reason": reason})


# -- kinds ------------------------------------------------------------------------------
@dataclass
class Kind:
    name: str
    level: int
    answer: str  # "str", "int", "list" or "none" (an action: only the state is checked)
    repeatable: bool
    uses: tuple[str, ...]  # shared steps, for analysis
    sample: Callable[[dict, random.Random], dict | None]
    text: Callable[[dict], str]
    solve: Callable[[dict, dict], Any]


def _pick(rng: random.Random, xs: list) -> Any:
    return rng.choice(xs) if xs else None


def _s_contact(w, rng):
    return {"contact": rng.choice(list(w["contacts"].values()))["name"]}


def _s_company(w, rng):
    return {"company": rng.choice(list(w["companies"].values()))["name"]}


def _cid(w, name):
    return next(c["id"] for c in w["contacts"].values() if c["name"] == name)


def _coid(w, name):
    return next(c["id"] for c in w["companies"].values() if c["name"] == name)


def _uid(w, name):
    return next(u["id"] for u in w["users"].values() if u["name"] == name)


KINDS: dict[str, Kind] = {}


def kind(name, level, answer, repeatable, uses):
    def deco(cls_fns):
        sample, text, solve = cls_fns()
        KINDS[name] = Kind(name, level, answer, repeatable, uses, sample, text, solve)
        return cls_fns
    return deco


@kind("contact_email", 1, "str", True, ("find_contact",))
def _k():
    def text(a):
        return f"What is {a['contact']}'s primary email address?"

    def solve(w, a):
        return w["contacts"][_cid(w, a["contact"])]["email"]
    return _s_contact, text, solve


@kind("contact_company", 1, "str", True, ("find_contact",))
def _k():
    def text(a):
        return f"Which company does {a['contact']} work for? Answer with the company's name."

    def solve(w, a):
        return w["companies"][w["contacts"][_cid(w, a["contact"])]["company_id"]]["name"]
    return _s_contact, text, solve


@kind("account_manager", 1, "str", True, ("find_company", "users"))
def _k():
    def text(a):
        return f"Who is the account manager for {a['company']}? Answer with their full name."

    def solve(w, a):
        return w["users"][w["companies"][_coid(w, a["company"])]["account_manager_id"]]["name"]
    return _s_company, text, solve


@kind("open_ticket_count", 1, "int", True, ("find_contact", "tickets"))
def _k():
    def sample(w, rng):
        with_open = [c for c in w["contacts"].values() if _open_tickets(w, c["id"])]
        pool = with_open if (with_open and rng.random() < 0.75) else list(w["contacts"].values())
        return {"contact": rng.choice(pool)["name"]}

    def text(a):
        return f"How many open tickets does {a['contact']} have? Answer with a number."

    def solve(w, a):
        return len(_open_tickets(w, _cid(w, a["contact"])))
    return sample, text, solve


@kind("latest_order_status", 1, "str", True, ("find_contact", "orders"))
def _k():
    def sample(w, rng):
        c = _pick(rng, [c for c in w["contacts"].values() if _orders(w, c["id"])])
        return c and {"contact": c["name"]}

    def text(a):
        return f"What is the status of {a['contact']}'s most recent order?"

    def solve(w, a):
        return _orders(w, _cid(w, a["contact"]))[0]["status"]
    return sample, text, solve


@kind("company_spend", 2, "int", True, ("find_company", "company_contacts", "orders"))
def _k():
    def sample(w, rng):
        return {"company": rng.choice(list(w["companies"].values()))["name"], "since": _since(rng.choice(SINCE_DAYS))}

    def text(a):
        return (f"How much have the contacts at {a['company']} spent on delivered orders placed on or after "
                f"{a['since']}? Answer in cents, as an integer.")

    def solve(w, a):
        return sum(_spend(w, c["id"], a["since"]) for c in _contacts_at(w, _coid(w, a["company"])))
    return sample, text, solve


@kind("agent_open_tickets", 2, "list", True, ("users", "tickets"))
def _k():
    def sample(w, rng):
        busy = [u for u in _agents(w)
                if sum(t["assignee_id"] == u["id"] and t["status"] == "open" for t in w["tickets"].values()) >= 2]
        u = _pick(rng, busy)
        return u and {"agent": u["name"]}

    def text(a):
        return (f"Which open tickets are assigned to {a['agent']}? Answer with a list of their IDs, most urgent first "
                f"(priority 1 is the most urgent), and oldest first among tickets of equal priority.")

    def solve(w, a):
        uid = _uid(w, a["agent"])
        ts = [t for t in w["tickets"].values() if t["assignee_id"] == uid and t["status"] == "open"]
        return [t["id"] for t in sorted(ts, key=lambda t: (t["priority"], t["created_on"], t["id"]))]
    return sample, text, solve


@kind("email_contact", 2, "none", True, ("find_contact", "email"))
def _k():
    def sample(w, rng):
        return {"contact": rng.choice(list(w["contacts"].values()))["name"],
                "subject": rng.choice(SUBJECTS), "body": rng.choice(BODIES)}

    def text(a):
        return (f"Email {a['contact']} at their primary address with the subject \"{a['subject']}\" "
                f"and the body \"{a['body']}\".")

    def solve(w, a):
        _send(w, w["contacts"][_cid(w, a["contact"])]["email"], a["subject"], a["body"])
    return sample, text, solve


@kind("reassign_tickets", 2, "none", False, ("users", "tickets", "ticket_update"))
def _k():
    def sample(w, rng):
        agents = _agents(w)
        busy = [u for u in agents if any(t["assignee_id"] == u["id"] and t["status"] == "open" for t in w["tickets"].values())]
        a = _pick(rng, busy)
        if a is None:
            return None
        b = rng.choice([u for u in agents if u["id"] != a["id"]])
        return {"from": a["name"], "to": b["name"]}

    def text(a):
        return f"Reassign every open ticket assigned to {a['from']} to {a['to']}."

    def solve(w, a):
        src, dst = _uid(w, a["from"]), _uid(w, a["to"])
        for t in w["tickets"].values():
            if t["assignee_id"] == src and t["status"] == "open":
                t["assignee_id"] = dst
                t["version"] += 1
    return sample, text, solve


@kind("tag_company_city", 2, "none", False, ("find_company", "company_contacts"))
def _k():
    def sample(w, rng):
        options = []
        for co in w["companies"].values():
            for city in sorted({c["city"] for c in _contacts_at(w, co["id"])}):
                for tag in TAGS:
                    if any(tag not in c["tags"] for c in _contacts_at(w, co["id"]) if c["city"] == city):
                        options.append((co["name"], city, tag))
        o = _pick(rng, options)
        return o and {"company": o[0], "city": o[1], "tag": o[2]}

    def text(a):
        return f"Tag every contact at {a['company']} who is based in {a['city']} with \"{a['tag']}\"."

    def solve(w, a):
        for c in _contacts_at(w, _coid(w, a["company"])):
            if c["city"] == a["city"] and a["tag"] not in c["tags"]:
                c["tags"] = sorted(c["tags"] + [a["tag"]])
    return sample, text, solve


@kind("refund_latest", 2, "none", False, ("find_contact", "orders", "refund"))
def _k():
    def sample(w, rng):
        c = _pick(rng, [c for c in w["contacts"].values() if _orders(w, c["id"], "delivered")])
        return c and {"contact": c["name"], "reason": rng.choice(REASONS)}

    def text(a):
        return f"Refund {a['contact']}'s most recent delivered order in full, with the reason \"{a['reason']}\"."

    def solve(w, a):
        _refund_latest(w, _cid(w, a["contact"]), a["reason"])
    return sample, text, solve


@kind("email_open_ticket_contacts", 3, "none", True, ("find_company", "company_contacts", "tickets", "email"))
def _k():
    def sample(w, rng):
        cos = [co for co in w["companies"].values() if any(_open_tickets(w, c["id"]) for c in _contacts_at(w, co["id"]))]
        co = _pick(rng, cos)
        return co and {"company": co["name"], "subject": rng.choice(SUBJECTS), "body": rng.choice(BODIES)}

    def text(a):
        return (f"Email every contact at {a['company']} who has at least one open ticket, at their primary address, "
                f"with the subject \"{a['subject']}\" and the body \"{a['body']}\".")

    def solve(w, a):
        for c in _contacts_at(w, _coid(w, a["company"])):
            if _open_tickets(w, c["id"]):
                _send(w, c["email"], a["subject"], a["body"])
    return sample, text, solve


@kind("escalate_urgent", 3, "none", False, ("find_company", "company_contacts", "tickets", "ticket_update", "users"))
def _k():
    def urgent(w, co_id):
        ids = {c["id"] for c in _contacts_at(w, co_id)}
        return [t for t in w["tickets"].values() if t["contact_id"] in ids and t["status"] == "open" and t["priority"] == 1]

    def sample(w, rng):
        co = _pick(rng, [co for co in w["companies"].values() if urgent(w, co["id"])])
        if co is None:
            return None
        return {"company": co["name"], "agent": rng.choice(_agents(w))["name"], "body": rng.choice(REPLIES)}

    def text(a):
        return (f"For every open ticket with priority 1 from a contact at {a['company']}, reply to the ticket with "
                f"\"{a['body']}\" and reassign it to {a['agent']}.")

    def solve(w, a):
        uid = _uid(w, a["agent"])
        for t in urgent(w, _coid(w, a["company"])):
            _reply(w, t, a["body"])
            if t["assignee_id"] != uid:
                t["assignee_id"] = uid
                t["version"] += 1
    return sample, text, solve


@kind("refund_and_apologise", 3, "none", False, ("find_contact", "orders", "refund", "email"))
def _k():
    def sample(w, rng):
        c = _pick(rng, [c for c in w["contacts"].values() if _orders(w, c["id"], "delivered")])
        return c and {"contact": c["name"], "reason": rng.choice(REASONS), "subject": "About your recent order",
                      "body": rng.choice(APOLOGIES)}

    def text(a):
        return (f"Refund {a['contact']}'s most recent delivered order in full with the reason \"{a['reason']}\", "
                f"then email them at their primary address with the subject \"{a['subject']}\" and the body "
                f"\"{a['body']}\".")

    def solve(w, a):
        cid = _cid(w, a["contact"])
        _refund_latest(w, cid, a["reason"])
        _send(w, w["contacts"][cid]["email"], a["subject"], a["body"])
    return sample, text, solve


@kind("top_spender", 3, "str", True, ("find_company", "company_contacts", "orders"))
def _k():
    def sample(w, rng):
        options = []
        for co in w["companies"].values():
            for d in SINCE_DAYS:
                since = _since(d)
                spends = sorted((_spend(w, c["id"], since) for c in _contacts_at(w, co["id"])), reverse=True)
                if spends and spends[0] > 0 and (len(spends) == 1 or spends[0] > spends[1]):
                    options.append({"company": co["name"], "since": since})
        return _pick(rng, options)

    def text(a):
        return (f"Which contact at {a['company']} has spent the most on delivered orders placed on or after "
                f"{a['since']}? Answer as \"Full Name <primary email>\".")

    def solve(w, a):
        cs = _contacts_at(w, _coid(w, a["company"]))
        best = max(cs, key=lambda c: _spend(w, c["id"], a["since"]))
        return f"{best['name']} <{best['email']}>"
    return sample, text, solve


@kind("manager_digest", 3, "none", True, ("find_company", "company_contacts", "tickets", "users", "email"))
def _k():
    def sample(w, rng):
        cos = [co for co in w["companies"].values() if any(_open_tickets(w, c["id"]) for c in _contacts_at(w, co["id"]))]
        co = _pick(rng, cos)
        return co and {"company": co["name"], "subject": rng.choice(DIGEST_SUBJECTS)}

    def text(a):
        return (f"Email {a['company']}'s account manager a list of the open tickets from the company's contacts, "
                f"with the subject \"{a['subject']}\" and a body of just the ticket IDs, one per line in ascending order.")

    def solve(w, a):
        co = w["companies"][_coid(w, a["company"])]
        ids = sorted(t["id"] for c in _contacts_at(w, co["id"]) for t in _open_tickets(w, c["id"]))
        _send(w, w["users"][co["account_manager_id"]]["email"], a["subject"], "\n".join(ids))
    return sample, text, solve


# -- sizes and schedules ------------------------------------------------------------------
SIZES: dict[str, dict[str, Any]] = {
    "sanity": dict(tasks=8, changes={}, kinds=["contact_email", "contact_company", "account_manager", "latest_order_status"]),
    "small": dict(tasks=24, changes={13: 2}, kinds=["contact_email", "contact_company", "open_ticket_count",
                                                    "email_contact", "email_open_ticket_contacts"]),
    "medium": dict(tasks=48, changes={17: 2, 33: 3}, kinds=[
        "contact_email", "contact_company", "account_manager", "open_ticket_count",
        "agent_open_tickets", "email_contact", "reassign_tickets", "tag_company_city",
        "email_open_ticket_contacts", "escalate_urgent"]),
    "realistic": dict(tasks=96, changes={33: 2, 65: 3}, kinds=list(KINDS)),
}


def version_at(size: str, position: int) -> int:
    v = 1
    for pos, ver in sorted(SIZES[size]["changes"].items()):
        if position >= pos:
            v = ver
    return v


def schedule(size: str) -> list[dict[str, Any]]:
    """The fixed order of kinds for a size: each kind first appears in level
    order over the first 40% of the stream, then recurs; about one task in ten
    repeats an earlier task word for word; every kind recurs after every change."""
    spec = SIZES[size]
    kinds, n = spec["kinds"], spec["tasks"]
    if size == "sanity":
        return [{"kind": k, "repeat_of": None} for k in kinds + kinds]
    for attempt in range(1000):
        rng = random.Random(f"schedule-{size}-{attempt}")
        n_rep = round(0.1 * n)
        intro_end = round(0.4 * n)
        first = {k: 1 + round(i * (intro_end - 1) / (len(kinds) - 1)) for i, k in enumerate(kinds)}
        base, extra = divmod(n - n_rep, len(kinds))
        bonus = set(rng.sample(kinds, extra))
        remaining = {k: base + (k in bonus) - 1 for k in kinds}
        rep_slots = set(rng.sample(range(intro_end + 1, n + 1), n_rep))
        slots: list[dict[str, Any]] = []
        ok = True
        for pos in range(1, n + 1):
            k_first = [k for k, p in first.items() if p == pos]
            if k_first:
                slots.append({"kind": k_first[0], "repeat_of": None})
                continue
            if pos in rep_slots:
                prior = [i + 1 for i, s in enumerate(slots) if s["repeat_of"] is None and KINDS[s["kind"]].repeatable]
                src = rng.choice(prior)
                slots.append({"kind": slots[src - 1]["kind"], "repeat_of": src})
                continue
            cands = [k for k in kinds if remaining[k] > 0 and first[k] < pos]
            if not cands:
                ok = False
                break
            k = rng.choices(cands, weights=[remaining[c] for c in cands])[0]
            remaining[k] -= 1
            slots.append({"kind": k, "repeat_of": None})
        if not ok or any(remaining.values()):
            continue
        # every kind recurs (as a fresh task) after every change
        for change in spec["changes"]:
            after = {s["kind"] for s in slots[change - 1:] if s["repeat_of"] is None}
            if after != set(kinds):
                ok = False
        if ok:
            return slots
    raise RuntimeError(f"no schedule found for {size}")


@dataclass
class Task:
    position: int
    kind: str
    level: int
    args: dict[str, Any]
    text: str
    version: int
    repeat_of: int | None
    first_of_kind: bool
    expected: Any
    before: dict[str, Any] = field(repr=False)
    after: dict[str, Any] = field(repr=False)  # observable state the reference leaves


@dataclass
class Stream:
    size: str
    seed: int
    tasks: list[Task]


def make_stream(size: str, seed: int) -> Stream:
    world = generate(seed, size)
    w = clone(world)
    tasks: list[Task] = []
    seen_kinds: set[str] = set()
    used: dict[str, list[dict]] = {}
    for pos, slot in enumerate(schedule(size), start=1):
        k = KINDS[slot["kind"]]
        if slot["repeat_of"] is not None:
            args = dict(tasks[slot["repeat_of"] - 1].args)
        else:
            rng = random.Random(f"args-{size}-{seed}-{pos}")
            args = None
            for _ in range(50):  # new arguments for each fresh task of a kind where possible
                args = k.sample(w, rng)
                if args is not None and args not in used.get(k.name, []):
                    break
            if args is None:
                raise RuntimeError(f"cannot sample {k.name} at {size}/{seed}/{pos}")
            used.setdefault(k.name, []).append(args)
        before = clone(w)
        expected = k.solve(w, args)
        tasks.append(Task(
            position=pos, kind=k.name, level=k.level, args=args, text=k.text(args),
            version=version_at(size, pos), repeat_of=slot["repeat_of"],
            first_of_kind=k.name not in seen_kinds, expected=expected, before=before, after=observable(w),
        ))
        seen_kinds.add(k.name)
    return Stream(size, seed, tasks)


# -- the check -------------------------------------------------------------------------
def answer_ok(kind: str, expected: Any, got: Any) -> bool:
    if kind == "none":
        return True
    if kind == "str":
        return isinstance(got, str) and " ".join(got.split()).casefold() == " ".join(str(expected).split()).casefold()
    if kind == "int":
        if isinstance(got, bool):
            return False
        if isinstance(got, int):
            return got == expected
        if isinstance(got, float):
            return got.is_integer() and int(got) == expected
        if isinstance(got, str):
            s = got.strip().replace(",", "")
            return bool(re.fullmatch(r"-?\d+", s)) and int(s) == expected
        return False
    if kind == "list":
        if isinstance(got, str):  # a list sent as text through the submit tool's untyped field
            got = _parse_list(got)
        return isinstance(got, (list, tuple)) and [str(x).strip() for x in got] == list(expected)
    raise ValueError(kind)


def _parse_list(text: str) -> Any:
    import ast
    import json

    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(text.strip())
        except (ValueError, SyntaxError, TypeError):
            continue
    return None


def check(task: Task, answer: Any, world_after: dict[str, Any]) -> tuple[bool, list[str]]:
    """Exact check of one task: the answer, and every change to the workspace."""
    reasons = []
    k = KINDS[task.kind]
    if not answer_ok(k.answer, task.expected, answer):
        reasons.append(f"answer: expected {task.expected!r}, got {answer!r}")
    reasons += diff(task.after, observable(world_after))
    return (not reasons), reasons
