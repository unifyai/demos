"""The workspace's data: users, companies, contacts, tickets and orders, from a seed.

A world is plain dicts and lists, so it deep-copies cheaply and compares
exactly. Everything an agent can change lives in it: tags, tickets, replies,
orders, refunds, emails sent and drafts.
"""

from __future__ import annotations

import copy
import datetime as dt
import random
from typing import Any

TODAY = "2026-09-15"

FIRST = [
    "Amara", "Bruno", "Celia", "Dmitri", "Elena", "Farid", "Greta", "Hugo", "Ines", "Jonas",
    "Kira", "Lionel", "Maya", "Nikhil", "Olga", "Pavel", "Quinn", "Rosa", "Stefan", "Tara",
    "Umar", "Vera", "Wendell", "Xenia", "Yusuf", "Zora", "Anders", "Beatriz", "Cyrus", "Delphine",
    "Emeka", "Fiona", "Gideon", "Hana", "Ivo", "Jun", "Kamala", "Leif", "Mirela", "Noor",
    "Otto", "Priya", "Rafael", "Selma", "Tomas", "Ulla", "Vikram", "Wren",
]
LAST = [
    "Abernathy", "Bjornsen", "Castellano", "Delacroix", "Eriksen", "Fairbanks", "Galloway", "Haddad",
    "Ishikawa", "Jablonski", "Kowalczyk", "Lindqvist", "Mbeki", "Novak", "Okafor", "Petrov",
    "Quintero", "Rasmussen", "Szabo", "Tanaka", "Underwood", "Varga", "Whitfield", "Yamamoto",
    "Zielinski", "Achterberg", "Brennan", "Carvalho", "Dunmore", "Esposito", "Fitzroy", "Gunnarsson",
    "Holloway", "Iversen", "Jovanovic", "Kaminski", "Lombardi", "Moreau", "Nakamura", "Oyelaran",
    "Pemberton", "Ravensworth", "Sandoval", "Thorne", "Ulrich", "Vasquez", "Winslow", "Xavier",
    "Yilmaz", "Zamora", "Ashdown", "Blackwood", "Coleridge", "Drummond", "Ellison", "Forsythe",
    "Greenhalgh", "Hartigan", "Ingram", "Jessop", "Kilbride", "Lockhart", "Merriweather", "Northcott",
]
COMPANIES = [
    "Alder & Finch", "Bluefin Logistics", "Copperleaf Studio", "Driftwood Analytics", "Ember Health",
    "Fernhill Foods", "Granite Ridge Capital", "Harbor Lane Legal", "Ironbark Engineering",
    "Juniper Retail", "Kestrel Aviation", "Larkspur Media", "Marlowe Textiles", "Nettlefield Farms",
    "Oakmoor Schools", "Pinecrest Hotels",
]
INDUSTRIES = ["logistics", "healthcare", "retail", "finance", "media", "manufacturing", "education", "hospitality"]
CITIES = ["Leeds", "Lisbon", "Toronto", "Osaka", "Denver", "Nairobi", "Krakow", "Perth", "Austin", "Bristol"]
TITLES = [
    "Operations Manager", "Head of Finance", "Procurement Lead", "IT Director", "Office Manager",
    "Customer Success Lead", "Founder", "Data Analyst", "Buyer", "Facilities Manager",
]
DOMAINS = ["mail.example", "post.example", "inbox.example"]
TICKET_SUBJECTS = [
    "Cannot log in", "Invoice shows the wrong amount", "Delivery is late", "Item arrived damaged",
    "Need to change the billing address", "Export to CSV fails", "Refund not received",
    "Password reset email never arrives", "Order shows the wrong size", "Question about bulk pricing",
    "Tracking link is broken", "Want to add a new user", "Duplicate charge on card",
    "Report is missing last week", "App crashes on upload",
]
SKUS = ["DESK-OAK", "CHAIR-ERG", "LAMP-LED", "MON-27", "DOCK-USB", "KB-MECH", "CAB-3D", "PLANT-FIG", "MAT-CUT", "SHELF-W"]
ORDER_STATUSES = ["pending", "paid", "shipped", "delivered", "cancelled", "refunded"]
TICKET_STATUSES = ["open", "pending", "solved", "closed"]

# How big a world each size gets.
SCALE = {
    "sanity": dict(users=5, companies=5, contacts=24, tickets=60, orders=90),
    "small": dict(users=5, companies=6, contacts=30, tickets=80, orders=90),
    "medium": dict(users=7, companies=9, contacts=55, tickets=140, orders=160),
    "realistic": dict(users=8, companies=12, contacts=80, tickets=180, orders=320),
}


def _no_substring_collisions(names: list[str]) -> bool:
    low = [n.lower() for n in names]
    return not any(a != b and a in b for a in low for b in low)


def _date(days_ago: int) -> str:
    return (dt.date.fromisoformat(TODAY) - dt.timedelta(days=days_ago)).isoformat()


def generate(seed: int, size: str) -> dict[str, Any]:
    """A world for one seed and size. Deterministic in (seed, size)."""
    sc = SCALE[size]
    rng = random.Random(f"world-{size}-{seed}")

    # Users: account managers and support agents, with names no contact has.
    pool = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(pool)
    user_names: list[str] = []
    for n in pool:
        if len(user_names) == sc["users"]:
            break
        if _no_substring_collisions(user_names + [n]) and n.split()[0] not in {u.split()[0] for u in user_names}:
            user_names.append(n)
    n_managers = max(2, sc["users"] // 3)
    users = {}
    for i, name in enumerate(user_names):
        uid = f"u_{11 + i}"
        first, last = name.lower().split()
        users[uid] = {
            "id": uid,
            "name": name,
            "email": f"{first}.{last}@brightdesk.example",
            "role": "account manager" if i < n_managers else "support agent",
        }
    managers = [u for u in users if users[u]["role"] == "account manager"]
    agents = [u for u in users if users[u]["role"] == "support agent"]

    company_names = rng.sample(COMPANIES, sc["companies"])
    companies = {}
    for i, name in enumerate(company_names):
        cid = f"co_{101 + i}"
        companies[cid] = {
            "id": cid,
            "name": name,
            "city": rng.choice(CITIES),
            "industry": rng.choice(INDUSTRIES),
            "account_manager_id": managers[i % len(managers)],
        }

    # Contacts: unique full names, none contained in another, none shared with a user.
    taken = set(user_names)
    contact_names: list[str] = []
    for n in pool:
        if len(contact_names) == sc["contacts"]:
            break
        if n in taken:
            continue
        if _no_substring_collisions(contact_names + user_names + [n]):
            contact_names.append(n)
    assert len(contact_names) == sc["contacts"], "name pool too small"
    # Company sizes vary, so some need more than one page of contacts.
    co_ids = list(companies)
    weights = [rng.choice([1, 1, 2, 2, 3]) for _ in co_ids]
    assignment = list(co_ids) + rng.choices(co_ids, weights=weights, k=sc["contacts"] - len(co_ids))
    rng.shuffle(assignment)
    contacts = {}
    for i, (name, co) in enumerate(zip(contact_names, assignment)):
        cid = f"c_{1001 + i}"
        first, last = name.lower().split()
        domain = rng.choice(DOMAINS)
        other = []
        if rng.random() < 0.4:
            other.append(f"{first[0]}{last}@{rng.choice([d for d in DOMAINS if d != domain])}")
        # Most contacts live where their company is; some elsewhere.
        city = companies[co]["city"] if rng.random() < 0.6 else rng.choice(CITIES)
        contacts[cid] = {
            "id": cid,
            "name": name,
            "company_id": co,
            "title": rng.choice(TITLES),
            "email": f"{first}.{last}@{domain}",
            "other_emails": other,
            "city": city,
            "tags": sorted(rng.sample(["vip", "newsletter", "beta", "partner"], k=rng.choice([0, 0, 1, 1, 2]))),
            "created_on": _date(rng.randint(200, 900)),
        }

    # Tickets: unique creation times, so "oldest first" is never a tie.
    contact_ids = list(contacts)
    minutes = rng.sample(range(0, 120 * 24 * 60), sc["tickets"])
    tickets = {}
    for i, m in enumerate(sorted(minutes, reverse=True)):
        tid = f"t_{5001 + i}"
        created = dt.datetime.fromisoformat(TODAY + "T09:00") - dt.timedelta(minutes=m)
        status = rng.choices(TICKET_STATUSES, weights=[40, 15, 25, 20])[0]
        last_reply_days = rng.randint(0, max(0, (dt.date.fromisoformat(TODAY) - created.date()).days))
        tickets[tid] = {
            "id": tid,
            "subject": rng.choice(TICKET_SUBJECTS),
            "status": status,
            "priority": rng.choices([1, 2, 3, 4], weights=[18, 25, 37, 20])[0],
            "assignee_id": rng.choice(agents),
            "contact_id": rng.choice(contact_ids),
            "created_on": created.strftime("%Y-%m-%d %H:%M"),
            "last_reply_on": _date(last_reply_days),
            "tags": [],
            "version": rng.randint(1, 6),
        }

    # Orders: each contact's orders have distinct dates, so "most recent" is never a tie.
    orders = {}
    per_contact: dict[str, set[int]] = {c: set() for c in contact_ids}
    oid_n = 7001
    for _ in range(sc["orders"]):
        c = rng.choice(contact_ids)
        days = rng.randint(1, 200)
        while days in per_contact[c]:
            days = rng.randint(1, 200)
        per_contact[c].add(days)
        items = []
        for sku in rng.sample(SKUS, rng.randint(1, 3)):
            items.append({"sku": sku, "qty": rng.randint(1, 4), "price_cents": rng.choice([1250, 2400, 3999, 4500, 6800, 8900, 12900])})
        status = rng.choices(ORDER_STATUSES, weights=[6, 10, 14, 50, 10, 10])[0]
        oid = f"o_{oid_n}"
        oid_n += 1
        orders[oid] = {
            "id": oid,
            "contact_id": c,
            "status": status,
            "total_cents": sum(it["qty"] * it["price_cents"] for it in items),
            "placed_on": _date(days),
            "items": items,
        }

    return {
        "today": TODAY,
        "users": users,
        "companies": companies,
        "contacts": contacts,
        "tickets": tickets,
        "orders": orders,
        "outbox": [],
        "replies": [],
        "refunds": [],
        "drafts": {},
        "approvals": {},
        "created": {"contacts": 0, "tickets": 0},
    }


def clone(world: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(world)


def _norm_text(s: Any) -> str:
    s = str(s).replace("\r\n", "\n").strip()
    return "\n".join(line.rstrip() for line in s.split("\n"))


def observable(world: dict[str, Any]) -> dict[str, Any]:
    """Everything a task can change, in a form that compares exactly. Ticket
    versions are left out: they count edits, which say nothing about the result."""
    return {
        "contacts": {cid: {k: (sorted(v) if k == "tags" else v) for k, v in c.items()} for cid, c in world["contacts"].items()},
        "tickets": {tid: {k: (sorted(v) if k == "tags" else v) for k, v in t.items() if k != "version"} for tid, t in world["tickets"].items()},
        "orders": {oid: o["status"] for oid, o in world["orders"].items()},
        "refunds": sorted((r["order_id"], r["amount_cents"], _norm_text(r["reason"])) for r in world["refunds"]),
        "replies": sorted((r["ticket_id"], _norm_text(r["body"])) for r in world["replies"]),
        "outbox": sorted((e["to"].strip().lower(), _norm_text(e["subject"]), _norm_text(e["body"])) for e in world["outbox"]),
        "companies": world["companies"],
        "users": world["users"],
    }


def diff(expected: dict[str, Any], got: dict[str, Any]) -> list[str]:
    """Human-readable differences between two observable states (for records)."""
    out = []
    for key in expected:
        a, b = expected[key], got.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    out.append(f"{key}[{k}]: expected {a.get(k)!r}, got {b.get(k)!r}")
        elif isinstance(a, list) and isinstance(b, list):
            missing = [x for x in a if x not in b]
            extra = [x for x in b if x not in a]
            if missing:
                out.append(f"{key}: missing {missing!r}")
            if extra:
                out.append(f"{key}: unexpected {extra!r}")
            if not missing and not extra:
                out.append(f"{key}: counts differ")
        else:
            out.append(f"{key}: expected {a!r}, got {b!r}")
    return out
