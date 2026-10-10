"""Synthetic histories: half a year of an assistant's work for one person.

Maya Chen runs a small design studio in London. Her assistant handles email,
calendar, tasks, documents, quotes, purchases and travel across about twenty
projects that start, pause, resume and end at different times, plus one-off
requests. Activity is uneven: weekends are quiet, a three-state chain makes
days busy, normal or quiet, and three breaks of five to sixteen days have no
activity at all.

Every interaction is rendered as the record an agent would log (the request,
the tool calls with their results, the reply) and keeps its structured facts,
from which every question's answer is computed exactly.

    history = generate(seed)          # deterministic in the seed
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Any

START = dt.datetime(2026, 1, 5)  # a Monday
DAYS = 186  # 26 weeks and 4 days: the last checkpoint falls early in week 27
USER = "Maya Chen"

# Per kind of project: roles of its three people, and the things its actions
# refer to. Quotes are (item, low, high, index of the person who quotes).
KINDS: dict[str, dict[str, Any]] = {
    "event": dict(
        roles=["venue manager", "caterer", "AV supplier"],
        meetings=["venue walkthrough", "catering tasting", "agenda review", "logistics check-in"],
        documents=["run of show", "attendee list", "venue contract", "seating plan"],
        quotes=[("venue hire", 2000, 9000, 0), ("catering", 800, 4000, 1), ("AV equipment", 300, 1500, 2), ("event photography", 400, 1800, 2)],
        tasks=["confirm the headcount", "send the save-the-date", "collect dietary requirements", "book airport transfers", "order name badges", "finalise the seating plan"],
        purchases=[("lanyards", 40, 200), ("signage", 100, 600), ("welcome packs", 150, 900)],
        research=["team activities nearby", "transport from the airport", "weather for the dates"],
    ),
    "hiring": dict(
        roles=["candidate", "recruiter", "candidate"],
        meetings=["screening call", "technical interview", "portfolio review", "reference call"],
        documents=["job description", "interview scorecard", "offer letter", "take-home brief"],
        quotes=[("recruiter fee", 4000, 15000, 1), ("job board package", 300, 1200, 1), ("background check", 60, 240, 1), ("relocation support", 1500, 6000, 1)],
        tasks=["post the job ad", "shortlist candidates", "book the interview panel", "send rejection emails", "prepare the offer", "write the onboarding plan"],
        purchases=[("laptop for the new hire", 1200, 2800), ("desk chair", 200, 700), ("software licences", 100, 900)],
        research=["salary benchmarks", "notice period norms", "visa sponsorship rules"],
    ),
    "fundraising": dict(
        roles=["investor", "lawyer", "accountant"],
        meetings=["investor call", "partner meeting", "due diligence session", "term sheet review"],
        documents=["pitch deck", "financial model", "cap table", "data room index"],
        quotes=[("legal fees for the round", 8000, 30000, 1), ("valuation report", 2000, 8000, 2), ("data room software", 300, 1500, 2), ("audit of last year's accounts", 3000, 12000, 2)],
        tasks=["update the financial model", "collect board consents", "upload the IP assignments", "answer the diligence questions", "sign the NDA", "chase the bank statements"],
        purchases=[("company search reports", 50, 300), ("courier for signed documents", 30, 150), ("printing of the deck", 60, 400)],
        research=["comparable rounds", "investor portfolio overlaps", "standard term sheet clauses"],
    ),
    "client": dict(
        roles=["client lead", "freelance developer", "client producer"],
        meetings=["kickoff", "design review", "user testing session", "handover meeting"],
        documents=["statement of work", "design mockups", "project timeline", "invoice"],
        quotes=[("freelance illustration", 1500, 6000, 1), ("user testing panel", 600, 2400, 1), ("stock photography", 100, 800, 1), ("hosting for a year", 200, 1200, 1)],
        tasks=["send the statement of work", "book the user testing room", "export the design files", "chase the signed contract", "prepare the handover pack", "send the final invoice"],
        purchases=[("font licence", 80, 600), ("prototype printing", 100, 700), ("usability test incentives", 150, 600)],
        research=["competitor websites", "accessibility guidelines", "the client's market"],
    ),
    "office": dict(
        roles=["letting agent", "contractor", "removals coordinator"],
        meetings=["site visit", "landlord meeting", "fit-out review", "move-day planning"],
        documents=["floor plan", "lease draft", "moving checklist", "furniture list"],
        quotes=[("removals", 1200, 5000, 2), ("fit-out works", 8000, 40000, 1), ("deep cleaning", 300, 1200, 1), ("broadband installation", 100, 600, 1)],
        tasks=["measure the meeting room", "order the new desks", "redirect the post", "book the electrician", "update the company address", "return the old keys"],
        purchases=[("shelving", 200, 900), ("meeting room table", 600, 2500), ("plants for the studio", 100, 500)],
        research=["parking near the new office", "business rates", "local lunch places"],
    ),
    "launch": dict(
        roles=["PR consultant", "photographer", "retail buyer"],
        meetings=["launch planning", "press briefing", "pricing review", "retailer pitch"],
        documents=["press release", "product sheet", "launch plan", "price list"],
        quotes=[("PR agency retainer", 3000, 12000, 0), ("product photography", 800, 3500, 1), ("launch video", 2000, 9000, 1), ("trade fair stand", 2500, 10000, 2)],
        tasks=["send samples to the press", "book the launch venue", "finalise the price list", "brief the photographer", "update the online shop", "prepare the retailer pack"],
        purchases=[("sample packaging", 100, 600), ("display plinths", 200, 900), ("launch invitations", 150, 700)],
        research=["design press contacts", "retail margins", "competitor pricing"],
    ),
    "finance": dict(
        roles=["accountant", "insurance broker", "bank manager"],
        meetings=["accountant call", "budget review", "insurance review", "bank meeting"],
        documents=["management accounts", "expense report", "insurance schedule", "budget forecast"],
        quotes=[("accountancy fee", 1500, 6000, 0), ("insurance premium", 800, 4000, 1), ("payroll software", 200, 900, 0), ("tax advice", 500, 2500, 0)],
        tasks=["reconcile the bank account", "file the receipts", "approve the payroll", "send the expense report", "renew the direct debits", "chase the unpaid invoices"],
        purchases=[("accounting software upgrade", 100, 500), ("receipt scanner", 80, 300), ("filing cabinet", 120, 450)],
        research=["R&D tax credit rules", "pension contribution limits", "business account fees"],
    ),
    "travel": dict(
        roles=["client host", "hotel concierge", "interpreter"],
        meetings=["itinerary review", "client dinner", "showroom visit", "trip debrief"],
        documents=["itinerary", "visa letter", "expense claims", "meeting notes"],
        quotes=[("hotel for the week", 900, 3500, 1), ("interpreter services", 400, 1600, 2), ("ground transport", 200, 900, 1), ("trade pass", 100, 600, 0)],
        tasks=["book the flights", "arrange travel insurance", "print the boarding passes", "exchange currency", "pack the samples", "file the travel expenses"],
        purchases=[("travel adapter set", 20, 80), ("sample carry case", 80, 300), ("gifts for the hosts", 50, 250)],
        research=["local etiquette", "trains from the airport", "restaurants near the hotel"],
    ),
    "legal": dict(
        roles=["solicitor", "trademark attorney", "opposing counsel"],
        meetings=["solicitor call", "mediation session", "filing review", "strategy meeting"],
        documents=["trademark application", "letter before action", "evidence bundle", "settlement draft"],
        quotes=[("solicitor fees", 2000, 12000, 0), ("filing fees", 200, 1500, 1), ("expert report", 1500, 7000, 0), ("document translation", 300, 1200, 1)],
        tasks=["gather the invoices as evidence", "sign the witness statement", "pay the filing fee", "review the settlement terms", "send the signed declaration", "diary the hearing date"],
        purchases=[("certified copies", 30, 200), ("courier to the court", 40, 150), ("legal research database", 100, 500)],
        research=["similar trademark cases", "small claims limits", "mediation services"],
    ),
    "marketing": dict(
        roles=["copywriter", "illustrator", "media buyer"],
        meetings=["campaign kickoff", "creative review", "media planning", "results review"],
        documents=["campaign brief", "content calendar", "media plan", "results report"],
        quotes=[("ad spend", 2000, 15000, 2), ("copywriting", 600, 2500, 0), ("illustrations", 500, 3000, 1), ("print run", 400, 2500, 2)],
        tasks=["approve the copy", "schedule the posts", "brief the illustrator", "set up the tracking links", "send the newsletter test", "compile the results"],
        purchases=[("social media scheduling tool", 50, 300), ("printed postcards", 100, 600), ("stock footage", 80, 400)],
        research=["audience segments", "best posting times", "competitor campaigns"],
    ),
}

# (project name, kind, currency, city or None). Every history touches about
# twenty of these; the rest are never touched and stand as distractors.
PROJECTS: list[tuple[str, str, str, str | None]] = [
    ("Lisbon offsite", "event", "EUR", "Lisbon"),
    ("Copenhagen summit booth", "event", "EUR", "Copenhagen"),
    ("studio tenth birthday party", "event", "GBP", None),
    ("senior designer hire", "hiring", "GBP", None),
    ("office manager hire", "hiring", "GBP", None),
    ("data engineer hire", "hiring", "GBP", None),
    ("Series A data room", "fundraising", "GBP", None),
    ("Innovate UK grant", "fundraising", "GBP", None),
    ("Fernhill Bakery website", "client", "GBP", None),
    ("Corvid Health app redesign", "client", "GBP", None),
    ("Atlas Rowing brand refresh", "client", "GBP", None),
    ("Marlow Library wayfinding", "client", "GBP", None),
    ("Clerkenwell office move", "office", "GBP", None),
    ("workshop refurbishment", "office", "GBP", None),
    ("Loom chair launch", "launch", "GBP", None),
    ("Kettle lamp launch", "launch", "GBP", None),
    ("year-end accounts", "finance", "GBP", None),
    ("insurance renewal", "finance", "GBP", None),
    ("Milan Design Week trip", "travel", "EUR", "Milan"),
    ("Tokyo client visit", "travel", "GBP", "Tokyo"),
    ("Kyoto supplier visit", "travel", "GBP", "Kyoto"),
    ("trademark registration", "legal", "GBP", None),
    ("supplier contract dispute", "legal", "GBP", None),
    ("newsletter relaunch", "marketing", "GBP", None),
    ("spring campaign", "marketing", "GBP", None),
    ("podcast sponsorship", "marketing", "GBP", None),
]
PROJECT_NAMES = [p[0] for p in PROJECTS]

FIRST = ["Ana", "Tomas", "Joana", "Priya", "Marcus", "Lena", "Omar", "Grace", "Hugo", "Ines", "Felix", "Nadia", "Ravi",
         "Clara", "Jonas", "Amara", "Theo", "Sofia", "Kenji", "Elif", "Mateo", "Hannah", "Yusuf", "Freya", "Diego", "Mina",
         "Oscar", "Leila", "Bruno", "Aisha", "Luca", "Nora", "Samir", "Isla", "Pavel", "Chloe", "Dev", "Maren", "Kofi",
         "Zara", "Emil", "Rosa", "Aiden", "Yara", "Niko", "Helen", "Arjun", "Beatriz", "Wim", "Saskia", "Tariq",
         "Ingrid", "Rafael", "Keiko", "Malik", "Petra", "Callum", "Anouk", "Victor", "Esme", "Joaquin", "Lotte", "Idris",
         "Carmen", "Rohan", "Astrid", "Gabriel", "Noor", "Henrik", "Lucia", "Kwame", "Fiona", "Stefan", "Yasmin", "Declan",
         "Mirela", "Ollie", "Tamsin", "Bilal", "Vera"]
LAST = ["Ruiz", "Pereira", "Lima", "Shah", "Webb", "Fischer", "Haddad", "Okafor", "Brandt", "Costa", "Novak", "Karimi",
        "Iyer", "Moreau", "Lindqvist", "Mensah", "Hartley", "Romano", "Sato", "Demir", "Alvarez", "Kowalski", "Aydin",
        "Larsen", "Navarro", "Park", "Quinn", "Farouk", "Vidal", "Bello", "Ricci", "Holm", "Nasser", "Murray", "Petrov",
        "Laurent", "Malhotra", "Berg", "Asante", "Kaur", "Strand", "Ortega", "Doyle", "Nakamura", "Virtanen", "Price",
        "Menon", "Sousa", "de Vries", "Jansen"]
ORGS = ["Casa do Rio", "Mesa Lisboa", "Brightwire AV", "Kestrel Ventures", "Hollis & Grant", "Northgate Partners",
        "Pinewood Recruiting", "Fernhill Bakery", "Corvid Health", "Atlas Rowing Club", "Marlow Library", "Granary Lettings",
        "Bowline Builders", "Swift Removals", "Ochre PR", "Lumen Studio Photo", "Harbour & Finch", "Clearline Accounts",
        "Meridian Insurance", "Thameside Bank", "Fiera Milano Services", "Sakura Hospitality", "Kanji Language Co",
        "Penrose Legal", "Markwell IP", "Dunmore Supplies", "Inkhouse Copy", "Paperkite Illustration", "Signal Media",
        "Oakline Interiors", "Copperleaf Events", "Riverside Catering", "Studio Halden", "Nordlys Fairs", "Tessellate Dev",
        "Bramble Research", "Quayside Hotels", "Lantern Productions", "Aldgate Trust", "Wren & Co"]
PERSONAL_EVENTS = ["Gym", "Lunch with Sam", "Dentist", "School run", "Yoga", "Call with Mum", "Haircut", "Running club",
                   "Coffee with Jo", "Pilates", "Dinner with Alex", "Piano lesson"]
CITIES = ["London", "Lisbon", "Milan", "Copenhagen", "Tokyo", "Kyoto", "Paris", "Berlin"]
DOMAINS = {"Casa do Rio": "casadorio.pt", "Mesa Lisboa": "mesalisboa.pt"}

# Relative weights of the actions on a project (subject to availability).
ACTION_WEIGHTS = {
    "meeting": 14, "move_meeting": 3, "email": 18, "quote": 8, "revise_quote": 3, "task_create": 12,
    "task_done": 10, "share": 7, "research": 9, "purchase": 5, "travel": 3, "contact": 5,
}


@dataclass
class Person:
    name: str
    role: str
    org: str

    @property
    def email(self) -> str:
        first, *rest = self.name.lower().split(" ")
        dom = DOMAINS.get(self.org) or "".join(ch for ch in self.org.lower() if ch.isalnum())[:18] + ".com"
        return f"{first}.{''.join(rest).replace(' ', '')}@{dom}"


@dataclass
class Interaction:
    idx: int  # 1-based, in time order
    ts: dt.datetime
    project: str | None
    action: str
    facts: dict[str, Any]
    text: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"idx": self.idx, "ts": self.ts.isoformat(), "project": self.project, "action": self.action,
                "facts": self.facts, "text": self.text}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Interaction":
        return Interaction(d["idx"], dt.datetime.fromisoformat(d["ts"]), d["project"], d["action"], d["facts"], d["text"])


@dataclass
class ProjectState:
    name: str
    kind: str
    currency: str
    city: str | None
    people: list[Person]
    spans: list[tuple[int, int]]  # active day ranges [a, b)
    weight: float
    meetings: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)  # (topic, person) -> latest
    quotes: dict[str, dict[str, Any]] = field(default_factory=dict)  # item -> latest
    revised: set[str] = field(default_factory=set)
    moved: set[tuple[str, str]] = field(default_factory=set)
    shared: dict[str, str] = field(default_factory=dict)  # document -> person
    open_tasks: list[str] = field(default_factory=list)
    task_ids: dict[str, str] = field(default_factory=dict)
    done_tasks: list[str] = field(default_factory=list)
    purchased: set[str] = field(default_factory=set)
    researched: int = 0

    def active(self, day: int) -> bool:
        return any(a <= day < b for a, b in self.spans)


def day_of(ts: dt.datetime) -> int:
    return (ts - START).days


def fmt_day(d: dt.date | dt.datetime) -> str:
    return f"{d:%a} {d.day} {d:%b %Y}"


def fmt_ts(ts: dt.datetime) -> str:
    return f"{ts:%a} {ts.day} {ts:%b %Y}, {ts:%H:%M}"


def _amount(rng: random.Random, lo: int, hi: int) -> int:
    step = 50 if hi >= 1000 else 10
    return int(round(rng.uniform(lo, hi) / step) * step)


def _money(currency: str, amount: int) -> str:
    sym = {"GBP": "£", "EUR": "EUR "}[currency]
    return f"{sym}{amount:,}"


def _plan_projects(rng: random.Random) -> list[tuple[str, list[tuple[int, int]], float]]:
    """Choose about twenty projects and their active spans over the half year.

    Three start in the first week; the others start at staggered points across
    the half year, so three to five are usually under way at once. About a
    third of the longer ones pause for two to five weeks and then resume."""
    names = PROJECT_NAMES[:]
    rng.shuffle(names)
    n = rng.randint(18, 21)
    chosen = names[:n]
    plans = []
    for i, name in enumerate(chosen):
        dur = rng.choice([7, 10, 14, 21, 28, 35, 42, 56, 70, 84])
        if i < 3:
            start = rng.randint(0, 5)
        else:
            slot = (i - 3 + 0.5) * (DAYS - 3) / (n - 3)
            start = int(min(DAYS - 7, max(0, slot - dur / 3 + rng.uniform(-10, 10))))
        end = min(DAYS, start + dur)
        spans = [(start, end)]
        if dur >= 28 and rng.random() < 0.35:
            mid = start + dur // 2
            pause = rng.randint(14, 35)
            spans = [(start, mid), (mid + pause, min(DAYS, end + pause))]
            spans = [(a, b) for a, b in spans if a < b and a < DAYS]
        plans.append((name, spans, rng.uniform(0.6, 1.6)))
    return plans


def _plan_breaks(rng: random.Random) -> list[tuple[int, int]]:
    """Three stretches with no activity at all: one long, two short."""
    while True:
        long_len = rng.randint(9, 16)
        a = rng.randint(35, 140 - long_len)
        breaks = [(a, a + long_len)]
        for _ in range(2):
            ln = rng.randint(5, 8)
            s = rng.randint(10, DAYS - 12 - ln)
            breaks.append((s, s + ln))
        breaks.sort()
        if all(breaks[i + 1][0] - breaks[i][1] >= 10 for i in range(len(breaks) - 1)):
            return breaks


def generate(seed: int, *, project_rate: float = 3.0, one_off_rate: float = 2.2) -> dict[str, Any]:
    rng = random.Random(7919 * seed + 17)
    kinds = {p[0]: p for p in PROJECTS}
    pairs = [(f, l) for f in FIRST for l in LAST]
    rng.shuffle(pairs)
    used_first: set[str] = set()
    people_pool = []
    for f, l in pairs:  # distinct first names keep people unambiguous
        if f in used_first:
            continue
        used_first.add(f)
        people_pool.append(f"{f} {l}")
    orgs = ORGS[:]
    rng.shuffle(orgs)

    projects: dict[str, ProjectState] = {}
    for k, (name, spans, w) in enumerate(_plan_projects(rng)):
        _, kind, currency, city = kinds[name]
        roles = KINDS[kind]["roles"]
        people = [Person(people_pool.pop(), roles[j], orgs[(3 * k + j) % len(orgs)]) for j in range(3)]
        projects[name] = ProjectState(name, kind, currency, city, people, spans, w)
    breaks = _plan_breaks(rng)

    interactions: list[Interaction] = []
    state = 1  # 0 quiet, 1 normal, 2 busy
    trans = {0: [0.55, 0.4, 0.05], 1: [0.2, 0.6, 0.2], 2: [0.05, 0.45, 0.5]}
    mult = {0: 0.3, 1: 1.0, 2: 2.0}
    for day in range(DAYS):
        state = rng.choices([0, 1, 2], trans[state])[0]
        date = START + dt.timedelta(days=day)
        if any(a <= day < b for a, b in breaks):
            continue
        day_mult = mult[state] * (0.15 if date.weekday() >= 5 else 1.0)
        active = [p for p in projects.values() if p.active(day)]
        # project work scales with what is under way; one-offs come at their own small rate
        n_proj = _poisson(rng, project_rate * day_mult * sum(p.weight for p in active))
        n_one = _poisson(rng, one_off_rate * day_mult)
        kinds_today = ["p"] * n_proj + ["o"] * n_one
        rng.shuffle(kinds_today)
        times = sorted(rng.uniform(7.5, 21.0) for _ in kinds_today)
        for h, kind in zip(times, kinds_today):
            ts = (date + dt.timedelta(hours=h)).replace(second=0, microsecond=0)
            it = None
            if kind == "p":
                p = rng.choices(active, [q.weight for q in active])[0]
                it = _project_action(rng, p, ts)
            if it is None:
                it = _one_off(rng, ts)
            it.idx = len(interactions) + 1
            interactions.append(it)
    for it in interactions:
        it.text = render(it, projects)
    return {
        "seed": seed,
        "start": START.isoformat(),
        "days": DAYS,
        "breaks": breaks,
        "projects": {n: {"kind": p.kind, "spans": p.spans, "people": [vars(x) for x in p.people]} for n, p in projects.items()},
        "interactions": interactions,
    }


def _poisson(rng: random.Random, lam: float) -> int:
    import math
    if lam <= 0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _future(rng: random.Random, ts: dt.datetime, lo: int, hi: int) -> dt.datetime:
    d = (ts + dt.timedelta(days=rng.randint(lo, hi))).replace(hour=0, minute=0)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d + dt.timedelta(hours=rng.choice([9, 10, 11, 12, 14, 15, 16, 17]), minutes=rng.choice([0, 30]))


def _project_action(rng: random.Random, p: ProjectState, ts: dt.datetime) -> Interaction | None:
    K = KINDS[p.kind]
    avail: dict[str, Any] = {}
    combos = [(t, x.name) for t in K["meetings"] for x in p.people if (t, x.name) not in p.meetings]
    if combos:
        avail["meeting"] = combos
    movable = [c for c in p.meetings if c not in p.moved and dt.date.fromisoformat(p.meetings[c]["date"]) > ts.date()]
    if movable:
        avail["move_meeting"] = movable
    avail["email"] = True
    q_items = [q for q in K["quotes"] if q[0] not in p.quotes]
    if q_items:
        avail["quote"] = q_items
    rev = [i for i in p.quotes if i not in p.revised]
    if rev:
        avail["revise_quote"] = rev
    t_new = [t for t in K["tasks"] if t not in p.open_tasks and t not in p.done_tasks]
    if t_new:
        avail["task_create"] = t_new
    if p.open_tasks:
        avail["task_done"] = p.open_tasks
    docs = [d for d in K["documents"] if d not in p.shared]
    if docs:
        avail["share"] = docs
    avail["research"] = K["research"]
    buys = [b for b in K["purchases"] if b[0] not in p.purchased]
    if buys:
        avail["purchase"] = buys
    if p.kind in ("travel", "event") or p.city:
        avail["travel"] = True
    avail["contact"] = True
    kinds = list(avail)
    action = rng.choices(kinds, [ACTION_WEIGHTS[k] for k in kinds])[0]
    person = rng.choice(p.people)
    f: dict[str, Any]
    if action == "meeting":
        topic, who = rng.choice(avail["meeting"])
        when = _future(rng, ts, 1, 21)
        f = {"topic": topic, "person": who, "date": when.date().isoformat(), "time": f"{when:%H:%M}", "event_id": f"evt_{rng.getrandbits(24):06x}"}
        p.meetings[(topic, who)] = dict(f)
    elif action == "move_meeting":
        topic, who = rng.choice(avail["move_meeting"])
        old = p.meetings[(topic, who)]
        when = _future(rng, ts, 2, 18)
        f = {"topic": topic, "person": who, "old_date": old["date"], "date": when.date().isoformat(), "time": f"{when:%H:%M}", "event_id": old["event_id"]}
        p.meetings[(topic, who)] = {"topic": topic, "person": who, "date": f["date"], "time": f["time"], "event_id": old["event_id"]}
        p.moved.add((topic, who))
    elif action == "email":
        f = {"person": person.name, "subject": rng.choice(_subjects(p))}
    elif action == "quote":
        item, lo, hi, vi = rng.choice(avail["quote"])
        f = {"item": item, "person": p.people[vi].name, "org": p.people[vi].org, "amount": _amount(rng, lo, hi), "currency": p.currency,
             "row": 2 + len(p.quotes)}
        p.quotes[item] = dict(f)
    elif action == "revise_quote":
        item = rng.choice(avail["revise_quote"])
        old = p.quotes[item]
        new = int(round(old["amount"] * rng.choice([0.8, 0.85, 0.9, 1.1, 1.15, 1.2]) / 10) * 10)
        f = {"item": item, "person": old["person"], "org": old["org"], "old_amount": old["amount"], "amount": new, "currency": p.currency,
             "row": old["row"]}
        p.quotes[item] = {**old, "amount": new}
        p.revised.add(item)
    elif action == "task_create":
        task = rng.choice(avail["task_create"])
        f = {"task": task, "due": _future(rng, ts, 3, 30).date().isoformat(), "task_id": f"T-{rng.randint(1000, 9999)}"}
        p.open_tasks.append(task)
        p.task_ids[task] = f["task_id"]
    elif action == "task_done":
        task = rng.choice(p.open_tasks)
        f = {"task": task, "task_id": p.task_ids[task]}
        p.open_tasks.remove(task)
        p.done_tasks.append(task)
    elif action == "share":
        doc = rng.choice(avail["share"])
        f = {"document": doc, "person": person.name}
        p.shared[doc] = person.name
    elif action == "research":
        f = {"topic": rng.choice(avail["research"])}
    elif action == "purchase":
        item, lo, hi = rng.choice(avail["purchase"])
        f = {"item": item, "amount": _amount(rng, lo, hi), "currency": p.currency, "vendor": rng.choice(ORGS)}
        p.purchased.add(item)
    elif action == "travel":
        city = p.city or rng.choice(CITIES)
        mode = "train" if city in ("Paris", "Berlin") or rng.random() < 0.2 else "flight"
        when = _future(rng, ts, 3, 25)
        f = {"mode": mode, "city": city, "date": when.date().isoformat(), "cost": _amount(rng, 90, 900), "currency": p.currency}
    else:  # contact
        field_ = rng.choice(["phone number", "job title", "email address"])
        f = {"person": person.name, "field": field_}
    return Interaction(0, ts, p.name, action, f)


def _subjects(p: ProjectState) -> list[str]:
    K = KINDS[p.kind]
    return [f"{p.name}: next steps", f"{p.name}: {K['documents'][0]}", f"{p.name}: timing",
            f"{p.name}: {K['meetings'][1]} follow-up", f"{p.name}: open questions"]


ONE_OFFS = ["weather", "convert", "reminder", "restaurant", "definition", "calendar_today", "personal_email", "unit"]


def _one_off(rng: random.Random, ts: dt.datetime) -> Interaction:
    kind = rng.choice(ONE_OFFS)
    f: dict[str, Any] = {"kind": kind}
    if kind == "weather":
        f["city"] = rng.choice(CITIES)
    elif kind == "convert":
        f["amount"] = rng.choice([40, 120, 250, 800, 1500])
    elif kind == "reminder":
        f["what"] = rng.choice(["call Mum", "buy a birthday card for Alex", "renew the gym membership", "book the car service", "water the plants", "pay the window cleaner"])
    elif kind == "restaurant":
        f["area"] = rng.choice(["Islington", "Clerkenwell", "Soho", "Hackney", "Borough"])
    elif kind == "definition":
        f["word"] = rng.choice(["kerning", "escrow", "pro forma", "indemnity", "chiaroscuro", "amortisation"])
    elif kind == "personal_email":
        f["to"] = rng.choice(["Sam", "Alex", "Jo", "Mum"])
    elif kind == "unit":
        f["q"] = rng.choice(["120 cm in inches", "5 miles in km", "30 C in F", "2 kg in pounds"])
    return Interaction(0, ts, None, "one_off", f)


# ---------------------------------------------------------------- rendering

def _pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def render(it: Interaction, projects: dict[str, ProjectState]) -> str:
    """The record an agent logs for one interaction."""
    rng = random.Random(it.idx * 1009 + it.ts.toordinal())
    head = f"#{it.idx:04d} · {fmt_ts(it.ts)}"
    f = it.facts
    if it.project is None:
        user, calls, reply = _render_one_off(rng, it)
    else:
        p = projects[it.project]
        people = {x.name: x for x in p.people}
        user, calls, reply = _render_project(rng, it, p, people)
    lines = [head, f"Maya: {user}", "Assistant actions:"]
    for call, result in calls:
        lines.append(f"  {call}")
        lines.append(f"    -> {result}")
    lines.append(f"Assistant: {reply}")
    return "\n".join(lines)


def _d(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d:%A} {d.day} {d:%B}"


def _cal_noise(rng: random.Random, iso: str, booked: str) -> tuple[str, str]:
    evs = rng.sample(PERSONAL_EVENTS, 2)
    hours = sorted(rng.sample([h for h in (8, 12, 13, 18, 19) if h != int(booked[:2])], 2))
    busy = ", ".join(f'{{"start": "{h:02d}:00", "title": "{e}"}}' for h, e in zip(hours, evs))
    return (f'calendar.free_busy(date="{iso}")', f'{{"busy": [{busy}]}}')


def _render_project(rng, it, p: ProjectState, people: dict[str, Person]):
    f, P = it.facts, it.project
    a = it.action
    cur = p.currency
    if a == "meeting":
        who = people[f["person"]]
        user = _pick(rng, [
            f"Can you set up the {f['topic']} with {who.name.split()[0]} for {_d(f['date'])} at {f['time']}? It's for the {P}.",
            f"Please book a {f['topic']} with {who.name} ({who.org}) on {_d(f['date'])}, {f['time']}.",
            f"{who.name.split()[0]} suggested {_d(f['date'])} at {f['time']} for the {f['topic']}. Lock it in please.",
        ])
        calls = [_cal_noise(rng, f["date"], f["time"]),
                 (f'calendar.create_event(title="{P}: {f["topic"]} with {who.name}", start="{f["date"]} {f["time"]}", '
                  f'duration_min={rng.choice([30, 45, 60])}, attendees=["{who.email}"])',
                  f'{{"status": "created", "event_id": "{f["event_id"]}"}}'),
                 (f'email.send(to="{who.email}", subject="{P}: {f["topic"]}", body="Hi {who.name.split()[0]}, '
                  f'confirming the {f["topic"]} on {_d(f["date"])} at {f["time"]}. Speak then, Maya")',
                  f'{{"status": "sent"}}')]
        reply = _pick(rng, [f"Done. The {f['topic']} with {who.name} is booked for {_d(f['date'])} at {f['time']}, and the invite is out.",
                            f"Booked: {f['topic']} with {who.name}, {_d(f['date'])} at {f['time']}. I sent {who.name.split()[0]} a confirmation."])
    elif a == "move_meeting":
        who = people[f["person"]]
        user = _pick(rng, [f"{who.name.split()[0]} can't do {_d(f['old_date'])} any more. Move the {f['topic']} to {_d(f['date'])} at {f['time']}.",
                           f"Please push the {P} {f['topic']} with {who.name} to {_d(f['date'])}, {f['time']}."])
        calls = [(f'calendar.find_event(query="{P}: {f["topic"]} with {who.name}")', f'{{"event_id": "{f["event_id"]}", "start": "{f["old_date"]}"}}'),
                 (f'calendar.update_event(event_id="{f["event_id"]}", start="{f["date"]} {f["time"]}")', '{"status": "updated", "notified": true}')]
        reply = f"Moved. The {f['topic']} with {who.name} is now on {_d(f['date'])} at {f['time']} (it was {_d(f['old_date'])}); {who.name.split()[0]} has the update."
    elif a == "email":
        who = people[f["person"]]
        body = _pick(rng, [
            f"Hi {who.name.split()[0]}, thanks for your note on the {P}. We're on track this side; could you send over anything outstanding by the end of the week? Best, Maya",
            f"Hi {who.name.split()[0]}, quick update on the {P}: the plan is unchanged and we'd like to keep the current dates. Let me know if anything on your side has moved. Thanks, Maya",
            f"Hello {who.name.split()[0]}, following up on our last conversation about the {P}. Could you confirm the details we discussed and flag any risks? Many thanks, Maya",
        ])
        user = _pick(rng, [f"Email {who.name.split()[0]} about the {P}, just a short update.",
                           f"Can you reply to {who.name} on the {P} thread and ask them to confirm where things are?",
                           f"Send {who.name} a note about the {P}."])
        calls = [(f'email.search(query="from:{who.email}", limit=1)', f'{{"results": [{{"subject": "{f["subject"]}", "snippet": "Hi Maya, just checking in on the {P}..."}}]}}'),
                 (f'email.send(to="{who.email}", subject="Re: {f["subject"]}", body="{body}")', '{"status": "sent"}')]
        reply = f"Sent to {who.name}, subject \"Re: {f['subject']}\"."
    elif a == "quote":
        who = people[f["person"]]
        money = _money(cur, f["amount"])
        user = _pick(rng, [f"{who.name.split()[0]} from {who.org} sent a quote for the {f['item']} on the {P}. Can you log it?",
                           f"Please add {who.org}'s quote for the {f['item']} to the {P} budget.",
                           f"Got a price from {who.name} for the {f['item']}. Put it in the {P} sheet."])
        calls = [(f'email.search(query="from:{who.email} quote", limit=1)',
                  f'{{"results": [{{"subject": "Quote: {f["item"]}", "snippet": "Hi Maya, as discussed our price for the {f["item"]} is {money} including VAT, valid for 30 days..."}}]}}'),
                 (f'sheets.append_row(sheet="{P} budget", row=["{f["item"]}", "{who.org}", "{f["amount"]}", "{cur}", "quoted"])', '{"status": "ok", "row": %d}' % f["row"])]
        reply = f"Logged {who.org}'s quote for the {f['item']}: {money} (from {who.name}), in the {P} budget sheet."
    elif a == "revise_quote":
        who = people[f["person"]]
        user = _pick(rng, [f"{who.name.split()[0]} revised the {f['item']} quote for the {P}. Update the budget.",
                           f"New number from {who.org} on the {f['item']}. Can you replace the old one?"])
        calls = [(f'email.search(query="from:{who.email} revised", limit=1)',
                  f'{{"results": [{{"subject": "Revised quote: {f["item"]}", "snippet": "Hi Maya, we have revised our price for the {f["item"]} to {_money(cur, f["amount"])}..."}}]}}'),
                 (f'sheets.update_row(sheet="{P} budget", row={f["row"]}, values={{"amount": "{f["amount"]}", "status": "revised"}})', '{"status": "ok"}')]
        reply = f"Updated: {who.org}'s quote for the {f['item']} is now {_money(cur, f['amount'])} (was {_money(cur, f['old_amount'])})."
    elif a == "task_create":
        user = _pick(rng, [f"Add a task to {f['task']} for the {P}, due {_d(f['due'])}.",
                           f"Remind me to {f['task']} ({P}) by {_d(f['due'])}.",
                           f"Put '{f['task']}' on my list for the {P}. Deadline {_d(f['due'])}."])
        calls = [(f'tasks.create(project="{P}", title="{f["task"].capitalize()}", due="{f["due"]}")', f'{{"status": "created", "task_id": "{f["task_id"]}"}}')]
        reply = f"Added to your {P} list: {f['task']}, due {_d(f['due'])}."
    elif a == "task_done":
        user = _pick(rng, [f"I've done the {P} task to {f['task']}. Tick it off.",
                           f"Mark '{f['task']}' as done on the {P} list, please.",
                           f"Finished: {f['task']} ({P})."])
        calls = [(f'tasks.find(project="{P}", query="{f["task"]}")', f'{{"task_id": "{f["task_id"]}", "status": "open"}}'),
                 (f'tasks.complete(task_id="{f["task_id"]}")', '{"status": "completed"}')]
        reply = f"Marked done: {f['task']} ({P})."
    elif a == "share":
        who = people[f["person"]]
        user = _pick(rng, [f"Share the {f['document']} for the {P} with {who.name.split()[0]}.",
                           f"Can {who.name} get access to the {P} {f['document']}?",
                           f"Send {who.name.split()[0]} the latest {f['document']} ({P})."])
        calls = [(f'drive.search(query="{P} {f["document"]}")', f'{{"files": [{{"name": "{P} - {f["document"]} (v{rng.randint(2, 6)})", "modified": "{(it.ts - dt.timedelta(days=rng.randint(0, 6))).date().isoformat()}"}}]}}'),
                 (f'drive.share(file="{P} - {f["document"]}", with_email="{who.email}", role="{rng.choice(["viewer", "commenter", "editor"])}")', '{"status": "shared"}')]
        reply = f"Shared the {f['document']} for the {P} with {who.name} ({who.org})."
    elif a == "research":
        user = _pick(rng, [f"Can you look into {f['topic']} for the {P}? Just a quick summary.",
                           f"Do some quick research on {f['topic']} ({P})."])
        snippets = []
        for k in range(3):
            snippets.append(f'{{"title": "{f["topic"].capitalize()}: {rng.choice(["a practical guide", "what to know", "an overview", "2026 update", "common pitfalls"])}", '
                            f'"snippet": "{rng.choice(_FILLER)} {rng.choice(_FILLER)}"}}')
        calls = [(f'web.search(query="{f["topic"]} {p.city or "London"}")', '{"results": [' + ", ".join(snippets) + "]}")]
        reply = (f"Here's the short version on {f['topic']} for the {P}. {rng.choice(_FILLER)} "
                 f"{rng.choice(_FILLER)} I've saved the links to the {P} folder.")
    elif a == "purchase":
        money = _money(cur, f["amount"])
        user = _pick(rng, [f"Order the {f['item']} for the {P}. Use the company card.",
                           f"We need {f['item']} for the {P}. Can you buy them from {f['vendor']}?"])
        calls = [(f'shop.search(vendor="{f["vendor"]}", query="{f["item"]}")', f'{{"best": {{"item": "{f["item"]}", "price": "{money}", "delivery": "{rng.randint(2, 7)} days"}}}}'),
                 (f'shop.order(vendor="{f["vendor"]}", item="{f["item"]}", payment="company card")', f'{{"status": "ordered", "order_id": "O-{rng.randint(10000, 99999)}"}}')]
        reply = f"Ordered the {f['item']} from {f['vendor']} for {money}, charged to the company card."
    elif a == "travel":
        money = _money(cur, f["cost"])
        user = _pick(rng, [f"Book me a {f['mode']} to {f['city']} on {_d(f['date'])} for the {P}.",
                           f"I need to get to {f['city']} on {_d(f['date'])} ({P}). Can you sort travel?"])
        calls = [(f'travel.search(mode="{f["mode"]}", to="{f["city"]}", date="{f["date"]}")', f'{{"options": {rng.randint(3, 9)}, "cheapest": "{money}"}}'),
                 (f'travel.book(mode="{f["mode"]}", to="{f["city"]}", date="{f["date"]}", fare="{money}")', f'{{"status": "booked", "ref": "{rng.getrandbits(20):05X}"}}')]
        reply = f"Booked your {f['mode']} to {f['city']} on {_d(f['date'])} for {money}. Tickets are in your inbox."
    else:  # contact
        who = people[f["person"]]
        user = _pick(rng, [f"{who.name.split()[0]} has a new {f['field']}. Update their contact card, it's in their signature.",
                           f"Can you update {who.name}'s {f['field']}? They mentioned it in the {P} thread."])
        calls = [(f'contacts.find(name="{who.name}")', f'{{"name": "{who.name}", "org": "{who.org}", "role": "{who.role}", "projects": ["{P}"]}}'),
                 (f'contacts.update(name="{who.name}", field="{f["field"]}", value=...)', '{"status": "updated"}')]
        reply = f"Updated {who.name}'s {f['field']} ({who.role} at {who.org}, on the {P})."
    return user, calls, reply


def _render_one_off(rng, it):
    f = it.facts
    k = f["kind"]
    if k == "weather":
        t = rng.randint(2, 24)
        return (f"What's the weather in {f['city']} tomorrow?",
                [(f'weather.forecast(city="{f["city"]}", days=1)', f'{{"high_c": {t}, "low_c": {t - rng.randint(3, 8)}, "rain_pct": {rng.randint(0, 90)}}}')],
                f"Tomorrow in {f['city']}: around {t}°C, {rng.choice(['dry', 'some showers', 'cloudy', 'sunny spells'])}.")
    if k == "convert":
        return (f"How much is {f['amount']} euros in pounds?", [(f'fx.convert(amount={f["amount"]}, from="EUR", to="GBP")', f'{{"result": {f["amount"] * 0.85:.2f}}}')],
                f"{f['amount']} euros is about £{f['amount'] * 0.85:.2f}.")
    if k == "reminder":
        return (f"Remind me to {f['what']} on Saturday.", [(f'reminders.create(text="{f["what"]}", when="Saturday 10:00")', '{"status": "created"}')],
                f"I'll remind you to {f['what']} on Saturday morning.")
    if k == "restaurant":
        return (f"Find somewhere nice for dinner in {f['area']} on Friday.", [(f'places.search(query="restaurant", area="{f["area"]}")', '{"results": 12}')],
                f"Three good options in {f['area']} with tables on Friday: a small Italian place, a Lebanese grill and a wine bar. Want me to book one?")
    if k == "definition":
        return (f"What does '{f['word']}' mean exactly?", [], f"'{f['word'].capitalize()}' is a standard term; here's a plain explanation with an example.")
    if k == "calendar_today":
        evs = rng.sample(PERSONAL_EVENTS, 2)
        return ("What's left in my calendar today?", [(f'calendar.list(date="{it.ts.date().isoformat()}")', f'{{"events": ["{evs[0]}", "{evs[1]}"]}}')],
                f"You have {evs[0]} and {evs[1]} left today.")
    if k == "personal_email":
        return (f"Email {f['to']} to say I'm running late tonight.", [(f'email.send(to="{f["to"].lower()}@gmail.com", subject="Running late", body="Sorry, running about 20 minutes late tonight. M")', '{"status": "sent"}')],
                f"Sent {f['to']} a note that you're running late.")
    return (f"Quick one: {f['q']}?", [], f"That's about {rng.randint(2, 200)}.")


_FILLER = [
    "Most sources agree the main constraint is lead time, so booking early matters more than price.",
    "Costs vary widely by season, and the cheapest options usually carry stricter cancellation terms.",
    "Several guides recommend getting at least three comparable quotes before committing.",
    "The rules changed last year, so older articles can be misleading on the details.",
    "Smaller providers are often more flexible, though they may need longer notice for changes.",
    "Reviews mention that clear written briefs avoid most of the usual delays.",
    "The typical range quoted online is broad, and local factors move it a lot.",
    "Industry surveys suggest demand peaks in spring and again in early autumn.",
    "A common recommendation is to keep a small contingency, around ten percent, for surprises.",
    "Official guidance is short and practical, with a checklist worth following.",
]
