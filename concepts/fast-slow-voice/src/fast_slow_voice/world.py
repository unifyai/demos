"""Sam Rivera's records, the tools over them, and the questions planted in them.

A world is generated from a script seed (the planted facts the script's
questions need) and a size (how many other records surround them). The same
seed gives the same planted facts at every size, so sizes are paired.

Everything here is invented. Phone numbers use the 555-01xx range, which is
reserved for fiction.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

TODAY = date(2026, 10, 12)  # a Monday
NOW_TEXT = "Monday 12 October 2026, 9:15 am (Chicago)"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SIZES: dict[str, dict[str, int]] = {
    # contacts, events, messages, invoices, trips (planted records count towards these)
    "sanity": {"contacts": 6, "events": 4, "messages": 4, "invoices": 3, "trips": 1},
    "small": {"contacts": 20, "events": 15, "messages": 25, "invoices": 8, "trips": 3},
    "medium": {"contacts": 100, "events": 70, "messages": 200, "invoices": 40, "trips": 6},
    "realistic": {"contacts": 300, "events": 200, "messages": 700, "invoices": 100, "trips": 12},
}

FIRST = """Priya Marcus Dana Jenna Tomas Grace Kofi Maria Oliver Hannah Wei Ahmed Sofia Lucas
Amara Ethan Chloe Mateo Nadia Ravi Elena Samuel Yuki Isaac Leila Diego Fatima Noah Ingrid Omar
Clara Jamal Mei Felix Zara Andre Rosa Viktor Imani Hugo Anika Kenji Lena Tariq Beatriz Owen
Maya Rafael Esther Jonah Aisha Peter Lucia Darius Freya Malik Sienna Arjun Tessa Emeka Iris
Paolo Nina Bruno Keira Hassan Julia Stefan Ama Caleb Rhea Mikhail Abby Joaquin Linh Desmond
Greta Yusuf Carmen Elliot Naomi Victor Sana Graham Adaeze Luis Petra Ronan Hana Theo Ines""".split()

LAST = """Raman Hale Okafor Park Becker Lopez Nowak Shah Bennett Kim Moreau Adeyemi Fischer
Novak Rossi Tanaka Haddad Silva Lindqvist Mensah Kowalski Duarte Chen Abara Varga Ito Brennan
Oyelaran Castillo Petrov Hughes Dubois Mahler Achebe Sato Romero Quinn Farouk Larsen Bianchi
Nakamura Owusu Delgado Sorensen Iyer Marsh Vance Coleman Reyes Whitfield Ibrahim Kaur Morales
Lindgren Osei Barros Kovac Holt Ferreira Aziz Callahan Ng Oduya Strand Pereira Jansen Malone
Rahman Toivonen Ortega Wolfe Bakshi Lund Eze Carvalho Mendes Frost Kagame Russo Brandt Yilmaz""".split()

COMPANIES = [
    ("Northwind Logistics", "northwind-logistics.example"), ("Halden Partners", "halden.example"),
    ("Brightwave Media", "brightwave.example"), ("Delta Freight", "deltafreight.example"),
    ("Corvid Systems", "corvid.example"), ("Mapleton Supply", "mapleton.example"),
    ("Juniper Health", "juniperhealth.example"), ("Tidewater Foods", "tidewater.example"),
    ("Granite Peak Capital", "granitepeak.example"), ("Lumen Print Co", "lumenprint.example"),
    ("Osprey Couriers", "osprey.example"), ("Redfern Electrical", "redfern.example"),
    ("Silverline Travel", "silverline.example"), ("Ashby & Cole", "ashbycole.example"),
    ("Quarry Lane Studio", "quarrylane.example"), ("Bluestem Software", "bluestem.example"),
    ("Kestrel Security", "kestrel.example"), ("Fernhill Catering", "fernhill.example"),
    ("Ironbridge Engineering", "ironbridge.example"), ("Pinecrest Realty", "pinecrest.example"),
    ("Saffron Office Supplies", "saffron.example"), ("Westgate Insurance", "westgate.example"),
    ("Harbor Point Legal", "harborpoint.example"), ("Cobalt Analytics", "cobalt.example"),
    ("Meridian Facilities", "meridianfm.example"), ("Oakleaf Furniture", "oakleaf.example"),
    ("Riverside Clinic", "riversideclinic.example"), ("Summit Cleaning", "summitclean.example"),
    ("Vantage Telecom", "vantagetel.example"), ("Wildflower Events", "wildflower.example"),
]
SAM_COMPANY = ("Larkspur Analytics", "larkspur.example")
CITIES_LOCAL = ["Chicago", "Evanston", "Oak Park", "Naperville", "Skokie", "Milwaukee"]
AREA = ["312", "773", "872", "630", "847", "414", "708", "224", "331", "262"]
ROLES_WORK = ["account manager", "operations lead", "procurement lead", "finance director",
              "project manager", "sales lead", "engineer", "analyst", "office manager", "partner"]
STREETS = ["Fulton", "Erie", "Halsted", "Kinzie", "Morgan", "Racine", "Ashland", "Damen",
           "Clybourn", "Division", "Wabash", "Grand", "Lake", "Ogden", "Elston", "Archer",
           "Cermak", "Roosevelt", "Superior", "Wells", "Canal", "Jackson", "Monroe", "Madison"]
RESTAURANTS = ["Marlowe's", "The Copper Fig", "Osteria Lume", "Hearth & Vine", "Saltwater",
               "Bramble", "The Gilded Pear", "Kiku", "Casa Morena", "Juniper Room",
               "The Long Table", "Pomelo", "Ember Grill", "Little Saint", "Avenue Bistro"]
TRIP_CITIES = [("Denver", "DEN"), ("Austin", "AUS"), ("Boston", "BOS"), ("Seattle", "SEA"),
               ("Atlanta", "ATL"), ("Phoenix", "PHX"), ("Nashville", "BNA"), ("Minneapolis", "MSP"),
               ("Detroit", "DTW"), ("San Diego", "SAN"), ("Toronto", "YYZ"), ("Portland", "PDX"),
               ("Raleigh", "RDU"), ("Pittsburgh", "PIT"), ("Kansas City", "MCI")]
AIRLINES = ["UA", "AA", "DL", "WN", "AS"]
APPOINTMENTS = ["dentist appointment", "physio session", "eye test", "haircut", "car service",
                "vet appointment", "parent-teacher meeting", "bank appointment"]
PLACE_EVENTS = ["supplier review", "team offsite", "board meeting", "site visit",
                "client workshop", "quarterly planning day", "budget review", "vendor demo"]
BASE_TITLES = ["Team standup", "Weekly ops review", "Gym", "Lunch", "Focus time", "Hiring panel",
               "Expense review", "Product sync", "Coffee catch-up", "All-hands", "Training session",
               "Call with the accountant", "Interview", "Strategy session", "School pickup"]


@dataclass
class Contact:
    id: str
    first: str
    last: str
    company: str
    role: str
    city: str
    mobile: str
    email: str

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


@dataclass
class Event:
    id: str
    date: str  # ISO
    start: str  # "3:20 pm"
    end: str
    title: str
    attendees: list[str]  # contact names
    location: str


@dataclass
class Message:
    id: str
    sender: str  # contact name or organisation
    date: str  # ISO date
    time: str
    channel: str  # email or text
    subject: str
    body: str


@dataclass
class Invoice:
    id: str
    vendor: str
    number: str
    amount: str  # "$2,340.50"
    issued: str
    due: str
    status: str  # paid or unpaid


@dataclass
class Trip:
    id: str
    kind: str  # flight
    carrier: str
    number: str  # "UA 1187"
    date: str
    depart: str
    origin: str
    destination: str
    confirmation: str


@dataclass
class Item:
    """One question (or request) planted in the world, with its gold answer."""

    id: str
    kind: str
    text: str  # what Sam says
    gold: str  # the answer as the records give it
    keys: list[str]  # normalised tokens that must all be heard in one agent utterance
    key_type: str  # digits, code, time, email, tokens
    difficulty: str  # easy or hard
    person: str | None = None  # the contact the question is about, if any
    thing: str = "answer"  # what Sam would call it when asking again ("number", "time", ...)
    extra: dict = field(default_factory=dict)


@dataclass
class World:
    seed: int
    size: str
    contacts: list[Contact]
    events: list[Event]
    messages: list[Message]
    invoices: list[Invoice]
    trips: list[Trip]

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "size": self.size,
            "contacts": [asdict(c) for c in self.contacts],
            "events": [asdict(e) for e in self.events],
            "messages": [asdict(m) for m in self.messages],
            "invoices": [asdict(i) for i in self.invoices],
            "trips": [asdict(t) for t in self.trips],
        }

    @classmethod
    def from_json(cls, d: dict) -> "World":
        return cls(
            seed=d["seed"],
            size=d["size"],
            contacts=[Contact(**c) for c in d["contacts"]],
            events=[Event(**e) for e in d["events"]],
            messages=[Message(**m) for m in d["messages"]],
            invoices=[Invoice(**i) for i in d["invoices"]],
            trips=[Trip(**t) for t in d["trips"]],
        )

    # ------------------------------------------------------------------ text
    def as_text(self) -> str:
        """All of Sam's records as plain text, for the long-context arm."""
        lines = ["CONTACTS (name | company | role | city | mobile | email)"]
        for c in sorted(self.contacts, key=lambda c: (c.last, c.first)):
            lines.append(f"- {c.name} | {c.company} | {c.role} | {c.city} | {c.mobile} | {c.email}")
        lines.append("")
        lines.append("CALENDAR (date | start-end | title | with | where)")
        for e in sorted(self.events, key=lambda e: (e.date, _minutes(e.start))):
            lines.append(f"- {_day(e.date)} | {e.start}-{e.end} | {e.title} | {', '.join(e.attendees) or '-'} | {e.location}")
        lines.append("")
        lines.append("MESSAGES (id | date time | channel | from | subject | body)")
        for m in sorted(self.messages, key=lambda m: (m.date, _minutes(m.time))):
            lines.append(f"- {m.id} | {_day(m.date)} {m.time} | {m.channel} | {m.sender} | {m.subject} | {m.body}")
        lines.append("")
        lines.append("INVOICES (vendor | number | amount | issued | due | status)")
        for i in sorted(self.invoices, key=lambda i: (i.vendor, i.issued)):
            lines.append(f"- {i.vendor} | {i.number} | {i.amount} | {i.issued} | {i.due} | {i.status}")
        lines.append("")
        lines.append("TRIPS (date | carrier | number | depart | from-to | confirmation)")
        for t in sorted(self.trips, key=lambda t: t.date):
            lines.append(f"- {_day(t.date)} | {t.carrier} | {t.number} | {t.depart} | {t.origin}-{t.destination} | {t.confirmation}")
        return "\n".join(lines)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True).encode()).hexdigest()[:16]


def _day(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{WEEKDAYS[d.weekday()][:3]} {d.isoformat()}"


def _minutes(t: str) -> int:
    m = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)", t)
    if not m:
        return 0
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return h * 60 + mi


def fmt_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    ap = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {ap}"


def time_keys(minutes: int) -> list[str]:
    h, m = divmod(minutes, 60)
    h12 = h % 12 or 12
    return [f"{h12}:{m:02d}"] if h == h12 else [f"{h12}:{m:02d}", f"{h}:{m:02d}"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {"type": "function", "function": {
        "name": "find_contacts",
        "description": "Search Sam's contacts by any words of a name, company, role or city. Returns up to 5 best matches with mobile and email.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calendar",
        "description": "Sam's calendar events from start_date to end_date inclusive (YYYY-MM-DD; end_date defaults to start_date). Returns up to 20 events.",
        "parameters": {"type": "object", "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["start_date"]}}},
    {"type": "function", "function": {
        "name": "search_messages",
        "description": "Search Sam's emails and texts by words in the sender, subject or body. Returns up to 6 best matches, newest first among equals, each with a snippet; use read_message for the full text.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "sender": {"type": "string", "description": "optional: words of the sender's name"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_message",
        "description": "The full text of one message, by its id.",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}},
    {"type": "function", "function": {
        "name": "invoices",
        "description": "Invoices Sam's company has received, newest first. Optional filters: vendor (words of the company name), status ('paid' or 'unpaid'). Returns up to 10.",
        "parameters": {"type": "object", "properties": {"vendor": {"type": "string"}, "status": {"type": "string", "enum": ["paid", "unpaid"]}}}}},
    {"type": "function", "function": {
        "name": "trips",
        "description": "Sam's upcoming trips (flights), optionally filtered by words of the destination city. Returns up to 10.",
        "parameters": {"type": "object", "properties": {"destination": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "send_text",
        "description": "Send a text message from Sam to one contact, named by full name (or mobile number).",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}}, "required": ["to", "text"]}}},
]
LOOKUP_TOOLS = [t for t in TOOL_SPECS if t["function"]["name"] != "send_text"]
ACTION_TOOLS = [t for t in TOOL_SPECS if t["function"]["name"] == "send_text"]

TOOL_LATENCY = {"find_contacts": 0.7, "calendar": 0.9, "search_messages": 2.2, "read_message": 0.6,
                "invoices": 1.2, "trips": 1.0, "send_text": 1.5}


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9@.&'-]+", s.lower().replace("’", "'")) if len(t) >= 2]


def _match_score(query_tokens: list[str], text: str) -> int:
    words = set(re.findall(r"[a-z0-9]+", text.lower().replace("'", "")))
    score = 0
    for q in query_tokens:
        q2 = re.sub(r"[^a-z0-9]", "", q)
        if q2 and (q2 in words or any(w.startswith(q2) for w in words if len(q2) >= 4)):
            score += 1
    return score


class Tools:
    """The tools over one world. Pure functions of the world and the arguments."""

    def __init__(self, world: World):
        self.w = world

    def call(self, name: str, args: dict) -> dict | list:
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**{k: v for k, v in (args or {}).items() if v not in (None, "")})
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}

    def t_find_contacts(self, query: str) -> list | dict:
        qt = _tokens(query)
        scored = []
        for c in self.w.contacts:
            s = _match_score(qt, f"{c.first} {c.last} {c.company} {c.role} {c.city}")
            if s:
                scored.append((-s, c.last, c.first, c))
        scored.sort(key=lambda x: x[:3])
        out = [{"name": c.name, "company": c.company, "role": c.role, "city": c.city, "mobile": c.mobile, "email": c.email}
               for *_, c in scored[:5]]
        return out if out else {"result": "no contacts match"}

    def t_calendar(self, start_date: str, end_date: str | None = None) -> list | dict:
        try:
            d0 = date.fromisoformat(start_date.strip()[:10])
            d1 = date.fromisoformat(end_date.strip()[:10]) if end_date else d0
        except ValueError:
            return {"error": "dates must be YYYY-MM-DD"}
        evs = [e for e in self.w.events if d0 <= date.fromisoformat(e.date) <= d1]
        evs.sort(key=lambda e: (e.date, _minutes(e.start)))
        out = [{"date": _day(e.date), "start": e.start, "end": e.end, "title": e.title,
                "with": e.attendees, "where": e.location} for e in evs[:20]]
        if len(evs) > 20:
            out.append({"note": f"{len(evs) - 20} more events; ask for a shorter range"})
        return out if out else {"result": "no events in that range"}

    def t_search_messages(self, query: str, sender: str | None = None) -> list | dict:
        qt = _tokens(query)
        st = _tokens(sender) if sender else []
        scored = []
        for m in self.w.messages:
            if st and _match_score(st, m.sender) == 0:
                continue
            s = _match_score(qt, f"{m.sender} {m.subject} {m.body}")
            if s or (st and not qt):
                scored.append((-s, m.date, m.time, m))
        # best score first, newest first among equals
        scored.sort(key=lambda x: (x[0], -date.fromisoformat(x[1]).toordinal(), -_minutes(x[2])))
        out = [{"id": m.id, "from": m.sender, "date": _day(m.date), "time": m.time, "channel": m.channel,
                "subject": m.subject, "snippet": m.body[:160]} for *_, m in scored[:6]]
        return out if out else {"result": "no messages match"}

    def t_read_message(self, id: str) -> dict:
        for m in self.w.messages:
            if m.id == id.strip():
                return {"id": m.id, "from": m.sender, "date": _day(m.date), "time": m.time,
                        "channel": m.channel, "subject": m.subject, "body": m.body}
        return {"error": f"no message with id {id}"}

    def t_invoices(self, vendor: str | None = None, status: str | None = None) -> list | dict:
        out = list(self.w.invoices)
        if vendor:
            vt = _tokens(vendor)
            out = [i for i in out if _match_score(vt, i.vendor) >= max(1, len(vt) - 1)]
        if status in ("paid", "unpaid"):
            out = [i for i in out if i.status == status]
        out.sort(key=lambda i: i.issued, reverse=True)
        res = [{"vendor": i.vendor, "number": i.number, "amount": i.amount, "issued": i.issued,
                "due": i.due, "status": i.status} for i in out[:10]]
        return res if res else {"result": "no invoices match"}

    def t_trips(self, destination: str | None = None) -> list | dict:
        out = [t for t in self.w.trips if date.fromisoformat(t.date) >= TODAY]
        if destination:
            dt = _tokens(destination)
            out = [t for t in out if _match_score(dt, t.destination) >= 1]
        out.sort(key=lambda t: t.date)
        res = [{"date": _day(t.date), "carrier": t.carrier, "number": t.number, "depart": t.depart,
                "from": t.origin, "to": t.destination, "confirmation": t.confirmation} for t in out[:10]]
        return res if res else {"result": "no trips match"}

    def t_send_text(self, to: str, text: str) -> dict:
        target = to.strip().lower()
        digits = re.sub(r"\D", "", target)
        hits = [c for c in self.w.contacts if c.name.lower() == target]
        if not hits and len(digits) >= 7:
            hits = [c for c in self.w.contacts if re.sub(r"\D", "", c.mobile).endswith(digits[-10:])]
        if not hits:
            firsts = [c for c in self.w.contacts if c.first.lower() == target]
            if len(firsts) == 1:
                hits = firsts
            elif len(firsts) > 1:
                return {"error": f"ambiguous: {', '.join(c.name for c in firsts)}; use the full name"}
        if not hits:
            return {"error": f"no contact named {to}"}
        c = hits[0]
        return {"status": "sent", "to": c.name, "mobile": c.mobile, "text": text}


def tool_latency(name: str, args: dict, script_seed: int, run_seed: int, sigma: float = 0.35) -> float:
    """Base latency times a log-normal factor drawn from a hash of the call, so the
    same call takes the same time in every arm."""
    h = hashlib.sha256(json.dumps([name, args, script_seed, run_seed], sort_keys=True).encode()).digest()
    rng = random.Random(int.from_bytes(h[:8], "big"))
    return round(TOOL_LATENCY.get(name, 1.0) * rng.lognormvariate(0.0, sigma), 3)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class _Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.used_names: set[str] = set()
        self.used_firsts: dict[str, int] = {}
        self.used_mobiles: set[str] = set()
        self.ids = {"c": 0, "e": 0, "m": 0, "i": 0, "t": 0}

    def nid(self, k: str) -> str:
        self.ids[k] += 1
        return f"{k}{self.ids[k]:04d}"

    def mobile(self) -> str:
        while True:
            m = f"{self.rng.choice(AREA)}-555-01{self.rng.randint(0, 99):02d}"
            if m not in self.used_mobiles:
                self.used_mobiles.add(m)
                return m

    def contact(self, first: str | None = None, last: str | None = None, company: tuple | None = None,
                role: str | None = None, unique_first: bool = False) -> Contact:
        for _ in range(1000):
            f = first or self.rng.choice(FIRST)
            l = last or self.rng.choice(LAST)
            if unique_first and first is None and f in self.used_firsts:
                continue
            if f"{f} {l}" not in self.used_names:
                break
        else:  # pragma: no cover
            raise RuntimeError("ran out of names")
        comp = company or self.rng.choice(COMPANIES)
        c = Contact(self.nid("c"), f, l, comp[0], role or self.rng.choice(ROLES_WORK),
                    self.rng.choice(CITIES_LOCAL), self.mobile(),
                    f"{f.lower()}.{l.lower()}@{comp[1]}")
        self.used_names.add(c.name)
        self.used_firsts[f] = self.used_firsts.get(f, 0) + 1
        return c


def _week_day(rng: random.Random, lo: int = 1, hi: int = 4) -> date:
    """A day this week, Tuesday (1) to Friday (4) by default."""
    return TODAY + timedelta(days=rng.randint(lo, hi))


def _amount(rng: random.Random) -> str:
    v = rng.randint(30000, 980000)  # cents
    return f"${v // 100:,}.{v % 100:02d}"


def _code(rng: random.Random, n: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        c = "".join(rng.choice(alphabet) for _ in range(n))
        if any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c):
            return c


LOOKUP_KINDS_EASY = ["phone", "meeting_time", "flight_number", "invoice_unpaid", "booking_ref", "meeting_place"]
LOOKUP_KINDS_HARD = ["phone_ambig", "code_latest", "meeting_who_phone", "invoice_via_contact"]


@dataclass
class Planted:
    """The planted records and items for one script, before the world is filled in."""

    contacts: list[Contact] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    invoices: list[Invoice] = field(default_factory=list)
    trips: list[Trip] = field(default_factory=list)
    reserved_slots: set = field(default_factory=set)  # (date, start) used by planted events
    reserved_titles: set = field(default_factory=set)  # (date, keyword)
    reserved_vendors: set = field(default_factory=set)
    reserved_restaurants: set = field(default_factory=set)
    reserved_streets: set = field(default_factory=set)
    reserved_trip_days: set = field(default_factory=set)  # (date, city)


class Planter:
    """Plants the records each question needs and returns the Item."""

    def __init__(self, g: _Gen, p: Planted):
        self.g, self.p, self.rng = g, p, g.rng
        self.n = 0

    def _id(self) -> str:
        self.n += 1
        return f"q{self.n}"

    def _free_slot(self, avoid_titles: str | None = None) -> tuple[date, int]:
        for _ in range(500):
            d = _week_day(self.rng)
            m = self.rng.choice(range(9 * 60, 17 * 60, 5))
            if m % 30 == 0:  # keep planted times off the hour and half hour
                continue
            if (d.isoformat(), fmt_time(m)) not in self.p.reserved_slots and \
                    (avoid_titles is None or (d.isoformat(), avoid_titles) not in self.p.reserved_titles):
                return d, m
        raise RuntimeError("no free slot")

    def phone(self) -> Item:
        c = self.g.contact(company=self.rng.choice(COMPANIES), unique_first=True)
        self.p.contacts.append(c)
        text = self.rng.choice([f"What's {c.name}'s mobile number?", f"Can you give me {c.name}'s cell number?",
                                f"I need {c.name}'s number, can you find it?"])
        return Item(self._id(), "phone", text, c.mobile, [re.sub(r"\D", "", c.mobile)], "digits", "easy",
                    person=c.name, thing="number")

    def phone_ambig(self) -> Item:
        first = self.rng.choice([f for f in FIRST if f not in self.g.used_firsts])
        comps = self.rng.sample(COMPANIES, 2)
        a = self.g.contact(first=first, company=comps[0])
        b = self.g.contact(first=first, company=comps[1])
        self.p.contacts += [a, b]
        short = comps[0][0].split()[0]
        text = self.rng.choice([f"What's {first}'s number? {first} from {short}, I mean.",
                                f"Can you get me {first}'s cell? The {first} at {short}."])
        return Item(self._id(), "phone_ambig", text, a.mobile, [re.sub(r"\D", "", a.mobile)], "digits", "hard",
                    person=a.name, thing="number", extra={"distractor": b.name})

    def email_followup(self, prev: Item) -> Item:
        c = next(c for c in self.p.contacts if c.name == prev.person)
        text = self.rng.choice(["And what's their email?", "Great, and their email address?", "And their email, please?"])
        return Item(self._id(), "email_followup", text, c.email, [c.email.lower()], "email", "hard",
                    person=c.name, thing="email", extra={"refers_to": prev.id})

    def meeting_time(self) -> Item:
        title = self.rng.choice([t for t in APPOINTMENTS if t not in {x[1] for x in self.p.reserved_titles}])
        d, m = self._free_slot()
        loc = f"{self.rng.randint(100, 2999)} N {self.rng.choice(STREETS)} St"
        e = Event(self.g.nid("e"), d.isoformat(), fmt_time(m), fmt_time(m + 45), title.capitalize(), [], loc)
        self.p.events.append(e)
        self.p.reserved_slots.add((e.date, e.start))
        self.p.reserved_titles.add((e.date, title))
        wd = WEEKDAYS[d.weekday()]
        text = self.rng.choice([f"What time is my {title} on {wd}?", f"When's the {title} on {wd} again?"])
        return Item(self._id(), "meeting_time", text, e.start, time_keys(m), "time", "easy", thing="time",
                    extra={"date": e.date})

    def meeting_place(self) -> Item:
        title = self.rng.choice([t for t in PLACE_EVENTS if t not in {x[1] for x in self.p.reserved_titles}])
        d, m = self._free_slot()
        num = self.rng.randint(100, 2999)
        street = self.rng.choice([s for s in STREETS if s not in self.p.reserved_streets])
        self.p.reserved_streets.add(street)
        loc = f"{num} W {street} St"
        e = Event(self.g.nid("e"), d.isoformat(), fmt_time(m), fmt_time(m + 90), title.capitalize(), [], loc)
        self.p.events.append(e)
        self.p.reserved_slots.add((e.date, e.start))
        self.p.reserved_titles.add((e.date, title))
        wd = WEEKDAYS[d.weekday()]
        text = self.rng.choice([f"Where's my {title} on {wd}?", f"What's the address for the {title} on {wd}?"])
        return Item(self._id(), "meeting_place", text, loc, [str(num), street.lower()], "tokens", "easy",
                    thing="address", extra={"date": e.date})

    def meeting_who_phone(self) -> Item:
        c = self.g.contact(company=self.rng.choice(COMPANIES), unique_first=True)
        self.p.contacts.append(c)
        d, m = self._free_slot()
        e = Event(self.g.nid("e"), d.isoformat(), fmt_time(m), fmt_time(m + 30), f"Meeting with {c.name}",
                  [c.name], self.rng.choice(["Zoom", "Sam's office", f"{c.company} office"]))
        self.p.events.append(e)
        self.p.reserved_slots.add((e.date, e.start))
        wd = WEEKDAYS[d.weekday()]
        t = fmt_time(m).replace(" am", "").replace(" pm", "")
        text = self.rng.choice([f"Who am I meeting at {t} on {wd}? And can you give me their number?",
                                f"I've got something at {t} on {wd}, who's that with? I'll need their number too."])
        return Item(self._id(), "meeting_who_phone", text, f"{c.name}, {c.mobile}", [re.sub(r"\D", "", c.mobile)],
                    "digits", "hard", person=c.name, thing="number", extra={"date": e.date, "name_key": c.last.lower()})

    def code_latest(self) -> Item:
        c = self.g.contact(company=self.rng.choice(COMPANIES), role="operations lead", unique_first=True)
        self.p.contacts.append(c)
        street = self.rng.choice([s for s in STREETS if s not in self.p.reserved_streets])
        self.p.reserved_streets.add(street)
        old, new = str(self.rng.randint(1000, 9999)), str(self.rng.randint(1000, 9999))
        while new[:2] == old[:2]:
            new = str(self.rng.randint(1000, 9999))
        d_old = TODAY - timedelta(days=self.rng.randint(20, 40))
        d_new = TODAY - timedelta(days=self.rng.randint(2, 6))
        self.p.messages.append(Message(self.g.nid("m"), c.name, d_old.isoformat(), "10:12 am", "text",
                                       f"{street} warehouse", f"Hi Sam, the gate code for the {street} St warehouse is {old}#. See you there."))
        self.p.messages.append(Message(self.g.nid("m"), c.name, d_new.isoformat(), "4:47 pm", "text",
                                       f"{street} warehouse gate", f"Heads up: we changed the gate code at the {street} St warehouse. The new code is {new}#. The old one no longer works."))
        text = self.rng.choice([f"What's the gate code for the {street} warehouse? {c.first} sent it to me at some point.",
                                f"{c.first} texted me the gate code for the {street} warehouse, what is it?"])
        return Item(self._id(), "code_latest", text, f"{new}#", [new], "digits", "hard", person=c.name, thing="code",
                    extra={"old": old})

    def invoice_unpaid(self) -> Item:
        comp = self.rng.choice([c for c in COMPANIES if c[0] not in self.p.reserved_vendors])
        self.p.reserved_vendors.add(comp[0])
        gold = self._vendor_invoices(comp[0])
        short = comp[0]
        text = self.rng.choice([f"How much do we still owe {short}?", f"What does the unpaid invoice from {short} come to?"])
        return Item(self._id(), "invoice_unpaid", text, gold, [re.sub(r"\D", "", gold)], "digits", "easy", thing="amount")

    def invoice_via_contact(self) -> Item:
        comp = self.rng.choice([c for c in COMPANIES if c[0] not in self.p.reserved_vendors])
        self.p.reserved_vendors.add(comp[0])
        c = self.g.contact(company=comp, unique_first=True)
        self.p.contacts.append(c)
        gold = self._vendor_invoices(comp[0])
        text = self.rng.choice([f"How much do we owe {c.name}'s company?",
                                f"{c.name}'s firm sent us an invoice we haven't paid. How much is it?"])
        return Item(self._id(), "invoice_via_contact", text, gold, [re.sub(r"\D", "", gold)], "digits", "hard",
                    person=c.name, thing="amount", extra={"vendor": comp[0]})

    def _vendor_invoices(self, vendor: str) -> str:
        n_paid = self.rng.randint(2, 3)
        unpaid_amt = _amount(self.rng)
        for k in range(n_paid):
            issued = TODAY - timedelta(days=30 * (k + 1) + self.rng.randint(0, 9))
            self.p.invoices.append(Invoice(self.g.nid("i"), vendor, f"INV-{self.rng.randint(1000, 9999)}",
                                           _amount(self.rng), issued.isoformat(), (issued + timedelta(days=30)).isoformat(), "paid"))
        issued = TODAY - timedelta(days=self.rng.randint(3, 12))
        self.p.invoices.append(Invoice(self.g.nid("i"), vendor, f"INV-{self.rng.randint(1000, 9999)}", unpaid_amt,
                                       issued.isoformat(), (issued + timedelta(days=30)).isoformat(), "unpaid"))
        return unpaid_amt

    def flight_number(self) -> Item:
        city, code = self.rng.choice(TRIP_CITIES)
        d = _week_day(self.rng)
        while (d.isoformat(), city) in self.p.reserved_trip_days:
            d = _week_day(self.rng)
        self.p.reserved_trip_days.add((d.isoformat(), city))
        carrier = self.rng.choice(AIRLINES)
        num = f"{carrier} {self.rng.randint(100, 2999)}"
        t = Trip(self.g.nid("t"), "flight", carrier, num, d.isoformat(), fmt_time(self.rng.choice(range(6 * 60, 20 * 60, 5))),
                 "ORD", f"{city} ({code})", _code(self.rng))
        self.p.trips.append(t)
        wd = WEEKDAYS[d.weekday()]
        text = self.rng.choice([f"What's my flight number for {city} on {wd}?", f"Which flight am I on to {city} on {wd}?"])
        return Item(self._id(), "flight_number", text, num, [num.replace(" ", "")], "code", "easy", thing="flight number",
                    extra={"city": city})

    def booking_ref(self) -> Item:
        r = self.rng.choice([x for x in RESTAURANTS if x not in self.p.reserved_restaurants])
        self.p.reserved_restaurants.add(r)
        d = _week_day(self.rng, 3, 4)
        ref = _code(self.rng)
        wd = WEEKDAYS[d.weekday()]
        sent = TODAY - timedelta(days=self.rng.randint(1, 8))
        self.p.messages.append(Message(self.g.nid("m"), r, sent.isoformat(), "2:05 pm", "email",
                                       f"Your table at {r} is confirmed",
                                       f"Dear Sam, your table for 4 at {r} on {wd} {d.day} October at 7:30 pm is confirmed. Booking reference: {ref}. We look forward to seeing you."))
        text = self.rng.choice([f"What's the booking reference for {r} on {wd}?",
                                f"Can you find the booking ref for dinner at {r} on {wd}?"])
        return Item(self._id(), "booking_ref", text, ref, [ref], "code", "easy", thing="reference",
                    extra={"restaurant": r})

    def text_late(self, person: str | None = None) -> Item:
        if person is None:
            c = self.g.contact(company=self.rng.choice(COMPANIES), unique_first=True)
            self.p.contacts.append(c)
            person = c.name
        text = self.rng.choice([f"Can you text {person} and tell them I'm running about ten minutes late?",
                                f"Could you send {person} a text saying I'll be ten minutes late?"])
        return Item(self._id(), "text_late", text, f"text sent to {person}", [], "action", "easy", person=person,
                    thing="text")

    def make(self, kind: str) -> Item:
        return getattr(self, kind)()


def fill_world(seed: int, size: str, planted: Planted, used_names: set[str], used_mobiles: set[str]) -> World:
    """Surround the planted records with others, up to the size's counts."""
    counts = SIZES[size]
    g = _Gen(int(hashlib.sha256(f"{seed}:{size}".encode()).hexdigest()[:12], 16))
    g.used_names = set(used_names)
    g.used_mobiles = set(used_mobiles)
    g.ids = {"c": 5000, "e": 5000, "m": 5000, "i": 5000, "t": 5000}
    rng = g.rng
    planted_firsts = {c.first for c in planted.contacts}
    # first names of ambiguous pairs must stay exactly two; any planted first name stays unique
    contacts = list(planted.contacts)
    while len(contacts) < counts["contacts"]:
        for _ in range(100):
            c = g.contact()
            if c.first not in planted_firsts:
                break
            g.used_names.discard(c.name)
        else:  # pragma: no cover
            raise RuntimeError("could not avoid planted first names")
        contacts.append(c)
    others = [c for c in contacts if c not in planted.contacts] or contacts

    events = list(planted.events)
    reserved_kw = {kw for (_, kw) in planted.reserved_titles}
    while len(events) < counts["events"]:
        d = TODAY + timedelta(days=rng.randint(-7, 18))
        if d.weekday() >= 5:
            continue
        m = rng.choice(range(8 * 60, 18 * 60, 30))  # base events sit on the hour and half hour
        if (d.isoformat(), fmt_time(m)) in planted.reserved_slots:
            continue
        title = rng.choice(BASE_TITLES)
        att: list[str] = []
        if title in ("Lunch", "Coffee catch-up", "Interview"):
            att = [rng.choice(others).name]
            title = f"{title} with {att[0].split()[0]}"
        elif rng.random() < 0.3:
            att = [rng.choice(others).name for _ in range(rng.randint(1, 3))]
        if any(kw in title.lower() for kw in reserved_kw):
            continue
        loc = rng.choice(["Zoom", "Sam's office", "Larkspur HQ, 4th floor", f"{rng.randint(100, 2999)} N {rng.choice(STREETS)} St"])
        events.append(Event(g.nid("e"), d.isoformat(), fmt_time(m), fmt_time(m + rng.choice([30, 45, 60])), title, att, loc))

    messages = list(planted.messages)
    free_streets = [s for s in STREETS if s not in planted.reserved_streets]
    free_rest = [r for r in RESTAURANTS if r not in planted.reserved_restaurants]
    while len(messages) < counts["messages"]:
        c = rng.choice(others)
        d = TODAY - timedelta(days=rng.randint(0, 45))
        t = fmt_time(rng.choice(range(7 * 60, 22 * 60, 1)))
        kind = rng.random()
        if kind < 0.08 and free_streets:
            s = rng.choice(free_streets)
            subj, body = f"{s} warehouse", f"Hi Sam, the gate code for the {s} St warehouse is {rng.randint(1000, 9999)}#."
            sender = c.name
        elif kind < 0.14 and free_rest:
            r = rng.choice(free_rest)
            wd = WEEKDAYS[rng.randint(0, 6)]
            subj, body = f"Your table at {r} is confirmed", f"Dear Sam, your table for 2 at {r} on {wd} at 8:00 pm is confirmed. Booking reference: {_code(rng)}."
            sender = r
        else:
            subj, body = rng.choice(_CHATTER)(rng, c, others)
            sender = c.name
        messages.append(Message(g.nid("m"), sender, d.isoformat(), t, rng.choice(["email", "text"]), subj, body))

    invoices = list(planted.invoices)
    vendors = [c for c in COMPANIES if c[0] not in planted.reserved_vendors]
    while len(invoices) < counts["invoices"]:
        v = rng.choice(vendors)[0]
        issued = TODAY - timedelta(days=rng.randint(5, 200))
        invoices.append(Invoice(g.nid("i"), v, f"INV-{rng.randint(1000, 9999)}", _amount(rng), issued.isoformat(),
                                (issued + timedelta(days=30)).isoformat(), rng.choice(["paid", "paid", "unpaid"])))

    trips = list(planted.trips)
    while len(trips) < counts["trips"]:
        city, code = rng.choice(TRIP_CITIES)
        d = TODAY + timedelta(days=rng.randint(1, 40))
        # a planted trip's city gets no other trips, so "my flight to Denver on Thursday" has one answer
        if city in {c for (_, c) in planted.reserved_trip_days} or any(
                (t.date == d.isoformat() and t.destination.startswith(city)) for t in trips):
            continue
        carrier = rng.choice(AIRLINES)
        trips.append(Trip(g.nid("t"), "flight", carrier, f"{carrier} {rng.randint(100, 2999)}", d.isoformat(),
                          fmt_time(rng.choice(range(6 * 60, 20 * 60, 5))), "ORD", f"{city} ({code})", _code(rng)))
    rng.shuffle(contacts)
    rng.shuffle(events)
    rng.shuffle(messages)
    return World(seed, size, contacts, events, messages, invoices, trips)


def _chat_lunch(rng, c, others):
    return "Lunch?", f"Hey Sam, are you free for lunch {rng.choice(WEEKDAYS[:5])}? {c.first}"


def _chat_update(rng, c, others):
    return f"Update from {c.company}", f"Hi Sam, quick update: the {rng.choice(['shipment', 'report', 'contract', 'proposal'])} is on track for the end of the month. Best, {c.first}"


def _chat_intro(rng, c, others):
    o = rng.choice(others)
    return "Introduction", f"Sam, meet {o.name} from {o.company}. I think you two should talk. {c.first}"


def _chat_running(rng, c, others):
    return "Running late", f"Running {rng.choice([5, 10, 15])} minutes late, sorry! {c.first}"


def _chat_thanks(rng, c, others):
    return "Thanks", f"Thanks for yesterday, Sam. Really useful. {c.first}"


def _chat_meeting(rng, c, others):
    return "Next week", f"Can we move our call to {rng.choice(WEEKDAYS[:5])}? {c.first}"


_CHATTER = [_chat_lunch, _chat_update, _chat_intro, _chat_running, _chat_thanks, _chat_meeting]


# ---------------------------------------------------------------------------
# Reference solver: checks every planted answer is reachable and unambiguous
# ---------------------------------------------------------------------------


def solve(item: Item, world: World) -> str | None:
    """Answer an item through the tools, as a careful agent would. Returns the gold
    value it finds, or None if the records do not determine it."""
    T = Tools(world)
    k = item.kind
    if k in ("phone", "phone_ambig", "email_followup"):
        c = next((c for c in world.contacts if c.name == item.person), None)
        if k == "phone":
            hits = T.t_find_contacts(item.person)
            same = [h for h in hits if h["name"] == item.person] if isinstance(hits, list) else []
            return same[0]["mobile"] if len(same) == 1 else None
        if k == "phone_ambig":
            first = item.person.split()[0]
            company_word = c.company.split()[0]
            hits = T.t_find_contacts(f"{first} {company_word}")
            same = [h for h in hits if h["name"].split()[0] == first and h["company"].split()[0] == company_word] if isinstance(hits, list) else []
            return same[0]["mobile"] if len(same) == 1 else None
        hits = T.t_find_contacts(item.person)
        same = [h for h in hits if h["name"] == item.person] if isinstance(hits, list) else []
        return same[0]["email"] if len(same) == 1 else None
    if k in ("meeting_time", "meeting_place", "meeting_who_phone"):
        evs = T.t_calendar(item.extra["date"])
        if not isinstance(evs, list):
            return None
        if k == "meeting_who_phone":
            m = re.search(r"at (\d{1,2}:\d{2})", item.text)
            hm = m.group(1) if m else ""
            at = [e for e in evs if e.get("start", "").startswith(hm + " ")]
            if len(at) != 1 or len(at[0]["with"]) != 1:
                return None
            hits = T.t_find_contacts(at[0]["with"][0])
            same = [h for h in hits if h["name"] == at[0]["with"][0]] if isinstance(hits, list) else []
            return f"{same[0]['name']}, {same[0]['mobile']}" if len(same) == 1 else None
        title_kw = next(t for t in (APPOINTMENTS + PLACE_EVENTS) if t in item.text.lower())
        match = [e for e in evs if title_kw in e.get("title", "").lower()]
        if len(match) != 1:
            return None
        return match[0]["start"] if k == "meeting_time" else match[0]["where"]
    if k == "code_latest":
        street = re.search(r"the (\w+) warehouse", item.text).group(1)
        msgs = [m for m in world.messages if f"{street} St warehouse" in m.body]
        msgs.sort(key=lambda m: (m.date, _minutes(m.time)), reverse=True)
        if not msgs or msgs[0].sender != item.person:
            return None
        m = re.search(r"(\d{4})#", msgs[0].body)
        return f"{m.group(1)}#" if m else None
    if k in ("invoice_unpaid", "invoice_via_contact"):
        vendor = item.extra.get("vendor") or next(c[0] for c in COMPANIES if c[0] in item.text)
        if k == "invoice_via_contact":
            hits = T.t_find_contacts(item.person)
            same = [h for h in hits if h["name"] == item.person] if isinstance(hits, list) else []
            if len(same) != 1 or same[0]["company"] != vendor:
                return None
        inv = T.t_invoices(vendor=vendor, status="unpaid")
        inv = [i for i in inv if i["vendor"] == vendor] if isinstance(inv, list) else []
        return inv[0]["amount"] if len(inv) == 1 else None
    if k == "flight_number":
        city = item.extra["city"]
        wd = next(w for w in WEEKDAYS if w in item.text)
        tr = T.t_trips(city)
        tr = [t for t in tr if t["date"].startswith(wd[:3])] if isinstance(tr, list) else []
        return tr[0]["number"] if len(tr) == 1 else None
    if k == "booking_ref":
        r = item.extra["restaurant"]
        msgs = [m for m in world.messages if m.sender == r]
        wd = next(w for w in WEEKDAYS if w in item.text)
        msgs = [m for m in msgs if f" {wd} " in m.body]
        if len(msgs) != 1:
            return None
        return re.search(r"Booking reference: (\w+)", msgs[0].body).group(1)
    if k == "text_late":
        res = T.t_send_text(item.person, "test")
        return f"text sent to {res['to']}" if res.get("status") == "sent" else None
    return None
