"""Maya's records: three sources searched with BM25, and template filler records.

Every scenario's target and distractor records go into one world together
with filler records from templates (about 320), so a search returns the kind
of noise a real inbox does. Filler never contains any scenario's `avoid`
fragments (checked in tests), so it neither adds a candidate nor settles one.
Record ids follow the timestamps within each source, so an id says nothing
about which record is a target.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from datetime import datetime

from .scenarios import SCENARIOS

SOURCES = ("messages", "documents", "calendar")
PREFIX = {"messages": "m", "documents": "d", "calendar": "c"}

PEOPLE = [
    "Grace Liu", "Felix Wong", "Imogen Price", "Kwame Mensah", "Lucia Romano", "Hugo Laurent",
    "Nina Kowalski", "Arjun Mehta", "Zoe Turner", "Malik Hassan", "Freya Lund", "Oscar Bell",
    "Rosa Jimenez", "Ethan Clarke", "Yuki Tanaka", "Callum Reid", "Bea Santos", "Dmitri Volkov",
    "Kofi Asante", "Alice Moreau",
]
PROJECTS = ["Atlas migration", "Q3 roadmap", "pricing page", "onboarding flow", "data warehouse",
            "mobile release", "customer survey", "hiring plan", "brand refresh", "analytics dashboard",
            "partner portal", "API docs", "search revamp", "help centre"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["09:30", "10:00", "11:30", "12:15", "14:00", "15:30", "16:45"]
HEADLINES = ["interest rates held again", "a heatwave on the way", "the new river footbridge opens",
             "a record year for cycling", "the city's new bus routes", "a museum's late openings",
             "the summer reading challenge", "a rooftop farm in Peckham", "electric bikes on the rise"]
BOOKS = ["The Quiet Coast", "North of Nowhere", "A Year of Small Things", "The Lantern House", "Salt and Iron"]
PODCASTS = ["Deep Work Daily", "The Long Read Aloud", "History in Ten", "Small Business Big Ideas"]
DISHES = ["shakshuka", "mushroom risotto", "Thai green curry", "lemon pasta", "lentil soup", "fish tacos"]
FILMS = ["the new Spielberg", "that heist film", "the sailing documentary", "the space thriller"]
TOPICS = ["remote work", "product analytics", "city gardening", "four-day weeks", "user research", "AI in healthcare"]
STEPS = ["a short proposal by next week", "a follow-up call in two weeks", "sharing the draft with the team",
         "going through the numbers with finance", "a pilot with two teams"]

MESSAGE_TEMPLATES = [
    ("chat", "WhatsApp", "Lunch on {day}? I'm free from {time}."),
    ("sms", "SMS", "Running about {n} minutes late, sorry!"),
    ("email", "{project}", "Hi Maya,\nCould you look over the {project} notes before {day}? Mostly the open questions at the end.\nThanks,\n{first}"),
    ("email", "{project} kickoff moved", "Hi all,\nThe {project} kickoff has moved to {day} at {time}. Same link as before.\n{first}"),
    ("chat", "WhatsApp", "Thanks again for the help with the {project}, it made a real difference."),
    ("call", "Call notes", "Call with {first}: talked through the {project}; next step is {step}."),
    ("chat", "WhatsApp", "Did you see the game last night? What a finish."),
    ("email", "Notes from today", "Hi Maya,\nNotes from today's {project} sync are in the shared folder. Action items are at the top.\n{first}"),
    ("sms", "SMS", "Can we push our catch-up to {day}? Something's come up."),
    ("chat", "WhatsApp", "Here's that podcast I mentioned: {podcast}. Episode {n} is the best one."),
    ("email", "Quick question", "Hi Maya,\nDo you know who owns the {project} now? I need edit access to the tracker.\n{first}"),
    ("sms", "SMS", "Home safe, thanks for a lovely evening!"),
    ("chat", "WhatsApp", "Are you around for a quick call at {time}?"),
    ("email", "Re: {project}", "Agree with your comments. I'll update the doc and circulate it on {day}.\n{first}"),
    ("chat", "WhatsApp", "I'm making that {dish} recipe tonight, wish me luck."),
    ("sms", "SMS", "Thanks so much for lending me the umbrella, I'll bring it back next week."),
    ("email", "Offsite ideas", "A few ideas for the team offsite: a cooking class, a river walk or an escape room. Thoughts?\n{first}"),
    ("call", "Call notes", "Call with {first}: catch-up; they're settling into the new role and enjoying it."),
    ("chat", "WhatsApp", "Happy Friday! Any plans for the weekend?"),
    ("chat", "WhatsApp", "Can you share the link to that article about {topic}?"),
    ("email", "Intro: {other}", "Hi Maya,\nIntroducing you to {other}, who's working on {topic}. I think you'd get on.\n{first}"),
    ("email", "{project}: first draft", "Hi Maya,\nThe first draft is in the folder. Comments welcome by {day}.\n{first}"),
    ("chat", "WhatsApp", "That trailer for {film} looks amazing, we should go."),
    ("chat", "WhatsApp", "Have you read {book}? Just finished it, couldn't put it down."),
    ("call", "Call notes", "Call with {first} about {topic}: lots of overlap with the {project}; agreed to swap notes."),
    ("email", "Stand-up moved", "Hi team,\nTomorrow's stand-up is at {time} instead, just this once.\n{first}"),
]
NEWSLETTERS = [
    ("The Weekly Brief", "This week", "This week: {h1}; {h2}; and a look at {h3}."),
    ("Riverside Library", "Your reservation", "The book you reserved, {book}, is waiting for you at Riverside Library until {day}."),
    ("Greenline Energy", "Your energy statement", "Your latest energy statement is ready to view online."),
    ("CityCycle", "Your membership", "Your CityCycle membership renews next month. Thanks for riding with us."),
    ("Northwind Bank", "Your statement", "Your monthly statement is now available in online banking."),
    ("Saturday 5k", "This Saturday", "This Saturday's 5k starts at 9am by the boathouse. Bring water!"),
]
DOCUMENT_TEMPLATES = [
    ("{project} - one-pager", "Problem, goals and open questions for the {project}. Owner: Maya Brooks. Next review in two weeks."),
    ("Notes - {project} sync", "Attendees: Maya, {first}. Decisions: keep the current scope; revisit metrics next sprint. Actions: {first} to share the tracker."),
    ("Reading list 2026", "{book}; {book2}; anything by the author of {book3}."),
    ("Running plan - 10k", "Three runs a week: one easy, one tempo, one long. Build to 12 km by the end of the month."),
    ("Spanish vocab - week {n}", "Phrases for asking for coffee, asking directions and small talk. Review on {day}."),
    ("Recipes to try", "{dish}; {dish2}; a proper sourdough."),
    ("Team offsite - agenda draft", "Morning: retro and goals. Afternoon: a river walk. Owner: {first}."),
    ("Interview questions - product manager", "Tell me about a launch that went wrong. How do you decide what not to build?"),
    ("Podcast notes - {podcast}", "Episode {n}: good points on focus and saying no. Try the two-list method."),
    ("{project} - retro notes", "Went well: shipping cadence. To improve: earlier user testing. Owner: {first}."),
    ("OKRs Q3 - draft", "Objective: make the {project} the default. Key results: adoption, satisfaction, time saved."),
    ("Garden plan", "Tomatoes along the fence, herbs by the back door, and more lavender."),
]
EVENT_TEMPLATES = [
    ("1:1 with {first}", "Video call", 30),
    ("{project} stand-up", "Video call", 15),
    ("Yoga", "Yoga Loft, Old Street", 60),
    ("Lunch with {first}", "Canteen", 60),
    ("Coffee with {first}", "Bean There, Exmouth Market", 30),
    ("Focus time", "", 120),
    ("Team retro", "Video call", 45),
    ("Haircut", "Cutting Room, Upper Street", 45),
    ("Spanish class", "Instituto Cervantes", 90),
    ("{project} demo", "Video call", 30),
    ("Quarterly planning", "Video call", 90),
    ("All-hands", "Main office", 60),
    ("Call with {first} about {topic}", "Phone", 30),
]


def _choices(rng: random.Random) -> dict:
    person = rng.choice(PEOPLE)
    other = rng.choice([p for p in PEOPLE if p != person])
    books = rng.sample(BOOKS, 3)
    dishes = rng.sample(DISHES, 2)
    h = rng.sample(HEADLINES, 3)
    return dict(
        day=rng.choice(DAYS), time=rng.choice(TIMES), n=rng.randint(2, 15), project=rng.choice(PROJECTS),
        first=person.split()[0], person=person, other=other, step=rng.choice(STEPS), podcast=rng.choice(PODCASTS),
        dish=dishes[0], dish2=dishes[1], film=rng.choice(FILMS), book=books[0], book2=books[1], book3=books[2],
        topic=rng.choice(TOPICS), h1=h[0], h2=h[1], h3=h[2],
    )


def filler_records(seed: int = 0, n_messages: int = 220, n_documents: int = 40, n_events: int = 60) -> list[dict]:
    """Template filler for the three sources, dated in the six weeks before today (events: June)."""
    rng = random.Random(f"filler-records-{seed}")
    out: list[dict] = []
    for _ in range(n_messages):
        c = _choices(rng)
        if rng.random() < 0.15:
            sender, subject, body = rng.choice(NEWSLETTERS)
            kind = "email"
        else:
            kind, subject, body = rng.choice(MESSAGE_TEMPLATES)
            sender = c["person"]
        date = _add_days(datetime(2026, 4, 28, rng.randint(7, 21), rng.choice([0, 5, 15, 20, 30, 40, 45, 50])), rng.randint(0, 41))
        out.append({"source": "messages", "kind": kind, "date": date, "sender": sender,
                    "to": "Maya Brooks", "subject": subject.format(**c), "body": body.format(**c)})
    for _ in range(n_documents):
        c = _choices(rng)
        title, body = rng.choice(DOCUMENT_TEMPLATES)
        date = _add_days(datetime(2026, 3, 2, rng.randint(9, 18), 0), rng.randint(0, 98))
        out.append({"source": "documents", "kind": "document", "date": date,
                    "sender": rng.choice(["Maya Brooks", c["person"]]), "to": "",
                    "subject": title.format(**c), "body": body.format(**c)})
    for _ in range(n_events):
        c = _choices(rng)
        title, location, minutes = rng.choice(EVENT_TEMPLATES)
        start = _add_days(datetime(2026, 6, 1, rng.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), rng.choice([0, 15, 30, 45])), rng.randint(0, 29))
        name = title.format(**c)
        out.append({"source": "calendar", "kind": "event", "date": start, "end": _add_minutes(start, minutes),
                    "subject": name, "location": location, "sender": "",
                    "to": "Maya Brooks" + (f", {c['person']}" if "{first}" in title else ""),
                    "body": f"{name}." + (f" Where: {location}." if location else "")})
    # No filler event may show a calendar scenario's answer time (it would muddle grading).
    times = [p for sc in SCENARIOS if sc["source"] == "calendar" for p in sc["target_value"] + sc["distractor_value"]]
    return [r for r in out if r["source"] != "calendar"
            or not any(re.search(p, render({**r, "id": "c"}), re.I) for p in times)]


def _add_days(d: datetime, days: int) -> str:
    from datetime import timedelta
    return (d + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M")


def _add_minutes(start: str, minutes: int) -> str:
    from datetime import timedelta
    return (datetime.fromisoformat(start) + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M")


def scenario_records() -> list[dict]:
    out = []
    for s in SCENARIOS:
        for role in ("target", "distractor"):
            r = dict(s[role])
            r.update(source=s["source"], scenario=s["id"], role=role)
            out.append(r)
    return out


def build_world(seed: int = 0) -> dict[str, list[dict]]:
    """All records by source, ids assigned in time order."""
    recs = scenario_records() + filler_records(seed)
    world: dict[str, list[dict]] = {s: [] for s in SOURCES}
    for r in sorted(recs, key=lambda r: (r["date"], r["subject"], r["body"])):
        world[r["source"]].append(r)
    for src, rs in world.items():
        for i, r in enumerate(rs, 1):
            r["id"] = f"{PREFIX[src]}-{i:04d}"
    return world


# ---------------------------------------------------------------- search

STOP = set("""a an and are as at be by can could did do does for from had has have her hers his how i i'm in is it
its me my of on or our please she so that the their them there these they this to up was we were what when where
which who why will with would you your maya maya's brooks find look looking need needs""".split())


def tokens(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def describe_date(iso: str) -> str:
    d = datetime.fromisoformat(iso)
    return d.strftime("%a %-d %b %Y %H:%M")


def render(r: dict) -> str:
    when = describe_date(r["date"])
    if r["kind"] == "event":
        end = datetime.fromisoformat(r["end"]).strftime("%H:%M")
        loc = f" | {r['location']}" if r.get("location") else ""
        return f"[{r['id']}] {when}-{end} | event | {r['subject']}{loc} | with {r['to']}"
    if r["kind"] == "document":
        return f"[{r['id']}] {when} | document | \"{r['subject']}\" by {r['sender']}\n{r['body']}"
    head = f"[{r['id']}] {when} | {r['kind']} | from {r['sender']} to {r['to']}"
    if r["kind"] == "email":
        head += f" | subject: {r['subject']}"
    return f"{head}\n{r['body']}"


def _index_text(r: dict) -> str:
    d = datetime.fromisoformat(r["date"])
    return " ".join([r.get("subject", ""), r.get("body", ""), r.get("sender", ""), r.get("to", ""),
                     r.get("location", ""), r["kind"], d.strftime("%A %B %-d"), str(d.day)])


class Store:
    """BM25 over one source (k1 1.2, b 0.75). search() returns up to k rendered records."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.docs = [Counter(tokens(_index_text(r))) for r in records]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(records)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[tuple[float, int]]:
        q = set(tokens(query))
        out = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * 2.2 / (f + 1.2 * (1 - 0.75 + 0.75 * self.lens[i] / self.avg))
            if s > 0:
                out.append((s, i))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def search(self, query: str, k: int = 5) -> str:
        hits = self.scores(query)[:k]
        if not hits:
            return "No matching records."
        return "\n\n".join(render(self.records[i]) for _, i in hits)


class World:
    def __init__(self, seed: int = 0):
        self.records = build_world(seed)
        self.stores = {src: Store(rs) for src, rs in self.records.items()}

    def search(self, source: str, query: str) -> str:
        return self.stores[source].search(query)
