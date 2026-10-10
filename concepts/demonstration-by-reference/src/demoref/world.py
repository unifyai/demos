"""The fictional workplace a session is recorded in: tasks, people, mail, calendar, chat, a doc.

Everything is drawn from a seeded random.Random, so a seed fixes the world.
Task titles come in families of three near-duplicates ("Draft Q3 on-call rota",
"Review Q3 on-call rota", "Draft Q4 on-call rota"), so telling which one a frame
shows means reading it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

FAMILIES: list[tuple[str, list[str]]] = [
    ("OPS", ["Draft Q3 on-call rota", "Review Q3 on-call rota", "Draft Q4 on-call rota"]),
    ("EVT", ["Send offsite invitations", "Send offsite reminders", "Send board dinner invitations"]),
    ("PPL", ["Crop team headshots", "Retouch team headshots", "Crop speaker headshots"]),
    ("FIN", ["Reconcile August payroll", "Reconcile July payroll", "Approve August payroll"]),
    ("VEN", ["Update Northwind supplier record", "Update Contoso supplier record", "Verify Northwind supplier record"]),
    ("EVT", ["Compare venue quotes", "Compare catering quotes", "Book venue for offsite"]),
    ("PPL", ["Translate onboarding guide (DE)", "Translate onboarding guide (FR)", "Proofread onboarding guide (DE)"]),
    ("FIN", ["File September expense report", "File August expense report", "Approve September expense report"]),
    ("OPS", ["Renew Figma licences", "Renew Slack licences", "Audit Figma licences"]),
    ("FIN", ["Plan Q4 hiring budget", "Plan Q4 travel budget", "Review Q4 hiring budget"]),
    ("OPS", ["Migrate wiki to new space", "Archive old wiki pages", "Migrate handbook to new space"]),
    ("OPS", ["Prepare board deck appendix", "Prepare board deck summary", "Review board deck appendix"]),
    ("PPL", ["Set up laptop for Maya", "Set up laptop for Leo", "Order laptop for Maya"]),
]
SINGLES: list[tuple[str, str]] = [("OPS", "Clean up shared drive")]

PROJECTS = {"OPS": "Operations", "EVT": "Events", "PPL": "People", "FIN": "Finance", "VEN": "Vendors"}
PEOPLE = ["Priya Shah", "Tom Becker", "Ana Lima", "Sam Okafor", "Jun Park", "Maya Cohen",
          "Leo Rossi", "Ines Dubois", "Kofi Mensah", "Rita Novak"]
STATUSES = ["Todo", "Todo", "In progress", "In progress", "In review", "Blocked"]
PRIORITIES = ["Low", "Medium", "Medium", "High", "Urgent"]
LABELS = ["admin", "quarterly", "finance", "people", "events", "vendors", "tooling", "board", "onboarding"]

CHECKLIST_VERBS = ["Collect", "Check", "Confirm", "Draft", "Share", "Update", "Send", "Review", "Agree", "Book"]
CHECKLIST_NOUNS = ["numbers from last quarter", "the template", "dates with the team", "first version",
                   "with Priya", "the shared sheet", "notes to finance", "the list of names",
                   "open questions", "sign-off", "the folder permissions", "a reminder"]
COMMENTS = [
    "I've started on this, will finish by Thursday.",
    "Waiting on numbers from finance before I can go further.",
    "Can we push this a week? Not urgent.",
    "Draft is in the shared folder.",
    "Looks good to me, a couple of small edits.",
    "Who owns the sign-off on this one?",
    "Moved the due date, see thread.",
    "Done the first half, the rest needs the new template.",
    "Pinged the vendor again this morning.",
    "Added the checklist from last time.",
]
DESC = [
    "Pull together everything we need for {t}. Use last quarter's version as the starting point and flag anything that changed.",
    "This came up in the ops sync. {t} before the end of the month so the team isn't blocked.",
    "Owner to coordinate with the people listed below. Keep notes in the shared folder and link them here.",
    "Small task but it keeps slipping. {t} and close this out once it's signed off.",
    "We agreed to do this together with the finance team. Check the numbers twice before sending anything out.",
]

MAIL_SUBJECTS = [
    ("Priya Shah", "Re: Q3 on-call rota", "I swapped two of the weekend shifts, can you check?"),
    ("Tom Becker", "Offsite guest list", "Here's the latest list for the invitations, 42 people so far."),
    ("Ana Lima", "Headshots from Tuesday", "Uploaded the raw files to the drive, cropping still to do."),
    ("Payroll bot", "August payroll ready for review", "The August run is ready. 3 items need attention."),
    ("Northwind Traders", "Updated bank details", "Please update our supplier record with the new details."),
    ("Jun Park", "Venue quotes", "Got three quotes back, the cheapest is the old brewery."),
    ("Ines Dubois", "Onboarding guide FR", "Translation is about half done, questions inline."),
    ("Expenses", "Reminder: September expenses", "Your September expense report is due on Friday."),
    ("Figma", "Your plan renews in 14 days", "Your organisation plan will renew automatically."),
    ("Kofi Mensah", "Q4 budget", "First pass at the hiring budget is in the sheet."),
    ("Rita Novak", "Lunch on Friday?", "A few of us are going to the noodle place, want to come?"),
    ("IT Desk", "Laptop order confirmed", "Your order for 2 laptops has been confirmed."),
    ("Sam Okafor", "Board deck", "Appendix tables are updated, summary still needs a pass."),
    ("Calendar", "Updated invitation: Ops sync", "The Ops sync has moved to 11:00."),
    ("Newsletter", "This week in ops", "Five things we learned from moving our wiki."),
    ("Leo Rossi", "First week", "Thanks for the welcome! Is there anything I should read first?"),
]
EVENTS = ["Standup", "Ops sync", "1:1 Priya", "Board prep", "Offsite planning", "Payroll review",
          "Lunch", "Vendor call", "Hiring panel", "Focus time", "Team retro", "Interview"]
CHAT_LINES = [
    ("Priya Shah", "morning! rota draft is in the drive"),
    ("Tom Becker", "has anyone heard back from the venue?"),
    ("Jun Park", "the brewery can do the 14th, not the 21st"),
    ("Ana Lima", "headshots: I'll need another hour tomorrow"),
    ("Kofi Mensah", "budget sheet is open for comments"),
    ("Sam Okafor", "board deck v3 uploaded"),
    ("Rita Novak", "lunch anyone?"),
    ("Ines Dubois", "FR guide: who can review the glossary?"),
    ("Maya Cohen", "my laptop hasn't arrived yet :("),
    ("Leo Rossi", "thanks all for the welcome"),
    ("Priya Shah", "payroll numbers look off for July, checking"),
    ("Tom Becker", "invites go out Monday unless someone shouts"),
]
DOC_PARAS = [
    "Q4 planning notes. This page collects what we agreed in the planning session and what is still open.",
    "Priorities for the quarter: keep the on-call rota stable, get the offsite booked early, and close the books on time each month.",
    "Hiring: two roles in operations and one in finance. Budget to be agreed with Kofi before the board meeting.",
    "Vendors: Northwind and Contoso both changed their bank details this quarter. Records must be checked before any payment run.",
    "Tooling: Figma and Slack renewals are due in October. Audit seats first; we are paying for at least six inactive accounts.",
    "People: onboarding guide translations (DE, FR) are half done. New starters Maya and Leo need laptops set up before day one.",
    "Events: venue and catering quotes are in. The brewery is cheapest but only has the 14th free.",
    "Open questions: who signs off the board deck appendix, and whether the wiki migration happens before or after the offsite.",
]


@dataclass
class Task:
    id: str
    title: str
    project: str
    family: int  # index into FAMILIES, or -1 for a single
    assignee: str
    status: str
    due: str
    priority: str
    labels: list[str]
    description: str
    checklist: list[tuple[str, bool]]
    comments: list[tuple[str, str, str]]  # (author, when, text)


@dataclass
class World:
    seed: int
    tasks: list[Task]
    mail: list[tuple[str, str, str, str]]  # (sender, subject, snippet, time)
    events: list[tuple[int, int, int, str]]  # (day 0-4, start half-hour from 8:00, length in half-hours, title)
    chat: list[tuple[str, str, str]]  # (author, time, text)
    doc: list[str]
    by_id: dict[str, Task] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {t.id: t for t in self.tasks}

    def siblings(self, task_id: str) -> list[str]:
        t = self.by_id[task_id]
        if t.family < 0:
            return []
        return [u.id for u in self.tasks if u.family == t.family and u.id != task_id]

    def listing(self) -> str:
        """The task list as the agent's own data, in plain text."""
        rows = ["ID | Title | Project | Assignee | Status | Due"]
        for t in sorted(self.tasks, key=lambda t: t.id):
            rows.append(f"{t.id} | {t.title} | {PROJECTS[t.project]} | {t.assignee} | {t.status} | {t.due}")
        return "\n".join(rows)


def make_world(seed: int) -> World:
    rng = random.Random(f"world-{seed}")
    used: set[int] = set()

    def new_id(project: str) -> str:
        while True:
            n = rng.randint(100, 2999)
            if n not in used:
                used.add(n)
                return f"{project}-{n}"

    tasks: list[Task] = []
    entries = [(fi, p, title) for fi, (p, titles) in enumerate(FAMILIES) for title in titles]
    entries += [(-1, p, title) for p, title in SINGLES]
    for fi, project, title in entries:
        n_check = rng.randint(3, 6)
        done = rng.randint(0, n_check - 1)
        checklist = [(f"{rng.choice(CHECKLIST_VERBS)} {rng.choice(CHECKLIST_NOUNS)}", i < done) for i in range(n_check)]
        comments = [(rng.choice(PEOPLE), f"{rng.randint(1, 9)}d ago", rng.choice(COMMENTS)) for _ in range(rng.randint(1, 3))]
        tasks.append(Task(
            id=new_id(project), title=title, project=project, family=fi,
            assignee=rng.choice(PEOPLE), status=rng.choice(STATUSES),
            due=f"Oct {rng.randint(10, 31)}", priority=rng.choice(PRIORITIES),
            labels=rng.sample(LABELS, rng.randint(1, 2)),
            description=rng.choice(DESC).format(t=title), checklist=checklist, comments=comments,
        ))
    mail = [(s, subj, snip, f"{rng.randint(8, 17)}:{rng.choice(['00', '12', '25', '40', '47', '58'])}")
            for s, subj, snip in rng.sample(MAIL_SUBJECTS, len(MAIL_SUBJECTS))]
    events = []
    for day in range(5):
        slot = rng.randint(0, 2)
        while slot < 18:
            length = rng.choice([1, 1, 2, 2, 3])
            if rng.random() < 0.6:
                events.append((day, slot, length, rng.choice(EVENTS)))
            slot += length + rng.randint(0, 3)
    chat = []
    minute = rng.randint(0, 30)
    for author, text in rng.sample(CHAT_LINES, len(CHAT_LINES)):
        minute += rng.randint(1, 9)
        chat.append((author, f"{9 + minute // 60}:{minute % 60:02d}", text))
    doc = DOC_PARAS[:1] + rng.sample(DOC_PARAS[1:], len(DOC_PARAS) - 1)
    return World(seed=seed, tasks=tasks, mail=mail, events=events, chat=chat, doc=doc)
