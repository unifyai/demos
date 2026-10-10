"""Scripted calls: what Sam says, and when.

A script is a greeting, five episodes and a goodbye. Each of Sam's turns has a
trigger that says when Sam speaks: after any agent speech, after a
substantive agent line, after the answers to named questions have been heard,
at a fixed time after Sam's previous turn, or cutting into the agent's answer.
The words never depend on the arm; only the timing does.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

from .world import (LOOKUP_KINDS_EASY, LOOKUP_KINDS_HARD, Item, Planted, Planter, World, _Gen,
                    fill_world, solve)


@dataclass
class Trigger:
    mode: str  # at | after_any | after_substance | after_answer | barge_answer
    gap: float = 0.6  # seconds of agent silence before Sam speaks
    patience: float = 6.0  # seconds after Sam's previous turn ended before Sam speaks anyway
    delay: float = 0.0  # 'at': after Sam's previous turn; 'barge_answer': into the answer
    items: list[str] = field(default_factory=list)
    # after_answer only: what must have happened since Sam's previous turn.
    # "none": nothing (the answers may have come before it); "any": some agent speech;
    # "answer": the answers heard again (Sam asked to hear them again).
    since_prev: str = "none"


@dataclass
class Turn:
    id: str
    text: str
    kind: str  # greeting question aside topic_change withdraw interrupt_repeat interrupt_resume repeat_request ack action closing
    trigger: Trigger
    expects_reply: bool
    episode: str
    asks: list[str] = field(default_factory=list)
    withdraws: list[str] = field(default_factory=list)


@dataclass
class Script:
    id: str
    seed: int
    episodes: list[str]
    turns: list[Turn]
    items: list[Item]
    planted: Planted
    used_names: list[str]
    used_mobiles: list[str]

    def item(self, iid: str) -> Item:
        return next(i for i in self.items if i.id == iid)

    def world(self, size: str) -> World:
        return fill_world(self.seed, size, self.planted, set(self.used_names), set(self.used_mobiles))

    def to_json(self) -> dict:
        return {
            "id": self.id, "seed": self.seed, "episodes": self.episodes,
            "turns": [asdict(t) for t in self.turns],
            "items": [asdict(i) for i in self.items],
            "planted": _planted_json(self.planted),
            "used_names": self.used_names, "used_mobiles": self.used_mobiles,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Script":
        from .world import Contact, Event, Invoice, Message, Trip
        p = d["planted"]
        planted = Planted(
            contacts=[Contact(**c) for c in p["contacts"]], events=[Event(**e) for e in p["events"]],
            messages=[Message(**m) for m in p["messages"]], invoices=[Invoice(**i) for i in p["invoices"]],
            trips=[Trip(**t) for t in p["trips"]],
            reserved_slots={tuple(x) for x in p["reserved_slots"]}, reserved_titles={tuple(x) for x in p["reserved_titles"]},
            reserved_vendors=set(p["reserved_vendors"]), reserved_restaurants=set(p["reserved_restaurants"]),
            reserved_streets=set(p["reserved_streets"]), reserved_trip_days={tuple(x) for x in p["reserved_trip_days"]},
        )
        turns = [Turn(**{**t, "trigger": Trigger(**t["trigger"])}) for t in d["turns"]]
        items = [Item(**i) for i in d["items"]]
        return cls(d["id"], d["seed"], d["episodes"], turns, items, planted, d["used_names"], d["used_mobiles"])


def _planted_json(p: Planted) -> dict:
    d = asdict(p)
    for k, v in d.items():
        if isinstance(v, set):
            d[k] = sorted((list(x) if isinstance(x, tuple) else x for x in v), key=str)
    return d


GREETINGS = ["Hi Ava, it's Sam. How are you doing today?", "Morning, Ava! How's it going?",
             "Hey Ava, Sam here. How are things?", "Hi Ava. Good morning, how are you?"]
ACKS = ["Great, thanks.", "Perfect, thank you.", "Got it, thanks.", "Okay, thanks.", "Lovely, thanks."]
ASIDES = ["Sorry, I'm just getting into the car, give me a second.", "Hang on, my coffee's just arrived.",
          "Ugh, the traffic this morning is terrible.", "Sorry, I'm just walking into the office."]
WITHDRAWS = ["Actually, never mind, I've just found it in my notes.", "Oh wait, don't worry about that, I've got it.",
             "Actually, forget that one, it doesn't matter now."]
INTERRUPT_REPEAT = ["Sorry, you cut out there. Can you say that again?", "Sorry, sorry, say that again?"]
INTERRUPT_RESUME = ["Hang on, sorry, someone's at the door. Okay, go on.", "Sorry, one sec. Okay, carry on."]
CLOSINGS = ["Okay, that's everything. Thanks, Ava, bye!", "Right, that's all for now. Thanks, bye."]

EPISODES = ["simple", "smalltalk_during", "topic_change", "withdraw", "interrupt", "repeat_request",
            "followup_ref", "multihop", "action"]
PERSON_KINDS = ["phone", "phone_ambig", "meeting_who_phone"]
MULTIHOP_KINDS = ["meeting_who_phone", "invoice_via_contact"]
LOOKUP_KINDS = LOOKUP_KINDS_EASY + LOOKUP_KINDS_HARD

ACK_AFTER = dict(gap=0.5, patience=25.0)


class _Builder:
    def __init__(self, sid: str, seed: int):
        self.sid, self.seed = sid, seed
        self.g = _Gen(seed)
        self.p = Planted()
        self.pl = Planter(self.g, self.p)
        self.rng = self.g.rng
        self.turns: list[Turn] = []
        self.items: list[Item] = []

    def _t(self, text, kind, trigger, expects, ep, asks=(), withdraws=()):
        t = Turn(f"t{len(self.turns) + 1}", text, kind, trigger, expects, ep, list(asks), list(withdraws))
        self.turns.append(t)
        return t

    def _item(self, kind: str) -> Item:
        it = self.pl.make(kind)
        self.items.append(it)
        return it

    def _opener(self) -> Trigger:
        """Trigger for the first turn of an episode, from the turn before it."""
        prev = self.turns[-1]
        if prev.kind == "greeting":
            return Trigger("after_substance", gap=0.6, patience=6.0)
        return Trigger("after_any", gap=0.8, patience=3.0)

    def greeting(self):
        self._t(self.rng.choice(GREETINGS), "greeting", Trigger("at", delay=0.5), True, "greeting")

    def closing(self):
        self._t(self.rng.choice(CLOSINGS), "closing", self._opener(), True, "closing")

    def ask(self, item: Item, ep: str, trigger: Trigger | None = None, kind: str = "question"):
        return self._t(item.text, kind, trigger or self._opener(), True, ep, asks=[item.id])

    def ack(self, ep: str, items: list[str], trigger: Trigger | None = None):
        self._t(self.rng.choice(ACKS), "ack", trigger or Trigger("after_answer", items=items, **ACK_AFTER), False, ep)

    # ---- episodes
    def ep_simple(self, kinds):
        x = self._item(kinds[0]); self.ask(x, "simple"); self.ack("simple", [x.id])

    def ep_smalltalk_during(self, kinds):
        x = self._item(kinds[0]); self.ask(x, "smalltalk_during")
        self._t(self.rng.choice(ASIDES), "aside", Trigger("after_any", gap=0.6, patience=3.0), False, "smalltalk_during")
        self.ack("smalltalk_during", [x.id])

    def ep_topic_change(self, kinds):
        x = self._item(kinds[0]); y = self._item(kinds[1])
        self.ask(x, "topic_change")
        text = "Oh, and while you're at it, " + y.text[0].lower() + y.text[1:]
        self._t(text, "topic_change", Trigger("after_any", gap=0.6, patience=3.0), True, "topic_change", asks=[y.id])
        self.ack("topic_change", [x.id, y.id])

    def ep_withdraw(self, kinds):
        x = self._item(kinds[0]); self.ask(x, "withdraw")
        self._t(self.rng.choice(WITHDRAWS), "withdraw", Trigger("after_any", gap=0.6, patience=3.0), True, "withdraw",
                withdraws=[x.id])
        x.extra["withdrawn"] = True

    def ep_interrupt(self, kinds):
        x = self._item(kinds[0]); self.ask(x, "interrupt")
        resume = self.rng.random() < 0.5
        text = self.rng.choice(INTERRUPT_RESUME if resume else INTERRUPT_REPEAT)
        self._t(text, "interrupt_resume" if resume else "interrupt_repeat",
                Trigger("barge_answer", delay=1.2, patience=25.0, items=[x.id]), True, "interrupt")
        self.ack("interrupt", [x.id], Trigger("after_answer", items=[x.id], since_prev="any" if resume else "answer", **ACK_AFTER))

    def ep_repeat_request(self, kinds):
        x = self._item(kinds[0]); self.ask(x, "repeat_request"); self.ack("repeat_request", [x.id])
        r = Item(f"{x.id}r", "repeat", f"Sorry, what was that {x.thing} again?", x.gold, list(x.keys), x.key_type,
                 x.difficulty, person=x.person, thing=x.thing, extra={"repeats": x.id})
        self.items.append(r)
        self._t(r.text, "repeat_request", Trigger("after_any", gap=0.8, patience=2.5), True, "repeat_request", asks=[r.id])
        self.ack("repeat_request", [r.id])

    def ep_followup_ref(self, kinds):
        x = self._item(kinds[0]); self.ask(x, "followup_ref")
        y = self.pl.email_followup(x); self.items.append(y)
        self._t(y.text, "question", Trigger("after_answer", items=[x.id], **ACK_AFTER), True, "followup_ref", asks=[y.id])
        self.ack("followup_ref", [y.id])

    def ep_multihop(self, kinds):
        self.ep_simple(kinds)
        self.turns[-2].episode = self.turns[-1].episode = "multihop"

    def ep_action(self, kinds):
        a = self.pl.text_late(); self.items.append(a)
        self._t(a.text, "action", self._opener(), True, "action", asks=[a.id])
        self.ack("action", [a.id], Trigger("after_substance", gap=0.5, patience=15.0))


def _assign(n_scripts: int, rng: random.Random) -> list[list[tuple[str, list[str]]]]:
    """Balanced episodes and question kinds across scripts: each script gets five
    distinct episode kinds, and question kinds are dealt in turn."""
    eps: list[str] = []
    while len(eps) < n_scripts * 5 + 20:
        perm = EPISODES[:]
        rng.shuffle(perm)
        eps += perm
    deck: list[str] = []

    def deal(allowed: list[str], avoid: set[str]) -> str:
        nonlocal deck
        for _ in range(3):
            for i, k in enumerate(deck):
                if k in allowed and k not in avoid:
                    return deck.pop(i)
            perm = LOOKUP_KINDS[:]
            rng.shuffle(perm)
            deck += perm
        choices = [k for k in allowed if k not in avoid] or allowed
        return rng.choice(choices)

    out = []
    pos = 0
    for _ in range(n_scripts):
        chosen: list[str] = []
        while len(chosen) < 5:
            e = eps[pos]
            pos += 1
            if e not in chosen:
                chosen.append(e)
        plan, used = [], set()
        for e in chosen:
            if e == "action":
                plan.append((e, []))
                continue
            allowed = PERSON_KINDS if e == "followup_ref" else MULTIHOP_KINDS if e == "multihop" else LOOKUP_KINDS
            k1 = deal(allowed, used)
            used.add(k1)
            ks = [k1]
            if e == "topic_change":
                k2 = deal(LOOKUP_KINDS, used)
                used.add(k2)
                ks.append(k2)
            plan.append((e, ks))
        out.append(plan)
    return out


def build_script(sid: str, seed: int, plan: list[tuple[str, list[str]]]) -> Script:
    b = _Builder(sid, seed)
    b.greeting()
    for ep, kinds in plan:
        getattr(b, "ep_" + ep)(kinds)
    b.closing()
    return Script(sid, seed, [e for e, _ in plan], b.turns, b.items, b.p, sorted(b.g.used_names), sorted(b.g.used_mobiles))


def build_sanity(sid: str, seed: int) -> Script:
    b = _Builder(sid, seed)
    b.greeting()
    for kind in ["phone", "meeting_time"]:
        x = b._item(kind)
        b.ask(x, "simple")
        b.ack("simple", [x.id], Trigger("after_answer", items=[x.id], gap=0.5, patience=40.0))
    b.closing()
    return Script(sid, seed, ["simple", "simple"], b.turns, b.items, b.p, sorted(b.g.used_names), sorted(b.g.used_mobiles))


def generate(base_seed: int, n: int) -> list[Script]:
    rng = random.Random(base_seed)
    plans = _assign(n, rng)
    return [build_script(f"s{base_seed}-{k:02d}", base_seed * 1000 + k, plans[k]) for k in range(n)]


def check(script: Script, sizes: list[str]) -> list[str]:
    """Problems with a script: a planted answer the tools cannot reach unambiguously."""
    problems = []
    for size in sizes:
        w = script.world(size)
        for it in script.items:
            if it.kind == "repeat":
                continue
            got = solve(it, w)
            if got != it.gold:
                problems.append(f"{script.id} {size} {it.id} {it.kind}: solver got {got!r}, gold {it.gold!r}")
    return problems


def scripts_hash(scripts: list[Script]) -> str:
    return hashlib.sha256(json.dumps([s.to_json() for s in scripts], sort_keys=True, default=list).encode()).hexdigest()[:16]
