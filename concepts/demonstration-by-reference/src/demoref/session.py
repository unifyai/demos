"""Generate a screen-share session: frames, what was said on each, and the ground truth.

A person shares their screen and, while doing other things, shows which tasks
they want in next week's queue and in what order ("first up, this one", "then
this", "and finally this"). The speech never names a task; which task is meant
is only visible in the frame on screen at the time (or, when the words come a
moment before the click, the frame after it).

Each item fixes a world and a *key script*: the tasks to queue, how each is
shown (its detail page, a pointer on its row in the list, or words that land a
frame before the page opens), and the speech that must be understood rather
than matched on a keyword: "no, not this one", "this one's already done",
"this can wait until the week after", a revisit of a queued task, and in some
items a later correction ("actually, scratch the second one, do this one
instead"). The key script is the same at every session length; only the
ordinary activity between its beats (scrolling, opening other tasks, searching,
mail, calendar, chat, a doc, with ordinary talk, much of it saying "this")
grows with the length. Frames are numbered in order, and a frame's annotation
is what was said while it was on screen.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .render import LIST_ROWS, ROW_H, list_tasks, visible_rows
from .world import FAMILIES, PROJECTS, World, make_world

ORDINALS = ["first", "second", "third", "fourth", "fifth"]
KEY_SAID = {
    "first": ["ok so first up next week is this one", "this one first", "let's start with this one",
              "so first, this", "the first one is this"],
    "middle": ["then this one", "after that, this", "next, this one here", "and then this", "then this one after it"],
    "last": ["and finally this one", "and last, this", "and to finish, this one", "and the last one is this"],
}
LAG_LEAD = {"first": ["so first up is", "ok so first", "let's start with"],
            "middle": ["and then", "after that", "next up is"],
            "last": ["and finally", "and last", "and to finish"]}
# Said on a wrong page while looking for the next task: about the screen now, not an earlier pick.
NOT_THIS = ["hmm, this isn't the one I'm looking for", "no, this isn't the one I'm after",
            "not this one, I'm looking for the other one", "wait, this isn't it, where's the next one"]
DONE = ["oh, and this one's already done, so ignore it", "this one we already finished, ignore it",
        "oh this one's done already, you can ignore that"]
LATER = ["this one can wait until the week after, not next week", "this one's for the week after, not next week",
         "this one can wait, it's not for next week"]
REVISIT = ["back to this one for a sec, see, it's due {due}", "and this one again, just so you can see the checklist",
           "oh and this one, see the comment from {who}"]
CORRECTION = ["actually, scratch the {ord} one, do this one instead", "oh wait, change of plan, instead of the {ord} one, this one",
              "actually no, swap the {ord} one for this one"]
INTRO = ["ok so I'll show you what I want done next week, in order", "right, let me show you what's on for next week"]
OUTRO = ["ok, I think that's everything", "right, that's it", "ok that's all of them"]

# Filler never affirms anything ("yeah" over an open task reads as picking it).
FILLER = ["um", "ok", "let me find it", "where was it", "hang on", "one sec", "right", "so", "hmm",
          "sorry, it's slow today", "let me scroll down", "bear with me", "where's it gone", "uh, where",
          "one moment", "it was here somewhere", "erm", "can you see my screen ok?",
          "sorry, my wifi is a bit slow"]
# Talk during ordinary activity. Much of it says "this", "first" or "then", but none of it picks a task
# for the queue: it is navigation, a question, or a task set aside.
NOISE_TRACKER = ["let me just check something on this one", "hmm, not sure who owns this one",
                 "this list is a mess", "where's the one I want", "these are all the finance ones",
                 "I did this one first thing last week", "first let me find it", "then I'll show you the rest",
                 "who made this one?", "oh, this has moved since yesterday", "the search is so slow",
                 "this view always resets", "I can never find anything in here", "this one has a long checklist"]
NOISE_APP = {
    "mail": ["sorry, that's just my inbox", "this is the email about it", "ignore that", "this one's from Priya",
             "let me just reply to this later", "this is the one about the venue"],
    "calendar": ["this is next week", "let me check when the offsite is", "so this is the week", "this is my calendar"],
    "chat": ["ignore that", "sorry, just chat", "this is the ops channel", "people are asking about this"],
    "doc": ["this is the planning doc", "this is what we agreed", "so this is the doc from last month"],
}
SEARCH_WORDS = ["rota", "payroll", "headshots", "invitations", "supplier", "quotes", "onboarding", "expense",
                "licences", "budget", "wiki", "board deck", "laptop"]
REQUEST = ("Please set next week's queue to the tasks I picked out for next week while I was sharing my screen, "
           "in the order I gave them. Replace whatever is in the queue now.")


@dataclass
class Frame:
    id: str
    t: float  # seconds from the start of the share
    said: str
    spec: dict
    role: str = "noise"  # intro, key, key-next, not-this, done, later, revisit, correction, outro, noise
    task: str | None = None  # the task this frame shows (open or under the pointer), if any

    @property
    def stamp(self) -> str:
        s = int(self.t)
        return f"{s // 60:02d}:{s % 60:02d}"


@dataclass
class Session:
    item: int
    size: int
    frames: list[Frame]
    answer: list[str]
    evidence: list[str]  # frame ids that show the queued tasks at the moments they were named
    request: str = REQUEST
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"item": self.item, "size": self.size, "answer": self.answer, "evidence": self.evidence,
                "request": self.request, "meta": self.meta,
                "frames": [asdict(f) for f in self.frames]}

    @staticmethod
    def from_json(d: dict) -> "Session":
        return Session(item=d["item"], size=d["size"], answer=d["answer"], evidence=d["evidence"],
                       request=d["request"], meta=d["meta"], frames=[Frame(**f) for f in d["frames"]])


# ----------------------------------------------------------------- beats
# A beat is a list of (spec, said, role, task) tuples, one per frame.

def _pointer_on_row(world: World, spec: dict, task_id: str, rng: random.Random) -> dict:
    """Scroll a list spec so task_id is visible, and put the pointer on its title."""
    tasks = [t.id for t in list_tasks(world, spec)]
    i = tasks.index(task_id)
    top = max(0, min(i - rng.randint(2, LIST_ROWS - 3), max(0, len(tasks) - LIST_ROWS)))
    spec = dict(spec, scroll=top)
    y = dict(visible_rows(world, spec))[task_id]
    # the tip high enough in the row that the whole arrow (20 px) stays inside it
    return dict(spec, hover=task_id, cursor=[rng.randint(380, 600), y + rng.randint(4, ROW_H - 23)])


def _list_spec(rng: random.Random, sidebar: str = "All tasks") -> dict:
    return {"app": "tracker", "view": "list", "sidebar": sidebar, "scroll": rng.choice([0, 0, 8, 17, 23]), "hover": None,
            "cursor": [rng.randint(240, 1260), rng.randint(90, 160)]}


def _detail(task_id: str, rng: random.Random, scroll: int = 0) -> dict:
    return {"app": "tracker", "view": "detail", "task": task_id, "scroll": scroll,
            "cursor": [rng.randint(300, 1250), rng.randint(130, 780)]}


NAV_SAID = ["", "", "", "um", "let me find it", "where is it", "one sec"]


def _nav(world: World, task_id: str, rng: random.Random) -> tuple:
    """The list on its way to a task: the row is in view, the pointer is not on any row yet."""
    spec = _pointer_on_row(world, _list_spec(rng), task_id, rng)
    spec = dict(spec, hover=None, cursor=[rng.randint(250, 1250), rng.randint(84, 158)])
    return (spec, rng.choice(NAV_SAID), "noise", None)


def key_beat(world: World, task_id: str, slot: str, style: str, rng: random.Random) -> list[tuple]:
    """A task named for the queue. Every beat starts by moving to it, so beats never run together."""
    if style == "detail":
        click = _pointer_on_row(world, _list_spec(rng), task_id, rng)
        click = dict(click, selected=task_id, hover=None)
        return [(click, rng.choice(NAV_SAID), "noise", task_id),
                (_detail(task_id, rng), rng.choice(KEY_SAID[slot]), "key", task_id)]
    if style == "hover":
        point = _pointer_on_row(world, _list_spec(rng), task_id, rng)
        frames = [_nav(world, task_id, rng), (point, rng.choice(KEY_SAID[slot]), "key", task_id)]
        frames.append(_settle(world, point, task_id, rng))
        return frames
    # lag: the words start while the pointer is still on its way; the page opens on the next frame
    spec, _, _, _ = _nav(world, task_id, rng)
    return [(spec, rng.choice(LAG_LEAD[slot]) + "…", "key", None),
            (_detail(task_id, rng), rng.choice(["this one", "…this one", "yeah, this"]), "key-next", task_id)]


def _settle(world: World, point: dict, task_id: str, rng: random.Random) -> tuple:
    """After pointing at a row, the pointer stays on it or the task opens: no other task appears
    on the very next frame, so the words cannot be read as being about something else."""
    if rng.random() < 0.5:
        return (_detail(task_id, rng), "", "noise", task_id)
    return (dict(point, cursor=[point["cursor"][0] + rng.randint(-30, 30), point["cursor"][1]]), "", "noise", task_id)


def shown_beat(world: World, task_id: str, said: str, role: str, rng: random.Random) -> list[tuple]:
    """A task shown with words about it that do not put it in the queue (not-this, done, later, revisit),
    or that swap it in (correction). Like a key beat, it starts by moving to the task."""
    if role == "correction" or rng.random() < 0.6:
        click = dict(_pointer_on_row(world, _list_spec(rng), task_id, rng), selected=task_id, hover=None)
        return [(click, rng.choice(NAV_SAID), "noise", task_id), (_detail(task_id, rng), said, role, task_id)]
    point = _pointer_on_row(world, _list_spec(rng), task_id, rng)
    return [_nav(world, task_id, rng), (point, said, role, task_id), _settle(world, point, task_id, rng)]


def noise_beat(world: World, rng: random.Random) -> list[tuple]:
    kind = rng.choices(["browse", "peek", "search", "mail", "calendar", "chat", "doc"], [5, 5, 2, 3, 1.5, 1.5, 1])[0]
    talk = lambda pool: rng.choice(pool) if rng.random() < 0.45 else (rng.choice(FILLER) if rng.random() < 0.4 else "")
    out: list[tuple] = []
    if kind == "browse":
        side = rng.choice(["All tasks"] * 3 + list(PROJECTS))
        spec = _list_spec(rng, side)
        for _ in range(rng.randint(2, 5)):
            tasks = [t.id for t in list_tasks(world, spec)]
            spec = dict(spec, scroll=rng.randint(0, max(0, len(tasks) - LIST_ROWS)), hover=None,
                        cursor=[rng.randint(240, 1260), rng.randint(90, 790)])
            if rng.random() < 0.5:
                spec = _pointer_on_row(world, spec, rng.choice(tasks), rng)
            out.append((spec, talk(NOISE_TRACKER), "noise", spec.get("hover")))
    elif kind == "peek":
        tid = rng.choice(world.tasks).id
        out.append((_detail(tid, rng), talk(NOISE_TRACKER), "noise", tid))
        if rng.random() < 0.5:
            out.append((_detail(tid, rng, scroll=1), talk(FILLER), "noise", tid))
    elif kind == "search":
        word = rng.choice(SEARCH_WORDS)
        for k in sorted(rng.sample(range(1, len(word) + 1), min(len(word), rng.randint(2, 3)))):
            out.append(({"app": "tracker", "view": "list", "search": word[:k], "typing": True, "sidebar": "All tasks",
                         "cursor": [rng.randint(900, 1150), rng.randint(96, 112)]}, talk(FILLER), "noise", None))
        spec = {"app": "tracker", "view": "list", "search": word, "sidebar": "All tasks", "scroll": 0, "hover": None,
                "cursor": [rng.randint(300, 1200), rng.randint(200, 600)]}
        hits = [t.id for t in list_tasks(world, spec)]
        if hits and rng.random() < 0.6:
            spec = _pointer_on_row(world, spec, rng.choice(hits), rng)
        out.append((spec, talk(NOISE_TRACKER), "noise", spec.get("hover")))
    elif kind == "mail":
        for _ in range(rng.randint(1, 4)):
            out.append(({"app": "mail", "open": rng.randrange(len(world.mail)), "scroll": rng.choice([0, 0, 3, 5]),
                         "cursor": [rng.randint(220, 1200), rng.randint(90, 780)]}, talk(NOISE_APP["mail"]), "noise", None))
    else:
        for _ in range(rng.randint(1, 3)):
            spec = {"app": kind, "cursor": [rng.randint(240, 1250), rng.randint(120, 780)]}
            if kind == "calendar":
                spec["week"] = rng.choice([0, 0, 1])
            elif kind == "chat":
                spec["channel"] = rng.choice(["ops", "ops", "general", "events"])
                spec["scroll"] = rng.randint(0, 3)
            else:
                spec["scroll"] = rng.choice([0, 0, 150, 300])
            out.append((spec, talk(NOISE_APP[kind]), "noise", None))
    return out


# ----------------------------------------------------------------- the key script

def key_script(world: World, item: int, simple: bool = False) -> tuple[list[list[tuple]], list[str], dict]:
    """The beats every length of this item shares, the answer, and a description.

    simple=True gives the sanity condition: three tasks, each on its own detail
    page with the words on the same frame, and no other talk about tasks.
    """
    rng = random.Random(f"script-{item}-{simple}")
    k = 3 if simple else 3 + item % 3  # 3, 4 and 5 tasks in turn across items
    fams = rng.sample(range(len(FAMILIES)), k)
    fam_tasks = {fi: [t.id for t in world.tasks if t.family == fi] for fi in fams}
    keys = [rng.choice(fam_tasks[fi]) for fi in fams]
    styles = ["detail"] * k if simple else [rng.choice(["detail", "detail", "hover", "lag"]) for _ in range(k)]
    beats: list[list[tuple]] = [[(_list_spec(rng), rng.choice(INTRO), "intro", None)]]
    meta: dict = {"k": k, "keys": keys, "styles": styles, "not_this": [], "correction": None}
    for p, (tid, style) in enumerate(zip(keys, styles)):
        slot = "first" if p == 0 else ("last" if p == k - 1 else "middle")
        if not simple and rng.random() < 0.2:
            sib = rng.choice(world.siblings(tid))
            beats.append(shown_beat(world, sib, rng.choice(NOT_THIS), "not-this", rng))
            meta["not_this"].append(sib)
            # "no, not this one" runs straight into the right one: no gap between these two beats
            beats[-1] = beats[-1] + key_beat(world, tid, slot, style, rng)
        else:
            beats.append(key_beat(world, tid, slot, style, rng))
    answer = list(keys)
    if not simple:
        others = [t.id for t in world.tasks if t.id not in keys]
        # one task set aside out loud: already done (any task; even items), or for the week after
        # (a near-duplicate of a pick; odd items)
        if item % 2 == 0:
            done_t, later_t = rng.choice(others), None
            extra = [shown_beat(world, done_t, rng.choice(DONE), "done", rng)]
        else:
            done_t, later_t = None, rng.choice([s for tid in keys for s in world.siblings(tid)])
            extra = [shown_beat(world, later_t, rng.choice(LATER), "later", rng)]
        meta.update(done=done_t, later=later_t)
        if rng.random() < 0.5:
            rv = rng.randrange(k)
            t = world.by_id[keys[rv]]
            said = rng.choice(REVISIT).format(due=t.due, who=t.comments[0][0].split()[0])
            meta["revisit"] = keys[rv]
            extra.append(("after", rv, shown_beat(world, keys[rv], said, "revisit", rng)))
        # place done/later anywhere after the intro, a revisit anywhere after its key beat
        for e in extra:
            if isinstance(e, tuple):
                _, rv, beat = e
                at = next(i for i, b in enumerate(beats)
                          if any(r[2] in ("key", "key-next") and r[3] == keys[rv] for r in b))
                pos = rng.randint(at + 1, len(beats))
            else:
                beat = e
                pos = rng.randint(1, len(beats))
            beats.insert(pos, beat)
        if item % 3 == 2:  # a late correction in a third of the items
            p = rng.randrange(k)
            sibs = [s for s in world.siblings(keys[p])
                    if s not in keys and s not in (done_t, later_t) and s not in meta["not_this"]]
            pool = sibs if (sibs and rng.random() < 0.5) else [
                t for t in others if t not in (done_t, later_t) and t not in meta["not_this"]]
            new = rng.choice(pool)
            said = rng.choice(CORRECTION).format(ord=ORDINALS[p])
            beats.append(shown_beat(world, new, said, "correction", rng))
            answer[p] = new
            meta["correction"] = {"position": p, "old": keys[p], "new": new}
    beats.append([(_list_spec(rng), rng.choice(OUTRO), "outro", None)])
    return beats, answer, meta


MIN_GAP = 2


def pause_frames(world: World, rng: random.Random, n: int) -> list[tuple]:
    """Frames between two beats of the script: the list or another app, no task open or under the
    pointer, and nothing said beyond filler. At every length, beats are at least this far apart."""
    out = []
    for _ in range(n):
        if rng.random() < 0.6:
            spec = dict(_list_spec(rng), cursor=[rng.randint(240, 1260), rng.randint(84, 160)])
        else:
            app = rng.choice(["mail", "calendar", "chat", "doc"])
            spec = {"app": app, "cursor": [rng.randint(240, 1250), rng.randint(120, 780)]}
            if app == "mail":
                spec.update(open=rng.randrange(len(world.mail)), scroll=0)
            elif app == "calendar":
                spec["week"] = 0
            elif app == "chat":
                spec.update(channel="ops", scroll=rng.randint(0, 3))
            else:
                spec["scroll"] = rng.choice([0, 150])
        out.append((spec, rng.choice(FILLER) if rng.random() < 0.4 else "", "noise", None))
    return out


def build_session(item: int, size: int | None, simple: bool = False) -> Session:
    """size=None keeps only the key script, with the minimum pauses between its beats."""
    world = make_world(item)
    beats, answer, meta = key_script(world, item, simple)
    rng = random.Random(f"noise-{item}-{size}-{simple}")
    # activity between beat i and beat i+1: a pause after the beat, then ordinary activity, then a pause
    # before the next beat, so no ordinary task talk ever sits right next to the script's beats
    after = [pause_frames(world, rng, 1) for _ in range(len(beats) - 1)]
    before = [pause_frames(world, rng, MIN_GAP) for _ in range(len(beats) - 1)]
    gaps: list[list[tuple]] = [[] for _ in range(len(beats) - 1)]
    n_script = sum(len(b) for b in beats) + sum(len(g) for g in after + before)
    if size is not None:
        need = size - n_script
        if need < 0:
            raise ValueError(f"item {item}: key script has {n_script} frames, more than size {size}")
        # spread the activity over the gaps at random (a few gaps get most of it, as in a real call)
        weights = [rng.expovariate(1.0) for _ in gaps]
        filled = [0] * len(gaps)
        while sum(filled) < need:
            g = rng.choices(range(len(gaps)), weights)[0]
            beat = noise_beat(world, rng)[: need - sum(filled)]
            gaps[g].extend(beat)
            filled[g] += len(beat)
    rows: list[tuple] = []
    for i, beat in enumerate(beats):
        rows.extend(beat)
        if i < len(gaps):
            rows.extend(after[i] + gaps[i] + before[i])
    frames: list[Frame] = []
    t = 0.0
    last_tracker: str | None = None
    for i, (spec, said, role, task) in enumerate(rows):
        spec = dict(spec)
        if spec["app"] == "tracker":
            last_tracker = spec.get("task") if spec.get("view") == "detail" else None
        else:
            spec["tracker_page"] = last_tracker
        frames.append(Frame(id=f"f{i:03d}", t=round(t, 1), said=said, spec=spec, role=role, task=task))
        t += rng.uniform(0.8, 2.2)
    evidence = [f.id for f in frames if f.role in ("key", "key-next", "correction") and f.task]
    meta.update(n_script=n_script, duration_s=round(t, 1))
    return Session(item=item, size=len(frames), frames=frames, answer=answer, evidence=evidence, meta=meta)


def save(session: Session, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session.to_json(), indent=1))


def load(path: Path) -> Session:
    return Session.from_json(json.loads(path.read_text()))


def world_for(session: Session) -> World:
    return make_world(session.item)
