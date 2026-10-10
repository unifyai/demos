"""Render a frame spec to a 1280x800 screenshot of a browser on a shared screen.

The browser has five tabs: a task tracker (list and detail views), mail, a
calendar, team chat and a planning doc. A spec is a small dict (see
session.py) and rendering is deterministic: the same world and spec give the
same pixels. Fonts are the DejaVu faces that ship with matplotlib, so every
machine renders the same glyphs.
"""

from __future__ import annotations

import io
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from .world import PEOPLE, PROJECTS, World

W, H = 1280, 800
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

INK = (32, 35, 41)
MUTED = (110, 116, 128)
FAINT = (160, 165, 175)
LINE = (232, 234, 238)
BLUE = (59, 102, 232)
HOVER = (238, 242, 255)
SELECTED = (224, 232, 253)
SIDEBAR = (246, 247, 249)

STATUS_COLORS = {
    "Todo": ((236, 238, 242), (80, 86, 98)),
    "In progress": ((222, 235, 255), (36, 86, 190)),
    "In review": ((240, 230, 255), (112, 66, 190)),
    "Blocked": ((255, 228, 228), (186, 48, 48)),
    "Done": ((222, 245, 230), (30, 130, 70)),
}
PRIORITY_COLORS = {"Low": (150, 156, 168), "Medium": (220, 160, 40), "High": (230, 110, 40), "Urgent": (210, 50, 50)}
PROJECT_COLORS = {"OPS": (59, 102, 232), "EVT": (232, 120, 59), "PPL": (46, 160, 110), "FIN": (150, 80, 200), "VEN": (200, 60, 110)}
AVATAR_COLORS = [(66, 133, 244), (219, 68, 55), (15, 157, 88), (171, 71, 188), (0, 172, 193),
                 (255, 112, 67), (92, 107, 192), (124, 179, 66), (240, 98, 146), (141, 110, 99)]

TABS = [("tracker", "All tasks – Tasklane", (59, 102, 232), "T"), ("mail", "Inbox (12) – Mail", (219, 68, 55), "M"),
        ("calendar", "Calendar – Week 42", (15, 157, 88), "C"), ("chat", "#ops – Teamchat", (120, 60, 140), "#"),
        ("doc", "Q4 planning notes – Docs", (66, 133, 244), "D")]

ROW_H = 34
LIST_TOP = 199
LIST_ROWS = 17


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False, mono: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    elif serif:
        name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    else:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def initials(name: str) -> str:
    return "".join(p[0] for p in name.split()[:2])


def avatar(d: ImageDraw.ImageDraw, x: int, y: int, name: str, r: int = 10) -> None:
    color = AVATAR_COLORS[PEOPLE.index(name) % len(AVATAR_COLORS)] if name in PEOPLE else (120, 120, 120)
    d.ellipse([x, y, x + 2 * r, y + 2 * r], fill=color)
    f = font(max(8, r - 1), bold=True)
    d.text((x + r, y + r), initials(name), font=f, fill=(255, 255, 255), anchor="mm")


def chip(d: ImageDraw.ImageDraw, x: int, y: int, text: str, bg: tuple, fg: tuple, size: int = 11) -> int:
    f = font(size, bold=True)
    w = int(d.textlength(text, font=f)) + 16
    d.rounded_rectangle([x, y, x + w, y + size + 9], radius=(size + 9) // 2, fill=bg)
    d.text((x + 8, y + (size + 9) // 2), text, font=f, fill=fg, anchor="lm")
    return w


def cursor(d: ImageDraw.ImageDraw, x: int, y: int) -> None:
    pts = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    poly = [(x + px, y + py) for px, py in pts]
    d.polygon(poly, fill=(0, 0, 0), outline=(255, 255, 255))
    d.line(poly + [poly[0]], fill=(255, 255, 255), width=1)


def browser(d: ImageDraw.ImageDraw, app: str, url: str, tracker_title: str = "All tasks – Tasklane") -> None:
    d.rectangle([0, 0, W, 36], fill=(222, 226, 231))
    x = 8
    for key, title, color, letter in TABS:
        active = key == app
        if key == "tracker":
            title = tracker_title if len(tracker_title) <= 22 else tracker_title[:21] + "…"
        if active:
            d.rounded_rectangle([x, 6, x + 214, 40], radius=8, fill=(255, 255, 255))
        d.rounded_rectangle([x + 12, 14, x + 28, 30], radius=4, fill=color)
        d.text((x + 20, 22), letter, font=font(10, bold=True), fill=(255, 255, 255), anchor="mm")
        d.text((x + 36, 22), title, font=font(12), fill=INK if active else MUTED, anchor="lm")
        d.text((x + 200, 22), "×", font=font(12), fill=FAINT, anchor="mm")
        if not active:
            d.line([x + 214, 12, x + 214, 30], fill=(196, 200, 206))
        x += 218
    d.text((x + 14, 22), "+", font=font(16), fill=MUTED, anchor="mm")
    d.rectangle([0, 36, W, 76], fill=(255, 255, 255))
    for i, glyph in enumerate(["←", "→", "↻"]):
        d.text((22 + i * 30, 56), glyph, font=font(16), fill=MUTED if i != 1 else FAINT, anchor="mm")
    d.rounded_rectangle([110, 44, 1170, 68], radius=12, fill=(241, 243, 244))
    d.text((132, 56), url, font=font(13), fill=(60, 64, 72), anchor="lm")
    d.ellipse([1200, 46, 1220, 66], fill=(66, 133, 244))
    d.text((1210, 56), "DL", font=font(8, bold=True), fill=(255, 255, 255), anchor="mm")
    d.text((1248, 56), "⋮", font=font(16), fill=MUTED, anchor="mm")
    d.line([0, 76, W, 76], fill=(218, 220, 224))


# ---------------------------------------------------------------- tracker

def tracker_sidebar(d: ImageDraw.ImageDraw, selected: str) -> None:
    d.rectangle([0, 77, 232, H], fill=SIDEBAR)
    d.line([232, 77, 232, H], fill=LINE)
    d.rounded_rectangle([18, 92, 42, 116], radius=6, fill=(59, 102, 232))
    d.text((30, 104), "A", font=font(13, bold=True), fill=(255, 255, 255), anchor="mm")
    d.text((52, 104), "Acme Ops", font=font(14, bold=True), fill=INK, anchor="lm")
    y = 140
    for item in ["My work", "Inbox", "Next week", "All tasks"]:
        if item == selected:
            d.rounded_rectangle([10, y - 13, 222, y + 13], radius=6, fill=(229, 233, 240))
        d.text((24, y), item, font=font(13, bold=item == selected), fill=INK, anchor="lm")
        y += 30
    y += 16
    d.text((24, y), "PROJECTS", font=font(10, bold=True), fill=FAINT, anchor="lm")
    y += 28
    for code, name in PROJECTS.items():
        if code == selected:
            d.rounded_rectangle([10, y - 13, 222, y + 13], radius=6, fill=(229, 233, 240))
        d.ellipse([24, y - 4, 32, y + 4], fill=PROJECT_COLORS[code])
        d.text((42, y), name, font=font(13), fill=INK, anchor="lm")
        y += 30
    d.text((24, H - 24), "Help & shortcuts", font=font(12), fill=MUTED, anchor="lm")


def id_key(task_id: str) -> tuple[str, int]:
    project, number = task_id.split("-")
    return project, int(number)


def list_tasks(world: World, spec: dict) -> list:
    tasks = sorted(world.tasks, key=lambda t: id_key(t.id))
    side = spec.get("sidebar", "All tasks")
    if side in PROJECTS:
        tasks = [t for t in tasks if t.project == side]
    q = (spec.get("search") or "").strip().lower()
    if q:
        tasks = [t for t in tasks if q in t.title.lower() or q in t.id.lower()]
    return tasks


def visible_rows(world: World, spec: dict) -> list[tuple[str, int]]:
    """(task id, row top y) for the rows a list spec shows. Used by session.py to aim the pointer."""
    tasks = list_tasks(world, spec)
    start = max(0, min(spec.get("scroll", 0), max(0, len(tasks) - LIST_ROWS)))
    return [(t.id, LIST_TOP + i * ROW_H) for i, t in enumerate(tasks[start:start + LIST_ROWS])]


def tracker_header(d: ImageDraw.ImageDraw, title: str, search: str, typing: bool) -> None:
    d.text((260, 104), title, font=font(18, bold=True), fill=INK, anchor="lm")
    d.rounded_rectangle([880, 90, 1160, 118], radius=8, fill=(255, 255, 255), outline=(208, 212, 220) if not typing else BLUE, width=1 if not typing else 2)
    d.ellipse([892, 98, 903, 109], outline=MUTED, width=2)
    d.line([902, 108, 907, 113], fill=MUTED, width=2)
    if search:
        d.text((916, 104), search, font=font(13), fill=INK, anchor="lm")
        if typing:
            x = 916 + int(d.textlength(search, font=font(13)))
            d.line([x + 1, 96, x + 1, 112], fill=INK, width=1)
    else:
        d.text((916, 104), "Search tasks", font=font(13), fill=FAINT, anchor="lm")
    d.rounded_rectangle([1172, 90, 1262, 118], radius=8, fill=BLUE)
    d.text((1217, 104), "New task", font=font(12, bold=True), fill=(255, 255, 255), anchor="mm")
    x = 260
    for i, tab in enumerate(["List", "Board", "Timeline"]):
        f = font(12, bold=i == 0)
        d.text((x, 146), tab, font=f, fill=INK if i == 0 else MUTED, anchor="lm")
        if i == 0:
            d.line([x, 160, x + d.textlength(tab, font=f), 160], fill=BLUE, width=2)
        x += int(d.textlength(tab, font=f)) + 26
    d.text((1080, 146), "Filter", font=font(12), fill=MUTED, anchor="lm")
    d.text((1140, 146), "Sort: ID ▾", font=font(12), fill=MUTED, anchor="lm")
    d.line([233, 164, W, 164], fill=LINE)


def tracker_list(img: Image.Image, d: ImageDraw.ImageDraw, world: World, spec: dict) -> None:
    side = spec.get("sidebar", "All tasks")
    tracker_sidebar(d, side)
    search = spec.get("search") or ""
    title = f"Search: “{search}”" if search and not spec.get("typing") else (PROJECTS.get(side, side))
    tracker_header(d, title, search, bool(spec.get("typing")))
    cols = [("ID", 260), ("TITLE", 362), ("ASSIGNEE", 742), ("STATUS", 920), ("DUE", 1050), ("PRIORITY", 1140)]
    for name, x in cols:
        d.text((x, 182), name, font=font(10, bold=True), fill=FAINT, anchor="lm")
    d.line([233, 196, W, 196], fill=LINE)
    rows = visible_rows(world, spec)
    for tid, y in rows:
        t = world.by_id[tid]
        if tid == spec.get("selected"):
            d.rectangle([233, y, W, y + ROW_H - 1], fill=SELECTED)
        elif tid == spec.get("hover"):
            d.rectangle([233, y, W, y + ROW_H - 1], fill=HOVER)
        cy = y + ROW_H // 2
        d.text((260, cy), t.id, font=font(12, mono=True), fill=MUTED, anchor="lm")
        d.text((362, cy), t.title, font=font(13), fill=INK, anchor="lm")
        avatar(d, 742, cy - 10, t.assignee)
        d.text((768, cy), t.assignee, font=font(12), fill=INK, anchor="lm")
        bg, fg = STATUS_COLORS[t.status]
        chip(d, 920, cy - 10, t.status, bg, fg)
        d.text((1050, cy), t.due, font=font(12), fill=INK, anchor="lm")
        pc = PRIORITY_COLORS[t.priority]
        for i in range(3):
            h = 4 + 3 * i
            on = i < {"Low": 1, "Medium": 2, "High": 3, "Urgent": 3}[t.priority]
            d.rectangle([1140 + i * 5, cy + 5 - h, 1143 + i * 5, cy + 5], fill=pc if on else LINE)
        d.text((1160, cy), t.priority, font=font(12), fill=INK, anchor="lm")
        d.line([233, y + ROW_H - 1, W, y + ROW_H - 1], fill=LINE)
    n = len(list_tasks(world, spec))
    msg = f"{n} tasks" if n else "No tasks match your search"
    d.text((260, H - 12), msg, font=font(11), fill=MUTED, anchor="lm")


def wrap(d: ImageDraw.ImageDraw, x: int, y: int, text: str, width_chars: int, f, fill, line_h: int) -> int:
    for line in textwrap.wrap(text, width_chars):
        d.text((x, y), line, font=f, fill=fill)
        y += line_h
    return y


def tracker_detail(img: Image.Image, d: ImageDraw.ImageDraw, world: World, spec: dict) -> None:
    t = world.by_id[spec["task"]]
    tracker_sidebar(d, spec.get("sidebar", "All tasks"))
    off = -170 * int(spec.get("scroll", 0))
    d.text((260, 98), PROJECTS[t.project], font=font(12), fill=MUTED, anchor="lm")
    x = 260 + int(d.textlength(PROJECTS[t.project], font=font(12))) + 8
    d.text((x, 98), "›", font=font(12), fill=FAINT, anchor="lm")
    d.text((x + 14, 98), t.id, font=font(12, mono=True), fill=MUTED, anchor="lm")
    d.text((1140, 98), "Share   ⋯", font=font(12), fill=MUTED, anchor="lm")
    body = Image.new("RGB", (700, 1100), (255, 255, 255))
    b = ImageDraw.Draw(body)
    b.text((0, 0), t.title, font=font(24, bold=True), fill=INK)
    bg, fg = STATUS_COLORS[t.status]
    cx = chip(b, 0, 46, t.status, bg, fg, size=12) + 12
    b.text((cx, 56), f"{t.priority} priority", font=font(12), fill=PRIORITY_COLORS[t.priority], anchor="lm")
    cx += int(b.textlength(f"{t.priority} priority", font=font(12))) + 18
    b.text((cx, 56), f"Due {t.due}", font=font(12), fill=INK, anchor="lm")
    y = 96
    b.text((0, y), "Description", font=font(13, bold=True), fill=INK)
    y = wrap(b, 0, y + 26, t.description, 82, font(13), (60, 64, 72), 21) + 18
    done = sum(1 for _, ok in t.checklist if ok)
    b.text((0, y), f"Checklist   {done}/{len(t.checklist)}", font=font(13, bold=True), fill=INK)
    b.rounded_rectangle([170, y + 6, 420, y + 12], radius=3, fill=LINE)
    b.rounded_rectangle([170, y + 6, 170 + int(250 * done / len(t.checklist)), y + 12], radius=3, fill=(46, 160, 110))
    y += 30
    for item, ok in t.checklist:
        if ok:
            b.rounded_rectangle([0, y + 1, 16, y + 17], radius=3, fill=BLUE)
            b.line([4, y + 9, 7, y + 13, 13, y + 5], fill=(255, 255, 255), width=2)
        else:
            b.rounded_rectangle([0, y + 1, 16, y + 17], radius=3, outline=(190, 195, 205), width=1)
        b.text((28, y + 9), item, font=font(13), fill=MUTED if ok else INK, anchor="lm")
        y += 28
    y += 16
    b.text((0, y), "Activity", font=font(13, bold=True), fill=INK)
    y += 30
    for author, when, text in t.comments:
        avatar(b, 0, y, author, r=12)
        b.text((34, y + 2), author, font=font(12, bold=True), fill=INK)
        b.text((34 + int(b.textlength(author, font=font(12, bold=True))) + 8, y + 2), when, font=font(12), fill=FAINT)
        y = wrap(b, 34, y + 22, text, 78, font(13), (60, 64, 72), 20) + 14
    b.rounded_rectangle([0, y, 690, y + 64], radius=8, outline=(214, 218, 224))
    b.text((14, y + 20), "Leave a comment…", font=font(13), fill=FAINT)
    crop = body.crop((0, max(0, -off), 700, max(0, -off) + (H - 124)))
    img.paste(crop, (260, 124))
    # right panel
    d.rounded_rectangle([980, 124, 1262, 600], radius=10, fill=(250, 251, 252), outline=LINE)
    y = 142
    fields = [("Assignee", t.assignee), ("Reporter", PEOPLE[(PEOPLE.index(t.assignee) + 3) % len(PEOPLE)]),
              ("Project", PROJECTS[t.project]), ("Labels", None), ("Due date", t.due), ("Priority", t.priority),
              ("Estimate", f"{2 + len(t.title) % 5}h"), ("Created", f"Sep {10 + len(t.title) % 18}")]
    for label, value in fields:
        d.text((998, y), label.upper(), font=font(10, bold=True), fill=FAINT)
        if label in ("Assignee", "Reporter"):
            avatar(d, 998, y + 18, value)
            d.text((1024, y + 28), value, font=font(13), fill=INK, anchor="lm")
        elif label == "Labels":
            lx = 998
            for lab in t.labels:
                lx += chip(d, lx, y + 18, lab, (238, 240, 244), (80, 86, 98)) + 6
        else:
            d.text((998, y + 28), value, font=font(13), fill=INK, anchor="lm")
        y += 56


# ---------------------------------------------------------------- other apps

def mail_view(img: Image.Image, d: ImageDraw.ImageDraw, world: World, spec: dict) -> None:
    d.rectangle([0, 77, 200, H], fill=(248, 249, 250))
    d.rounded_rectangle([14, 92, 130, 124], radius=12, fill=(194, 231, 255))
    d.text((72, 108), "✎ Compose", font=font(13, bold=True), fill=INK, anchor="mm")
    y = 150
    for i, (name, n) in enumerate([("Inbox", "12"), ("Starred", ""), ("Snoozed", ""), ("Sent", ""), ("Drafts", "3"), ("Archive", "")]):
        if i == 0:
            d.rounded_rectangle([8, y - 13, 192, y + 13], radius=13, fill=(211, 227, 253))
        d.text((28, y), name, font=font(13, bold=i == 0), fill=INK, anchor="lm")
        if n:
            d.text((176, y), n, font=font(12, bold=True), fill=INK, anchor="rm")
        y += 30
    d.line([620, 77, 620, H], fill=LINE)
    sel = spec.get("open", 0)
    start = spec.get("scroll", 0)
    y = 84
    for i, (sender, subj, snip, when) in list(enumerate(world.mail))[start:start + 11]:
        if i == sel:
            d.rectangle([201, y, 619, y + 63], fill=(232, 240, 254))
        d.text((220, y + 14), sender, font=font(13, bold=True), fill=INK, anchor="lm")
        d.text((604, y + 14), when, font=font(11), fill=MUTED, anchor="rm")
        d.text((220, y + 33), subj, font=font(12), fill=INK, anchor="lm")
        d.text((220, y + 50), (snip[:52] + "…") if len(snip) > 52 else snip, font=font(11), fill=MUTED, anchor="lm")
        d.line([201, y + 63, 619, y + 63], fill=LINE)
        y += 64
    sender, subj, snip, when = world.mail[sel]
    d.text((650, 108), subj, font=font(20, bold=True), fill=INK, anchor="lm")
    avatar(d, 650, 136, sender if sender in PEOPLE else "x", r=16)
    d.text((694, 146), sender, font=font(13, bold=True), fill=INK, anchor="lm")
    d.text((694, 164), "to me", font=font(11), fill=MUTED, anchor="lm")
    d.text((1250, 146), f"Today, {when}", font=font(11), fill=MUTED, anchor="rm")
    y = 200
    body = [f"Hi,", snip, "Let me know if anything looks wrong or if you need more from me before the end of the week.",
            "Thanks,", sender.split()[0]]
    for para in body:
        y = wrap(d, 650, y, para, 66, font(13), (50, 54, 60), 21) + 12
    d.rounded_rectangle([650, y + 20, 760, y + 52], radius=16, outline=(200, 204, 210))
    d.text((705, y + 36), "↩ Reply", font=font(12), fill=INK, anchor="mm")
    d.rounded_rectangle([772, y + 20, 892, y + 52], radius=16, outline=(200, 204, 210))
    d.text((832, y + 36), "↪ Forward", font=font(12), fill=INK, anchor="mm")


def calendar_view(img: Image.Image, d: ImageDraw.ImageDraw, world: World, spec: dict) -> None:
    wk = spec.get("week", 0)
    days = [f"{name} {13 + 7 * wk + i}" for i, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])]
    d.text((30, 104), f"October {13 + 7 * wk} – {17 + 7 * wk}, 2026", font=font(18, bold=True), fill=INK, anchor="lm")
    d.rounded_rectangle([1080, 90, 1150, 118], radius=8, outline=(208, 212, 220))
    d.text((1115, 104), "Today", font=font(12), fill=INK, anchor="mm")
    d.text((1180, 104), "‹  ›", font=font(16), fill=MUTED, anchor="lm")
    left, top, colw, hh = 90, 160, 236, 31
    for i, day in enumerate(days):
        d.text((left + i * colw + colw // 2, 142), day, font=font(12, bold=True), fill=INK, anchor="mm")
    for h in range(10):
        y = top + h * 2 * hh
        d.text((70, y), f"{8 + h}:00", font=font(11), fill=MUTED, anchor="rm")
        d.line([left, y, left + 5 * colw, y], fill=LINE)
    for i in range(6):
        d.line([left + i * colw, top, left + i * colw, H], fill=LINE)
    palette = [(66, 133, 244), (15, 157, 88), (244, 160, 0), (171, 71, 188), (0, 172, 193)]
    for day, slot, length, title in world.events:
        x0 = left + day * colw + 4
        y0 = top + (slot + 14 * wk) % 18 * hh + 2
        y1 = min(H - 2, y0 + length * hh - 4)
        c = palette[(len(title) + day) % len(palette)]
        d.rounded_rectangle([x0, y0, x0 + colw - 8, y1], radius=5, fill=tuple(min(255, v + (255 - v) * 2 // 3) for v in c), outline=c)
        d.text((x0 + 8, y0 + 6), title, font=font(12, bold=True), fill=INK)
        if length > 1:
            d.text((x0 + 8, y0 + 24), f"{8 + (slot // 2)}:{'30' if slot % 2 else '00'}", font=font(11), fill=MUTED)


def chat_view(img: Image.Image, d: ImageDraw.ImageDraw, world: World, spec: dict) -> None:
    d.rectangle([0, 77, 220, H], fill=(35, 39, 48))
    d.text((20, 104), "Acme", font=font(15, bold=True), fill=(255, 255, 255), anchor="lm")
    y = 150
    channel = spec.get("channel", "ops")
    for ch in ["general", "ops", "events", "finance", "random"]:
        if ch == channel:
            d.rounded_rectangle([8, y - 13, 212, y + 13], radius=6, fill=(18, 100, 163))
        d.text((24, y), f"# {ch}", font=font(13, bold=ch == channel), fill=(255, 255, 255) if ch == channel else (190, 194, 202), anchor="lm")
        y += 28
    d.text((248, 104), f"# {channel}", font=font(16, bold=True), fill=INK, anchor="lm")
    d.line([221, 128, W, 128], fill=LINE)
    start = spec.get("scroll", 0)
    y = 148
    for author, when, text in world.chat[start:start + 9]:
        d.rounded_rectangle([248, y, 284, y + 36], radius=6, fill=AVATAR_COLORS[PEOPLE.index(author) % 10])
        d.text((266, y + 18), initials(author), font=font(12, bold=True), fill=(255, 255, 255), anchor="mm")
        d.text((298, y + 2), author, font=font(13, bold=True), fill=INK)
        d.text((298 + int(d.textlength(author, font=font(13, bold=True))) + 8, y + 4), when, font=font(11), fill=MUTED)
        d.text((298, y + 22), text, font=font(13), fill=(50, 54, 60))
        y += 64
    d.rounded_rectangle([248, H - 64, 1256, H - 16], radius=8, outline=(200, 204, 210))
    d.text((264, H - 40), f"Message #{channel}", font=font(13), fill=FAINT, anchor="lm")


def doc_view(img: Image.Image, d: ImageDraw.ImageDraw, world: World, spec: dict) -> None:
    d.rectangle([0, 77, W, H], fill=(241, 243, 244))
    d.rectangle([0, 77, W, 112], fill=(255, 255, 255))
    x = 20
    for item in ["File", "Edit", "View", "Insert", "Format", "Tools"]:
        d.text((x, 95), item, font=font(12), fill=INK, anchor="lm")
        x += int(d.textlength(item, font=font(12))) + 18
    d.rectangle([290, 124, 990, H], fill=(255, 255, 255), outline=(220, 222, 226))
    off = -int(spec.get("scroll", 0))
    y = 170 + off
    d.text((350, y), "Q4 planning notes", font=font(26, bold=True, serif=True), fill=INK)
    y += 58
    for para in world.doc:
        y = wrap(d, 350, y, para, 64, font(14, serif=True), (40, 44, 50), 24) + 16


VIEWS = {"mail": mail_view, "calendar": calendar_view, "chat": chat_view, "doc": doc_view}
URLS = {"mail": "mail.acme.com/inbox", "calendar": "calendar.acme.com/week/42", "chat": "teamchat.acme.com/acme/ops",
        "doc": "docs.acme.com/d/q4-planning-notes"}


def render(world: World, spec: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    app = spec["app"]
    # The tracker tab keeps the title of the last tracker page shown, as a browser does.
    last = spec.get("tracker_page")
    tracker_title = f"{last} {world.by_id[last].title}" if last else "All tasks – Tasklane"
    if app == "tracker":
        if spec["view"] == "detail":
            url = f"tasklane.acme.com/acme/tasks/{spec['task']}"
            tracker_title = f"{spec['task']} {world.by_id[spec['task']].title}"
        else:
            q = spec.get("search") or ""
            url = "tasklane.acme.com/acme/tasks" + (f"?q={q.replace(' ', '+')}" if q else "")
            tracker_title = "All tasks – Tasklane"
    else:
        url = URLS[app]
    browser(d, app, url, tracker_title)
    if app == "tracker":
        (tracker_detail if spec["view"] == "detail" else tracker_list)(img, d, world, spec)
    else:
        VIEWS[app](img, d, world, spec)
    if spec.get("cursor"):
        cursor(d, *spec["cursor"])
    return img


def encode(img: Image.Image, fmt: str = "webp_lossless") -> bytes:
    """Frames travel as lossless WebP: the exact pixels, at about a quarter of PNG's bytes."""
    buf = io.BytesIO()
    if fmt == "webp_lossless":
        img.save(buf, "WEBP", lossless=True, quality=80, method=4)
    elif fmt == "png":
        img.save(buf, "PNG", optimize=True)
    else:
        raise ValueError(fmt)
    return buf.getvalue()
