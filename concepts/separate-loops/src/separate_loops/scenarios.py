"""The data: task instances generated from seeds.

An instance is one conversation: the sites' data, the person's request, the
person's scripted messages with their triggers, the facts the person knows
(for answering the agent's questions), and what the checks expect.

Families (each a different site): A invitations (messaging), B booking,
C ordering (shop). Sizes: sanity, small, medium, realistic, and dual (a
realistic main task plus a second task requested partway, then cancelled).
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any

from .sites import fmt_time

FIRST = ["Priya", "Tom", "Jess", "Alex", "Hannah", "Leo", "Maya", "Ravi", "Nina", "Owen", "Chloe", "Dev", "Isla",
         "Marcus", "Zoe", "Felix", "Amara", "Ben", "Lucia", "Kofi", "Erin", "Hugo", "Mei", "Jonah", "Aisha", "Rory",
         "Elena", "Tariq", "Freya", "Callum", "Ines", "Yusuf", "Grace", "Mateo", "Lena", "Oscar", "Ruth", "Kai",
         "Nadia", "Theo", "Alice", "Gabriel", "Holly", "Ivan", "Jade", "Luca", "Molly", "Nikhil", "Orla", "Pablo"]
LAST = ["Shah", "Okafor", "Moreno", "Chen", "Webb", "Fischer", "Lindqvist", "Nair", "Ross", "Wu", "Byrne", "Kaur",
        "Petrov", "Adeyemi", "Novak", "Hughes", "Costa", "Larsen", "Haddad", "Murphy", "Tanaka", "Silva", "Brennan",
        "Osei", "Kowalski", "Dubois", "Reyes", "Ahmed", "Walsh", "Jensen", "Morgan", "Patel", "Khan", "Price",
        "Bianchi", "Ndlovu", "Grant", "Iyer", "Lowe", "Romero"]
BIOS = ["Into cycling and jazz.", "Runs a bakery on weekends.", "Plays five-a-side on Tuesdays.", "Learning Spanish.",
        "Keeps bees.", "Loves board games.", "Amateur photographer.", "Volunteers at the food bank.",
        "Training for a marathon.", "Collects vinyl.", "Teaches yoga.", "Restores old bikes."]
GROUPS = ["book club", "climbing group", "choir", "allotment society", "running club", "film club", "quiz team"]
EVENTS = ["a summer barbecue", "a quiz night", "a picnic", "a birthday dinner", "a board-game evening",
          "an end-of-season party", "a curry night"]
PLACES = ["my place, 14 Elm Road", "the Crown pub on Church Street", "Victoria Park by the bandstand",
          "the community hall on Mill Lane", "my flat, 3B Albion Court"]
TIME_PAIRS = [(18 * 60, 19 * 60 + 30), (19 * 60, 18 * 60 + 30), (17 * 60 + 30, 19 * 60), (18 * 60 + 30, 20 * 60),
              (12 * 60, 13 * 60 + 30), (19 * 60 + 30, 18 * 60)]
UNRELATED = [("Unrelated, but what's 15% of 240?", "36"),
             ("Quick unrelated question: how many minutes are there in 3.5 hours?", "210"),
             ("Random question: what's 17 times 23?", "391"),
             ("Unrelated: what's the capital of Australia?", "Canberra"),
             ("Off topic: how many days are there in a leap year?", "366"),
             ("Unrelated: what's 12 squared?", "144"),
             ("Random one: what's the boiling point of water in Fahrenheit?", "212"),
             ("Quick one: how many seconds are there in an hour?", "3600")]
CUISINES = [("Italian", "Shoreditch"), ("Thai", "Islington"), ("Greek", "Camden"), ("Indian", "Brixton"),
            ("Japanese", "Soho"), ("Lebanese", "Hackney")]
RESTAURANT_NAMES = ["Juniper", "The Fig Tree", "Copper Pot", "Little Lantern", "Blue Door", "Saltbox", "The Larder",
                    "Ember", "Harbour Room", "Olive & Ash", "The Orchard", "Wildflower", "Pepper Mill", "The Courtyard",
                    "Lark", "The Willow", "Sable", "Hearth", "The Pantry", "Five Spoons", "Silver Birch", "Oak & Barrel",
                    "Kettle Yard", "The Lantern", "Moss", "North Star", "The Granary", "Laurel", "Ivy Room",
                    "Two Doors Down", "The Little Table", "Market Hall", "Plum", "The Corner Room", "The Grove", "Harvest",
                    "Blackbird", "Red Kite", "The Long Table", "Bramble"]
STREETS = {  # streets in each area, so a restaurant's address agrees with the search
    "Shoreditch": ["Rivington St", "Curtain Rd", "Old St", "Great Eastern St", "Redchurch St", "Calvert Ave",
                   "Hoxton St", "Charlotte Rd", "Leonard St", "Paul St", "Phipp St", "Scrutton St", "Holywell Ln",
                   "Shoreditch High St", "Club Row", "Chance St", "Boundary St", "Ebor St"],
    "Soho": ["Dean St", "Frith St", "Greek St", "Old Compton St", "Wardour St", "Berwick St", "Poland St",
             "Broadwick St", "Lexington St", "Beak St", "Brewer St", "Rupert St", "Meard St", "Romilly St",
             "Bateman St", "Carlisle St", "Kingly St", "D'Arblay St"],
    "Islington": ["Upper St", "Essex Rd", "Liverpool Rd", "Cross St", "Camden Passage", "Theberton St", "Canonbury Ln",
                  "Barnsbury St", "Duncan St", "Charlton Pl", "Gaskin St", "Almeida St", "Islington Park St",
                  "St Peter's St", "Colebrooke Row", "Danbury St", "Noel Rd", "Packington St"],
    "Camden": ["Parkway", "Camden High St", "Chalk Farm Rd", "Kentish Town Rd", "Delancey St", "Arlington Rd",
               "Inverness St", "Jamestown Rd", "Pratt St", "Plender St", "Bayham St", "Greenland Rd", "Albert St",
               "Gloucester Cres", "Oval Rd", "Castlehaven Rd", "Hawley Cres", "Royal College St"],
    "Brixton": ["Atlantic Rd", "Coldharbour Ln", "Electric Ave", "Acre Ln", "Brixton Hill", "Ferndale Rd",
                "Railton Rd", "Dorrell Pl", "Market Row", "Pope's Rd", "Brixton Station Rd", "Effra Rd", "Saltoun Rd",
                "Vining St", "Rushcroft Rd", "Tunstall Rd", "Mayall Rd", "Somerleyton Rd"],
    "Hackney": ["Mare St", "Chatsworth Rd", "Broadway Market", "Well St", "Morning Ln", "Amhurst Rd", "Dalston Ln",
                "Graham Rd", "Kingsland Rd", "Lower Clapton Rd", "Homerton High St", "Wilton Way", "Navarino Rd",
                "Richmond Rd", "Clarence Rd", "Paragon Rd", "Darnley Rd", "Median Rd"],
}
PRODUCTS = [  # (what the person asks for, the shop's product name, extra search words)
    ("A4 copier paper (box of 5 reams)", "A4 copier paper, box of 5 reams", "printer paper"),
    ("black ballpoint pens (box of 50)", "Ballpoint pens, black, box of 50", "biro"),
    ("yellow sticky notes (pack of 12 pads)", "Sticky notes, yellow, pack of 12 pads", "post-it"),
    ("whiteboard markers (pack of 8)", "Whiteboard markers, assorted, pack of 8", "dry wipe"),
    ("AA batteries (pack of 24)", "AA batteries, alkaline, pack of 24", "battery"),
    ("staples (box of 5,000)", "Staples 26/6, box of 5000", "stapler"),
    ("lever-arch files (pack of 10)", "Lever arch files, A4, pack of 10", "folder"),
    ("bin bags (roll of 50)", "Bin bags, black, roll of 50", "rubbish"),
    ("coffee beans (1 kg bag)", "Coffee beans, 1kg bag", "espresso"),
    ("paper towels (pack of 24 rolls)", "Paper towels, pack of 24 rolls", "kitchen roll"),
    ("hand soap refills (pack of 6)", "Hand soap refills, pack of 6", "liquid"),
    ("C5 envelopes (box of 500)", "C5 envelopes, white, box of 500", "envelope"),
    ("highlighters (pack of 10)", "Highlighters, assorted, pack of 10", "highlighter pens"),
    ("printer labels (pack of 100 sheets)", "Printer labels, A4, pack of 100 sheets", "sticker"),
    ("USB-C cables (pack of 3)", "USB-C cables, 1m, pack of 3", "charger"),
    ("tea bags (box of 1,100)", "Tea bags, box of 1100", "tea"),
    ("dry-wipe erasers (pack of 4)", "Whiteboard erasers, pack of 4", "dry wipe eraser"),
    ("masking tape (pack of 6 rolls)", "Masking tape, pack of 6 rolls", "tape"),
]
BRANDS = ["OfficeBasics", "Prima", "Staplex", "Northline", "ValueLine", "Crest"]
UK_STREETS = ["Tabernacle Street", "Featherstone Street", "Bunhill Row", "Clifton Street", "Worship Street"]

# Per size: (main items, away members for A); minimum gap and maximum wait between scripted messages; run caps.
ITEMS = {"sanity": {"A": 4, "B": 4, "C": 4}, "small": {"A": 5, "B": 5, "C": 5},
         "medium": {"A": 10, "B": 10, "C": 7}, "realistic": {"A": 16, "B": 16, "C": 10},
         "second": {"A": 6, "B": 6, "C": 6}}
AWAY = {"sanity": 1, "small": 1, "medium": 2, "realistic": 3, "second": 1}
GAPS = {"sanity": (3, 30), "small": (6, 45), "medium": (8, 90), "realistic": (8, 150), "dual": (8, 150)}
CAPS = {"sanity": 240, "small": 480, "medium": 900, "realistic": 1800, "dual": 2100}
LATENCY = {"sanity": "sanity", "small": "small", "medium": "medium", "realistic": "realistic", "dual": "realistic"}

# The person's script: (kind, progress count on the main task, or ("second", n), or None for "20 s after").
SCRIPTS: dict[str, dict[str, list[tuple[str, Any]]]] = {
    "sanity": {f: [("status", 2)] for f in "ABC"},
    "small": {"A": [("status", 1), ("correction", 2), ("unrelated", 3)],
              "B": [("status", 2), ("correction", 3), ("unrelated", 4)],
              "C": [("status", 1), ("correction", 2), ("unrelated", 3)]},
    "medium": {"A": [("status", 2), ("correction", 3), ("addition", 5), ("pause", 6), ("resume", None), ("unrelated", 7)],
               "B": [("status", 2), ("correction", 3), ("addition", 5), ("pause", 6), ("resume", None), ("unrelated", 8)],
               "C": [("status", 1), ("correction", 2), ("addition", 3), ("pause", 4), ("resume", None), ("unrelated", 5)]},
    "realistic": {"A": [("status", 2), ("correction", 4), ("addition", 6), ("pause", 8), ("resume", None),
                        ("unrelated", 9), ("status2", 11), ("change", 12)],
                  "B": [("status", 3), ("correction", 5), ("addition", 7), ("pause", 9), ("resume", None),
                        ("unrelated", 11), ("status2", 13), ("change", 15)],
                  "C": [("status", 1), ("correction", 2), ("addition", 3), ("pause", 5), ("resume", None),
                        ("unrelated", 6), ("status2", 7), ("change", 8)]},
    "dual": {"A": [("status", 2), ("correction", 4), ("second", 5), ("status_both", ("second", 2)), ("unrelated", 8),
                   ("cancel_second", ("second", 3)), ("addition", 10)],
             "B": [("status", 3), ("correction", 5), ("second", 6), ("status_both", ("second", 2)), ("unrelated", 9),
                   ("cancel_second", ("second", 3)), ("addition", 11)],
             "C": [("status", 1), ("correction", 2), ("second", 3), ("status_both", ("second", 2)), ("unrelated", 5),
                   ("cancel_second", ("second", 3)), ("addition", 6)]},
}
SECOND_OF = {"A": "B", "B": "C", "C": "A"}
PROGRESS = {  # what counts as progress on a task, for triggers
    "A": {"site": "messaging", "tool": "messaging_send_message", "distinct": "to"},
    "B": {"site": "booking", "tool": "booking_check_availability", "distinct": None},
    "C": {"site": "shop", "tool": "shop_add_to_cart", "distinct": None},
}
LABEL = {"A": "invitations", "B": "table booking", "C": "office order"}


def _date(rng: random.Random, weekdays: tuple[int, ...]) -> dt.date:
    """A real date between 12 October and 27 November 2026 on one of the weekdays (Monday is 0)."""
    days = [dt.date(2026, 10, 12) + dt.timedelta(d) for d in range(47)]
    return rng.choice([d for d in days if d.weekday() in weekdays])


def _day_text(d: dt.date) -> str:
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


def _people(rng: random.Random, n: int, avoid: set[str]) -> list[tuple[str, str]]:
    firsts = [f for f in FIRST if f not in avoid]
    rng.shuffle(firsts)
    return [(f, rng.choice(LAST)) for f in firsts[:n]]


# --------------------------------------------------------------------------- #
#  A: invitations                                                             #
# --------------------------------------------------------------------------- #


def build_A(rng: random.Random, n: int, n_away: int, gap: bool, kinds: list[str], person: tuple[str, str]) -> dict[str, Any]:
    people = _people(rng, n + 8, avoid={"Sam", person[0]})
    members = [f"{f} {l}" for f, l in people[:n]]
    others = [f"{f} {l}" for f, l in people[n:]]
    group = rng.choice(GROUPS)
    away = set(rng.sample(members[2:], n_away))  # the first two are never away
    sams = ["Sam Rivera", "Sam Oyelaran"] if gap else ["Sam Rivera"]
    when = _date(rng, (3, 4, 5, 6))
    contacts = []
    for full in members + others + sams:
        f = full.split()[0]
        back = when + dt.timedelta(days=rng.randint(3, 20))  # away over the event, so "skip anyone away" is clear
        status = f"away until {back.day} {back.strftime('%B')}" if full in away else "active"
        contacts.append({"name": full, "first_name": f, "status": status, "city": "London", "bio": rng.choice(BIOS)})
    other_group = rng.choice([g for g in GROUPS if g != group])
    groups = {group: members, other_group: others[:6]}
    old, new = rng.choice(TIME_PAIRS)
    event, place = rng.choice(EVENTS), rng.choice(PLACES)
    day = _day_text(when)
    active = [m for m in members if m not in away]
    drop = [m for m in active[-4:]][-1] if "change" in kinds else None
    sam_txt = "my neighbour Sam" if gap else "my neighbour Sam Rivera"
    request = (f"Could you invite everyone in my {group}, and {sam_txt}, to {event} on {day} at {fmt_time(old)} at "
               f"{place}? Please send each person their own short message, using their first name. Check each "
               f"group member's profile before you message them, and skip anyone who is away; tell me who you skipped.")
    texts = {
        "status": "How's it going? Who have you messaged so far?",
        "status2": "How many people are left to invite?",
        "correction": (f"Oh no, I got the time wrong: it's {fmt_time(new)}, not {fmt_time(old)}. Please make sure "
                       f"everyone has the right time, including anyone you've already messaged."),
        "addition": "Also, please ask everyone to bring a dish to share, including the people you've already messaged.",
        "change": (f"Actually, don't invite {drop}; I'll ask them myself. If you've already messaged them, leave it."
                   if drop else ""),
    }
    facts = ("When you say 'my neighbour Sam' you mean Sam Rivera, not any other Sam."
             if gap else "Your neighbour is Sam Rivera.")
    expect = {"family": "A", "group": group, "members": members, "away": sorted(away), "sam": "Sam Rivera",
              "wrong_sam": "Sam Oyelaran" if gap else None, "old_time": old, "new_time": new,
              "drop": drop, "event": event, "day": day, "gap": gap}
    return {"site": "messaging", "data": {"contacts": contacts, "groups": groups}, "request": request,
            "texts": texts, "facts": facts, "expect": expect, "label": LABEL["A"]}


# --------------------------------------------------------------------------- #
#  B: booking                                                                 #
# --------------------------------------------------------------------------- #


def build_B(rng: random.Random, n: int, gap: bool, kinds: list[str], person: tuple[str, str]) -> dict[str, Any]:
    cuisine, area = rng.choice(CUISINES)
    party = rng.choice([3, 4, 5])
    final_party = party + 2 if "correction" in kinds else party
    k = 3
    names = rng.sample(RESTAURANT_NAMES, n + 4)
    streets = rng.sample(STREETS[area], n)
    prices = sorted(rng.sample(range(22, 60), n))
    roles: list[str] = []
    if "correction" in kinds:
        roles.append("trap_party")   # room for the first party size only
    if "addition" in kinds:
        roles.append("trap_veg")     # room for the final party, too few vegetarian mains
    if "change" in kinds:
        roles.append("dropped")      # would be the answer, until the person rules it out
    roles.append("answer")
    full = n - len(roles)
    n_full_cheap = min(1, full)      # one full restaurant cheaper than everything
    restaurants = []
    i = 0
    for j in range(n_full_cheap):
        restaurants.append({"price": prices[i], "max_party": party - 1, "veg": rng.randint(3, 5), "role": "full"}); i += 1
    for role in roles:
        if role == "trap_party":
            r = {"max_party": party + 1, "veg": rng.randint(k, 5)}
        elif role == "trap_veg":
            r = {"max_party": final_party + rng.randint(0, 2), "veg": rng.randint(0, k - 1)}
        else:
            r = {"max_party": final_party + rng.randint(0, 4), "veg": rng.randint(k, 5)}
        restaurants.append({"price": prices[i], **r, "role": role}); i += 1
    while i < n:  # the rest: dearer than the answer, or with no room
        if rng.random() < 0.5:
            r = {"max_party": rng.randint(1, party - 1), "veg": rng.randint(0, 5), "role": "full"}
        else:
            r = {"max_party": rng.randint(party, final_party + 4), "veg": rng.randint(0, 5), "role": "dearer"}
        restaurants.append({"price": prices[i], **r}); i += 1
    rng.shuffle(restaurants)
    for j, r in enumerate(restaurants):
        r.update({"id": f"R{j + 1:02d}", "name": names[j], "cuisine": cuisine, "area": area,
                  "street": streets[j], "mains": rng.randint(7, 12)})
    # A few restaurants elsewhere, so the search has something to filter.
    for j in range(4):
        c2, a2 = rng.choice([c for c in CUISINES if c != (cuisine, area)])
        restaurants.append({"id": f"R{n + j + 1:02d}", "name": names[n + j], "cuisine": c2, "area": a2,
                            "street": rng.choice(STREETS[a2]), "price": rng.randint(22, 60), "max_party": 10,
                            "veg": 4, "mains": 9, "role": "elsewhere"})
    when = _date(rng, (3, 4, 5))
    weekday, date = when.strftime("%A"), when.day
    t = rng.choice([19 * 60, 19 * 60 + 30, 20 * 60])
    phone = f"07700 900{rng.randint(100, 999)}"
    friend = " ".join(_people(rng, 1, avoid={person[0]})[0])
    ok = [r for r in restaurants if r["role"] != "elsewhere" and r["max_party"] >= final_party
          and ("addition" not in kinds or r["veg"] >= k)]
    ok.sort(key=lambda r: r["price"])
    dropped = ok[0]["name"] if "change" in kinds else None
    answer = ok[1] if "change" in kinds else ok[0]
    assert answer["role"] == "answer", (answer, ok)
    who = f"{person[0]} {person[1]}"
    phone_txt = "" if gap else f" My number is {phone}."
    request = (f"Please find me a table for {party} at a {cuisine} restaurant in {area} on {_day_text(when)} at "
               f"{fmt_time(t)}. Check availability at every place the search finds, hold the cheapest one that has a "
               f"table for us, in my name ({who}), and then share the booking with {friend}.{phone_txt}")
    texts = {
        "status": "How's it going? Which places have you checked so far?",
        "status2": "How many places are left to check?",
        "correction": f"Sorry, change of plan: there'll be {party + 2} of us, not {party}.",
        "addition": f"Oh, and it needs decent vegetarian options: at least {k} vegetarian mains.",
        "change": f"Actually, rule out {dropped}; I had a bad experience there." if dropped else "",
    }
    facts = f"Your name is {who}. Your mobile number is {phone}."
    expect = {"family": "B", "party": party, "final_party": final_party, "veg_min": k if "addition" in kinds else 0,
              "answer_id": answer["id"], "answer": answer["name"], "dropped": dropped,
              "dropped_id": ok[0]["id"] if dropped else None, "phone": phone, "friend": friend, "name": who,
              "weekday": weekday.lower(), "date": date, "time": t, "gap": gap}
    data = {"restaurants": restaurants, "slot": {"weekday": weekday.lower(), "date": date, "time": t}}
    return {"site": "booking", "data": data, "request": request, "texts": texts, "facts": facts, "expect": expect,
            "label": LABEL["B"]}


# --------------------------------------------------------------------------- #
#  C: ordering                                                                #
# --------------------------------------------------------------------------- #


def build_C(rng: random.Random, n: int, gap: bool, kinds: list[str]) -> dict[str, Any]:
    types = rng.sample(PRODUCTS, n + 1)
    req, extra = types[:n], types[n]
    catalog = []
    best: dict[str, str] = {}
    pid = 100
    for label, pname, words in types:
        brands = rng.sample(BRANDS, 3)
        base = rng.randint(3, 40)
        prices = sorted({round(base * f + rng.random(), 2) for f in (1.0, 1.25, 1.6)})
        while len(prices) < 3:
            prices.append(round(prices[-1] + 1.5, 2))
        cheapest_out = rng.random() < 0.4
        opts = []
        for bi, (b, p) in enumerate(zip(brands, prices)):
            pid += 1
            opts.append({"id": f"P{pid}", "name": f"{pname}, {b}", "type": words, "price": p,
                         "in_stock": not (bi == 0 and cheapest_out)})
        rng.shuffle(opts)
        catalog.extend(opts)
        best[label] = min((o for o in opts if o["in_stock"]), key=lambda o: o["price"])["id"]
    qty = {label: rng.randint(1, 6) for label, _, _ in types}
    p_label = req[0][0]
    q2 = qty[p_label] + rng.choice([2, 3, 4])
    y_label = req[-1][0] if "change" in kinds else None
    po = f"PO-{rng.randint(1000, 9999)}"
    addr = f"{rng.randint(2, 90)} {rng.choice(UK_STREETS)}, London EC2A {rng.randint(1, 9)}{rng.choice('ABDEFGHJ')}{rng.choice('LNPQRSTU')}"
    lines = "; ".join(f"{qty[l]} x {l}" for l, _, _ in req)
    po_txt = "" if gap else f" Our purchase-order number is {po}."
    request = (f"Please order these for the office from the stationery shop: {lines}. For each one, pick the cheapest "
               f"option that's in stock. Then check out with delivery to {addr}, and tell me the order number and "
               f"the total.{po_txt}")
    texts = {
        "status": "How's it going? What's in the cart so far?",
        "status2": "How many items are left to add?",
        "correction": f"Make that {q2} x {p_label}, not {qty[p_label]}.",
        "addition": f"Also add {qty[extra[0]]} x {extra[0]}, please.",
        "change": f"Actually, drop the {y_label}; we have plenty." if y_label else "",
    }
    expected: dict[str, int] = {}
    for label, _, _ in req:
        if label == y_label:
            continue
        expected[best[label]] = q2 if (label == p_label and "correction" in kinds) else qty[label]
    if "addition" in kinds:
        expected[best[extra[0]]] = qty[extra[0]]
    facts = f"The purchase-order number for this order is {po}."
    expect = {"family": "C", "order": expected, "po": po, "address": addr, "p_item": best[p_label], "p_old": qty[p_label],
              "p_new": q2, "y_item": best[y_label] if y_label else None, "z_item": best[extra[0]],
              "best": best, "qty": qty, "gap": gap}
    return {"site": "shop", "data": {"catalog": catalog}, "request": request, "texts": texts, "facts": facts,
            "expect": expect, "label": LABEL["C"]}


def _filler(rng: random.Random, fam: str) -> dict[str, Any]:
    """Data for a site that plays no part in the instance, so its tools still work."""
    if fam == "A":
        return build_A(rng, 4, 0, False, [], ("Zed", "Q"))["data"]
    if fam == "B":
        return build_B(rng, 4, False, [], ("Zed", "Q"))["data"]
    return build_C(rng, 4, False, [])["data"]


def _build(fam: str, rng: random.Random, size: str, gap: bool, kinds: list[str], person: tuple[str, str]) -> dict[str, Any]:
    n = ITEMS[size][fam]
    if fam == "A":
        return build_A(rng, n, AWAY[size], gap, kinds, person)
    if fam == "B":
        return build_B(rng, n, gap, kinds, person)
    return build_C(rng, n, gap, kinds)


def _friend_in_contacts(rng: random.Random, messaging: dict[str, Any], bookings: list[dict[str, Any]]) -> None:
    """The friend a booking is shared with is in the address book, and nobody else there shares their first name."""
    for b in bookings:
        e = b["expect"]
        taken = {c["first_name"] for c in messaging["contacts"]} | {e["name"].split()[0]}
        first = e["friend"].split()[0]
        if first in taken:
            first = next(f for f in FIRST if f not in taken)
        friend = f"{first} {e['friend'].split()[1]}"
        b["request"] = b["request"].replace(e["friend"], friend)
        e["friend"] = friend
        messaging["contacts"].append({"name": friend, "first_name": first, "status": "active", "city": "London",
                                      "bio": rng.choice(BIOS)})


def make_instance(condition: str, fam: str, seed: int) -> dict[str, Any]:
    # One random stream per part, so a change to how one part is drawn never shifts another.
    def stream(part: str) -> random.Random:
        return random.Random(f"{condition}-{fam}-{seed}-{part}")
    rng = stream("main")
    person = (rng.choice(FIRST), rng.choice(LAST))
    script_spec = SCRIPTS[condition][fam]
    kinds = [k for k, _ in script_spec]
    size = "realistic" if condition == "dual" else condition
    gap = condition != "sanity"
    main = _build(fam, rng, size, gap, kinds, person)
    second = None
    if condition == "dual":
        sf = SECOND_OF[fam]
        second = _build(sf, stream("second"), "second", False, [], person)
        second["family"] = sf
    sites = {}
    for f, site in (("A", "messaging"), ("B", "booking"), ("C", "shop")):
        if main["site"] == site:
            sites[site] = main["data"]
        elif second and second["site"] == site:
            sites[site] = second["data"]
        else:
            sites[site] = _filler(stream(f"filler-{f}"), f)
    _friend_in_contacts(stream("friend"), sites["messaging"], [x for x in (main, second) if x and x["site"] == "booking"])
    rng = stream("script")
    min_gap, max_wait = GAPS[condition]
    unrelated_q, unrelated_a = rng.choice(UNRELATED)
    script = []
    for kind, trig in script_spec:
        if kind == "resume":
            trigger: dict[str, Any] = {"after_s": 20}
            text = rng.choice(["OK, all good, please carry on.", "Right, sorted. Carry on.", "OK, go ahead again."])
        else:
            if isinstance(trig, tuple):
                trigger = {**PROGRESS[second["family"]], "count": trig[1], "task": "second"}
            else:
                trigger = {**PROGRESS[fam], "count": trig, "task": "main"}
            if kind == "pause":
                text = rng.choice(["Hold on a moment, please pause. I need to check something.",
                                   "Wait, can you pause for a minute? I need to check something."])
            elif kind == "unrelated":
                text = unrelated_q
            elif kind == "second":
                r = second["request"]
                text = "Separately, could you also do this? " + r
            elif kind == "status_both":
                text = f"How are both things going, the {main['label']} and the {second['label']}?"
            elif kind == "cancel_second":
                text = f"Actually, cancel the {second['label']}; I'll sort it out myself. Keep going with the {main['label']}."
            else:
                text = main["texts"][kind]
        script.append({"kind": kind, "text": text, "trigger": trigger, "min_gap": min_gap, "max_wait": max_wait})
    facts = main["facts"]
    if second:
        facts += " " + second["facts"]
    return {
        "id": f"{condition}-{fam}-s{seed}", "condition": condition, "family": fam, "seed": seed,
        "latency": LATENCY[condition], "cap_s": CAPS[condition], "idle_s": 10,
        "sites": sites, "request": main["request"], "script": script, "facts": facts,
        "person_name": f"{person[0]} {person[1]}", "unrelated_answer": unrelated_a,
        "main": {"family": fam, "site": main["site"], "label": main["label"], "expect": main["expect"],
                 "progress": PROGRESS[fam]},
        "second": None if not second else {"family": second["family"], "site": second["site"],
                                           "label": second["label"], "expect": second["expect"],
                                           "progress": PROGRESS[second["family"]]},
    }


SEEDS = {"sanity": [0], "small": [0, 1, 2], "medium": [0, 1, 2], "realistic": [0, 1, 2, 3], "dual": [0, 1, 2]}


def generate(condition: str, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    return [make_instance(condition, fam, s) for s in (seeds if seeds is not None else SEEDS[condition]) for fam in "ABC"]
