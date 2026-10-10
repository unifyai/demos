"""Three simulated sites: tools with state, a time per action, and a log.

Every action takes a time drawn from the instance seed, the tool, its
arguments and how many times that exact call has been made, so the same action
takes the same time in every arm. Each worker (the single loop, or one task's
work loop) has its own session on each site, which does one action at a time,
as a browser would. An action's effect applies when it ends, so an action that
is cancelled before it ends leaves no trace on the site.

Outcomes are read from the sites' state and the log, never from what the agent
says.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

# Seconds per action, by kind of action and by size.
PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "sanity": {"list": (0.3, 0.8), "read": (0.2, 0.5), "write": (0.3, 0.6), "big": (0.5, 0.8)},
    "small": {"list": (1.0, 2.0), "read": (0.5, 1.5), "write": (1.0, 2.0), "big": (2.0, 3.0)},
    "medium": {"list": (3.0, 6.0), "read": (2.0, 5.0), "write": (3.0, 6.0), "big": (6.0, 10.0)},
    "realistic": {"list": (6.0, 12.0), "read": (3.0, 8.0), "write": (5.0, 12.0), "big": (15.0, 30.0)},
}


class Clock:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def now(self) -> float:
        return round(time.monotonic() - self.t0, 3)


def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _tool(name: str, kind: str, description: str, params: dict[str, Any], required: list[str]):
    def deco(fn):
        fn._tool = {"name": name, "kind": kind, "description": description, "params": params, "required": required}
        return fn
    return deco


def _s(desc: str) -> dict[str, Any]:
    return {"type": "string", "description": desc}


def _i(desc: str) -> dict[str, Any]:
    return {"type": "integer", "description": desc}


# --------------------------------------------------------------------------- #
#  Parsing helpers shared with the checks                                     #
# --------------------------------------------------------------------------- #

_TIME = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?![\w])", re.I)


def times_in(text: str) -> set[int]:
    """Clock times mentioned in a text, as minutes after midnight. A bare number
    counts only with minutes or am/pm ("7:30", "7pm", "19:30"); "7:30" without
    am/pm counts as both 7:30 and 19:30."""
    out: set[int] = set()
    for m in _TIME.finditer(text or ""):
        h, mins, ampm = int(m.group(1)), m.group(2), m.group(3)
        if mins is None and ampm is None:
            continue
        mm = int(mins) if mins else 0
        if mm > 59 or h > 23:
            continue
        if ampm:
            if h > 12 or h == 0:
                continue
            pm = ampm.lower().startswith("p")
            out.add(((h % 12) + (12 if pm else 0)) * 60 + mm)
        else:
            out.add(h * 60 + mm)
            if h <= 12:
                out.add(((h % 12) + 12) * 60 + mm)
    return out


def fmt_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    ap = "pm" if h >= 12 else "am"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d}{ap}" if m else f"{h12}{ap}"


def digits(s: Any) -> str:
    return re.sub(r"\D", "", str(s or ""))


# --------------------------------------------------------------------------- #
#  Sites                                                                      #
# --------------------------------------------------------------------------- #


class Site:
    name = ""

    def tools(self) -> list[Callable]:
        return [getattr(self, a) for a in dir(self) if hasattr(getattr(self, a), "_tool")]

    def state(self) -> dict[str, Any]:
        raise NotImplementedError


class Messaging(Site):
    """Groups, profiles, an address book, and messages."""

    name = "messaging"

    def __init__(self, data: dict[str, Any]):
        self.contacts = {c["name"]: c for c in data["contacts"]}
        self.groups = data["groups"]
        self.sent: list[dict[str, Any]] = []

    def _resolve(self, who: str) -> tuple[str | None, str | None]:
        n = norm(who)
        exact = [c for c in self.contacts if norm(c) == n]
        if exact:
            return exact[0], None
        first = [c for c in self.contacts if norm(self.contacts[c]["first_name"]) == n]
        if len(first) == 1:
            return first[0], None
        if len(first) > 1:
            return None, f"More than one contact is called {who!r}: {', '.join(first)}. Use the full name."
        part = [c for c in self.contacts if n and n in norm(c)]
        if len(part) == 1:
            return part[0], None
        if len(part) > 1:
            return None, f"More than one contact matches {who!r}: {', '.join(part)}. Use the full name."
        return None, f"No contact called {who!r}."

    @_tool("messaging_list_group", "list", "Open one of your groups on the messaging site and list its members.",
           {"group": _s("The group's name.")}, ["group"])
    def list_group(self, group: str) -> dict[str, Any]:
        g = norm(group)
        for name, members in self.groups.items():
            if norm(name) == g or g in norm(name) or norm(name) in g:
                return {"group": name, "members": members}
        return {"error": f"No group called {group!r}. Your groups: {', '.join(self.groups)}."}

    @_tool("messaging_get_profile", "read", "Open a contact's profile page: their first name, whether they are away, their city and a short bio.",
           {"name": _s("The contact's full name.")}, ["name"])
    def get_profile(self, name: str) -> dict[str, Any]:
        who, err = self._resolve(name)
        if err:
            return {"error": err}
        c = self.contacts[who]
        return {"name": who, "first_name": c["first_name"], "status": c["status"], "city": c["city"], "bio": c["bio"]}

    @_tool("messaging_find_contact", "read", "Search your address book by name. Lists every contact whose name contains the text.",
           {"query": _s("A name or part of a name.")}, ["query"])
    def find_contact(self, query: str) -> dict[str, Any]:
        q = norm(query)
        hits = [c for c in self.contacts if q and (q in norm(c) or any(q == w for w in norm(c).split()))]
        return {"matches": [{"name": c, "city": self.contacts[c]["city"],
                             "groups": [g for g, ms in self.groups.items() if c in ms]} for c in hits]}

    @_tool("messaging_send_message", "write", "Send a message to one contact.",
           {"to": _s("The contact's full name."), "text": _s("The message.")}, ["to", "text"])
    def send_message(self, to: str, text: str) -> dict[str, Any]:
        who, err = self._resolve(to)
        if err:
            return {"error": err}
        if not str(text or "").strip():
            return {"error": "The message is empty."}
        mid = f"M{len(self.sent) + 1}"
        self.sent.append({"id": mid, "to": who, "text": str(text)})
        return {"sent": True, "message_id": mid, "to": who}

    @_tool("messaging_list_sent", "read", "List every message you have sent on the messaging site, oldest first.", {}, [])
    def list_sent(self) -> dict[str, Any]:
        return {"messages": [{"message_id": m["id"], "to": m["to"], "text": m["text"]} for m in self.sent]}

    def state(self) -> dict[str, Any]:
        return {"sent": self.sent}


_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class Booking(Site):
    """Restaurant search, availability, menus, holds and sharing a booking."""

    name = "booking"

    def __init__(self, data: dict[str, Any]):
        self.rest = {r["id"]: r for r in data["restaurants"]}
        self.slot = data["slot"]  # {"weekday": "friday", "date": 17, "time": 1170}
        self.holds: dict[str, dict[str, Any]] = {}
        self.shares: list[dict[str, Any]] = []

    def _find(self, rid: str) -> dict[str, Any] | None:
        r = self.rest.get(str(rid).strip().upper())
        if r:
            return r
        n = norm(rid)
        hits = [r for r in self.rest.values() if n and (norm(r["name"]) == n or n in norm(r["name"]))]
        return hits[0] if len(hits) == 1 else None

    def _slot_ok(self, day: str, time_: str) -> bool:
        d = norm(day)
        day_ok = self.slot["weekday"] in d or re.search(rf"\b{self.slot['date']}\b", d) is not None
        other_day = any(w in d for w in _DAYS if w != self.slot["weekday"])
        return day_ok and not other_day and self.slot["time"] in times_in(str(time_))

    @_tool("booking_search", "list", "Search the booking site for restaurants by cuisine and area. Lists each one's id, name, street and price per head.",
           {"cuisine": _s("For example 'Italian'."), "area": _s("For example 'Shoreditch'.")}, ["cuisine", "area"])
    def search(self, cuisine: str, area: str) -> dict[str, Any]:
        c, a = norm(cuisine), norm(area)
        hits = [r for r in self.rest.values() if c and a and (c in norm(r["cuisine"]) or norm(r["cuisine"]) in c)
                and (a in norm(r["area"]) or norm(r["area"]) in a)]
        return {"results": [{"restaurant_id": r["id"], "name": r["name"], "street": r["street"],
                             "price_per_head_gbp": r["price"]} for r in hits]}

    @_tool("booking_check_availability", "read", "Check whether a restaurant has a table for a party on a day and time.",
           {"restaurant_id": _s("The restaurant's id."), "day": _s("For example 'Friday 17 October'."),
            "time": _s("For example '7:30pm'."), "party_size": _i("How many people.")},
           ["restaurant_id", "day", "time", "party_size"])
    def check_availability(self, restaurant_id: str, day: str, time: str, party_size: int) -> dict[str, Any]:
        r = self._find(restaurant_id)
        if r is None:
            return {"error": f"No restaurant {restaurant_id!r}."}
        try:
            n = int(party_size)
        except (TypeError, ValueError):
            return {"error": "party_size must be a number."}
        ok = self._slot_ok(day, time) and n <= r["max_party"]
        return {"restaurant_id": r["id"], "name": r["name"], "day": day, "time": time, "party_size": n, "available": ok}

    @_tool("booking_get_menu", "read", "Open a restaurant's menu: how many mains it has and how many of them are vegetarian.",
           {"restaurant_id": _s("The restaurant's id.")}, ["restaurant_id"])
    def get_menu(self, restaurant_id: str) -> dict[str, Any]:
        r = self._find(restaurant_id)
        if r is None:
            return {"error": f"No restaurant {restaurant_id!r}."}
        return {"restaurant_id": r["id"], "name": r["name"], "mains": r["mains"], "vegetarian_mains": r["veg"]}

    @_tool("booking_hold_table", "write", "Hold a table at a restaurant. Returns a hold id.",
           {"restaurant_id": _s("The restaurant's id."), "day": _s("The day."), "time": _s("The time."),
            "party_size": _i("How many people."), "name": _s("The name the booking is under."),
            "phone": _s("A contact phone number for the booking.")},
           ["restaurant_id", "day", "time", "party_size", "name"])
    def hold_table(self, restaurant_id: str, day: str, time: str, party_size: int, name: str, phone: str = "") -> dict[str, Any]:
        r = self._find(restaurant_id)
        if r is None:
            return {"error": f"No restaurant {restaurant_id!r}."}
        try:
            n = int(party_size)
        except (TypeError, ValueError):
            return {"error": "party_size must be a number."}
        if not (self._slot_ok(day, time) and n <= r["max_party"]):
            return {"error": f"{r['name']} has no table for {n} then."}
        if len(digits(phone)) < 10:
            return {"error": "A contact phone number is required to hold a table."}
        hid = f"H{len(self.holds) + 1}"
        self.holds[hid] = {"hold_id": hid, "restaurant_id": r["id"], "restaurant": r["name"], "day": day, "time": time,
                           "party_size": n, "name": name, "phone": phone, "active": True}
        return {k: v for k, v in self.holds[hid].items() if k != "active"}

    @_tool("booking_cancel_hold", "write", "Cancel a table hold.", {"hold_id": _s("The hold id.")}, ["hold_id"])
    def cancel_hold(self, hold_id: str) -> dict[str, Any]:
        h = self.holds.get(str(hold_id).strip().upper())
        if not h or not h["active"]:
            return {"error": f"No active hold {hold_id!r}."}
        h["active"] = False
        return {"cancelled": h["hold_id"]}

    @_tool("booking_share", "write", "Share a held booking's details with someone.",
           {"hold_id": _s("The hold id."), "with_name": _s("Who to share it with.")}, ["hold_id", "with_name"])
    def share(self, hold_id: str, with_name: str) -> dict[str, Any]:
        h = self.holds.get(str(hold_id).strip().upper())
        if not h or not h["active"]:
            return {"error": f"No active hold {hold_id!r}."}
        self.shares.append({"hold_id": h["hold_id"], "with": str(with_name)})
        return {"shared": h["hold_id"], "with": with_name}

    def state(self) -> dict[str, Any]:
        return {"holds": list(self.holds.values()), "shares": self.shares}


_STOP = {"the", "and", "for", "with", "of", "box", "boxes", "pack", "packs", "set", "sets", "x", "a", "an", "to",
         "in", "on", "some", "please", "office"}


def _words(s: str) -> set[str]:
    return {w.rstrip("s") for w in norm(s).split() if len(w) >= 3 and w not in _STOP}


class Shop(Site):
    """A stationery shop: search, item pages, a cart and checkout."""

    name = "shop"

    def __init__(self, data: dict[str, Any]):
        self.items = {i["id"]: i for i in data["catalog"]}
        self.cart: dict[str, int] = {}
        self.orders: list[dict[str, Any]] = []

    def _cart_view(self) -> dict[str, Any]:
        lines = [{"item_id": k, "name": self.items[k]["name"], "quantity": q, "price_gbp": self.items[k]["price"]}
                 for k, q in self.cart.items()]
        return {"cart": lines, "total_gbp": round(sum(self.items[k]["price"] * q for k, q in self.cart.items()), 2)}

    def _item(self, item_id: str) -> dict[str, Any] | None:
        return self.items.get(str(item_id).strip().upper())

    @_tool("shop_search", "list", "Search the shop. Lists matching items with their ids and prices (stock is shown on each item's page).",
           {"query": _s("What to look for.")}, ["query"])
    def search(self, query: str) -> dict[str, Any]:
        q = _words(query)
        scored = []
        for it in self.items.values():
            s = len(q & _words(it["name"] + " " + it["type"]))
            if s:
                scored.append((-s, it["price"], it["id"]))
        scored.sort()
        return {"results": [{"item_id": self.items[i]["id"], "name": self.items[i]["name"],
                             "price_gbp": self.items[i]["price"]} for _, _, i in scored[:8]]}

    @_tool("shop_get_item", "read", "Open an item's page: its price, whether it is in stock, and delivery time.",
           {"item_id": _s("The item id.")}, ["item_id"])
    def get_item(self, item_id: str) -> dict[str, Any]:
        it = self._item(item_id)
        if not it:
            return {"error": f"No item {item_id!r}."}
        return {"item_id": it["id"], "name": it["name"], "price_gbp": it["price"], "in_stock": it["in_stock"],
                "delivery": "2 working days" if it["in_stock"] else "out of stock"}

    @_tool("shop_add_to_cart", "write", "Add an item to the cart.",
           {"item_id": _s("The item id."), "quantity": _i("How many.")}, ["item_id", "quantity"])
    def add_to_cart(self, item_id: str, quantity: int) -> dict[str, Any]:
        it = self._item(item_id)
        if not it:
            return {"error": f"No item {item_id!r}."}
        if not it["in_stock"]:
            return {"error": f"{it['name']} is out of stock."}
        try:
            q = int(quantity)
        except (TypeError, ValueError):
            return {"error": "quantity must be a number."}
        if q < 1:
            return {"error": "quantity must be at least 1."}
        self.cart[it["id"]] = self.cart.get(it["id"], 0) + q
        return self._cart_view()

    @_tool("shop_set_quantity", "write", "Change how many of an item are in the cart (0 removes it).",
           {"item_id": _s("The item id."), "quantity": _i("The new quantity.")}, ["item_id", "quantity"])
    def set_quantity(self, item_id: str, quantity: int) -> dict[str, Any]:
        it = self._item(item_id)
        if not it or it["id"] not in self.cart:
            return {"error": f"{item_id!r} is not in the cart."}
        try:
            q = int(quantity)
        except (TypeError, ValueError):
            return {"error": "quantity must be a number."}
        if q <= 0:
            del self.cart[it["id"]]
        else:
            self.cart[it["id"]] = q
        return self._cart_view()

    @_tool("shop_remove_from_cart", "write", "Remove an item from the cart.", {"item_id": _s("The item id.")}, ["item_id"])
    def remove_from_cart(self, item_id: str) -> dict[str, Any]:
        it = self._item(item_id)
        if not it or it["id"] not in self.cart:
            return {"error": f"{item_id!r} is not in the cart."}
        del self.cart[it["id"]]
        return self._cart_view()

    @_tool("shop_view_cart", "read", "Show the cart and its total.", {}, [])
    def view_cart(self) -> dict[str, Any]:
        return self._cart_view()

    @_tool("shop_checkout", "big", "Place the order for everything in the cart.",
           {"delivery_address": _s("Where to deliver."), "po_number": _s("The purchase-order number.")},
           ["delivery_address"])
    def checkout(self, delivery_address: str, po_number: str = "") -> dict[str, Any]:
        if not self.cart:
            return {"error": "The cart is empty."}
        if not re.fullmatch(r"\s*PO-?\s?\d{4}\s*", str(po_number or ""), re.I):
            return {"error": "A purchase-order number (like PO-1234) is required to check out."}
        view = self._cart_view()
        oid = f"ORD-{len(self.orders) + 1:03d}"
        self.orders.append({"order_id": oid, "lines": dict(self.cart), "total_gbp": view["total_gbp"],
                            "address": str(delivery_address), "po_number": str(po_number)})
        self.cart = {}
        return {"order_id": oid, "total_gbp": view["total_gbp"], "lines": view["cart"]}

    def state(self) -> dict[str, Any]:
        return {"cart": self.cart, "orders": self.orders}


SITE_CLASSES = {"messaging": Messaging, "booking": Booking, "shop": Shop}


class Env:
    """All three sites, one log, and per-worker sessions."""

    def __init__(self, inst: dict[str, Any], clock: Clock, speed: float = 1.0):
        self.inst = inst
        self.clock = clock
        self.speed = speed  # tests run the sites faster
        self.profile = PROFILES[inst["latency"]]
        self.sites: dict[str, Site] = {k: cls(inst["sites"][k]) for k, cls in SITE_CLASSES.items()}
        self.fns: dict[str, tuple[Site, Callable, dict[str, Any]]] = {}
        for site in self.sites.values():
            for fn in site.tools():
                self.fns[fn._tool["name"]] = (site, fn, fn._tool)
        self.log: list[dict[str, Any]] = []
        self.listeners: list[Callable[[str, dict[str, Any]], None]] = []
        self.locks: dict[tuple[str, str], asyncio.Lock] = defaultdict(asyncio.Lock)
        self.seen_calls: dict[str, int] = defaultdict(int)
        self.inflight = 0

    def schemas(self) -> list[dict[str, Any]]:
        out = []
        for name, (_, _, t) in sorted(self.fns.items()):
            out.append({"type": "function", "function": {
                "name": name, "description": t["description"],
                "parameters": {"type": "object", "properties": t["params"], "required": t["required"]}}})
        return out

    def has(self, name: str) -> bool:
        return name in self.fns

    def _latency(self, name: str, args: dict[str, Any], kind: str) -> float:
        key = json.dumps([name, args], sort_keys=True, default=str)
        self.seen_calls[key] += 1
        h = hashlib.sha256(f"{self.inst['seed']}|{key}|{self.seen_calls[key]}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        lo, hi = self.profile[kind]
        return (lo + (hi - lo) * u) * self.speed

    async def call(self, worker: str, name: str, arguments: str | dict[str, Any] | None,
                   gate: Callable[[], str | None] | None = None) -> str:
        """Run one tool call for a worker and return its result as text.

        A worker's session does one action at a time, so the calls of one
        response queue. `gate` is asked when an action's turn comes, before it
        starts: if it gives a reason (a new message, a pause, a stop), the
        action is not started and its result says why."""
        if name not in self.fns:
            return json.dumps({"error": f"No tool {name!r}."})
        site, fn, t = self.fns[name]
        try:
            args = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except (ValueError, json.JSONDecodeError):
            return json.dumps({"error": "The arguments are not a JSON object."})
        missing = [p for p in t["required"] if args.get(p) in (None, "")]
        unknown = [p for p in args if p not in t["params"]]
        if missing or unknown:
            return json.dumps({"error": f"Missing arguments {missing}" if missing else f"Unknown arguments {unknown}"})
        rec: dict[str, Any] = {"i": len(self.log), "site": site.name, "tool": name, "kind": t["kind"], "args": args,
                               "worker": worker, "t_queued": self.clock.now(), "t_start": None, "t_end": None,
                               "result": None, "ok": None, "cancelled": False, "skipped": None}
        self.log.append(rec)
        self.inflight += 1
        try:
            async with self.locks[(site.name, worker)]:
                reason = gate() if gate else None
                if reason:
                    rec["skipped"] = reason
                    return json.dumps({"not_started": f"This action was not started: {reason} before its turn "
                                                      f"came. Call it again if it is still needed."})
                rec["t_start"] = self.clock.now()
                for cb in self.listeners:
                    cb("start", rec)
                await asyncio.sleep(self._latency(name, args, t["kind"]))
                out = fn(**args)
                rec["t_end"] = self.clock.now()
                rec["result"] = out
                rec["ok"] = "error" not in out
                for cb in self.listeners:
                    cb("end", rec)
                return json.dumps(out)
        except asyncio.CancelledError:
            rec["cancelled"] = True
            raise
        finally:
            self.inflight -= 1

    def state(self) -> dict[str, Any]:
        return {k: s.state() for k, s in self.sites.items()}
