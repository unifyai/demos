"""Correct solutions to every kind, written only against the API (any version).

They stand in for a perfect agent: every task of every stream must pass the
check when solved this way, which tests the API, the reference solutions and
the check together. `naive=True` drops pagination, to show that it matters.
"""

from __future__ import annotations

from typing import Any


class Scripted:
    def __init__(self, ops: dict[str, Any], naive: bool = False):
        self.o = ops
        self.naive = naive

    # shared steps
    def pages(self, fn, **kw) -> list[dict]:
        out, page = [], 1
        while True:
            r = fn(page=page, **kw)
            out += r["results"]
            if self.naive or page >= r["pages"]:
                return out
            page += 1

    def find_contacts(self, **kw) -> list[dict]:
        out, cursor = [], None
        while True:
            r = self.o["find_contacts"](cursor=cursor, **kw)
            out += r["items"]
            cursor = r["next_cursor"]
            if self.naive or cursor is None:
                return out

    def v2(self) -> bool:
        return "find_contacts" in self.o

    def contact_id(self, name: str) -> str:
        hits = self.find_contacts(name=name) if self.v2() else self.pages(self.o["search_contacts"], query=name)
        return next(h["id"] for h in hits if h["name"] == name)

    def company_id(self, name: str) -> str:
        return next(c["id"] for c in self.o["search_companies"](name) if c["name"] == name)

    def company_contacts(self, company_id: str) -> list[str]:
        if self.v2():
            return [c["id"] for c in self.find_contacts(company_id=company_id)]
        return [c["id"] for c in self.pages(self.o["list_company_contacts"], company_id=company_id)]

    def user_id(self, name: str) -> str:
        return next(u["id"] for u in self.o["list_users"]() if u["name"] == name)

    def send(self, to: str, subject: str, body: str) -> None:
        if "create_draft" in self.o:
            self.o["send_draft"](self.o["create_draft"](to, subject, body))
        else:
            self.o["send_email"](to, subject, body)

    def open_tickets(self, contact_id: str) -> list[dict]:
        return self.pages(self.o["search_tickets"], status="open", contact_id=contact_id)

    def spend(self, contact_id: str, since: str) -> int:
        orders = self.pages(self.o["list_orders"], contact_id=contact_id, status="delivered")
        return sum(o["total_cents"] for o in orders if o["placed_on"] >= since)

    def refund_latest(self, contact_id: str, reason: str) -> None:
        o = self.pages(self.o["list_orders"], contact_id=contact_id, status="delivered")[0]
        code = None
        if o["total_cents"] > 10000:
            code = self.o["request_refund_approval"](o["id"], o["total_cents"])
        self.o["refund_order"](o["id"], o["total_cents"], reason, approval_code=code)

    # kinds
    def solve(self, kind: str, a: dict[str, Any]) -> Any:
        o = self.o
        if kind == "contact_email":
            return o["get_contact"](self.contact_id(a["contact"]))["email"]
        if kind == "contact_company":
            return o["get_company"](o["get_contact"](self.contact_id(a["contact"]))["company_id"])["name"]
        if kind == "account_manager":
            mid = o["get_company"](self.company_id(a["company"]))["account_manager_id"]
            return next(u["name"] for u in o["list_users"]() if u["id"] == mid)
        if kind == "open_ticket_count":
            return len(self.open_tickets(self.contact_id(a["contact"])))
        if kind == "latest_order_status":
            return o["list_orders"](self.contact_id(a["contact"]))["results"][0]["status"]
        if kind == "company_spend":
            return sum(self.spend(c, a["since"]) for c in self.company_contacts(self.company_id(a["company"])))
        if kind == "agent_open_tickets":
            ts = [o["get_ticket"](t["id"]) for t in self.pages(o["search_tickets"], status="open", assignee_id=self.user_id(a["agent"]))]
            return [t["id"] for t in sorted(ts, key=lambda t: (t["priority"], t["created_on"]))]
        if kind == "email_contact":
            self.send(o["get_contact"](self.contact_id(a["contact"]))["email"], a["subject"], a["body"])
            return None
        if kind == "reassign_tickets":
            src, dst = self.user_id(a["from"]), self.user_id(a["to"])
            ids = [t["id"] for t in self.pages(o["search_tickets"], status="open", assignee_id=src)]
            for tid in ids:
                o["update_ticket"](tid, o["get_ticket"](tid)["version"], assignee_id=dst)
            return None
        if kind == "tag_company_city":
            for cid in self.company_contacts(self.company_id(a["company"])):
                if o["get_contact"](cid)["city"] == a["city"]:
                    o["tag_contact"](cid, a["tag"])
            return None
        if kind == "refund_latest":
            self.refund_latest(self.contact_id(a["contact"]), a["reason"])
            return None
        if kind == "email_open_ticket_contacts":
            for cid in self.company_contacts(self.company_id(a["company"])):
                if self.open_tickets(cid):
                    self.send(o["get_contact"](cid)["email"], a["subject"], a["body"])
            return None
        if kind == "escalate_urgent":
            uid = self.user_id(a["agent"])
            for cid in self.company_contacts(self.company_id(a["company"])):
                for t in self.open_tickets(cid):
                    if o["get_ticket"](t["id"])["priority"] == 1:
                        o["reply_to_ticket"](t["id"], a["body"])
                        o["update_ticket"](t["id"], o["get_ticket"](t["id"])["version"], assignee_id=uid)
            return None
        if kind == "refund_and_apologise":
            cid = self.contact_id(a["contact"])
            self.refund_latest(cid, a["reason"])
            self.send(o["get_contact"](cid)["email"], a["subject"], a["body"])
            return None
        if kind == "top_spender":
            cids = self.company_contacts(self.company_id(a["company"]))
            best = max(cids, key=lambda c: self.spend(c, a["since"]))
            c = o["get_contact"](best)
            return f"{c['name']} <{c['email']}>"
        if kind == "manager_digest":
            co = o["get_company"](self.company_id(a["company"]))
            ids = sorted(t["id"] for c in self.company_contacts(co["id"]) for t in self.open_tickets(c))
            email = next(u["email"] for u in o["list_users"]() if u["id"] == co["account_manager_id"])
            self.send(email, a["subject"], "\n".join(ids))
            return None
        raise KeyError(kind)
