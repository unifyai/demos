"""The workspace's API: the operations an agent's code calls, in three versions.

v1 is the starting API. v2 replaces search_contacts and list_company_contacts
with one cursor-paginated find_contacts. v3 replaces send_email with
create_draft followed by send_draft. A removed operation is still callable
and fails with 410 Gone, as a retired endpoint does. Which operations a size
exposes is fixed per size (OPS_BY_SIZE).

Every operation returns fresh copies, so code cannot change the world except
through the API, and raises ApiError with a message that says why it failed.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Callable

PAGE = 5
REFUND_APPROVAL_OVER = 10000


class ApiError(Exception):
    pass


class OpLimit(BaseException):
    """Too many API calls in one task (a runaway loop). Not catchable as Exception."""


# name -> (first version, last version or None)
VERSIONS: dict[str, tuple[int, int | None]] = {
    "search_contacts": (1, 1),
    "list_company_contacts": (1, 1),
    "find_contacts": (2, None),
    "send_email": (1, 2),
    "create_draft": (3, None),
    "send_draft": (3, None),
}

OPS_BY_SIZE: dict[str, list[str]] = {
    "sanity": ["today", "search_contacts", "list_company_contacts", "get_contact", "search_companies",
               "get_company", "list_users", "list_orders", "get_order"],
    "small": ["today", "search_contacts", "list_company_contacts", "find_contacts", "get_contact",
              "tag_contact", "create_contact", "search_companies", "get_company", "list_users",
              "search_tickets", "get_ticket", "send_email"],
    "medium": ["today", "search_contacts", "list_company_contacts", "find_contacts", "get_contact",
               "tag_contact", "create_contact", "search_companies", "get_company", "list_users",
               "get_user", "search_tickets", "get_ticket", "update_ticket", "reply_to_ticket",
               "create_ticket", "send_email", "create_draft", "send_draft"],
    "realistic": ["today", "search_contacts", "list_company_contacts", "find_contacts", "get_contact",
                  "tag_contact", "create_contact", "search_companies", "get_company", "list_users",
                  "get_user", "search_tickets", "get_ticket", "update_ticket", "reply_to_ticket",
                  "create_ticket", "list_orders", "get_order", "request_refund_approval",
                  "refund_order", "cancel_order", "send_email", "create_draft", "send_draft"],
}


# Operations that change nothing (looking is not doing).
READ_ONLY = {"today", "search_contacts", "list_company_contacts", "find_contacts", "get_contact",
             "search_companies", "get_company", "list_users", "get_user", "search_tickets", "get_ticket",
             "list_orders", "get_order"}


def available(name: str, version: int) -> bool:
    lo, hi = VERSIONS.get(name, (1, None))
    return version >= lo and (hi is None or version <= hi)


class Api:
    def __init__(self, world: dict[str, Any], version: int, size: str, call_limit: int = 600):
        self.w = world
        self.version = version
        self.size = size
        self.calls = 0
        self.call_limit = call_limit
        self.log: list[str] = []

    # -- plumbing -----------------------------------------------------------
    def _tick(self, name: str) -> None:
        self.calls += 1
        self.log.append(name)
        if self.calls > self.call_limit:
            raise OpLimit(f"More than {self.call_limit} API calls in one task.")

    @staticmethod
    def _out(x: Any) -> Any:
        return copy.deepcopy(x)

    @staticmethod
    def _page(items: list[Any], page: Any) -> dict[str, Any]:
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ApiError(f"400 Bad Request: page must be a positive integer, not {page!r}")
        pages = max(1, -(-len(items) // PAGE))
        return {"results": items[(page - 1) * PAGE: page * PAGE], "page": page, "pages": pages}

    def _contact(self, contact_id: Any) -> dict[str, Any]:
        c = self.w["contacts"].get(contact_id)
        if c is None:
            raise ApiError(f"404 Not Found: no contact with id {contact_id!r}")
        return c

    def _company(self, company_id: Any) -> dict[str, Any]:
        c = self.w["companies"].get(company_id)
        if c is None:
            raise ApiError(f"404 Not Found: no company with id {company_id!r}")
        return c

    def _ticket(self, ticket_id: Any) -> dict[str, Any]:
        t = self.w["tickets"].get(ticket_id)
        if t is None:
            raise ApiError(f"404 Not Found: no ticket with id {ticket_id!r}")
        return t

    def _order(self, order_id: Any) -> dict[str, Any]:
        o = self.w["orders"].get(order_id)
        if o is None:
            raise ApiError(f"404 Not Found: no order with id {order_id!r}")
        return o

    @staticmethod
    def _text(name: str, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ApiError(f"400 Bad Request: {name} must be a non-empty string")
        return value

    def functions(self) -> dict[str, Callable[..., Any]]:
        """The operations as plain functions, for an agent's namespace. Operations
        removed in this version are present and fail with 410 Gone."""
        out = {}
        for name in OPS_BY_SIZE[self.size]:
            lo, _ = VERSIONS.get(name, (1, None))
            if self.version < lo:
                continue  # not released yet: not in the namespace
            out[name] = self._wrap(name)
        return out

    def _wrap(self, name: str) -> Callable[..., Any]:
        method = getattr(self, name)
        api = self

        def op(*args: Any, **kwargs: Any) -> Any:
            api._tick(name)
            if not available(name, api.version):
                raise ApiError(f"410 Gone: {name}() was removed in API v{api.version}. See the API reference.")
            try:
                inspect.signature(method).bind(*args, **kwargs)
            except TypeError as e:
                raise ApiError(f"400 Bad Request: {name}(): {e}") from None
            return method(*args, **kwargs)

        op.__name__ = name
        op.__doc__ = method.__doc__
        op.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
        return op

    def reference(self) -> str:
        """The API reference for this version and size, as the agent sees it."""
        lines = []
        for name in OPS_BY_SIZE[self.size]:
            if not available(name, self.version):
                continue
            method = getattr(self, name)
            sig = str(inspect.signature(method)).replace("'", "")  # annotations are strings here
            doc = inspect.getdoc(method) or ""
            lines.append(f"{name}{sig}\n" + "\n".join("    " + d for d in doc.splitlines()))
        return "\n".join(lines)

    # -- general ----------------------------------------------------------------
    def today(self) -> str:
        """Today's date in the workspace, as "YYYY-MM-DD"."""
        return self.w["today"]

    # -- contacts and companies ----------------------------------------------------
    def search_contacts(self, query: str, page: int = 1) -> dict:
        """Contacts whose full name contains query (case-insensitive), 5 per page.
        Returns {"results": [{"id", "name"}], "page": int, "pages": int}."""
        q = self._text("query", query).strip().lower()
        hits = [{"id": c["id"], "name": c["name"]} for c in self.w["contacts"].values() if q in c["name"].lower()]
        return self._out(self._page(hits, page))

    def list_company_contacts(self, company_id: str, page: int = 1) -> dict:
        """A company's contacts, 5 per page.
        Returns {"results": [{"id", "name"}], "page": int, "pages": int}."""
        self._company(company_id)
        hits = [{"id": c["id"], "name": c["name"]} for c in self.w["contacts"].values() if c["company_id"] == company_id]
        return self._out(self._page(hits, page))

    def find_contacts(self, name: str | None = None, company_id: str | None = None, cursor: str | None = None) -> dict:
        """Contacts matching every filter given (name: the full name contains it,
        case-insensitive; company_id: works there), 5 at a time.
        Returns {"items": [{"id", "name", "company_id"}], "next_cursor": str or None};
        pass next_cursor back as cursor to get the next 5."""
        if name is None and company_id is None:
            raise ApiError("400 Bad Request: give name or company_id")
        if company_id is not None:
            self._company(company_id)
        q = self._text("name", name).strip().lower() if name is not None else None
        hits = [
            {"id": c["id"], "name": c["name"], "company_id": c["company_id"]}
            for c in self.w["contacts"].values()
            if (q is None or q in c["name"].lower()) and (company_id is None or c["company_id"] == company_id)
        ]
        start = 0
        if cursor is not None:
            if not isinstance(cursor, str) or not cursor.startswith("cur_") or not cursor[4:].isdigit():
                raise ApiError(f"400 Bad Request: invalid cursor {cursor!r}")
            start = int(cursor[4:])
        nxt = start + PAGE
        return self._out({"items": hits[start:nxt], "next_cursor": f"cur_{nxt}" if nxt < len(hits) else None})

    def get_contact(self, contact_id: str) -> dict:
        """Returns {"id", "name", "company_id", "title", "email", "other_emails",
        "city", "tags", "created_on"}. email is the contact's primary address."""
        return self._out(self._contact(contact_id))

    def tag_contact(self, contact_id: str, tag: str) -> None:
        """Adds a tag to a contact (no change if it already has it)."""
        c = self._contact(contact_id)
        tag = self._text("tag", tag)
        if tag not in c["tags"]:
            c["tags"] = sorted(c["tags"] + [tag])

    def create_contact(self, name: str, company_id: str, email: str) -> str:
        """Creates a contact and returns its id."""
        self._company(company_id)
        self.w["created"]["contacts"] += 1
        cid = f"c_{9001 + self.w['created']['contacts']}"
        self.w["contacts"][cid] = {
            "id": cid, "name": self._text("name", name), "company_id": company_id, "title": "",
            "email": self._text("email", email), "other_emails": [], "city": "", "tags": [],
            "created_on": self.w["today"],
        }
        return cid

    def search_companies(self, query: str) -> list:
        """Companies whose name contains query (case-insensitive): [{"id", "name"}]."""
        q = self._text("query", query).strip().lower()
        return self._out([{"id": c["id"], "name": c["name"]} for c in self.w["companies"].values() if q in c["name"].lower()])

    def get_company(self, company_id: str) -> dict:
        """Returns {"id", "name", "city", "industry", "account_manager_id"}."""
        return self._out(self._company(company_id))

    def list_users(self) -> list:
        """The workspace's users (account managers and support agents):
        [{"id", "name", "email", "role"}]."""
        return self._out(list(self.w["users"].values()))

    def get_user(self, user_id: str) -> dict:
        """Returns {"id", "name", "email", "role"}."""
        u = self.w["users"].get(user_id)
        if u is None:
            raise ApiError(f"404 Not Found: no user with id {user_id!r}")
        return self._out(u)

    # -- tickets ----------------------------------------------------------------
    def search_tickets(self, status: str | None = None, assignee_id: str | None = None,
                       contact_id: str | None = None, page: int = 1) -> dict:
        """Tickets matching every filter given, oldest first, 5 per page.
        status is one of "open", "pending", "solved", "closed".
        Returns {"results": [{"id", "subject", "status"}], "page": int, "pages": int}."""
        if status is not None and status not in ("open", "pending", "solved", "closed"):
            raise ApiError(f"400 Bad Request: status must be one of open, pending, solved, closed, not {status!r}")
        ts = sorted(self.w["tickets"].values(), key=lambda t: (t["created_on"], t["id"]))
        hits = [
            {"id": t["id"], "subject": t["subject"], "status": t["status"]}
            for t in ts
            if (status is None or t["status"] == status)
            and (assignee_id is None or t["assignee_id"] == assignee_id)
            and (contact_id is None or t["contact_id"] == contact_id)
        ]
        return self._out(self._page(hits, page))

    def get_ticket(self, ticket_id: str) -> dict:
        """Returns {"id", "subject", "status", "priority", "assignee_id", "contact_id",
        "created_on", "last_reply_on", "tags", "version"}. priority is 1 (urgent) to 4 (low)."""
        return self._out(self._ticket(ticket_id))

    def update_ticket(self, ticket_id: str, version: int, status: str | None = None,
                      assignee_id: str | None = None, priority: int | None = None) -> dict:
        """Changes the fields given and returns the updated ticket. version must be
        the ticket's current version."""
        t = self._ticket(ticket_id)
        if version != t["version"]:
            raise ApiError(f"409 Conflict: ticket {ticket_id} is at version {t['version']}, not {version!r}; get it again")
        if status is not None and status not in ("open", "pending", "solved", "closed"):
            raise ApiError(f"400 Bad Request: status must be one of open, pending, solved, closed, not {status!r}")
        if assignee_id is not None and (assignee_id not in self.w["users"] or self.w["users"][assignee_id]["role"] != "support agent"):
            raise ApiError(f"400 Bad Request: {assignee_id!r} is not a support agent's user id")
        if priority is not None and priority not in (1, 2, 3, 4):
            raise ApiError(f"400 Bad Request: priority must be 1, 2, 3 or 4, not {priority!r}")
        if status is not None:
            t["status"] = status
        if assignee_id is not None:
            t["assignee_id"] = assignee_id
        if priority is not None:
            t["priority"] = priority
        t["version"] += 1
        return self._out(t)

    def reply_to_ticket(self, ticket_id: str, body: str) -> None:
        """Posts a public reply on a ticket."""
        t = self._ticket(ticket_id)
        self.w["replies"].append({"ticket_id": ticket_id, "body": self._text("body", body), "on": self.w["today"]})
        t["last_reply_on"] = self.w["today"]
        t["version"] += 1

    def create_ticket(self, contact_id: str, subject: str) -> str:
        """Opens a new ticket for a contact and returns its id."""
        self._contact(contact_id)
        self.w["created"]["tickets"] += 1
        tid = f"t_{8001 + self.w['created']['tickets']}"
        agents = [u for u in self.w["users"].values() if u["role"] == "support agent"]
        self.w["tickets"][tid] = {
            "id": tid, "subject": self._text("subject", subject), "status": "open", "priority": 3,
            "assignee_id": agents[0]["id"], "contact_id": contact_id,
            "created_on": self.w["today"] + " 09:00", "last_reply_on": self.w["today"], "tags": [], "version": 1,
        }
        return tid

    # -- orders -------------------------------------------------------------------
    def list_orders(self, contact_id: str, status: str | None = None, page: int = 1) -> dict:
        """A contact's orders, newest first, 5 per page. status is one of "pending",
        "paid", "shipped", "delivered", "cancelled", "refunded".
        Returns {"results": [{"id", "status", "total_cents", "placed_on"}], "page": int, "pages": int}."""
        self._contact(contact_id)
        if status is not None and status not in ("pending", "paid", "shipped", "delivered", "cancelled", "refunded"):
            raise ApiError(f"400 Bad Request: unknown order status {status!r}")
        os_ = sorted((o for o in self.w["orders"].values() if o["contact_id"] == contact_id),
                     key=lambda o: (o["placed_on"], o["id"]), reverse=True)
        hits = [{"id": o["id"], "status": o["status"], "total_cents": o["total_cents"], "placed_on": o["placed_on"]}
                for o in os_ if status is None or o["status"] == status]
        return self._out(self._page(hits, page))

    def get_order(self, order_id: str) -> dict:
        """Returns {"id", "contact_id", "status", "total_cents", "placed_on",
        "items": [{"sku", "qty", "price_cents"}]}."""
        return self._out(self._order(order_id))

    def request_refund_approval(self, order_id: str, amount_cents: int) -> str:
        """Returns an approval code for refunding amount_cents of an order."""
        o = self._order(order_id)
        if not isinstance(amount_cents, int) or isinstance(amount_cents, bool) or not 0 < amount_cents <= o["total_cents"]:
            raise ApiError(f"400 Bad Request: amount_cents must be an integer from 1 to {o['total_cents']}")
        code = f"APR-{order_id[2:]}-{amount_cents}"
        self.w["approvals"][code] = (order_id, amount_cents)
        return code

    def refund_order(self, order_id: str, amount_cents: int, reason: str, approval_code: str | None = None) -> None:
        """Refunds a delivered order; its status becomes "refunded"."""
        o = self._order(order_id)
        if o["status"] != "delivered":
            raise ApiError(f"409 Conflict: only delivered orders can be refunded; {order_id} is {o['status']}")
        if not isinstance(amount_cents, int) or isinstance(amount_cents, bool) or not 0 < amount_cents <= o["total_cents"]:
            raise ApiError(f"400 Bad Request: amount_cents must be an integer from 1 to {o['total_cents']}")
        reason = self._text("reason", reason)
        if amount_cents > REFUND_APPROVAL_OVER:
            if approval_code is None:
                raise ApiError(f"403 Forbidden: refunds over {REFUND_APPROVAL_OVER} cents need an approval_code "
                               "from request_refund_approval(order_id, amount_cents)")
            if self.w["approvals"].get(approval_code) != (order_id, amount_cents):
                raise ApiError(f"403 Forbidden: approval_code {approval_code!r} is not valid for this refund")
        o["status"] = "refunded"
        self.w["refunds"].append({"order_id": order_id, "amount_cents": amount_cents, "reason": reason})

    def cancel_order(self, order_id: str, reason: str) -> None:
        """Cancels a pending or paid order."""
        o = self._order(order_id)
        if o["status"] not in ("pending", "paid"):
            raise ApiError(f"409 Conflict: only pending or paid orders can be cancelled; {order_id} is {o['status']}")
        self._text("reason", reason)
        o["status"] = "cancelled"

    # -- email --------------------------------------------------------------------
    def send_email(self, to: str, subject: str, body: str) -> None:
        """Sends an email."""
        self._send(to, subject, body)

    def create_draft(self, to: str, subject: str, body: str) -> str:
        """Creates an email draft and returns its id. Nothing is sent until send_draft."""
        to, subject, body = self._check_mail(to, subject, body)
        did = f"d_{len(self.w['drafts']) + 1}"
        self.w["drafts"][did] = {"to": to, "subject": subject, "body": body, "sent": False}
        return did

    def send_draft(self, draft_id: str) -> None:
        """Sends a draft made by create_draft."""
        d = self.w["drafts"].get(draft_id)
        if d is None:
            raise ApiError(f"404 Not Found: no draft with id {draft_id!r}")
        if d["sent"]:
            raise ApiError(f"409 Conflict: draft {draft_id} was already sent")
        d["sent"] = True
        self.w["outbox"].append({"to": d["to"], "subject": d["subject"], "body": d["body"]})

    def _check_mail(self, to: Any, subject: Any, body: Any) -> tuple[str, str, str]:
        to = self._text("to", to)
        if "@" not in to or " " in to.strip():
            raise ApiError(f"400 Bad Request: {to!r} is not an email address")
        return to.strip(), self._text("subject", subject), self._text("body", body)

    def _send(self, to: Any, subject: Any, body: Any) -> None:
        to, subject, body = self._check_mail(to, subject, body)
        self.w["outbox"].append({"to": to, "subject": subject, "body": body})
