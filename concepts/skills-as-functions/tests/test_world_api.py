import pytest

from skills.api import OPS_BY_SIZE, Api, ApiError, OpLimit, available
from skills.world import clone, generate, observable


def test_generate_is_deterministic_and_seeded():
    assert generate(1, "realistic") == generate(1, "realistic")
    assert generate(1, "realistic") != generate(2, "realistic")


@pytest.mark.parametrize("size", ["sanity", "small", "medium", "realistic"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_names_are_unique_and_never_inside_another(size, seed):
    w = generate(seed, size)
    names = [c["name"].lower() for c in w["contacts"].values()] + [u["name"].lower() for u in w["users"].values()]
    assert len(set(names)) == len(names)
    assert not any(a != b and a in b for a in names for b in names)
    cos = [c["name"].lower() for c in w["companies"].values()]
    assert not any(a != b and a in b for a in cos for b in cos)
    # each contact's orders have distinct dates, and tickets distinct creation times
    for cid in w["contacts"]:
        dates = [o["placed_on"] for o in w["orders"].values() if o["contact_id"] == cid]
        assert len(dates) == len(set(dates))
    created = [t["created_on"] for t in w["tickets"].values()]
    assert len(created) == len(set(created))


def test_some_companies_need_more_than_one_page():
    w = generate(1, "realistic")
    sizes = [sum(c["company_id"] == co for c in w["contacts"].values()) for co in w["companies"]]
    assert max(sizes) > 5 and min(sizes) >= 1


def test_search_and_list_paginate_in_v1():
    w = generate(1, "realistic")
    api = Api(w, 1, "realistic")
    f = api.functions()
    co = max(w["companies"], key=lambda c: sum(x["company_id"] == c for x in w["contacts"].values()))
    first = f["list_company_contacts"](co)
    assert first["page"] == 1 and first["pages"] >= 2 and len(first["results"]) == 5
    every = []
    for page in range(1, first["pages"] + 1):
        every += f["list_company_contacts"](co, page=page)["results"]
    assert {c["id"] for c in every} == {c["id"] for c in w["contacts"].values() if c["company_id"] == co}
    assert f["list_company_contacts"](co, page=first["pages"] + 1)["results"] == []
    name = next(iter(w["contacts"].values()))["name"]
    assert f["search_contacts"](name)["results"] == [{"id": next(iter(w["contacts"])), "name": name}]


def test_v2_replaces_contact_search_with_cursor_find():
    w = generate(1, "realistic")
    v1, v2 = Api(clone(w), 1, "realistic").functions(), Api(clone(w), 2, "realistic").functions()
    assert "find_contacts" not in v1
    with pytest.raises(ApiError, match="410 Gone"):
        v2["search_contacts"]("a")
    with pytest.raises(ApiError, match="410 Gone"):
        v2["list_company_contacts"]("co_101")
    got, cursor = [], None
    while True:
        r = v2["find_contacts"](company_id="co_101", cursor=cursor)
        got += r["items"]
        cursor = r["next_cursor"]
        if cursor is None:
            break
    assert [c["id"] for c in got] == [c["id"] for c in w["contacts"].values() if c["company_id"] == "co_101"]
    with pytest.raises(ApiError, match="400"):
        v2["find_contacts"]()
    with pytest.raises(ApiError, match="400"):
        v2["find_contacts"](name="a", cursor="garbage")


def test_v3_replaces_send_email_with_drafts():
    w = generate(1, "realistic")
    api = Api(w, 3, "realistic")
    f = api.functions()
    with pytest.raises(ApiError, match="410 Gone"):
        f["send_email"]("a@b.example", "s", "b")
    d = f["create_draft"]("a@b.example", "s", "b")
    assert w["outbox"] == []
    f["send_draft"](d)
    assert w["outbox"] == [{"to": "a@b.example", "subject": "s", "body": "b"}]
    with pytest.raises(ApiError, match="409"):
        f["send_draft"](d)
    assert available("send_email", 2) and not available("send_email", 3)


def test_reference_shows_only_current_operations():
    w = generate(1, "realistic")
    r1, r2, r3 = (Api(w, v, "realistic").reference() for v in (1, 2, 3))
    assert "search_contacts(" in r1 and "find_contacts(" not in r1
    assert "find_contacts(" in r2 and "search_contacts(" not in r2 and "send_email(" in r2
    assert "create_draft(" in r3 and "send_email(" not in r3


def test_operation_counts_per_size():
    counts = {s: {v: sum(available(n, v) for n in OPS_BY_SIZE[s]) for v in (1, 2, 3)} for s in OPS_BY_SIZE}
    assert counts["sanity"][1] == 9
    assert counts["small"][1] == 12 and counts["small"][2] == 11
    assert counts["medium"][1] == 16
    assert counts["realistic"][1] == 21


def test_refund_rules():
    w = generate(1, "realistic")
    f = Api(w, 1, "realistic").functions()
    big = next(o for o in w["orders"].values() if o["status"] == "delivered" and o["total_cents"] > 10000)
    small = next(o for o in w["orders"].values() if o["status"] == "delivered" and o["total_cents"] <= 10000)
    shipped = next(o for o in w["orders"].values() if o["status"] == "shipped")
    with pytest.raises(ApiError, match="409"):
        f["refund_order"](shipped["id"], shipped["total_cents"], "x")
    with pytest.raises(ApiError, match="403"):
        f["refund_order"](big["id"], big["total_cents"], "x")
    with pytest.raises(ApiError, match="403"):
        f["refund_order"](big["id"], big["total_cents"], "x", approval_code="APR-nope")
    code = f["request_refund_approval"](big["id"], big["total_cents"])
    f["refund_order"](big["id"], big["total_cents"], "x", approval_code=code)
    f["refund_order"](small["id"], small["total_cents"], "y")
    assert w["orders"][big["id"]]["status"] == "refunded" and len(w["refunds"]) == 2
    with pytest.raises(ApiError, match="409"):
        f["refund_order"](small["id"], small["total_cents"], "again")


def test_ticket_versions_and_replies():
    w = generate(1, "realistic")
    f = Api(w, 1, "realistic").functions()
    t = f["get_ticket"]("t_5001")
    f["reply_to_ticket"]("t_5001", "hello")
    with pytest.raises(ApiError, match="409 Conflict"):
        f["update_ticket"]("t_5001", t["version"], priority=2)
    t2 = f["update_ticket"]("t_5001", f["get_ticket"]("t_5001")["version"], priority=2)
    assert t2["priority"] == 2 and t2["version"] == t["version"] + 2
    manager = next(u for u in w["users"].values() if u["role"] == "account manager")
    with pytest.raises(ApiError, match="not a support agent"):
        f["update_ticket"]("t_5001", t2["version"], assignee_id=manager["id"])


def test_bad_arguments_are_400_and_results_are_copies():
    w = generate(1, "realistic")
    f = Api(w, 1, "realistic").functions()
    with pytest.raises(ApiError, match="400"):
        f["get_contact"]()
    with pytest.raises(ApiError, match="400"):
        f["search_contacts"]("a", pages=2)
    with pytest.raises(ApiError, match="404"):
        f["get_contact"]("c_0")
    c = f["get_contact"]("c_1001")
    c["tags"].append("hacked")
    assert "hacked" not in w["contacts"]["c_1001"]["tags"]


def test_call_limit_is_not_catchable_as_exception():
    w = generate(1, "sanity")
    api = Api(w, 1, "sanity", call_limit=3)
    f = api.functions()
    for _ in range(3):
        f["today"]()
    with pytest.raises(OpLimit):
        try:
            f["today"]()
        except Exception:  # noqa: BLE001
            pytest.fail("OpLimit was caught as an Exception")


def test_observable_ignores_versions_but_not_changes():
    w = generate(1, "realistic")
    a = clone(w)
    a["tickets"]["t_5001"]["version"] += 5
    assert observable(a) == observable(w)
    a["contacts"]["c_1001"]["tags"] = sorted(a["contacts"]["c_1001"]["tags"] + ["zzz"])
    assert observable(a) != observable(w)
