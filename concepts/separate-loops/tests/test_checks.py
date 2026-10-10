"""The measures, on synthetic records built from real instances."""

from __future__ import annotations

from separate_loops.checks import evaluate
from separate_loops.scenarios import make_instance
from separate_loops.sites import fmt_time


def _act(i, site, tool, args, t, result, worker="flat", kind="write", dur=1.0):
    return {"i": i, "site": site, "tool": tool, "kind": kind, "args": args, "worker": worker, "t_queued": t,
            "t_start": t, "t_end": t + dur, "result": result, "ok": "error" not in result, "cancelled": False,
            "skipped": None}


def _rec(inst, arm, person, agent, actions, state, calls=(), tasks=()):
    return {"run_id": f"x|{inst['id']}|{arm}", "instance": inst["id"], "condition": inst["condition"],
            "family": inst["family"], "seed": inst["seed"], "arm": arm, "model": "m", "end": {"t": 500, "reason": "idle"},
            "person": person, "agent": agent, "calls": list(calls), "tasks": list(tasks), "actions": actions,
            "state": state, "errors": []}


def _person(inst, times):
    out = [{"id": 0, "kind": "request", "text": inst["request"], "t": 0.0, "how": None}]
    for i, (s, t) in enumerate(zip(inst["script"], times)):
        out.append({"id": i + 1, "kind": s["kind"], "text": s["text"], "t": t, "how": "progress"})
    return out


def _empty_state():
    return {"messaging": {"sent": []}, "booking": {"holds": [], "shares": []}, "shop": {"cart": {}, "orders": []}}


def test_invitations_right_time_dish_drop_and_stale():
    inst = make_instance("realistic", "A", 0)
    e = inst["main"]["expect"]
    old, new = fmt_time(e["old_time"]), fmt_time(e["new_time"])
    # status 10, correction 20, addition 30, pause 40, resume 60, unrelated 70, status2 80, change 90
    person = _person(inst, [10, 20, 30, 40, 60, 70, 80, 90])
    person.append({"id": 9, "kind": "answer", "text": "Sam Rivera.", "t": 15.0, "how": "responder"})
    active = [m for m in e["members"] if m not in e["away"] and m != e["drop"]]
    acts, sent, t = [], [], 0.0
    for k, m in enumerate(active + ["Sam Rivera"]):
        t += 2
        text = f"Hi, {old}" if t < 25 else f"Hi, {new}, bring a dish"
        acts.append(_act(len(acts), "messaging", "messaging_send_message", {"to": m, "text": text}, t,
                         {"sent": True, "to": m}))
        sent.append({"to": m, "text": text})
    # corrections for those who got the old time, after the correction
    for m in [s["to"] for s in sent if old in s["text"] and new not in s["text"]]:
        t += 2
        acts.append(_act(len(acts), "messaging", "messaging_send_message", {"to": m, "text": f"Now {new}; bring a dish"}, t,
                         {"sent": True, "to": m}))
        sent.append({"to": m, "text": f"Now {new}; bring a dish"})
    st = _empty_state()
    st["messaging"]["sent"] = sent
    ev = evaluate(_rec(inst, "flat", person, [], acts, st), inst)
    assert ev["checks"] == {"invited_all": True, "no_wrong_recipients": True, "right_time": True, "dish_for_all": True}
    assert ev["correct"]
    # Sends at 22 and 24 (after the correction at 20) still said the old time.
    assert ev["stale"]["correction"] == 2 and ev["stale"]["change"] == 0
    assert ev["gap"]["guessed"] is False  # Sam was messaged after the person said which one
    # Messaging the dropped member after the change, and the wrong Sam, fail.
    acts.append(_act(len(acts), "messaging", "messaging_send_message", {"to": e["drop"], "text": new}, 95,
                     {"sent": True, "to": e["drop"]}))
    acts.append(_act(len(acts), "messaging", "messaging_send_message", {"to": "Sam Oyelaran", "text": new}, 96,
                     {"sent": True, "to": "Sam Oyelaran"}))
    ev = evaluate(_rec(inst, "flat", person, [], acts, st), inst)
    assert not ev["checks"]["no_wrong_recipients"] and ev["stale"]["change"] == 1
    assert ev["gap"]["wrong_reached_site"]


def test_booking_answer_party_phone_and_stale_checks():
    inst = make_instance("realistic", "B", 1)
    e = inst["main"]["expect"]
    person = _person(inst, [10, 20, 30, 40, 60, 70, 80, 90])
    person.append({"id": 9, "kind": "answer", "text": f"It's {e['phone']}.", "t": 100.0, "how": "responder"})
    day, t = f"{e['weekday']} {e['date']}", fmt_time(e["time"])
    acts = [_act(0, "booking", "booking_check_availability", {"restaurant_id": "R01", "day": day, "time": t, "party_size": e["party"]}, 25, {"available": True}, kind="read"),
            _act(1, "booking", "booking_check_availability", {"restaurant_id": "R02", "day": day, "time": t, "party_size": e["final_party"]}, 26, {"available": True}, kind="read"),
            _act(2, "booking", "booking_hold_table", {"restaurant_id": e["answer_id"], "day": day, "time": t, "party_size": e["final_party"], "name": e["name"]}, 95, {"error": "phone"}),
            _act(3, "booking", "booking_hold_table", {"restaurant_id": e["answer_id"], "day": day, "time": t, "party_size": e["final_party"], "name": e["name"], "phone": e["phone"]}, 105,
                 {"hold_id": "H1", "restaurant_id": e["answer_id"]}),
            _act(4, "booking", "booking_check_availability", {"restaurant_id": e["dropped_id"], "day": day, "time": t, "party_size": e["final_party"]}, 92, {"available": True}, kind="read")]
    st = _empty_state()
    st["booking"]["holds"] = [{"hold_id": "H1", "restaurant_id": e["answer_id"], "party_size": e["final_party"], "phone": e["phone"], "active": True}]
    st["booking"]["shares"] = [{"hold_id": "H1", "with": e["friend"]}]
    agent = [{"i": 0, "t": 110.0, "text": f"Held {e['answer']} for {e['final_party']}.", "loop": "flat", "seen": 9}]
    ev = evaluate(_rec(inst, "flat", person, agent, acts, st), inst)
    assert ev["correct"] and ev["stale"] == {"correction": 1, "addition": 0, "change": 1}
    assert ev["gap"] == {"guessed": False, "wrong_reached_site": False, "fact_given": True, "fact_t": 100.0}
    assert ev["reported_auto"]
    # A made-up phone is a guess, and a second active hold fails the check.
    acts.append(_act(5, "booking", "booking_hold_table", {"restaurant_id": "R03", "day": day, "time": t, "party_size": 9, "name": "x", "phone": "07123 456789"}, 50,
                     {"hold_id": "H2", "restaurant_id": "R03"}))
    st["booking"]["holds"].append({"hold_id": "H2", "restaurant_id": "R03", "party_size": 9, "phone": "07123 456789", "active": True})
    ev = evaluate(_rec(inst, "flat", person, agent, acts, st), inst)
    assert ev["gap"]["guessed"] and ev["gap"]["wrong_reached_site"] and not ev["checks"]["one_hold"]


def test_order_items_and_stale_quantity_and_po():
    inst = make_instance("medium", "C", 0)
    e = inst["main"]["expect"]
    person = _person(inst, [10, 20, 30, 40, 60, 70])
    acts = [_act(0, "shop", "shop_add_to_cart", {"item_id": e["p_item"], "quantity": e["p_old"]}, 25, {"cart": []}),
            _act(1, "shop", "shop_checkout", {"delivery_address": e["address"], "po_number": "PO-0000"}, 80, {"order_id": "ORD-001"}, kind="big")]
    st = _empty_state()
    lines = dict(e["order"])
    st["shop"]["orders"] = [{"order_id": "ORD-001", "lines": lines, "total_gbp": 1.0, "address": e["address"], "po_number": "PO-0000"}]
    ev = evaluate(_rec(inst, "handle", person, [], acts, st), inst)
    assert ev["checks"]["right_items"] and ev["checks"]["right_address"] and not ev["checks"]["right_po"]
    assert ev["stale"]["correction"] == 1 and ev["gap"]["guessed"]


def test_reply_and_reach_times_for_flat_and_handle():
    inst = make_instance("small", "A", 0)
    person = _person(inst, [10, 20, 30])
    agent = [{"i": 0, "t": 5.0, "text": "Started.", "loop": "flat", "seen": 0},
             {"i": 1, "t": 14.0, "text": "Two done.", "loop": "flat", "seen": 1},
             {"i": 2, "t": 33.0, "text": "36.", "loop": "flat", "seen": 3}]
    calls = [{"loop": "flat", "t0": 12.0, "t1": 14.0, "seen": 1, "relays": [], "discarded": False, "cost": 0.001, "prompt": 1, "completion": 1, "latency": 2.0, "overhead": 0.0},
             {"loop": "flat", "t0": 23.0, "t1": 25.0, "seen": 2, "relays": [], "discarded": False, "cost": 0.001, "prompt": 1, "completion": 1, "latency": 2.0, "overhead": 0.0}]
    acts = [_act(0, "messaging", "messaging_list_sent", {}, 40, {"messages": []}, kind="read")]
    ev = evaluate(_rec(inst, "flat", person, agent, acts, _empty_state(), calls), inst)
    m = {x["kind"]: x for x in ev["messages"]}
    assert m["status"]["reply_s"] == 4.0 and m["correction"]["reach_s"] == 3.0 and m["unrelated"]["reply_s"] == 3.0
    # handle: the correction reaches the work through an interjection made after the conversation saw it
    calls = [{"loop": "work:t1", "t0": 21.0, "t1": 22.0, "seen": -1, "relays": ["t1:desc"], "discarded": False, "cost": 0, "prompt": 1, "completion": 1, "latency": 1.0, "overhead": 0.0},
             {"loop": "work:t1", "t0": 27.5, "t1": 29.0, "seen": -1, "relays": ["t1:desc", "t1:m1"], "discarded": False, "cost": 0, "prompt": 1, "completion": 1, "latency": 1.5, "overhead": 0.0}]
    tasks = [{"id": "t1", "seen": 0, "interjections": [{"id": "t1:m1", "t": 22.5, "text": "7pm", "seen": 2}]}]
    ev = evaluate(_rec(inst, "handle", person, agent, acts, _empty_state(), calls, tasks), inst)
    assert {x["kind"]: x for x in ev["messages"]}["correction"]["reach_s"] == 7.5


def test_pause_and_cancel_steps():
    inst = make_instance("dual", "C", 0)
    times = [10, 20, 30, 40, 50, 60, 70]
    person = _person(inst, times)
    t_cancel = times[[s["kind"] for s in inst["script"]].index("cancel_second")]
    acts = [_act(0, "messaging", "messaging_send_message", {"to": "X", "text": "hi"}, t_cancel - 5, {"sent": True, "to": "X"}),
            _act(1, "messaging", "messaging_get_profile", {"name": "Y"}, t_cancel + 1, {"name": "Y"}, kind="read"),
            _act(2, "messaging", "messaging_send_message", {"to": "Y", "text": "hi"}, t_cancel + 3, {"sent": True, "to": "Y"})]
    ev = evaluate(_rec(inst, "flat", person, [], acts, _empty_state()), inst)
    assert ev["cancel_steps"] == 2 and ev["cancel_side_effects"] == 1 and not ev["checks"]["second_cancelled"]
    inst = make_instance("medium", "A", 0)
    person = _person(inst, [10, 20, 30, 40, 60, 70])
    acts = [_act(0, "messaging", "messaging_list_sent", {}, 41, {"messages": []}, kind="read"),
            _act(1, "messaging", "messaging_list_sent", {}, 61, {"messages": []}, kind="read")]
    ev = evaluate(_rec(inst, "flat", person, [], acts, _empty_state()), inst)
    assert ev["pause_steps"] == 1 and ev["resumed"]
