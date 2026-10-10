"""The sites, the scripted person, the generator and a whole run, with the model stubbed."""

from __future__ import annotations

import asyncio
import json

from conftest import SPEED, FakeLLM, last_text, say, tc

from separate_loops.core import Run
from separate_loops.person import Person, parse_reply
from separate_loops.runner import run_one
from separate_loops.scenarios import generate, make_instance
from separate_loops.sites import Clock, Env, times_in


def _env(cond="realistic", fam="A", seed=0, speed=SPEED):
    inst = make_instance(cond, fam, seed)
    return Env(inst, Clock(), speed=speed), inst


def call(env, tool_, worker_="w", **args):
    return json.loads(asyncio.run(env.call(worker_, tool_, json.dumps(args))))


def test_times_in():
    assert times_in("see you at 7:30pm") == {19 * 60 + 30}
    assert times_in("at 6pm, not 7.30 PM") == {18 * 60, 19 * 60 + 30}
    assert times_in("19:30 sharp") == {19 * 60 + 30}
    assert times_in("7:30") == {7 * 60 + 30, 19 * 60 + 30}
    assert times_in("Saturday 18 October, 6 people") == set()


def test_messaging_resolves_names_and_refuses_the_ambiguous_sam():
    env, inst = _env()
    assert "More than one contact" in call(env, "messaging_send_message", to="Sam", text="hi")["error"]
    assert call(env, "messaging_send_message", to="Sam Rivera", text="hi")["to"] == "Sam Rivera"
    m = inst["main"]["expect"]["members"][0]
    assert call(env, "messaging_send_message", to=m.split()[0], text="hi")["to"] == m
    assert len(call(env, "messaging_find_contact", query="Sam")["matches"]) == 2
    assert [s["to"] for s in env.state()["messaging"]["sent"]] == ["Sam Rivera", m]


def test_booking_needs_the_phone_and_the_right_slot():
    env, inst = _env(fam="B")
    e = inst["main"]["expect"]
    day = f"{e['weekday']} {e['date']}"
    t = f"{(e['time'] // 60) - 12}:{e['time'] % 60:02d}pm"
    r = call(env, "booking_check_availability", restaurant_id=e["answer_id"], day=day, time=t, party_size=e["final_party"])
    assert r["available"] is True
    assert call(env, "booking_check_availability", restaurant_id=e["answer_id"], day="Monday", time=t,
                party_size=e["final_party"])["available"] is False
    assert "phone" in call(env, "booking_hold_table", restaurant_id=e["answer_id"], day=day, time=t,
                           party_size=e["final_party"], name=e["name"])["error"]
    h = call(env, "booking_hold_table", restaurant_id=e["answer_id"], day=day, time=t, party_size=e["final_party"],
             name=e["name"], phone=e["phone"])
    assert h["hold_id"] == "H1"
    assert call(env, "booking_share", hold_id="H1", with_name=e["friend"])["shared"] == "H1"


def test_shop_stock_and_purchase_order():
    env, inst = _env(fam="C")
    e = inst["main"]["expect"]
    items = inst["sites"]["shop"]["catalog"]
    out = next((i for i in items if not i["in_stock"]), None)
    if out:
        assert "out of stock" in call(env, "shop_add_to_cart", item_id=out["id"], quantity=1)["error"]
    call(env, "shop_add_to_cart", item_id=e["p_item"], quantity=2)
    assert "purchase-order" in call(env, "shop_checkout", delivery_address="x")["error"]
    o = call(env, "shop_checkout", delivery_address=e["address"], po_number=e["po"])
    assert o["order_id"] == "ORD-001" and env.state()["shop"]["cart"] == {}


def test_search_finds_each_requested_product_first():
    for seed in range(3):
        env, inst = _env(fam="C", seed=seed)
        for label, item in inst["main"]["expect"]["best"].items():
            res = call(env, "shop_search", query=label.split(" (")[0])["results"]
            names = {i["id"]: i["name"] for i in inst["sites"]["shop"]["catalog"]}
            top = names[res[0]["item_id"]].split(",")[0]
            assert top == names[item].split(",")[0], (label, res[:3])


def test_bad_arguments_are_errors_not_exceptions():
    env, _ = _env()
    assert "not a JSON object" in json.loads(asyncio.run(env.call("w", "messaging_send_message", "{oops")))["error"]
    assert "Missing" in call(env, "messaging_send_message", to="Sam Rivera")["error"]
    assert "Unknown" in call(env, "messaging_list_sent", foo=1)["error"]
    assert "No tool" in call(env, "nope")["error"]


def test_latency_is_the_same_for_the_same_action_and_one_worker_does_one_thing_at_a_time():
    env1, _ = _env(speed=1)
    env2, _ = _env(speed=1)
    a = env1._latency("messaging_get_profile", {"name": "X"}, "read")
    assert a == env2._latency("messaging_get_profile", {"name": "X"}, "read")
    assert 3.0 <= a <= 8.0
    env, _ = _env()

    async def go():
        await asyncio.gather(env.call("w", "messaging_list_sent", "{}"), env.call("w", "messaging_list_sent", "{}"),
                             env.call("v", "messaging_list_sent", "{}"))
    asyncio.run(go())
    w = sorted((a for a in env.log if a["worker"] == "w"), key=lambda a: a["t_start"])
    v = next(a for a in env.log if a["worker"] == "v")
    assert w[1]["t_start"] >= w[0]["t_end"]
    assert v["t_start"] < w[0]["t_end"]


def test_generator_is_deterministic_and_the_answers_follow_the_script():
    assert make_instance("realistic", "B", 2) == make_instance("realistic", "B", 2)
    for cond in ("sanity", "small", "medium", "realistic", "dual"):
        for inst in generate(cond):
            e = inst["main"]["expect"]
            kinds = [s["kind"] for s in inst["script"]]
            if inst["family"] == "B":
                rest = {r["id"]: r for r in inst["sites"]["booking"]["restaurants"]}
                ans = rest[e["answer_id"]]
                assert ans["max_party"] >= e["final_party"] and ans["veg"] >= e["veg_min"]
                cheaper = [r for r in rest.values() if r["price"] < ans["price"] and r["role"] != "elsewhere"
                           and r["max_party"] >= e["final_party"] and r["veg"] >= e["veg_min"]]
                assert [r["name"] for r in cheaper] == ([e["dropped"]] if e["dropped"] else [])
                if "correction" in kinds:  # the correction changes the answer
                    first = min((r for r in rest.values() if r["role"] != "elsewhere" and r["max_party"] >= e["party"]),
                                key=lambda r: r["price"])
                    assert first["max_party"] < e["final_party"]
            if inst["family"] == "C":
                assert (e["y_item"] is None) == ("change" not in kinds)
            assert ("Sam\"" not in inst["request"])
            if cond == "dual":
                assert inst["second"]["family"] != inst["family"]


def test_person_fires_on_progress_falls_back_on_a_stall_and_resumes_on_time():
    inst = make_instance("medium", "A", 0)
    clock = Clock()
    env = Env(inst, clock, speed=SPEED)
    run = Run("t", "m", FakeLLM(lambda *a: say('{"reply": null}')), clock)
    got = []
    p = Person(run, env, inst, got.append, speed=SPEED)
    members = inst["main"]["expect"]["members"]

    async def go():
        main = asyncio.create_task(p.main())
        await asyncio.sleep(0.02)
        assert [m["kind"] for m in run.person_msgs] == ["request"]
        for m in members[:2]:  # status fires at 2 distinct recipients
            await env.call("w", "messaging_send_message", json.dumps({"to": m, "text": "hi"}))
        await asyncio.sleep(inst["script"][0]["min_gap"] * SPEED + 0.05)
        assert run.person_msgs[-1]["kind"] == "status" and run.person_msgs[-1]["how"] == "progress"
        # No more progress: the next message comes on the fallback.
        await asyncio.sleep(inst["script"][1]["max_wait"] * SPEED + 0.1)
        assert run.person_msgs[-1]["kind"] == "correction" and run.person_msgs[-1]["how"] == "fallback"
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)

    asyncio.run(go())


def test_responder_answers_a_question_after_a_delay_and_ignores_updates():
    inst = make_instance("small", "A", 0)
    clock = Clock()
    env = Env(inst, clock, speed=SPEED)

    def person_model(loop, m, t):
        newest = m[-1]["content"].split("The assistant's newest messages:")[1]
        return say('{"reply": "Sam Rivera."}' if "Which Sam" in newest else '{"reply": null}')

    run = Run("t", "m", FakeLLM(person_model), clock)
    got = []
    p = Person(run, env, inst, got.append, speed=SPEED)

    async def go():
        main = asyncio.create_task(p.main())
        await asyncio.sleep(0.01)
        run.say("Working on it, two sent.", "conv", 0)
        await asyncio.sleep(0.06)
        assert [g["text"] for g in got] == [inst["request"]]
        run.say("Which Sam do you mean, Sam Rivera or Sam Oyelaran?", "conv", 0)
        await asyncio.sleep(0.08)
        main.cancel()
        await asyncio.gather(main, return_exceptions=True)

    asyncio.run(go())
    ans = [m for m in run.person_msgs if m["kind"] == "answer"]
    assert [a["text"] for a in ans] == ["Sam Rivera."]
    asked = run.agent_msgs[1]["t"]
    assert ans[0]["t"] >= asked + 2.0 * SPEED
    assert parse_reply('```json\n{"reply": null}\n```') is None and parse_reply('{"reply": "Yes"}') == "Yes"


def test_a_whole_run_ends_when_the_script_is_done_and_all_is_quiet():
    inst = make_instance("sanity", "A", 0)
    members = [m for m in inst["main"]["expect"]["members"] if m not in inst["main"]["expect"]["away"]]

    def policy(loop, m, t):
        if loop == "person":
            return say('{"reply": null}')
        sent = sum(1 for x in m if x["role"] == "tool")
        if m[-1]["role"] == "user" and "How's it going" in last_text(m):
            return say(f"{sent} sent so far.", *([tc("messaging_send_message", to=members[sent], text="Hi, 6pm")]
                                                   if sent < len(members) else []))
        if sent < len(members):
            return say("", tc("messaging_send_message", to=members[sent], text="Hi, 6pm"))
        return say("All invited.")

    rec = asyncio.run(run_one(inst, "flat", "fake/model", FakeLLM(policy, delay=0.002), run_id="r", speed=SPEED))
    assert rec["end"]["reason"] == "idle", rec["end"]
    assert [m["kind"] for m in rec["person"]] == ["request", "status"]
    assert len(rec["state"]["messaging"]["sent"]) == len(members)
    assert "All invited." in [m["text"] for m in rec["agent"]]
    json.dumps(rec)  # the record is plain JSON


def test_responder_drops_a_reply_thought_out_before_the_person_said_something_else():
    inst = make_instance("medium", "A", 0)
    clock = Clock()
    env = Env(inst, clock, speed=SPEED)

    def person_model(loop, m, t):
        paused = "pause and have not yet told it to carry on" in m[-1]["content"]
        return say('{"reply": "Not yet."}' if paused else '{"reply": null}')

    run = Run("t", "m", FakeLLM(person_model, delay=0.05), clock)
    p = Person(run, env, inst, lambda m: None, speed=SPEED)

    async def go():
        responder = asyncio.create_task(p._responder())
        p.paused = True
        run.say("Shall I carry on?", "conv", 0)
        await asyncio.sleep(0.03)  # the responder is thinking, with the person paused
        p.paused = False
        p.send("OK, carry on.", "resume", "time")
        await asyncio.sleep(0.3)
        responder.cancel()
        await asyncio.gather(responder, return_exceptions=True)

    asyncio.run(go())
    assert [m["text"] for m in run.person_msgs] == ["OK, carry on."]
    assert p.log[0].get("dropped") and p.log[-1]["reply"] is None


def test_anthropic_cache_breakpoints_mark_system_previous_end_and_last_without_mutating():
    from separate_loops.core import anthropic_cache
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "r1"}, {"role": "user", "content": "more"}]
    out = anthropic_cache(msgs)
    marked = [i for i, m in enumerate(out) if isinstance(m["content"], list)]
    assert marked == [0, 1, 4]
    assert all(isinstance(m["content"], (str, type(None))) for m in msgs)  # the loop's own context is untouched
    assert out[4]["content"][0] == {"type": "text", "text": "more", "cache_control": {"type": "ephemeral"}}
