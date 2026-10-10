import copy

import pytest

from fast_slow_voice.judge import item_brief, judge_messages, parse_judgement
from fast_slow_voice.metrics import call_metrics, invented_values, known_values
from fast_slow_voice.render import judge_transcript
from fast_slow_voice.script import build_sanity


def _rec(persons, utts, tool_calls=()):
    return {"persons": persons, "utterances": utts, "tool_calls": list(tool_calls), "notes": [], "calls": [],
            "flags": [], "t_end": max([u["t_end"] or 0 for u in utts] + [p["t_end"] for p in persons]),
            "turns_spoken": len(persons), "turns_total": len(persons)}


def _p(idx, turn, kind, text, t0, t1, asks=(), withdraws=(), expects=True, barged=False):
    return {"idx": idx, "turn": turn, "kind": kind, "text": text, "t_start": t0, "t_end": t1, "t_heard": t1 + 0.5,
            "barged": barged, "asks": list(asks), "withdraws": list(withdraws), "expects_reply": expects, "episode": "x"}


def _u(i, text, t0, t1, heard=None, cut=False, source="single", dropped=False):
    return {"id": i, "text": text, "source": source, "kind": "voice", "label": None, "t_queued": t0, "t_ready": t0,
            "t_start": t0, "t_end": t1, "heard": text if heard is None else heard, "cut": cut, "dropped": dropped, "call": None}


def test_call_metrics_on_a_handmade_call():
    s = copy.deepcopy(build_sanity("m-0", 11))
    w = s.world("sanity")
    q1, q2 = s.item("q1"), s.item("q2")
    persons = [
        _p(0, "t1", "greeting", "Hi Ava", 0.5, 1.5),
        _p(1, "t2", "question", q1.text, 3.0, 5.0, asks=["q1"]),
        _p(2, "t3", "ack", "Thanks", 12.0, 12.6, expects=False),
        _p(3, "t4", "question", q2.text, 14.0, 16.0, asks=["q2"]),
        _p(4, "t5", "withdraw", "Never mind", 17.5, 18.2, withdraws=["q2"]),
        _p(5, "t6", "closing", "Bye", 30.0, 30.5),
    ]
    utts = [
        _u(1, "Hi Sam!", 2.5, 2.9),
        _u(2, "Hello again!", 2.95, 2.99),  # a second substantive reply to the greeting
        _u(3, "One sec, let me check.", 6.0, 7.0),
        _u(4, f"The number is {q1.gold}.", 9.0, 11.0),
        _u(5, "One sec.", 17.0, 17.4),
        _u(6, f"It's at {q2.gold}, and the code is 9999#.", 19.0, 21.0),  # stale, and an invented code
        _u(7, "Bye!", 31.5, 31.8),
    ]
    m = call_metrics(_rec(persons, utts), s, w)
    items = {r["item"]: r for r in m["items"]}
    assert items["q1"]["delivered"] and items["q1"]["t_answer"] == pytest.approx(4.0)
    assert items["q2"]["withdrawn"] and items["q2"]["stale"] and not items["q2"]["answered_before_withdraw"]
    assert m["n_wanted"] == 1 and m["n_correct"] == 1 and m["n_stale"] == 1
    ttfw = {t["turn"]: t["ttfw"] for t in m["ttfw"]}
    assert ttfw["t1"] == pytest.approx(1.0) and ttfw["t2"] == pytest.approx(1.0) and ttfw["t6"] == pytest.approx(1.0)
    assert "t3" not in ttfw
    assert m["longest_silence"] == pytest.approx(9.0)  # 21.0 to 30.0, before the goodbye
    assert m["waiting_silence"] == [pytest.approx(3.0)]  # 5-6 and 7-9
    assert m["double_reply_proxy"] == 1
    assert [v["value"] for v in m["invented"]] == ["9999#"]


def test_cut_answers_do_not_count_as_heard():
    s = copy.deepcopy(build_sanity("m-1", 12))
    w = s.world("sanity")
    q1 = s.item("q1")
    persons = [_p(0, "t1", "question", q1.text, 0.5, 2.0, asks=["q1"]),
               _p(1, "t2", "interrupt_repeat", "Sorry?", 5.0, 6.0, barged=True)]
    utts = [_u(1, f"Her mobile is {q1.gold}.", 4.0, 5.2, heard="Her mobile is", cut=True)]
    m = call_metrics(_rec(persons, utts), s, w)
    assert not m["items"][0]["delivered"]
    assert "[cut off by Sam]" in judge_transcript(_rec(persons, utts))


def test_invented_values_against_the_records():
    s = build_sanity("m-2", 13)
    w = s.world("sanity")
    known = known_values(w, ["my number is 312-555-0199"])
    c = w.contacts[0]
    assert invented_values(f"{c.name}'s mobile is {c.mobile}.", known) == []
    assert invented_values("Call 312-555-0199.", known) == []  # Sam said it
    assert invented_values("It's 773-555-0000 and costs $123.45 at 4:17 am.", known) == ["773-555-0000", "$123.45", "4:17 am"]


def test_judge_sees_no_arm_and_parses_robustly():
    s = build_sanity("m-3", 14)
    w = s.world("sanity")
    rec = _rec([_p(0, "t1", "question", s.item("q1").text, 0.5, 2.0, asks=["q1"])],
               [_u(1, "One sec.", 3.0, 3.5, source="fast")])
    rec["arm"] = "relay"
    msgs = judge_messages(rec, s, w)
    body = msgs[0]["content"] + msgs[1]["content"]
    assert "relay" not in body and "fast" not in body.lower().replace("breakfast", "")
    assert s.item("q1").gold in item_brief(s, w, rec)
    j = parse_judgement('Here: {"naturalness": "4", "naturalness_reason": "ok", "double_replies": [{"time": 1}], "redundant_fillers": "2"}')
    assert j["naturalness"] == 4 and len(j["double_replies"]) == 1 and j["contradictions"] == [] and j["redundant_fillers"] == 2
    assert parse_judgement("no json here") is None


def test_judge_transcript_handles_a_line_the_call_ended_on():
    s = build_sanity("m-4", 15)
    persons = [_p(0, "t1", "closing", "Bye", 0.5, 1.5)]
    u = _u(1, "Before you go, the code is 1234#.", 3.0, None)
    rec = _rec(persons, [_u(0, "Hi", 2.0, 2.4)])
    rec["utterances"].append(u)
    rec["t_end"] = 4.0
    out = judge_transcript(rec)
    assert "[the call ended here]" in out and "Before you" in out and "1234" not in out
