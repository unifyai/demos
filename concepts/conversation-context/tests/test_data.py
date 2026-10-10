"""Invariants of the scenarios, the filler, the world and the conversations."""

from __future__ import annotations

import re

import pytest

from conversation_context.data import KINDS, LENGTHS, conversation, items, pool, present, settle_distance
from conversation_context.grade import matches
from conversation_context.scenarios import COMPLETE_IDS, SCENARIOS
from conversation_context.world import World, filler_records, render, scenario_records

WORLD = World(0)
ALL = [r for rs in WORLD.records.values() for r in rs]


def avoid(text: str) -> list[str]:
    return [f for s in SCENARIOS for f in s["avoid"] if re.search(r"\b" + f, text, re.I)]


def test_sizes():
    assert len(SCENARIOS) == 24 and len(COMPLETE_IDS) == 8
    its = items()
    assert len(its) == 56 and len({i["id"] for i in its}) == 56
    assert sum(1 for i in its if i["kind"] == "complete") == 8


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s["id"])
def test_scenario_patterns(s):
    t, d = render({**s["target"], "id": "x"}), render({**s["distractor"], "id": "y"})
    assert matches(t, s["target_value"]) and not matches(t, s["distractor_value"])
    assert matches(d, s["distractor_value"]) and not matches(d, s["target_value"])
    assert matches(s["target_answer"], s["target_value"]) and matches(s["distractor_answer"], s["distractor_value"])
    # the terse question and request leave the detail out; the complete ones name it
    assert not matches(s["question"], s["detail"]) and not matches(s["request"], s["detail"])
    assert matches(s["question_complete"], s["detail"]) and matches(s["request_complete"], s["detail"])
    # the settling exchange states it, the open exchange doesn't
    assert matches(" ".join(s["settle"]), s["detail"]) and not matches(" ".join(s["open"]), s["detail"])
    # the question points back to what Maya said, so the conversation settles it or leaves it open at any distance
    assert re.search(r"I mentioned|I said", s["question"]), s["question"]
    # settling and open lines differ only in the deciding detail (same speaker turns, same reply where it fits)
    assert len(s["settle"]) == len(s["open"]) == 2
    # the answers themselves never appear in the conversation
    for line in s["settle"] + s["open"] + (s["question"], s["question_complete"]):
        assert not matches(line, s["target_value"] + s["distractor_value"]), line


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s["id"])
def test_answers_are_unique_among_what_a_search_returns(s):
    """Among the records a search for the item's question or request returns, each answer
    pattern matches only its own record. (Times also occur as other records' send times.)"""
    store = WORLD.stores[s["source"]]
    for q in (s["request"], s["request_complete"], s["question"], s["question_complete"]):
        for _, i in store.scores(q)[:5]:
            r = store.records[i]
            if matches(render(r), s["target_value"]):
                assert (r.get("scenario"), r.get("role")) == (s["id"], "target"), (q, render(r))
            if matches(render(r), s["distractor_value"]):
                assert (r.get("scenario"), r.get("role")) == (s["id"], "distractor"), (q, render(r))


@pytest.mark.parametrize("s", [s for s in SCENARIOS if s["source"] != "calendar"], ids=lambda s: s["id"])
def test_message_and_document_answers_occur_in_one_record_each(s):
    def content(r):
        return " ".join(str(r.get(k, "")) for k in ("subject", "body", "location"))
    assert [r.get("scenario") for r in ALL if matches(content(r), s["target_value"])] == [s["id"]]
    assert [r.get("scenario") for r in ALL if matches(content(r), s["distractor_value"])] == [s["id"]]


@pytest.mark.parametrize("s", SCENARIOS, ids=lambda s: s["id"])
def test_the_terse_request_finds_both_candidates(s):
    top = WORLD.search(s["source"], s["request"])
    assert matches(top, s["target_value"]) and matches(top, s["distractor_value"]), s["request"]


def test_target_is_the_later_record_in_half_the_scenarios():
    later = sum(1 for s in SCENARIOS if s["target"]["date"] > s["distractor"]["date"])
    assert later == 12


def test_filler_never_touches_a_scenario():
    for ex in pool():
        assert not avoid(" ".join(m["text"] for m in ex)), ex
    for r in filler_records(0):
        assert not avoid(" ".join(str(v) for v in r.values())), r


def test_scenario_records_carry_their_source():
    recs = scenario_records()
    assert len(recs) == 48 and all(r["source"] in ("messages", "documents", "calendar") for r in recs)


@pytest.mark.parametrize("length", list(LENGTHS))
def test_kinds_share_filler_and_position(length):
    for s in SCENARIOS[:6]:
        for seed in range(3):
            st, op, co = (conversation(s, k, length, seed) for k in KINDS)
            assert len(st) == len(op) == len(co)
            i = next(k for k, m in enumerate(st) if m["content"] == s["settle"][0])
            assert op[i]["content"] == s["open"][0] and op[i + 1]["content"] == s["open"][1]
            assert st[:i] == op[:i] and st[i + 2:] == op[i + 2:]  # only the settling exchange differs
            assert st[-1]["content"] == s["question"] == op[-1]["content"]
            assert co[-1]["content"] == s["question_complete"]
            assert st[:-1] == co[:-1]


def test_short_settles_two_messages_before_the_question_and_long_further():
    for s in SCENARIOS:
        assert settle_distance(s, "short", 0) == 2
        assert settle_distance(s, "long", 0) >= 40


def test_seeds_redraw_the_conversation():
    s = SCENARIOS[0]
    assert conversation(s, "settled", "long", 0) != conversation(s, "settled", "long", 1)
    assert conversation(s, "settled", "long", 0) == conversation(s, "settled", "long", 0)


def test_presence_is_balanced():
    for seed in range(3):
        assert sum(present(i, seed) for i in range(24)) == 12
    assert present(0, 0) != present(0, 1)
