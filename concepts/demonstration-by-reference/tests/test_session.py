"""The data: sessions are deterministic, the ground truth follows from the script, and frames render the spec."""

import hashlib
import json

import pytest

from demoref.render import encode, list_tasks, render, visible_rows
from demoref.session import build_session
from demoref.world import make_world


def test_world_is_deterministic_and_ids_unique():
    a, b = make_world(5), make_world(5)
    assert [t.__dict__ for t in a.tasks] == [t.__dict__ for t in b.tasks]
    assert len({t.id for t in a.tasks}) == len(a.tasks) == 40
    assert make_world(6).tasks[0].id != a.tasks[0].id or make_world(6).tasks[1].id != a.tasks[1].id


@pytest.mark.parametrize("item", range(6))
@pytest.mark.parametrize("size", [64, 128, 256, 500])
def test_session_size_and_determinism(item, size):
    s1, s2 = build_session(item, size), build_session(item, size)
    assert len(s1.frames) == size
    assert json.dumps(s1.to_json()) == json.dumps(s2.to_json())
    assert [f.id for f in s1.frames] == [f"f{i:03d}" for i in range(size)]
    assert all(b.t > a.t for a, b in zip(s1.frames, s1.frames[1:]))


@pytest.mark.parametrize("item", range(12))
def test_key_script_is_the_same_at_every_length(item):
    """Only the ordinary activity between the beats grows; what was shown and said about the queue does not."""
    def script(s):
        return [(f.role, f.task, f.said) for f in s.frames if f.role != "noise"]
    base = script(build_session(item, None))
    for size in (64, 128, 256, 500):
        s = build_session(item, size)
        assert script(s) == base
        assert s.answer == build_session(item, None).answer


@pytest.mark.parametrize("item", range(12))
def test_answer_follows_from_what_was_shown(item):
    s = build_session(item, 256)
    by_id = {f.id: f for f in s.frames}
    shown = [by_id[e].task for e in s.evidence if by_id[e].role in ("key", "key-next")]
    assert shown == s.meta["keys"]  # the queued tasks were shown in order
    answer = list(shown)
    c = s.meta["correction"]
    if c:
        corr = [by_id[e] for e in s.evidence if by_id[e].role == "correction"]
        assert len(corr) == 1 and corr[0].task == c["new"]
        assert ["first", "second", "third", "fourth", "fifth"][c["position"]] in corr[0].said
        answer[c["position"]] = c["new"]
    assert answer == s.answer
    assert len(set(s.answer)) == len(s.answer)
    # the words never name a task
    world = make_world(item)
    for f in s.frames:
        for t in world.tasks:
            assert t.id not in f.said and t.title.lower() not in f.said.lower()


def test_negatives_are_not_in_the_answer():
    for item in range(12):
        s = build_session(item, 128)
        for f in s.frames:
            if f.role in ("not-this", "done", "later"):
                assert f.task not in s.answer


def test_sanity_condition_is_simple():
    for item in range(12):
        s = build_session(item, None, simple=True)
        roles = [f.role for f in s.frames]
        assert roles.count("key") == 3 and "key-next" not in roles and "not-this" not in roles
        assert s.meta["correction"] is None
        for f in s.frames:
            if f.role == "key":
                assert f.spec["view"] == "detail" and f.spec["task"] == f.task


def test_pointer_is_on_the_row_it_names():
    world = make_world(2)
    s = build_session(2, 256)
    for f in s.frames:
        if f.spec.get("app") == "tracker" and f.spec.get("view") == "list" and f.spec.get("hover"):
            rows = dict(visible_rows(world, f.spec))
            y = rows[f.spec["hover"]]
            assert y < f.spec["cursor"][1] and f.spec["cursor"][1] + 20 < y + 34  # the whole arrow is in the row
            assert f.spec["hover"] in [t.id for t in list_tasks(world, f.spec)]


def test_render_is_deterministic():
    world = make_world(1)
    s = build_session(1, 64)
    for f in s.frames[:6]:
        a = hashlib.sha256(encode(render(world, f.spec))).hexdigest()
        b = hashlib.sha256(encode(render(world, f.spec))).hexdigest()
        assert a == b
    assert render(world, s.frames[0].spec).size == (1280, 800)
