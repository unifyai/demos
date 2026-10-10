import pytest

from scripted import Scripted
from skills.api import Api
from skills.tasks import KINDS, SIZES, answer_ok, check, make_stream, schedule
from skills.world import clone

SEEDS = [1, 2, 3, 4, 5]


@pytest.mark.parametrize("size", list(SIZES))
def test_schedule_shape(size):
    spec = SIZES[size]
    sch = schedule(size)
    assert len(sch) == spec["tasks"]
    firsts = []
    for s in sch:
        if s["kind"] not in firsts:
            firsts.append(s["kind"])
    assert firsts == spec["kinds"]  # first occurrences in level order
    levels = [KINDS[k].level for k in firsts]
    assert levels == sorted(levels)
    for i, s in enumerate(sch, start=1):
        if s["repeat_of"] is not None:
            assert s["repeat_of"] < i and sch[s["repeat_of"] - 1]["repeat_of"] is None
            assert KINDS[s["kind"]].repeatable
    for change in spec["changes"]:
        assert {s["kind"] for s in sch[change - 1:] if s["repeat_of"] is None} == set(spec["kinds"])
    assert schedule(size) == sch  # fixed


@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("seed", SEEDS)
def test_scripted_api_solutions_pass_every_task(size, seed):
    """A correct program written only against the API passes every task, in
    every version: the API, the references and the check agree."""
    stream = make_stream(size, seed)
    for t in stream.tasks:
        world = clone(t.before)
        ops = Api(world, t.version, size).functions()
        answer = Scripted(ops).solve(t.kind, t.args)
        ok, reasons = check(t, answer, world)
        assert ok, (t.position, t.kind, t.text, reasons)


@pytest.mark.parametrize("seed", SEEDS)
def test_repeats_have_the_same_text_and_fresh_tasks_new_arguments(seed):
    stream = make_stream("realistic", seed)
    for t in stream.tasks:
        if t.repeat_of:
            assert t.text == stream.tasks[t.repeat_of - 1].text
    fresh = [(t.kind, t.text) for t in stream.tasks if not t.repeat_of]
    assert len(fresh) == len(set(fresh))


def test_streams_differ_by_seed_and_are_deterministic():
    a, b, c = make_stream("realistic", 1), make_stream("realistic", 1), make_stream("realistic", 2)
    assert [t.text for t in a.tasks] == [t.text for t in b.tasks]
    assert [t.text for t in a.tasks] != [t.text for t in c.tasks]
    assert [t.kind for t in a.tasks] == [t.kind for t in c.tasks]  # same schedule


def test_pagination_matters_somewhere():
    """Ignoring pagination fails some tasks in every realistic stream."""
    for seed in SEEDS:
        stream = make_stream("realistic", seed)
        fails = 0
        for t in stream.tasks:
            world = clone(t.before)
            ops = Api(world, t.version, "realistic").functions()
            try:
                answer = Scripted(ops, naive=True).solve(t.kind, t.args)
            except Exception:  # noqa: BLE001
                fails += 1
                continue
            fails += not check(t, answer, world)[0]
        assert fails >= 5, (seed, fails)


def test_check_catches_wrong_answers_and_stray_changes():
    stream = make_stream("realistic", 1)
    t = next(x for x in stream.tasks if x.kind == "contact_email")
    w = clone(t.before)
    assert check(t, t.expected, w)[0]
    assert not check(t, "nobody@x.example", w)[0]
    w["outbox"].append({"to": "a@b.example", "subject": "s", "body": "b"})
    ok, reasons = check(t, t.expected, w)
    assert not ok and any("outbox" in r for r in reasons)
    act = next(x for x in stream.tasks if x.kind == "email_contact")
    assert not check(act, None, clone(act.before))[0]  # nothing sent


def test_answer_normalisation():
    assert answer_ok("str", "Ada Lovelace", " ada  lovelace ")
    assert not answer_ok("str", "Ada Lovelace", "Ada")
    assert answer_ok("int", 1200, 1200) and answer_ok("int", 1200, 1200.0) and answer_ok("int", 1200, "1,200")
    assert not answer_ok("int", 1200, True) and not answer_ok("int", 1, True) and not answer_ok("int", 1200, 12.5)
    assert answer_ok("list", ["t_1", "t_2"], ("t_1", "t_2")) and not answer_ok("list", ["t_1", "t_2"], ["t_2", "t_1"])
    assert answer_ok("none", None, "anything")
    # a list sent as text through the submit tool's untyped answer field
    assert answer_ok("list", ["t_1", "t_2"], "['t_1', 't_2']") and answer_ok("list", ["t_1"], '["t_1"]')
    assert not answer_ok("list", ["t_1", "t_2"], "['t_2', 't_1']") and not answer_ok("list", ["t_1"], "t_1")
