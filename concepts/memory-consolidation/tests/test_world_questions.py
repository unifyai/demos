"""The generator and the questions: deterministic, consistent, exact, and no
answer depends on anything after its checkpoint."""

import datetime as dt

import pytest

from memcon.questions import (Question, checkpoints, last_gap, make_questions, projects_in, sanity_questions, score, visible)
from memcon.world import DAYS, PROJECT_NAMES, START, Interaction, day_of, generate


@pytest.fixture(scope="module")
def h0():
    return generate(0)


def test_generate_is_deterministic_and_ordered(h0):
    again = generate(0)
    assert [it.text for it in again["interactions"]] == [it.text for it in h0["interactions"]]
    its = h0["interactions"]
    assert [it.idx for it in its] == list(range(1, len(its) + 1))
    assert all(a.ts <= b.ts for a, b in zip(its, its[1:]))
    assert generate(1)["interactions"][0].text != its[0].text


def test_breaks_are_silent_and_records_name_their_project(h0):
    for a, b in h0["breaks"]:
        assert not [it for it in h0["interactions"] if a <= day_of(it.ts) < b]
    for it in h0["interactions"]:
        if it.project:
            assert it.project in it.text
        assert it.text.startswith(f"#{it.idx:04d} · ")


def test_ids_are_stable_across_the_records_that_refer_to_them(h0):
    tasks, events, rows = {}, {}, {}
    for it in h0["interactions"]:
        f = it.facts
        if it.action == "task_create":
            tasks[(it.project, f["task"])] = f["task_id"]
            assert f["task_id"] in it.text
        elif it.action == "task_done":
            assert tasks[(it.project, f["task"])] == f["task_id"] and f["task_id"] in it.text
        elif it.action == "meeting":
            events[(it.project, f["topic"], f["person"])] = f["event_id"]
            assert f["event_id"] in it.text
            booked = f["time"][:2]
            busy = it.text.split("calendar.free_busy", 1)[1].split("\n", 2)[1]
            assert f'"start": "{booked}:00"' not in busy
        elif it.action == "move_meeting":
            assert events[(it.project, f["topic"], f["person"])] == f["event_id"] and f["event_id"] in it.text
        elif it.action == "quote":
            assert (it.project, f["row"]) not in rows.values()
            rows[(it.project, f["item"])] = (it.project, f["row"])
            assert f'"row": {f["row"]}' in it.text
        elif it.action == "revise_quote":
            assert rows[(it.project, f["item"])] == (it.project, f["row"]) and f"row={f['row']}" in it.text


def test_facts_are_consistent(h0):
    quotes, meetings, shares = {}, {}, set()
    for it in h0["interactions"]:
        f = it.facts
        if it.action == "quote":
            assert (it.project, f["item"]) not in quotes
            quotes[(it.project, f["item"])] = f["amount"]
            assert f"{f['amount']:,}" in it.text
        elif it.action == "revise_quote":
            assert quotes[(it.project, f["item"])] == f["old_amount"]
            quotes[(it.project, f["item"])] = f["amount"]
        elif it.action == "meeting":
            assert (it.project, f["topic"], f["person"]) not in meetings
            meetings[(it.project, f["topic"], f["person"])] = f["date"]
        elif it.action == "move_meeting":
            assert meetings[(it.project, f["topic"], f["person"])] == f["old_date"]
            meetings[(it.project, f["topic"], f["person"])] = f["date"]
        elif it.action == "share":
            assert (it.project, f["document"]) not in shares
            shares.add((it.project, f["document"]))


def mk(idx, ts, project=None, action="email", facts=None):
    return Interaction(idx, ts, project, action, facts or {}, f"#{idx:04d}")


def test_project_and_count_questions_use_only_the_past():
    t = START + dt.timedelta(days=40, hours=12)
    its = [mk(1, t - dt.timedelta(days=30), "Lisbon offsite"), mk(2, t - dt.timedelta(days=10), "spring campaign"),
           mk(3, t - dt.timedelta(days=3), "trademark registration"), mk(4, t - dt.timedelta(hours=2), None),
           mk(5, t - dt.timedelta(hours=1), "Loom chair launch"), mk(6, t + dt.timedelta(hours=1), "insurance renewal")]
    qs = {q.qtype: q for q in make_questions(0, "x", t, its)}
    assert "to today" in qs["T_7d"].text and qs["T_7d"].meta["start"] == (t.replace(hour=0, minute=0) - dt.timedelta(days=6)).isoformat()
    assert qs["T_today"].gold == ["Loom chair launch"]
    assert qs["T_7d"].gold == ["Loom chair launch", "trademark registration"]
    assert qs["T_28d"].gold == ["Loom chair launch", "spring campaign", "trademark registration"]
    assert qs["T_all"].gold == sorted(["Lisbon offsite", "spring campaign", "trademark registration", "Loom chair launch"])
    assert "insurance renewal" not in qs["T_all"].gold  # after now
    assert "C_10" not in qs  # fewer than 10 interactions so far


def test_details_take_the_latest_value_before_now():
    t0 = START + dt.timedelta(days=40)
    q = {"item": "venue hire", "person": "Ana Ruiz", "org": "Casa do Rio", "currency": "EUR"}
    its = [mk(1, t0, "Lisbon offsite", "quote", {**q, "amount": 4000}),
           mk(2, t0 + dt.timedelta(days=5), "Lisbon offsite", "revise_quote", {**q, "old_amount": 4000, "amount": 4600}),
           mk(3, t0 + dt.timedelta(days=9), "Lisbon offsite", "revise_quote", {**q, "old_amount": 4600, "amount": 5000})]
    now = t0 + dt.timedelta(days=7)
    from memcon.questions import _detail_candidates
    c = [c for c in _detail_candidates(visible(its, now), now) if c["kind"] == "quote"]
    assert len(c) == 1 and c[0]["gold"] == 4600 and c[0]["idx"] == 2
    assert 1.9 < c[0]["age_days"] < 2.1


def test_last_gap_is_the_most_recent_silent_stretch_that_activity_followed():
    days = [0, 1, 2, 9, 10, 11, 18, 25]  # gaps: 3..8 (6 days), 12..17 (6 days), 19..24 (6 days)
    its = [mk(k + 1, START + dt.timedelta(days=d, hours=10)) for k, d in enumerate(days)]
    now = START + dt.timedelta(days=25, hours=12)
    assert last_gap(visible(its, now), now) == (19, 25)
    now = START + dt.timedelta(days=24, hours=12)
    assert last_gap(visible(its, now), now) == (12, 18)  # the stretch after day 18 has not ended
    assert last_gap(its, now) == (12, 18)  # and later activity is ignored even if passed in
    assert last_gap(its[:3], START + dt.timedelta(days=5)) is None


def test_checkpoints_and_question_mix():
    h = generate(3)
    its = h["interactions"]
    cps = checkpoints(3, its, h["breaks"])
    assert set(cps) == {"2w", "8w", "16w", "26w", "break"}
    a, b = max(h["breaks"], key=lambda x: x[1] - x[0])
    assert day_of(cps["break"]) >= b
    assert sum(1 for it in its if day_of(it.ts) == day_of(cps["break"]) and it.ts < cps["break"]) <= 3
    for label, now in cps.items():
        qs = make_questions(3, label, now, its)
        assert all(q.checkpoint == label and q.qid.startswith(f"s3-{label}-") and q.now == now.isoformat() for q in qs)
        assert {q.family for q in qs} >= {"T", "C", "D", "N"}
        assert len({q.qid for q in qs}) == len(qs)
        for q in qs:
            if q.fmt == "projects":
                assert set(q.gold) <= set(projects_in(visible(its, now)))


def test_sanity_details_come_from_the_last_ten_interactions():
    h = generate(113)
    its = [it for it in h["interactions"] if day_of(it.ts) < 4]
    now = START + dt.timedelta(days=3, hours=20, minutes=30)
    qs = sanity_questions(113, now, its)
    assert sum(len(it.text) for it in its) < 30_000
    assert [q for q in qs if q.family == "D"] and all(q.meta["age_days"] < 2 for q in qs if q.family == "D")


def Q(fmt, gold):
    return Question("q", 0, "x", START.isoformat(), "T", "t", "?", fmt, gold)


def test_scoring():
    q = Q("projects", ["Lisbon offsite", "spring campaign"])
    assert score(q, ["lisbon offsite", "Spring Campaign"]) == 1.0
    assert score(q, ["Lisbon offsite"]) == pytest.approx(2 / 3)
    assert score(q, ["Lisbon offsite", "spring campaign", "made up"]) == pytest.approx(0.8)
    assert score(q, []) == 0.0 and score(q, None) == 0.0 and score(q, "Lisbon offsite") == pytest.approx(2 / 3)
    assert score(Q("projects", []), []) == 1.0 and score(Q("projects", []), ["Lisbon offsite"]) == 0.0
    assert score(Q("amount", 4350), "EUR 4,350") == 1.0 and score(Q("amount", 4350), 4300) == 0.0
    assert score(Q("count", 3), "3") == 1.0 and score(Q("count", 3), 2) == 0.0 and score(Q("count", 0), 0) == 1.0
    assert score(Q("date", "2026-03-05"), "2026-03-05") == 1.0 and score(Q("date", "2026-03-05"), "5 March") == 0.0
    assert score(Q("person", "Ana Ruiz"), "Ana Ruiz (Casa do Rio)") == 1.0 and score(Q("person", "Ana Ruiz"), "Ana") == 0.0
    assert score(Q("project", "spring campaign"), "Spring campaign") == 1.0


def test_option_list_covers_every_project():
    assert len(PROJECT_NAMES) == len(set(PROJECT_NAMES)) == 26
