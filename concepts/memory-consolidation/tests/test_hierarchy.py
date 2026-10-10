"""The mechanism: every summary reads exactly its window, is scheduled exactly
when its inputs exist, runs off the caller's path, and the snapshot shows the
right summaries and nothing from the future."""

import asyncio
import datetime as dt
import re

import pytest

from memcon.hierarchy import (BUDGET, COUNT_LEVELS, NO_ACTIVITY, RAW_SHOW, TIME_LEVELS, Hierarchy, load, node_id, save)
from memcon.world import START, Interaction, fmt_day


def mk(idx: int, ts: dt.datetime, project: str | None = "P") -> Interaction:
    return Interaction(idx, ts, project, "email", {}, f"#{idx:04d} record of interaction {idx}")


def history(days: int, per_day: dict[int, int]) -> list[Interaction]:
    """per_day: day -> number of interactions that day (others none)."""
    its = []
    for d in range(days):
        for k in range(per_day.get(d, 0)):
            its.append(mk(len(its) + 1, START + dt.timedelta(days=d, hours=9, minutes=10 * k)))
    return its


class Recorder:
    """Stub summariser: the summary of a window lists the interaction numbers it
    traces back to, so higher levels can be checked against their windows."""

    def __init__(self):
        self.calls = []

    async def __call__(self, node, prompt, words):
        self.calls.append((node.id, prompt))
        ids = sorted({int(x) for x in re.findall(r"(?:#|ids )(\d{4})", prompt.split("\n\n", 1)[1])}
                     | {int(x) for x in re.findall(r"ids=([\d,]+)", prompt) for x in x.split(",") if x})
        return {"text": "ids=" + ",".join(str(i) for i in ids), "latency_s": 0.01, "cost": 0.0, "prompt_tokens": 1, "completion_tokens": 1}


def traced(text: str) -> set[int]:
    m = re.search(r"ids=([\d,]*)", text)
    return {int(x) for x in m.group(1).split(",") if x} if m else set()


async def build(its, end):
    rec = Recorder()
    h = Hierarchy(rec)
    for it in its:
        h.observe(it)
    h.advance(end)
    await h.drain()
    return h, rec


def test_every_summary_reads_exactly_its_window():
    # 13 weeks with uneven activity, including empty days and an empty week
    per_day = {d: (d % 5) for d in range(91)}
    for d in range(21, 28):
        per_day[d] = 0
    its = history(91, per_day)
    h, rec = asyncio.run(build(its, START + dt.timedelta(days=91)))
    by_day = {}
    for it in its:
        by_day.setdefault((it.ts - START).days, []).append(it.idx)
    # day level: exactly the records of that calendar day
    for d in range(91):
        n = h.nodes[node_id("time", 1, d)]
        assert n.inputs == by_day.get(d, [])
        if by_day.get(d):
            assert traced(n.text) == set(by_day[d])
        else:
            assert n.text == NO_ACTIVITY and not n.called
    # each level above: exactly its fan-in children, in order, and it traces back to exactly its window
    spans = [1, 7, 28, 84]
    for lv in range(2, 5):
        fan = TIME_LEVELS[lv - 1][1]
        for n in [x for x in h.nodes.values() if x.axis == "time" and x.level == lv]:
            assert n.inputs == [node_id("time", lv - 1, n.block * fan + k) for k in range(fan)]
            a, b = n.block * spans[lv - 1], (n.block + 1) * spans[lv - 1]
            window = {i for d in range(a, b) for i in by_day.get(d, [])}
            assert (traced(n.text) if n.called else set()) == window
    week3 = h.nodes[node_id("time", 2, 3)]
    assert week3.n == 0 and week3.text == NO_ACTIVITY and not week3.called
    assert len([x for x in h.nodes.values() if x.axis == "time" and x.level == 4]) == 1  # one 12-week block in 13 weeks


def test_count_levels_read_exactly_their_blocks():
    its = history(30, {d: 9 for d in range(30)})  # 270 interactions
    h, _ = asyncio.run(build(its, START + dt.timedelta(days=30)))
    for n in [x for x in h.nodes.values() if x.axis == "count"]:
        size = [10, 40, 120, 480][n.level - 1]
        window = set(range(n.block * size + 1, (n.block + 1) * size + 1))
        assert traced(n.text) == window
        assert (n.first, n.last, n.n) == (min(window), max(window), size)
        if n.level > 1:
            fan = COUNT_LEVELS[n.level - 1][1]
            assert n.inputs == [node_id("count", n.level - 1, n.block * fan + k) for k in range(fan)]
    assert len([x for x in h.nodes.values() if x.axis == "count" and x.level == 1]) == 27
    assert len([x for x in h.nodes.values() if x.axis == "count" and x.level == 2]) == 6
    assert len([x for x in h.nodes.values() if x.axis == "count" and x.level == 3]) == 2


def test_base_reads_each_record_once_and_higher_levels_read_no_raw_records():
    its = history(28, {d: 3 for d in range(28)})
    h, rec = asyncio.run(build(its, START + dt.timedelta(days=28)))
    raw_reads = {}
    for nid, prompt in rec.calls:
        n = h.nodes[nid]
        recs = re.findall(r"#(\d{4}) record", prompt)
        if n.level == 1:
            for r in recs:
                raw_reads.setdefault((n.axis, int(r)), 0)
                raw_reads[(n.axis, int(r))] += 1
        else:
            assert not recs, f"{nid} read raw records"
    assert all(v == 1 for v in raw_reads.values())
    assert {i for (axis, i) in raw_reads if axis == "time"} == {it.idx for it in its}
    assert {i for (axis, i) in raw_reads if axis == "count"} == set(range(1, 81))  # 84 interactions: 8 full blocks


def test_scheduled_exactly_when_inputs_exist_and_never_before():
    async def go():
        h = Hierarchy(Recorder())
        its = history(15, {d: 2 for d in range(15)})
        for it in its[:9]:
            h.observe(it)
        assert node_id("count", 1, 0) not in h.nodes
        h.observe(its[9])
        assert node_id("count", 1, 0) in h.nodes
        # day 6 (Sunday) closes only when the clock passes its midnight
        h.advance(START + dt.timedelta(days=6, hours=23, minutes=59))
        assert node_id("time", 1, 5) in h.nodes and node_id("time", 1, 6) not in h.nodes
        assert node_id("time", 2, 0) not in h.nodes
        h.advance(START + dt.timedelta(days=7))
        assert node_id("time", 1, 6) in h.nodes and node_id("time", 2, 0) in h.nodes
        await h.drain()
    asyncio.run(go())


def test_write_path_never_blocks_the_caller():
    async def go():
        gate = asyncio.Event()

        async def slow(node, prompt, words):
            await gate.wait()
            return {"text": "s", "latency_s": 0, "cost": 0, "prompt_tokens": 0, "completion_tokens": 0}

        h = Hierarchy(slow)
        its = history(3, {0: 10, 1: 10})
        t0 = asyncio.get_running_loop().time()
        for it in its:
            h.observe(it)
        h.advance(START + dt.timedelta(days=3))
        await asyncio.sleep(0)  # let the scheduled tasks start; all but the empty day wait on the model
        text, info = h.snapshot(START + dt.timedelta(days=3))  # a lookup, while summaries are still being written
        assert asyncio.get_running_loop().time() - t0 < 0.05
        assert "#0020" in text and info["summaries"] == 1  # only the empty day 2, written without a call
        gate.set()
        await h.drain()
        text, info = h.snapshot(START + dt.timedelta(days=3))
        assert info["summaries"] > 1
    asyncio.run(go())


def test_snapshot_shows_latest_k_per_level_and_nothing_from_the_future():
    its = history(70, {d: 4 for d in range(70) if d % 7 < 5})  # weekdays only, 200 interactions
    h, _ = asyncio.run(build(its, START + dt.timedelta(days=70)))
    now = START + dt.timedelta(days=45, hours=12)
    text, info = h.snapshot(now)
    seen = set()
    for m in re.finditer(r"ids=([\d,]*)", text):
        seen |= {int(x) for x in m.group(1).split(",") if x}
    seen |= {int(x) for x in re.findall(r"#(\d{4}) record", text)}
    visible = {it.idx for it in its if it.ts < now}
    assert seen <= visible and max(visible) in seen
    days = re.findall(r"\[Day: (\w+ \d+ \w+ \d+)", text)
    assert len(days) == 7 and days[-1] == fmt_day(START + dt.timedelta(days=44))  # day 44 is the last closed day
    assert len(re.findall(r"\[Week:", text)) == 4
    assert len(re.findall(r"\[4 weeks:", text)) == 1  # only one 4-week block has closed by day 45
    raw = re.findall(r"#(\d{4}) record", text)
    assert len(raw) == RAW_SHOW and int(raw[-1]) == max(visible)
    assert info["dropped"] == 0 and len(text) <= BUDGET
    # views
    t_text, _ = h.snapshot(now, ("time",))
    c_text, _ = h.snapshot(now, ("count",))
    assert "[Day:" in t_text and "[Interactions #" not in t_text
    assert "[Interactions #" in c_text and "[Day:" not in c_text


def test_snapshot_respects_the_budget():
    its = history(10, {d: 12 for d in range(10)})
    h, _ = asyncio.run(build(its, START + dt.timedelta(days=10)))
    now = START + dt.timedelta(days=9, hours=23)
    text, info = h.snapshot(now, budget=600)
    assert len(text) <= 600 and info["dropped"] > 0


def test_save_and_load_round_trip():
    its = history(14, {d: 5 for d in range(14)})
    h, _ = asyncio.run(build(its, START + dt.timedelta(days=14)))
    h2 = load(save(h), its)
    now = START + dt.timedelta(days=12, hours=10)
    assert h.snapshot(now) == h2.snapshot(now)


def test_out_of_order_interactions_are_rejected():
    h = Hierarchy(Recorder())
    its = history(1, {0: 2})
    with pytest.raises(AssertionError):
        h.observe(its[1])


def test_resume_continues_exactly_where_the_log_stood():
    its = history(21, {d: 6 for d in range(21)})
    full, _ = asyncio.run(build(its, START + dt.timedelta(days=21)))
    at = START + dt.timedelta(days=9, hours=9, minutes=25)  # mid-morning: some of day 9 logged, block of 10 part-filled

    async def go():
        from memcon.hierarchy import resume
        h = resume(save(full), its, at, Recorder())
        assert {n.id for n in h.nodes.values()} == {n.id for n in full.nodes.values()
                                                    if (n.end_dt <= at if n.axis == "time" else n.last <= len(h.raw))}
        for it in its[len(h.raw):]:
            h.observe(it)
        h.advance(START + dt.timedelta(days=21))
        await h.drain()
        return h
    h = asyncio.run(go())
    assert set(h.nodes) == set(full.nodes)
    for nid, n in h.nodes.items():
        assert n.inputs == full.nodes[nid].inputs and traced(n.text) == traced(full.nodes[nid].text)
