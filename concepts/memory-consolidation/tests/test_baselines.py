"""Every baseline: reads only the past, keeps to the budget, and does what it says."""

import asyncio
import datetime as dt
import json
import re

import numpy as np

from memcon import baselines as B
from memcon.answer import messages, parse
from memcon.questions import Question
from memcon.retrieve import BM25, fill, rrf_order
from memcon.world import START, Interaction


def mk(idx, ts, text=None, project="P"):
    return Interaction(idx, ts, project, "email", {}, text or f"#{idx:04d} record {idx} " + "x" * 80)


def log(n, per_day=10):
    return [mk(i + 1, START + dt.timedelta(days=i // per_day, hours=8, minutes=5 * (i % per_day))) for i in range(n)]


def ids(text):
    return [int(x) for x in re.findall(r"#(\d{4}) record", text)]


def reply(text, latency=0.5):
    return {"text": text, "latency_s": latency, "cost": 0.001, "prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0}


def test_recent_takes_the_newest_records_that_fit_in_order():
    its = log(100)
    now = its[59].ts + dt.timedelta(minutes=1)
    r = B.recent(its, now, budget=1000)
    got = ids(r.memory)
    assert got == list(range(60 - len(got) + 1, 61)) and len(r.memory) <= 1000 and len(got) >= 5


def test_full_has_every_record_before_now():
    its = log(30)
    now = its[19].ts + dt.timedelta(minutes=1)
    assert ids(B.full(its, now).memory) == list(range(1, 21))


def test_rolling_updates_from_itself_and_exactly_the_new_block():
    its = log(35)
    seen = []

    async def chat(tag, msgs, max_tokens):
        p = msgs[-1]["content"]
        seen.append(p)
        prev = re.search(r"Current running summary \([^)]*\):\n(.*?)\n\nNew interactions", p, re.S).group(1)
        new = ids(p.split("raw records:", 1)[1])
        return reply(f"{prev.strip()}|{new[0]}-{new[-1]}" if prev != "(empty)" else f"{new[0]}-{new[-1]}")

    vs = asyncio.run(B.build_rolling(its, chat, every=10))
    assert [v["last"] for v in vs] == [10, 20, 30]
    assert vs[-1]["text"] == "1-10|11-20|21-30"
    assert "empty: this is the start of the log" in seen[0] and "up to interaction #0010" in seen[1]
    # the snapshot: the latest version before now plus the raw records since
    now = its[24].ts + dt.timedelta(minutes=1)
    r = B.rolling(vs, its, now)
    assert "1-10|11-20" in r.memory and "21-30" not in r.memory and ids(r.memory) == [21, 22, 23, 24, 25]
    # resuming continues from the last version
    vs2 = asyncio.run(B.build_rolling(its, chat, every=10, resume=vs[:1]))
    assert [v["last"] for v in vs2] == [10, 20, 30] and vs2[-1]["text"] == "1-10|11-20|21-30"


def test_bm25_and_fusion():
    docs = ["the venue hire quote from Casa do Rio", "lunch with Sam", "catering quote", "venue walkthrough booked"]
    s = BM25(docs).scores("Casa do Rio venue quote")
    assert int(np.argmax(s)) == 0 and s[1] == 0
    order = rrf_order([np.array([3.0, 0.0, 1.0, 2.0]), np.array([0.9, 0.1, 0.2, 0.95])])
    assert order[:2] == [0, 3] or order[:2] == [3, 0]
    assert fill([2, 0, 1], [50, 50, 30], budget=90, sep=2) == [2, 0]


def test_retrieval_searches_only_the_past_within_budget_in_time_order():
    its = log(60)
    its[10].text = "#0011 record 11 the Casa do Rio venue quote " + "y" * 40
    its[50].text = "#0051 record 51 the Casa do Rio venue quote " + "y" * 40  # after now
    vecs = np.eye(60, 8)[:, :8] + 0.01
    now = its[40].ts + dt.timedelta(minutes=1)
    r = B.retrieval(its, vecs, np.ones(8), "Casa do Rio venue quote", now, budget=500)
    got = ids(r.memory)
    assert 11 in got and 51 not in got and all(i <= 41 for i in got)
    assert got == sorted(got) and len(r.memory) <= 500


class FakeLLM:
    """Stub extraction and update for the Mem0 store."""

    def __init__(self):
        self.prompts = []

    async def chat(self, tag, msgs, max_tokens):
        p = msgs[-1]["content"]
        self.prompts.append((tag, p))
        if tag == "mem0/extract":
            rec = p.split("New interaction:\n", 1)[1].split("\n\nExtract", 1)[0]
            n = ids(rec)[0]
            return reply(json.dumps({"facts": [f"fact {n}"]}))
        mems = re.findall(r"^(\d+): (.*)$", p.split("New facts:", 1)[0], re.M)
        fact = re.findall(r"^\d+\. (fact (\d+))$", p, re.M)[0]
        n = int(fact[1])
        if n == 3 and mems:  # fact 3 replaces the memory of fact 1
            k = [k for k, t in mems if t == "fact 1"][0]
            return reply(json.dumps({"operations": [{"op": "UPDATE", "id": int(k), "text": "fact 1 revised by 3"}]}))
        if n == 4 and mems:
            k = [k for k, t in mems if t == "fact 2"][0]
            return reply(json.dumps({"operations": [{"op": "DELETE", "id": int(k)}]}))
        return reply(json.dumps({"operations": [{"op": "ADD", "text": fact[0]}]}))

    async def embed(self, tag, texts):
        return [[1.0, float(len(t)), 0.5] for t in texts]


def test_mem0_extracts_per_exchange_and_adds_updates_deletes():
    its = log(6)
    fake = FakeLLM()
    store = B.Mem0(fake.chat, fake.embed)
    snaps = {}
    cps = {"a": its[3].ts, "b": its[5].ts + dt.timedelta(hours=1)}
    asyncio.run(B.build_mem0(its, store, cps, on_checkpoint=lambda label, s: snaps.__setitem__(label, s.state())))
    extract = [p for t, p in fake.prompts if t == "mem0/extract"]
    assert len(extract) == 6
    # each extraction sees the new interaction and the two before it, for context only
    ctx, new = extract[3].split("New interaction:\n", 1)
    assert ids(ctx) == [2, 3] and ids(new.split("\n\nExtract")[0]) == [4]
    # checkpoint a is taken before interaction 4 is processed
    assert [m["text"] for m in snaps["a"]] == ["fact 1 revised by 3", "fact 2"]
    assert [m["text"] for m in snaps["b"]] == ["fact 1 revised by 3", "fact 5", "fact 6"]
    assert {m["text"]: m["updated"] for m in snaps["b"]}["fact 1 revised by 3"] == its[2].ts.date().isoformat()
    # ids in the update prompt are small local numbers
    upd = [p for t, p in fake.prompts if t == "mem0/update"][-1]
    assert re.findall(r"^(\d+): ", upd, re.M) == [str(k) for k in range(1, len(re.findall(r"^(\d+): ", upd, re.M)) + 1)]


def test_mem0_recall_within_budget_in_date_order():
    mems = [{"id": i, "text": f"memory {i} about " + ("venue quote" if i % 3 == 0 else "lunch"), "created": f"2026-01-{i:02d}",
             "updated": f"2026-01-{i:02d}"} for i in range(1, 25)]
    vecs = np.random.default_rng(0).normal(size=(24, 4))
    r = B.mem0_recall(mems, vecs, np.ones(4), "venue quote", budget=200)
    lines = r.memory.splitlines()[1:]
    dates = [l[3:13] for l in lines]
    assert dates == sorted(dates) and len(r.memory) <= 200 and any("venue quote" in l for l in lines)


def test_query_time_routes_selects_and_maps_only_the_selection():
    its = log(200, per_day=10)  # 20 days
    now = its[-1].ts + dt.timedelta(hours=1)
    calls = []

    def make_chat(route):
        async def chat(tag, msgs, max_tokens):
            p = msgs[-1]["content"]
            calls.append((tag, p))
            if tag == "query_time/route":
                return reply(json.dumps(route), 0.3)
            if tag == "query_time/map":
                got = ids(p.split("\n\nQuestion:", 1)[0])
                return reply(f"notes on {got[0]}-{got[-1]}", 1.0 + len(got) / 1000)
            return reply("merged notes", 0.7)
        return chat

    # a short period fits the budget: raw records, no map calls
    r = asyncio.run(B.query_time(its, "q?", now, make_chat({"kind": "last", "n": 5})))
    assert ids(r.memory) == [196, 197, 198, 199, 200] and r.info["map_calls"] == 0 and r.info["critical_s"] == 0.3
    # a long period: map calls over chunks that tile exactly the selection
    calls.clear()
    a, b = its[20].ts, its[150].ts
    r = asyncio.run(B.query_time(its, "q?", now, make_chat({"kind": "period", "start": f"{a:%Y-%m-%d %H:%M}", "end": f"{b:%Y-%m-%d %H:%M}"}), budget=2000))
    maps = [ids(p.split("\n\nQuestion:", 1)[0]) for t, p in calls if t == "query_time/map"]
    assert [i for m in maps for i in m] == list(range(21, 151))
    assert r.info["selected"] == 130 and r.info["map_calls"] == len(maps) >= 1
    assert abs(r.info["critical_s"] - (0.3 + max(1.0 + len(m) / 1000 for m in maps))) < 1e-9
    # unparseable routes fall back to the whole log
    async def bad(tag, msgs, max_tokens):
        return reply("not json") if tag == "query_time/route" else reply("notes")
    r = asyncio.run(B.query_time(its, "q?", now, bad, budget=2000))
    assert r.info["route"]["kind"] == "all" and r.info["selected"] == 200


def test_query_time_merges_notes_that_overflow_the_budget():
    its = [mk(i + 1, START + dt.timedelta(hours=i), "#%04d record " % (i + 1) + "z" * 5000) for i in range(200)]
    now = its[-1].ts + dt.timedelta(hours=1)
    rounds = []

    async def chat(tag, msgs, max_tokens):
        if tag == "query_time/route":
            return reply('{"kind": "all"}')
        if tag == "query_time/merge":
            rounds.append(1)
            return reply("m" * 100)
        return reply("n" * 3000)

    r = asyncio.run(B.query_time(its, "q?", now, chat, budget=8000))
    assert r.info["map_calls"] == 6 and rounds and len(r.memory) <= 8000 and r.info["merge_rounds"] >= 1


def test_answer_prompt_is_the_same_apart_from_memory_and_parse_is_robust():
    q = Question("q", 0, "x", START.isoformat(), "T", "T_7d", "Which projects?", "projects", ["a"])
    m1 = messages(q, "MEMORY ONE", START, 12)
    m2 = messages(q, "MEMORY TWO", START, 12, cache=True)
    assert m1[1] == m2[1] and m1[0]["content"].replace("ONE", "TWO") == m2[0]["content"][0]["text"]
    assert m2[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert parse('```json\n{"working": "x", "answer": ["a", "b"]}\n```') == (["a", "b"], True)
    assert parse('{"answer": 4350}') == (4350, True)
    assert parse('Sure: "answer": "2026-03-05" and more') == ("2026-03-05", False)
    assert parse("no idea") == (None, False)


def test_mem0_stops_and_resumes_without_changing_the_result():
    its = log(6)
    cps = {"a": its[3].ts, "b": its[5].ts + dt.timedelta(hours=1)}
    fake = FakeLLM()
    whole = B.Mem0(fake.chat, fake.embed)
    snaps_whole = {}
    asyncio.run(B.build_mem0(its, whole, cps, on_checkpoint=lambda l, s: snaps_whole.__setitem__(l, s.state())))
    fake2 = FakeLLM()
    part = B.Mem0(fake2.chat, fake2.embed)
    snaps = {}
    asyncio.run(B.build_mem0(its, part, cps, on_checkpoint=lambda l, s: snaps.__setitem__(l, s.state()), stop=2))
    assert snaps == {}  # interaction 3 is not processed yet, so checkpoint a (before #4) is not reached
    asyncio.run(B.build_mem0(its, part, cps, start=2, on_checkpoint=lambda l, s: snaps.__setitem__(l, s.state()), stop=3))
    assert snaps == {"a": snaps_whole["a"]}  # stopped exactly at checkpoint a: the store is as a sees it
    rest = {k: v for k, v in cps.items() if k not in snaps}
    asyncio.run(B.build_mem0(its, part, rest, start=3, on_checkpoint=lambda l, s: snaps.__setitem__(l, s.state())))
    assert snaps == snaps_whole


def test_rolling_stop():
    its = log(35)

    async def chat(tag, msgs, max_tokens):
        return reply("s")

    assert [v["last"] for v in asyncio.run(B.build_rolling(its, chat, every=10, stop=25))] == [10, 20]
