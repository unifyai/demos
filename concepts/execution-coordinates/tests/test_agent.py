"""The agent loop, the arms' tools, scoring and the generator, with the model stubbed."""

import ast
import asyncio
import json

import pandas as pd
import pytest
from conftest import FakeLLM, tool_call

from execution_coordinates import data as D
from execution_coordinates.agent import handle, parse_answer, system_prompt, with_cache
from execution_coordinates.executor import Executor
from execution_coordinates.scoring import correct, matches, signature_causes
from execution_coordinates.tools import ARMS, Coordinate, Ctx, PerTaskModes, Shared
from execution_coordinates.worker import names_read


def test_parse_answer():
    assert parse_answer("1234.5") == 1234.5
    assert parse_answer(12) == 12.0
    assert parse_answer('"1,234.50"') == 1234.5
    assert parse_answer("EUR 1,234.50") == 1234.5
    assert parse_answer('{"EMEA": "1,000.5", "APAC": 2}') == {"EMEA": 1000.5, "APAC": 2.0}
    assert parse_answer('{"EMEA": "n/a"}') is None
    assert parse_answer(None) is None and parse_answer(True) is None


def test_scoring_tolerances_and_region_keys():
    num = {"answer_type": "number", "truth": 100.0}
    assert correct(num, 100.04) and not correct(num, 100.06) and not correct(num, None)
    reg = {"answer_type": "regions", "truth": {"UK & Ireland": 1.0, "EMEA": 2.0}}
    assert correct(reg, {"uk and ireland": 1.0, "EMEA ": 2.0})
    assert not correct(reg, {"UK & Ireland": 1.0}) and not correct(reg, 3.0)
    cnt = {"answer_type": "count", "truth": 7}
    assert correct(cnt, 7.0) and not correct(cnt, 8)
    sig = {"answer_type": "number", "truth": 1.0,
           "signatures": [{"cause": "leak:b", "how": "x", "value": 5.0}, {"cause": "leak:z", "how": "x", "value": 5.0},
                          {"cause": "own_whatif", "how": "rate", "value": 5.0}]}
    assert signature_causes(sig, 5.0, {"a", "b"}) == ["leak:b (x)", "own_whatif (rate)"]
    assert signature_causes(sig, 1.0, {"a", "b"}) == []
    assert matches({"a": 1}, {"a": 1.0}, "regions")


def test_names_read_counts_conditional_bindings():
    t = ast.parse("if 'df' not in globals():\n    df = load()\nx = 1\ny = df.sum() + x\nz += 1\nfor i in r:\n    w = i\nw")
    assert set(names_read(t)) == {"df", "globals", "load", "z", "r", "i", "w"}
    assert "x" not in names_read(t)


def test_cache_marks_only_for_anthropic_and_do_not_touch_the_transcript():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}, {"role": "tool", "content": "out",
                                                                               "tool_call_id": "1"}]
    marked = with_cache("anthropic/claude-haiku-5.5", msgs)
    assert marked[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert marked[3]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(marked[1]["content"], str)  # only the latest message with text
    assert msgs[0]["content"] == "sys" and msgs[3]["content"] == "out"
    assert with_cache("openai/gpt-6-luna", msgs) is msgs


def test_system_prompt_differs_only_in_the_context_note():
    a, b = system_prompt(20, 20, "threads"), system_prompt(20, 20, "inbox")
    assert a != b and "about 20 seconds" in a and "about 1 second " in system_prompt(1, 1, "threads")
    assert a.split(" You work by")[1] == b.split(" You work by")[1]


def _loop(tmp_path, policy, toolset_cls=Shared, max_calls=25):
    async def go():
        ex = Executor(tmp_path / "ex")
        try:
            llm = FakeLLM(policy)
            oc = await handle(llm, "stub/m", "sys", [], "question", toolset_cls(ex), Ctx("acme", "acme#1"),
                              max_calls=max_calls)
            return oc, llm
        finally:
            await ex.close()

    return asyncio.run(go())


def test_loop_runs_tools_then_records_the_answer(tmp_path):
    def policy(messages, tools):
        if messages[-1]["role"] == "user":
            return tool_call("run_python", {"code": "x = 6 * 7\nx"})
        return tool_call("final_answer", {"answer": json.loads(messages[-1]["content"])["value"]})

    oc, llm = _loop(tmp_path, policy)
    assert oc.stopped == "answered" and oc.answer == 42.0
    roles = [m["role"] for m in oc.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]
    assert oc.messages[-1]["content"] == "Answer recorded."
    assert len(oc.calls) == 2 and oc.tools[0]["record"]["value"] == "42"
    assert llm.seen[0]["tool_choice"] == "auto" and llm.seen[0]["messages"][0]["role"] == "system"


def test_loop_nudges_then_gives_up(tmp_path):
    oc, llm = _loop(tmp_path, lambda m, t: {"role": "assistant", "content": "The answer is 4."})
    assert oc.stopped == "no_answer" and oc.answer is None and len(oc.calls) == 3
    assert [m["role"] for m in oc.messages].count("user") == 3


def test_loop_reports_bad_arguments_and_unknown_tools_and_stops_at_max_calls(tmp_path):
    calls = {"n": 0}

    def policy(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "run_python", "arguments": "{not json"}}]}
        return tool_call("teleport", {}, calls["n"])

    oc, _ = _loop(tmp_path, policy, max_calls=3)
    assert oc.stopped == "max_calls" and len(oc.calls) == 3
    assert "not a valid JSON object" in oc.messages[2]["content"]
    assert "Unknown tool" in oc.messages[4]["content"]


def test_coordinate_tool_passes_choices_and_errors_through(tmp_path):
    async def go():
        ex = Executor(tmp_path / "ex")
        try:
            t = Coordinate(ex)
            ctx = Ctx("acme", "acme#1")
            a = await t.call("run_python", {"code": "x = 1", "mode": "stateful", "session": "acme"}, ctx)
            b = await t.call("run_python", {"code": "x = 2", "mode": "read_only", "session": "acme"}, ctx)
            c = await t.call("run_python", {"code": "x", "mode": "stateful", "session": "acme"}, ctx)
            d = await t.call("run_python", {"code": "x", "mode": "read_only", "session": "nope"}, ctx)
            e = await t.call("run_python", {"code": "x"}, ctx)
            f = await t.call("list_sessions", {}, ctx)
            g = await t.call("inspect_session", {"session": "acme"}, ctx)
            return [json.loads(x[0]) for x in (a, b, c, d, e, f, g)]
        finally:
            await ex.close()

    a, b, c, d, e, f, g = asyncio.run(go())
    assert a["mode"] == "stateful" and a["session"] == "acme" and a["new_session"] is True
    assert b["mode"] == "read_only" and "new_session" not in b
    assert c["value"] == "1"
    assert set(d) == {"error", "suggestion"} and set(e) == {"error", "suggestion"}
    assert f["sessions"][0]["session"] == "acme" and g["names"][0]["name"] == "x"


def test_per_task_modes_uses_the_clients_interpreter(tmp_path):
    async def go():
        ex = Executor(tmp_path / "ex")
        try:
            t = PerTaskModes(ex)
            early = json.loads((await t.call("run_python", {"code": "1", "mode": "read_only"}, Ctx("acme", "a")))[0])
            await t.call("run_python", {"code": "x = 'acme'", "mode": "stateful"}, Ctx("acme", "a"))
            await t.call("run_python", {"code": "x = 'globex'", "mode": "stateful"}, Ctx("globex", "g"))
            ro = json.loads((await t.call("run_python", {"code": "x = 'no'\nx", "mode": "read_only"},
                                          Ctx("acme", "a")))[0])
            sl = json.loads((await t.call("run_python", {"code": "'x' in globals()", "mode": "stateless"},
                                          Ctx("acme", "a")))[0])
            back = json.loads((await t.call("run_python", {"code": "x", "mode": "stateful"}, Ctx("acme", "a")))[0])
            return early, ro, sl, back
        finally:
            await ex.close()

    early, ro, sl, back = asyncio.run(go())
    assert "not been started" in early["error"]
    assert ro["value"] == "'no'" and sl["value"] == "False" and back["value"] == "'acme'"
    assert "session" not in back


def test_tool_schemas_differ_only_in_the_mechanism():
    ex = None
    schemas = {arm: cls(ex).schemas for arm, (cls, _) in ARMS.items()}
    for arm, s in schemas.items():
        names = [t["function"]["name"] for t in s]
        assert names[0] == "run_python" and names[-1] == "final_answer"
        assert s[-1] == schemas["fresh"][-1]
    assert [t["function"]["name"] for t in schemas["coordinate"]] == [
        "run_python", "list_sessions", "inspect_session", "final_answer"]
    assert schemas["long_shared"] == schemas["shared"] and schemas["long_coordinate"] == schemas["coordinate"]


# ─────────────────────────────── the generator ───────────────────────────────
def test_generator_truths_match_an_independent_parse(tiny, tmp_path):
    spec = tiny
    for c in spec["clients"]:
        text = pd.read_csv(tmp_path / "data" / "tiny" / "seed1" / f"{c['id']}.csv", dtype=str, keep_default_na=False)
        client = D.Client(**{k: c[k] for k in ("idx", "id", "name", "person", "locale", "rep", "currencies", "rates",
                                                "prefix", "regions")})
        ref = D.reference_parse(text, client)
        scores = pd.read_csv(tmp_path / "data" / "tiny" / "seed1" / f"{c['id']}_risk.csv")
        p = c["params"]
        reqs = {r["id"]: r for r in c["requests"]}
        assert reqs["r1_net_revenue"]["truth"] == D.net_revenue(ref, c["rates"], c["prefix"], p["m1"])
        assert reqs["r2_by_region"]["truth"] == D.net_revenue_by_region(ref, c["rates"], c["prefix"], p["m1"])
        assert reqs["r4_avg_card"]["truth"] == D.avg_card_charge(ref, c["rates"], c["prefix"], p["m2"])
        hr = D.high_risk(scores, c["prefix"], p["thr"])
        assert reqs["r5_risk_count"]["truth"] == len(hr)
        assert not (scores["risk"] == p["thr"]).any()  # no ties at the threshold
        assert reqs["r6_risk_revenue"]["truth"] == D.net_revenue(ref, c["rates"], c["prefix"], p["m3"], customers=hr)
        assert sum(r.startswith("r3_") for r in reqs) == 1


def test_interleavings_keep_each_clients_order_and_mix_them():
    for k in (1, 2, 4, 8):
        order = D.interleave([6] * k, 3, k)
        assert len(order) == 6 * k
        for ci in range(k):
            assert [ri for c, ri in order if c == ci] == list(range(6))
    assert D.interleave([6] * 4, 3, 4) == D.interleave([6] * 4, 3, 4)
    assert D.interleave([6] * 4, 3, 4) != D.interleave([6] * 4, 4, 4)


@pytest.mark.parametrize("style,expect", [("comma_point", "1,234.50"), ("point_comma", "1.234,50"),
                                          ("space_comma", "1 234,50"), ("apos_point", "1'234.50"),
                                          ("plain_point", "1234.50")])
def test_amount_formats(style, expect):
    assert D.fmt_amount(1234.5, style) == expect


def test_an_answer_sent_alongside_code_is_not_recorded(tmp_path):
    def policy(messages, tools):
        if messages[-1]["role"] == "user":  # code and a placeholder answer in one message
            return {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "run_python", "arguments": '{"code": "6 * 7"}'}},
                {"id": "b", "type": "function", "function": {"name": "final_answer", "arguments": '{"answer": "0"}'}}]}
        return tool_call("final_answer", {"answer": json.loads(messages[-2]["content"])["value"]})

    oc, _ = _loop(tmp_path, policy)
    assert oc.stopped == "answered" and oc.answer == 42.0
    assert oc.messages[3]["content"].startswith("Not recorded")
    assert [t.get("early", False) for t in oc.tools] == [False, True, False]
