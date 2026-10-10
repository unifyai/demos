"""The mechanism and every arm, with the model stubbed."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest
from conftest import FakeLLM, call, say

from conversation_context import agents as A
from conversation_context.agents import (ARMS, FULL_HEADER, NAIVE_HEADER, SUMMARY_HEADER, Person, Run, first_turn,
                                         fixed_prefix, render_exchange, run_from, run_whole)
from conversation_context.data import conversation
from conversation_context.scenarios import BY_ID
from conversation_context.world import World

WORLD = World(0)
S01 = BY_ID["s01"]
CONV = conversation(S01, "settled", "short", 0)


def scripted(script: dict[str, list[dict]]):
    counters: dict[str, int] = {}

    def policy(agent, system, messages, tools):
        i = counters.get(agent, 0)
        counters[agent] = i + 1
        steps = script[agent]
        return copy.deepcopy(steps[min(i, len(steps) - 1)])
    return policy


def chain_script(answer: str = "LH 2473") -> dict[str, list[dict]]:
    """Depth 3, nobody asks: each agent hands on once and relays the answer."""
    return {
        "top#1": [call("start_task", request="Find the flight number for Maya's upcoming flight."), say(f"It's {answer}.")],
        "task#1": [call("ask_records", request="Find Maya's flight number."), say(answer)],
        "records#1": [call("search_messages", request="flight number"), say(answer)],
        "searcher#1": [call("search", query="flight number"), say(answer)],
        "summarizer@top#1": [say("SUMMARY-0: Maya is going to the Berlin conference.")],
        "summarizer@task#1": [say("SUMMARY-1: flight for Berlin.")],
        "summarizer@records#1": [say("SUMMARY-2: Berlin flight.")],
    }


def go(arm: str, script: dict, depth: int = 3, present: bool = True, conv=CONV, terse: bool = False):
    llm = FakeLLM(scripted(script))
    run = Run(llm, "x/model", ARMS[arm], depth, WORLD, Person(present, S01["note"]), "test", conv)
    if terse:
        final = asyncio.run(run_from(run, fixed_prefix(depth, "messages", S01["request"])))
    else:
        final = asyncio.run(run_whole(run))
    return run, llm, final


def system_of(llm: FakeLLM, agent: str) -> str:
    return next(c["messages"][0]["content"] for c in llm.calls if c["agent"] == agent)


def first_user(llm: FakeLLM, agent: str) -> str:
    return next(c["messages"][1]["content"] for c in llm.calls if c["agent"] == agent)


# ------------------------------------------------------------ the idea: nested, read-only, hidden


def test_full_nests_one_layer_per_handover_at_depth_3():
    run, llm, final = go("full", chain_script())
    assert final == "It's LH 2473."
    task, records, searcher = (system_of(llm, n) for n in ("task#1", "records#1", "searcher#1"))
    # each callee gets the header and one more layer than its caller
    for sysmsg, n_layers in ((task, 1), (records, 2), (searcher, 3)):
        assert FULL_HEADER in sysmsg
        assert sysmsg.count("### Layer ") == n_layers
        assert sysmsg.count("[handed to you]") == 1
    # the root is Maya's conversation, settling line included, ending with the handover
    assert "### Layer 1: Maya's conversation with her assistant" in searcher
    assert f"Maya: {S01['settle'][0]}" in searcher
    assert 'Assistant -> start_task: "Find the flight number for Maya\'s upcoming flight."   [handed on: layer 2]' in searcher
    assert "### Layer 2: The work of the task agent on its request" in searcher
    assert 'Task agent -> ask_records: "Find Maya\'s flight number."   [handed on: layer 3]' in searcher
    assert 'Records agent -> search_messages: "flight number"   [handed to you]' in searcher
    # the request still arrives as the callee's first message, not in the background
    assert first_user(llm, "searcher#1") == "flight number"
    assert run.agents[3].layers[0]["title"] == "Maya's conversation with her assistant"


def test_naive_keeps_the_layers_and_changes_only_the_header():
    _, full, _ = go("full", chain_script())
    _, naive, _ = go("full_naive", chain_script())
    f, n = system_of(full, "searcher#1"), system_of(naive, "searcher#1")
    assert NAIVE_HEADER in n and FULL_HEADER not in n
    assert f.split(FULL_HEADER)[1] == n.split(NAIVE_HEADER)[1]


def test_no_tool_lets_the_model_write_context():
    for tool in list(A.DELEGATE.values()) + list(A.LEAF.values()) + [A.SEARCH, A.ASK_CALLER_TOOL, A.ASK_PERSON_TOOL]:
        props = tool["function"]["parameters"]["properties"]
        assert len(props) == 1 and set(props) <= {"request", "query", "question"}


def test_alone_and_brief_pass_nothing_but_the_request():
    for arm in ("alone", "brief"):
        run, llm, final = go(arm, chain_script())
        assert final == "It's LH 2473."
        for name in ("task#1", "records#1", "searcher#1"):
            s = system_of(llm, name)
            assert "Background" not in s and "Layer" not in s and S01["settle"][0] not in s


def test_brief_adds_only_the_request_writing_rule():
    for role in ("top", "task", "records"):
        prompt = {"top": A.top_prompt, "task": A.task_prompt, "records": lambda d, a: A.records_prompt(a)}[role]
        with_rule, without = prompt(3, ARMS["brief"]), prompt(3, ARMS["alone"])
        assert "stands alone" in with_rule and "stands alone" not in without
        assert with_rule.replace(A.BRIEF.format(what="this conversation" if role == "top" else "what you have seen") + "\n", "") == without
    assert A.searcher_prompt("messages") == A.searcher_prompt("messages")  # searchers hand nothing on


def test_identical_prompts_across_alone_summary_full():
    for arm in ("summary", "full", "full_naive"):
        assert A.top_prompt(3, ARMS[arm]) == A.top_prompt(3, ARMS["alone"])
        assert A.task_prompt(3, ARMS[arm]) == A.task_prompt(3, ARMS["alone"])
        assert A.records_prompt(ARMS[arm]) == A.records_prompt(ARMS["alone"])


def test_summary_is_written_at_every_handover_and_rolled_forward():
    run, llm, final = go("summary", chain_script())
    names = [c["agent"] for c in llm.calls]
    assert [n for n in names if n.startswith("summarizer")] == ["summarizer@top#1", "summarizer@task#1", "summarizer@records#1"]
    assert SUMMARY_HEADER in system_of(llm, "task#1") and "SUMMARY-0" in system_of(llm, "task#1")
    assert "SUMMARY-2" in system_of(llm, "searcher#1") and "SUMMARY-0" not in system_of(llm, "searcher#1")
    # the first summariser sees the whole conversation and the request; the second sees the summary it rolls forward
    first = next(c for c in llm.calls if c["agent"] == "summarizer@top#1")["messages"][1]["content"]
    assert S01["settle"][0] in first and "Find the flight number for Maya's upcoming flight." in first
    second = next(c for c in llm.calls if c["agent"] == "summarizer@task#1")["messages"][1]["content"]
    assert "SUMMARY-0" in second and S01["settle"][0] not in second
    assert "Layer" not in system_of(llm, "searcher#1")


def test_single_searches_itself_with_the_whole_conversation():
    script = {"single#1": [call("search_messages", query="flight Berlin"), say("LH 2473")]}
    run, llm, final = go("single", script)
    assert final == "LH 2473"
    names = {t["function"]["name"] for t in llm.calls[0]["tools"]}
    assert names == {"search_messages", "search_documents", "search_calendar", "ask_person"}
    tool_result = run.agents[0].messages[-2]["content"]
    assert "LH 2473" in tool_result  # the real store answered
    assert len(run.agents) == 1


def test_depth_wiring():
    assert {t["function"]["name"] for t in A.tools_for("top", 1, ARMS["full"])} == {"search_messages", "search_documents", "search_calendar", "ask_person"}
    assert {t["function"]["name"] for t in A.tools_for("top", 2, ARMS["full"])} == {"start_task", "ask_person"}
    assert {t["function"]["name"] for t in A.tools_for("task", 2, ARMS["full"])} == {"search_messages", "search_documents", "search_calendar", "ask_caller"}
    assert {t["function"]["name"] for t in A.tools_for("task", 3, ARMS["full"])} == {"ask_records", "ask_caller"}
    assert {t["function"]["name"] for t in A.tools_for("searcher", 3, ARMS["full"])} == {"search", "ask_caller"}


def test_depth_1_full_gives_the_searcher_one_layer():
    script = {"top#1": [call("search_messages", request="Find the flight number."), say("LH 2473")],
              "searcher#1": [call("search", query="flight"), say("LH 2473")]}
    run, llm, final = go("full", script, depth=1)
    s = system_of(llm, "searcher#1")
    assert s.count("### Layer ") == 1 and "[handed to you]" in s and S01["settle"][0] in s


# ------------------------------------------------------------ questions bubble up, inside each caller's thread


def bubbling_script(top_answers_itself: bool) -> dict:
    """The searcher asks; each caller asks its own caller in its own thread; the answer comes back down."""
    s = chain_script()
    top_answer = [say("She means Berlin.")] if top_answers_itself else [call("ask_person", 9, question="Which flight?"), say("She means Berlin.")]
    s["top#1"] = [s["top#1"][0]] + top_answer + [say("It's LH 2473.")]
    s["task#1"] = [s["task#1"][0], call("ask_caller", 3, question="Which flight?"), say("Berlin"), say("LH 2473")]
    s["records#1"] = [s["records#1"][0], call("ask_caller", 2, question="Which flight does Maya mean?"), say("Berlin"), say("LH 2473")]
    s["searcher#1"] = [call("search", query="flight number"), call("ask_caller", question="Berlin or Lisbon?"), say("LH 2473")]
    return s


@pytest.mark.parametrize("present", [True, False])
def test_question_bubbles_to_the_person_and_back(present):
    run, llm, final = go("alone", bubbling_script(top_answers_itself=False), present=present)
    assert final == "It's LH 2473."
    assert [q["from"] for q in run.questions] == ["searcher#1", "records#1", "task#1"]
    assert [q["to"] for q in run.questions] == ["records#1", "task#1", "top#1"]
    assert all(q["escalated"] for q in run.questions)
    assert len(run.person_questions) == 1 and run.person_questions[0]["answered"] is present
    # the searcher got the answer through each caller
    searcher_last = [c for c in llm.calls if c["agent"] == "searcher#1"][-1]["messages"][-1]
    assert searcher_last["role"] == "tool" and searcher_last["content"] == "The records agent answered: Berlin"


def test_a_caller_remembers_its_answer_and_gets_the_result_after_it():
    run, llm, final = go("alone", bubbling_script(top_answers_itself=False), present=True)
    top = run.agents[0]
    own = top.messages[len(CONV):]
    # start_task call -> its slot holds the question -> the top asks Maya -> her reply -> its answer -> the result
    assert own[0]["tool_calls"][0]["function"]["name"] == "start_task"
    assert own[1]["role"] == "tool" and "The task agent you gave this to is still working on it" in own[1]["content"]
    assert "Your next reply goes to the task agent, not to Maya" in own[1]["content"]
    assert own[2]["tool_calls"][0]["function"]["name"] == "ask_person"
    assert own[3]["role"] == "tool" and own[3]["content"] == 'Maya replied: "The Berlin one, for the conference."'
    assert own[4] == {"role": "assistant", "content": "[Reply to the task agent] She means Berlin."}
    assert own[5]["role"] == "user" and own[5]["content"].startswith('[Result of start_task: "Find the flight number')
    assert own[6] == {"role": "assistant", "content": "It's LH 2473."}
    # the final reply's call saw all of it
    last_call = [c for c in llm.calls if c["agent"] == "top#1"][-1]["messages"]
    assert any("Maya replied" in str(m.get("content")) for m in last_call)
    # a caller's thread is valid for the API: every tool call has exactly one tool message
    for a in run.agents:
        ids = [tc["id"] for m in a.messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []]
        tools = [m["tool_call_id"] for m in a.messages if m["role"] == "tool"]
        assert sorted(ids) == sorted(tools), a.name


def test_top_can_answer_from_the_conversation_without_the_person():
    run, llm, final = go("alone", bubbling_script(top_answers_itself=True))
    assert run.person_questions == [] and run.questions[-1]["escalated"] is False
    assert run.questions[-1]["answer"] == "She means Berlin."
    assert final == "It's LH 2473."


def test_person_reply_texts():
    assert Person(True, "The Berlin one.").reply() == 'Maya replied: "The Berlin one."'
    assert "No reply" in Person(False, "x").reply()


def test_answering_happens_in_the_callers_own_thread_with_its_background():
    run, llm, _ = go("full", bubbling_script(top_answers_itself=True))
    answering = [c for c in llm.calls if c["agent"] == "records#1"][1]
    assert answering["messages"][0]["content"] == system_of(llm, "records#1")  # same system, background included
    last = answering["messages"][-1]
    assert last["role"] == "tool" and 'The search agent you gave this to is still working on it and asks you: "Berlin or Lisbon?"' in last["content"]
    assert "ask your own caller with ask_caller" in last["content"]
    top_turn = [c for c in llm.calls if c["agent"] == "top#1"][1]["messages"][-1]["content"]
    assert "ask Maya with ask_person" in top_turn


def test_a_second_question_arrives_as_a_message_and_results_merge():
    s = bubbling_script(top_answers_itself=True)
    s["searcher#1"] = [call("search", query="x"), call("ask_caller", question="Berlin or Lisbon?"),
                       call("ask_caller", 7, question="And the date?"), say("LH 2473")]
    s["records#1"] = [s["records#1"][0], say("Berlin"), say("16 June"), say("LH 2473")]
    run, llm, final = go("alone", s)
    rec = run.agents[2]
    own = rec.messages[1:]
    assert own[1]["role"] == "tool" and "Berlin or Lisbon?" in own[1]["content"]
    assert own[2] == {"role": "assistant", "content": "[Reply to the search agent] Berlin"}
    assert own[3]["role"] == "user" and "And the date?" in own[3]["content"]
    assert own[4] == {"role": "assistant", "content": "[Reply to the search agent] 16 June"}
    assert own[5]["role"] == "user" and own[5]["content"].startswith("[Result of search_messages")


def test_parallel_calls_all_get_a_slot_when_one_of_them_is_asked_about():
    llm = FakeLLM(scripted({"records#1": [say("Berlin")]}))
    run = Run(llm, "x", ARMS["alone"], 3, WORLD, Person(True, "n"), "t", CONV)
    caller = run.new_agent("records", 2, "sys", [
        {"role": "user", "content": "req"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "search_messages", "arguments": '{"request": "x"}'}},
            {"id": "b", "type": "function", "function": {"name": "search_calendar", "arguments": '{"request": "y"}'}},
            {"id": "c", "type": "function", "function": {"name": "search_documents", "arguments": '{"request": "z"}'}}]},
        {"role": "tool", "tool_call_id": "a", "content": "done a"}], n_fixed=1)
    asker = run.new_agent("searcher", 3, "s", [], caller=caller, handover_id="b")
    answer = asyncio.run(run.clarify(asker, "which?"))
    assert answer == "The records agent answered: Berlin"
    tail = {m["tool_call_id"]: m["content"] for m in caller.messages if m["role"] == "tool"}
    assert tail["a"] == "done a" and "which?" in tail["b"] and tail["c"].startswith("[Not started yet")
    assert caller.answered == {"b", "c"} and caller.messages[-1] == {"role": "assistant", "content": "[Reply to the search agent] Berlin"}


def test_render_shows_events_but_never_as_maya():
    run, llm, _ = go("full", bubbling_script(top_answers_itself=True))
    lines = render_exchange(run.agents[0], None)
    assert not any(l.startswith("Maya: [") for l in lines)
    assert any(l.startswith("  <- [Result of start_task") for l in lines)


def test_render_exchange_keeps_results_of_earlier_parallel_calls_and_marks_the_handover():
    llm = FakeLLM(scripted({}))
    run = Run(llm, "x", ARMS["full"], 3, WORLD, Person(True, "n"), "t", CONV)
    top = run.new_agent("top", 0, "sys", n_fixed=1, messages=[{"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "p", "type": "function", "function": {"name": "ask_person", "arguments": '{"question": "which?"}'}},
            {"id": "t", "type": "function", "function": {"name": "start_task", "arguments": '{"request": "do it"}'}}]},
        {"role": "tool", "tool_call_id": "p", "content": "Maya replied: \"Berlin\""}])
    lines = render_exchange(top, "t")
    assert lines[0] == "Maya: hi" and lines[1] == 'Assistant -> ask_person: "which?"'
    assert lines[2] == '  <- Maya replied: "Berlin"'
    assert lines[3].startswith('Assistant -> start_task: "do it"') and lines[3].endswith(A.HANDOVER)


# ------------------------------------------------------------ first turns and fixed requests


def test_first_turn_stops_at_the_handover_and_run_from_carries_on():
    script = chain_script()
    llm = FakeLLM(scripted(script))
    run = Run(llm, "x", ARMS["alone"], 3, WORLD, Person(True, "n"), "t", CONV)
    added, reply = asyncio.run(first_turn(run))
    assert reply is None and len(added) == 1 and added[0]["tool_calls"][0]["function"]["name"] == "start_task"
    assert [c["agent"] for c in llm.calls] == ["top#1"]  # nothing executed yet
    llm2 = FakeLLM(scripted({k: v[1:] if k == "top#1" else v for k, v in script.items()}))
    run2 = Run(llm2, "x", ARMS["full"], 3, WORLD, Person(True, "n"), "t2", CONV)
    final = asyncio.run(run_from(run2, added))
    assert final == "It's LH 2473." and "top#1" in [c["agent"] for c in llm2.calls]
    assert FULL_HEADER in system_of(llm2, "task#1")


def test_first_turn_executes_a_question_to_the_person_before_handing_on():
    script = chain_script()
    script["top#1"] = [call("ask_person", question="Which flight?"), call("start_task", request="Find the Berlin flight number.")]
    llm = FakeLLM(scripted(script))
    run = Run(llm, "x", ARMS["alone"], 3, WORLD, Person(False, "n"), "t", CONV)
    added, reply = asyncio.run(first_turn(run))
    assert [m["role"] for m in added] == ["assistant", "tool", "assistant"] and "No reply" in added[1]["content"]
    assert len(run.person_questions) == 1


def test_fixed_prefix_hands_the_terse_request_to_the_right_place():
    p3 = fixed_prefix(3, "calendar", "Find it.")
    p1 = fixed_prefix(1, "calendar", "Find it.")
    assert p3[0]["tool_calls"][0]["function"]["name"] == "start_task"
    assert p1[0]["tool_calls"][0]["function"]["name"] == "search_calendar"
    assert json.loads(p1[0]["tool_calls"][0]["function"]["arguments"]) == {"request": "Find it."}


def test_terse_run_at_depth_3():
    script = chain_script()
    script["top#1"] = [say("It's LH 2473.")]
    run, llm, final = go("full", script, terse=True)
    assert final == "It's LH 2473."
    assert 'Assistant -> start_task: "Find the flight number for Maya\'s upcoming flight."' in system_of(llm, "searcher#1")


def test_step_limit_is_recorded_not_fatal():
    script = chain_script()
    script["searcher#1"] = [call("search", query="x")]  # never replies
    run, llm, final = go("alone", script)
    assert run.step_limits == ["searcher#1"]
    assert final == "It's LH 2473."
    assert sum(1 for c in llm.calls if c["agent"] == "searcher#1") == A.MAX_STEPS


def test_unknown_tool_and_missing_args_are_reported_to_the_model():
    script = chain_script()
    script["searcher#1"] = [call("start_task", request="x"), call("search"), say("LH 2473")]
    run, llm, final = go("alone", script)
    results = [m["content"] for m in run.agents[3].messages if m["role"] == "tool"]
    assert results[0].startswith("Error: there is no tool") and results[1] == "Error: query is required."


def test_transcript_drops_the_conversation():
    run, llm, final = go("full", chain_script())
    t = A.transcript(run, len(CONV))
    assert t[0]["role"] == "top" and all(m.get("content") != CONV[0]["content"] for m in t[0]["messages"])
    assert "[the conversation, as in the item" in t[3]["background"] and "### Layer 3" in t[3]["background"]


def test_caller_and_worker_models_and_the_summary_writer():
    llm = FakeLLM(scripted(chain_script()))
    run = Run(llm, "x/caller", ARMS["summary"], 3, WORLD, Person(True, S01["note"]), "t", CONV, worker_model="y/worker")
    asyncio.run(run_whole(run))
    by_agent = {c["agent"]: c["model"] for c in llm.calls}
    assert by_agent["top#1"] == "x/caller"
    assert by_agent["task#1"] == by_agent["records#1"] == by_agent["searcher#1"] == "y/worker"
    assert by_agent["summarizer@top#1"] == "x/caller"  # the agent handing over writes the summary
    assert by_agent["summarizer@task#1"] == by_agent["summarizer@records#1"] == "y/worker"
    single = Run(FakeLLM(scripted({"single#1": [say("LH 2473")]})), "x/caller", ARMS["single"], 3, WORLD,
                 Person(True, "n"), "t", CONV, worker_model="y/worker")
    asyncio.run(run_whole(single))
    assert {c["model"] for c in single.llm.calls} == {"x/caller"}


def test_model_person_answers_from_the_note_and_away_means_no_call():
    def policy(agent, system, messages, tools):
        if agent == "maya":
            assert "The Berlin one, for the conference." in system and messages[-1]["content"] == "Which flight?"
            return say("The Berlin one, for my conference.")
        return copy.deepcopy(bubbling_script(top_answers_itself=False)[agent][counts.setdefault(agent, [0])[0]])
    counts: dict = {}

    def counted(agent, system, messages, tools):
        if agent != "maya":
            counts.setdefault(agent, [0])
            msg = policy(agent, system, messages, tools)
            counts[agent][0] += 1
            return msg
        return policy(agent, system, messages, tools)

    for present in (True, False):
        counts.clear()
        llm = FakeLLM(counted)
        run = Run(llm, "x", ARMS["alone"], 3, WORLD, Person(present, S01["note"], model="p/person"), "t", CONV)
        asyncio.run(run_whole(run))
        maya_calls = [c for c in llm.calls if c["agent"] == "maya"]
        assert len(maya_calls) == (1 if present else 0)
        reply = run.person_questions[0]["reply"]
        assert reply == ('Maya replied: "The Berlin one, for my conference."' if present else Person(False, "x").reply())


def test_worker_reasoning_reaches_workers_only():
    seen = []

    class Recorder(FakeLLM):
        async def chat(self, model, messages, **kw):
            seen.append(((kw.get("meta") or {}).get("agent", ""), kw.get("reasoning")))
            return await super().chat(model, messages, **kw)

    llm = Recorder(scripted(chain_script()))
    run = Run(llm, "x/caller", ARMS["summary"], 3, WORLD, Person(True, S01["note"]), "t", CONV,
              worker_model="y/worker", worker_reasoning="high")
    asyncio.run(run_whole(run))
    by = dict(seen)
    assert by["top#1"] is None and by["summarizer@top#1"] is None
    assert by["task#1"] == by["searcher#1"] == by["summarizer@task#1"] == {"effort": "high"}
