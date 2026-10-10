"""Each arm, run through the real agent loop with the model stubbed."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from skills.agent import MAX_STEPS, run_task
from skills.arms import ARMS, Library, Retrieval, bm25_rank, global_names, listing_entry
from skills.tasks import make_stream


@dataclass
class FakeReply:
    message: dict
    text: str = ""
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 100
    completion_tokens: int = 10
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.001
    latency_s: float = 0.1
    raw: dict = field(default_factory=dict)


class FakeLLM:
    """Replies from a script: a list of code strings (one tool call each) for
    agent calls, and a fixed text for notes calls. Records every request."""

    def __init__(self, codes: list[str | None], notes_text: str | list[tuple[str, str]] = "NOTES v1"):
        self.codes = list(codes)
        # notes replies: a text, or a list of (text, finish_reason) served in order
        self.notes = [(notes_text, "stop")] if isinstance(notes_text, str) else list(notes_text)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, model, messages, **kw):
        self.requests.append({"messages": json.loads(json.dumps(messages)), **{k: v for k, v in kw.items() if k != "meta"}, "meta": kw.get("meta")})
        if "tools" not in kw:
            text, finish = self.notes.pop(0) if len(self.notes) > 1 else self.notes[0]
            return FakeReply(message={"role": "assistant", "content": text}, text=text,
                             raw={"choices": [{"finish_reason": finish}]})
        code = self.codes.pop(0)
        if code is None:
            return FakeReply(message={"role": "assistant", "content": "thinking"}, text="thinking")
        if isinstance(code, tuple) and code[0] == "submit":  # ("submit", answer): the submit tool
            tc = {"id": f"call_{len(self.requests)}", "type": "function", "function": {"name": "submit", "arguments": json.dumps({"answer": code[1]})}}
            return FakeReply(message={"role": "assistant", "content": None, "tool_calls": [tc]}, tool_calls=[tc])
        tc = {"id": f"call_{len(self.requests)}", "type": "function", "function": {"name": "run_python", "arguments": json.dumps({"code": code})}}
        return FakeReply(message={"role": "assistant", "content": None, "tool_calls": [tc]}, tool_calls=[tc])


def run(arm, llm, task, size="realistic"):
    async def go():
        r, ctx = await run_task(llm, "fake/model", arm, task, size, {"t": task.position})
        mem = await arm.after_task(llm, "fake/model", task, r, ctx, {"t": task.position})
        return r, mem, ctx.ns
    return asyncio.run(go())


STREAM = make_stream("realistic", 1)
EMAIL_TASK = next(t for t in STREAM.tasks if t.kind == "contact_email" and t.version == 1)
EMAIL_TASK_2 = next(t for t in STREAM.tasks if t.kind == "contact_email" and t.version == 1 and t.position > EMAIL_TASK.position)
EMAIL_TASK_V2 = next(t for t in STREAM.tasks if t.kind == "contact_email" and t.version == 2)

FIND_V1 = '''def find_contact_id(name):
    """Return the id of the contact with this exact full name."""
    page = 1
    while True:
        r = search_contacts(name, page=page)
        for c in r["results"]:
            if c["name"] == name:
                return c["id"]
        if page >= r["pages"]:
            raise ValueError(name)
        page += 1

def primary_email(name: str) -> str:
    """Return the primary email address of the contact with this full name."""
    return get_contact(find_contact_id(name))["email"]
'''
FIND_V2 = '''def find_contact_id(name):
    """Return the id of the contact with this exact full name."""
    cursor = None
    while True:
        r = find_contacts(name=name, cursor=cursor)
        for c in r["items"]:
            if c["name"] == name:
                return c["id"]
        cursor = r["next_cursor"]
        if cursor is None:
            raise ValueError(name)
'''


def name_of(task):
    return task.args["contact"]


def test_every_arm_passes_a_correct_solution_and_fails_a_wrong_one():
    for arm_name, cls in ARMS.items():
        llm = FakeLLM([FIND_V1 + f"submit(primary_email({name_of(EMAIL_TASK)!r}))", save_code("primary_email", name_of(EMAIL_TASK))])
        r, mem, ns = run(cls("realistic"), llm, EMAIL_TASK)
        assert r.passed and r.steps == 1, arm_name
        llm = FakeLLM(["submit('wrong@x.example')"])
        r, mem, ns = run(cls("realistic"), llm, EMAIL_TASK)
        assert not r.passed and r.submitted, arm_name


def test_tool_choice_cache_marks_and_same_base_prompt_for_every_arm():
    bases = set()
    for cls in ARMS.values():
        llm = FakeLLM(["submit(1)"])
        run(cls("realistic"), llm, EMAIL_TASK)
        req = llm.requests[0]
        assert req["tool_choice"] == "required" and req["temperature"] == 0
        sys_parts = req["messages"][0]["content"]
        assert all(p.get("cache_control") for p in sys_parts)
        assert req["messages"][-1]["role"] == "user" and req["messages"][-1]["content"][-1]["cache_control"]
        bases.add(sys_parts[0]["text"])
    assert len(bases) == 1  # the shared part is identical across arms


def test_step_limit_and_text_replies():
    llm = FakeLLM([None] + ["print(1)"] * (MAX_STEPS - 1))
    r, _, _ = run(ARMS["none"]("realistic"), llm, EMAIL_TASK)
    assert r.steps == MAX_STEPS and not r.submitted and not r.passed and r.reasons[0] == "never submitted"
    assert any(m["role"] == "user" and "run_python" in m["content"] for m in r.messages)


def save_code(fn_name, *args):
    return f"save({fn_name}, {', '.join(repr(a) for a in args)})"


def test_library_saves_verified_functions_after_a_correct_task_only():
    lib = Library("realistic")
    # a failed task: no save step at all (the fake would run out of code if one were asked for)
    run(lib, FakeLLM(["submit('wrong@x.example')"]), EMAIL_TASK)
    assert lib.functions == {}
    # a correct task solved by hand, then saved as functions
    llm = FakeLLM([f"submit({EMAIL_TASK.expected!r})", FIND_V1 + save_code("primary_email", name_of(EMAIL_TASK))])
    r, mem, _ = run(lib, llm, EMAIL_TASK)
    assert r.passed and mem["save"] == "saved on attempt 1"
    assert set(lib.functions) == {"find_contact_id", "primary_email"} and mem["stored"] == ["find_contact_id", "primary_email"]
    save_req = llm.requests[-1]
    assert save_req["meta"]["role"] == "save" and "save(your_function" in save_req["messages"][-1]["content"]
    assert len(mem["memory_calls"]) == 1


def test_library_save_is_retried_when_the_rerun_differs():
    lib = Library("realistic")
    wrong = FIND_V1.replace('return get_contact(find_contact_id(name))["email"]', 'return get_contact(find_contact_id(name))["name"]')
    llm = FakeLLM([f"submit({EMAIL_TASK.expected!r})",
                   wrong + save_code("primary_email", name_of(EMAIL_TASK)),
                   FIND_V1 + save_code("primary_email", name_of(EMAIL_TASK))])
    r, mem, _ = run(lib, llm, EMAIL_TASK)
    first = next(m for m in mem["save_turns"] if m["role"] == "tool")["content"]
    assert "Not saved: it returned" in first and EMAIL_TASK.expected in first
    assert mem["save"] == "saved on attempt 2" and "find_contact_id" in lib.functions
    assert 'get_contact(find_contact_id(name))["email"]' in lib.functions["primary_email"]["source"]


def test_library_is_preloaded_listed_and_skips_saving_when_it_did_the_task():
    lib = Library("realistic")
    run(lib, FakeLLM([f"submit({EMAIL_TASK.expected!r})", FIND_V1 + save_code("primary_email", name_of(EMAIL_TASK))]), EMAIL_TASK)
    llm = FakeLLM([f"print(search_contacts('x'))\nsubmit(primary_email({name_of(EMAIL_TASK_2)!r}))"])
    r, mem, ns = run(lib, llm, EMAIL_TASK_2)
    assert r.passed and r.library_calls == ["find_contact_id", "primary_email"]
    assert mem["save"] == "not needed (done by library functions)" and len(llm.requests) == 1
    shown = llm.requests[0]["messages"][0]["content"][1]["text"]
    assert "def primary_email(name: str) -> str:" in shown and "Return the primary email address" in shown
    assert "search_contacts(name, page=page)" not in shown  # bodies are not listed


def test_library_repair_after_the_interface_change_reaches_every_caller():
    lib = Library("realistic")
    run(lib, FakeLLM([f"submit({EMAIL_TASK.expected!r})", FIND_V1 + save_code("primary_email", name_of(EMAIL_TASK))]), EMAIL_TASK)
    # v2: the stored search fails with 410; the model sees the source, repairs the shared step in the task
    llm = FakeLLM([f"submit(primary_email({name_of(EMAIL_TASK_V2)!r}))",
                   FIND_V2 + f"submit(primary_email({name_of(EMAIL_TASK_V2)!r}))",
                   save_code("primary_email", name_of(EMAIL_TASK_V2))])
    r, mem, _ = run(lib, llm, EMAIL_TASK_V2)
    first_tool = next(m for m in r.messages if m["role"] == "tool")["content"]
    assert "410 Gone" in first_tool and "Source of the library functions in this traceback" in first_tool
    assert "def find_contact_id(name):" in first_tool
    assert set(r.library_failed) == {"find_contact_id", "primary_email"}
    assert "These library functions failed during this task" in llm.requests[-1]["messages"][-1]["content"]
    assert r.passed and mem["replaced"] == ["find_contact_id"] and mem["stored"] == []
    assert "find_contacts(name=name" in lib.functions["find_contact_id"]["source"]
    # the unchanged caller now works in one step, with no save step
    llm = FakeLLM([f"submit(primary_email({name_of(EMAIL_TASK_V2)!r}))"])
    r, mem, _ = run(lib, llm, EMAIL_TASK_V2)
    assert r.passed and r.steps == 1 and mem["save"].startswith("not needed")


def test_library_verifies_actions_by_their_changes():
    t = next(x for x in STREAM.tasks if x.kind == "email_contact" and x.version == 1)
    a = t.args
    fn = ('def email_contact(name, subject, body):\n    """Email a contact at their primary address."""\n'
          '    c = get_contact(search_contacts(name)["results"][0]["id"])\n    send_email(c["email"], subject, body)\n')
    twice = fn + "    send_email(c['email'], subject, body)\n"
    lib = Library("realistic")
    llm = FakeLLM([fn + f"email_contact({a['contact']!r}, {a['subject']!r}, {a['body']!r})\nsubmit(None)",
                   twice + save_code("email_contact", a["contact"], a["subject"], a["body"]),
                   fn + save_code("email_contact", a["contact"], a["subject"], a["body"])])
    r, mem, _ = run(lib, llm, t)
    assert r.passed
    first = next(m for m in mem["save_turns"] if m["role"] == "tool")["content"]
    assert "changed the workspace differently" in first and "outbox" in first
    assert mem["save"] == "saved on attempt 2" and lib.functions["email_contact"]["source"].count("send_email") == 1


def test_library_rejects_functions_that_need_task_state_and_keeps_imports():
    lib = Library("realistic")
    setup = ("import datetime as dt\nfrom collections import Counter\nLIMIT = 5\ntarget = get_contact('c_1001')\n"
             "def uses_alias():\n    \"\"\"d\"\"\"\n    return dt.date(2026, 1, 1).isoformat(), Counter('aa'), LIMIT\n"
             "def uses_state():\n    \"\"\"s\"\"\"\n    return target['email']\n"
             "def calls_bad():\n    \"\"\"c\"\"\"\n    return uses_state()\n"
             "def find_id_inline(name):\n    \"\"\"f\"\"\"\n    return search_contacts(name)['results'][0]['id']\n"
             "def task_email(name):\n    \"\"\"t\"\"\"\n    uses_alias()\n    return get_contact(find_id_inline(name))['email']\n")
    llm = FakeLLM([setup + f"submit(task_email({name_of(EMAIL_TASK)!r}))", save_code("calls_bad"),
                   save_code("task_email", name_of(EMAIL_TASK))])
    r, mem, _ = run(lib, llm, EMAIL_TASK)
    first = next(m for m in mem["save_turns"] if m["role"] == "tool")["content"]
    assert "Not saved: save() needs a function" in first and "not stored" in first
    assert set(lib.functions) == {"uses_alias", "find_id_inline", "task_email"}
    assert set(mem["rejected"]) == {"uses_state", "calls_bad"}
    src = lib.functions["uses_alias"]["source"]
    assert "import datetime as dt" in src and "from collections import Counter" in src and "LIMIT = 5" in src
    r, mem, _ = run(lib, FakeLLM(["print(uses_alias())\nsubmit(task_email(" + repr(name_of(EMAIL_TASK_2)) + "))"]), EMAIL_TASK_2)
    assert "('2026-01-01', Counter({'a': 2}), 5)" in r.cells[0]["seen"] and r.passed


def test_library_rejects_rebound_preloaded_names_and_shadowing():
    lib = Library("realistic")
    code = ("from datetime import datetime\n"
            "def parse(s):\n    \"\"\"p\"\"\"\n    return datetime.strptime(s, '%Y-%m-%d')\n"
            "def get_contact(x):\n    \"\"\"shadow\"\"\"\n    return x\n")
    llm = FakeLLM([code + f"submit({EMAIL_TASK.expected!r})", save_code("parse", "2026-01-01"), save_code("get_contact", "x")])
    r, mem, _ = run(lib, llm, EMAIL_TASK)
    assert r.passed and lib.functions == {} and mem["save"] == "failed"
    assert set(mem["rejected"]) >= {"parse", "get_contact"}


def test_global_names_sees_nested_code():
    ns = {}
    exec("def f(x):\n    return [g(y) for y in x] + [h for h in range(2)]", ns)
    assert global_names(ns["f"].__code__) == {"g", "range"}


def test_listing_entry_keeps_signature_and_docstring():
    src = 'import re\n\ndef f(a: str, b: int = 2) -> list:\n    """Line one.\n\n    Line two."""\n    return [a] * b\n'
    e = listing_entry(src)
    assert e.startswith("def f(a: str, b: int=2) -> list:") and "Line one." in e and "return" not in e


def test_notes_arm_rewrites_its_notes_from_the_outcome():
    arm = ARMS["notes"]("realistic")
    llm = FakeLLM([f"print('hi')\nsubmit({EMAIL_TASK.expected!r})"], notes_text="To find an email: search_contacts then get_contact.")
    r, mem, _ = run(arm, llm, EMAIL_TASK)
    assert arm.notes == "To find an email: search_contacts then get_contact."
    notes_req = llm.requests[-1]
    prompt = notes_req["messages"][1]["content"]
    assert "was checked and was correct" in prompt and "print('hi')" in prompt and "hi" in prompt
    assert len(mem["memory_calls"]) == 1 and mem["notes_words"] == 7
    llm = FakeLLM(["submit('x')"], notes_text="v2")
    r, mem, _ = run(arm, llm, EMAIL_TASK_2)
    assert "To find an email" in llm.requests[0]["messages"][0]["content"][1]["text"]
    assert "was NOT correct" in llm.requests[-1]["messages"][1]["content"] and arm.notes == "v2"


def test_retrieval_stores_passed_code_only_and_shows_similar_tasks():
    arm = Retrieval("realistic")
    run(arm, FakeLLM(["print(1/0)", "submit('wrong@x')"]), EMAIL_TASK)
    assert arm.store == []
    run(arm, FakeLLM(["print(1/0)", f"x = 1\nsubmit({EMAIL_TASK.expected!r})"]), EMAIL_TASK)
    assert arm.store[0]["code"] == f"x = 1\nsubmit({EMAIL_TASK.expected!r})"  # the failing cell is left out
    llm = FakeLLM(["submit(1)"])
    run(arm, llm, EMAIL_TASK_2)
    shown = llm.requests[0]["messages"][0]["content"][1]["text"]
    assert EMAIL_TASK.text in shown and "x = 1" in shown


def test_bm25_prefers_the_same_kind_over_the_same_name():
    docs = ["What is Ada Kowalczyk's primary email address?",
            "Refund Ada Kowalczyk's most recent delivered order in full, with the reason \"Late delivery\".",
            "Refund Bruno Novak's most recent delivered order in full, with the reason \"Wrong item sent\"."]
    s = bm25_rank("Refund Ada Kowalczyk's most recent delivered order in full, with the reason \"Duplicate order\".", docs[:1] + docs[2:])
    assert s[1] > s[0]


def test_retrieval_finds_the_same_kind_on_real_streams():
    """Diagnostic: how often the top example is of the task's own kind, when one exists."""
    for seed in (1, 2, 3):
        st = make_stream("realistic", seed)
        hit = total = 0
        for i, t in enumerate(st.tasks):
            prior = st.tasks[:i]
            if not any(p.kind == t.kind for p in prior):
                continue
            arm = Retrieval("realistic")
            arm.store = [{"position": p.position, "text": p.text, "code": ""} for p in prior]
            top = arm.retrieve(t.text)[0]
            hit += st.tasks[top["position"] - 1].kind == t.kind
            total += 1
        assert hit / total >= 0.9, (seed, hit, total)


def test_fullctx_keeps_every_turn_with_outcomes_and_moves_the_cache_mark():
    arm = ARMS["fullctx"]("realistic")
    run(arm, FakeLLM(["print('first')", f"submit({EMAIL_TASK.expected!r})"]), EMAIL_TASK)
    llm = FakeLLM(["submit('nope')"])
    run(arm, llm, EMAIL_TASK_2)
    msgs = llm.requests[0]["messages"]
    assert msgs[1]["content"] == f"Task: {EMAIL_TASK.text}" or msgs[1]["content"][-1]["text"] == f"Task: {EMAIL_TASK.text}"
    assert any(m.get("role") == "tool" and "first" in (m.get("content") or "") for m in msgs)
    last_user = msgs[-1]["content"][-1]["text"]
    assert last_user.startswith("The result of the previous task was checked and was correct.") and EMAIL_TASK_2.text in last_user
    assert isinstance(msgs[1]["content"], list) and msgs[1]["content"][-1]["cache_control"]  # previous task's start is marked
    llm = FakeLLM(["submit(1)"])
    run(arm, llm, EMAIL_TASK)
    assert "was NOT correct" in llm.requests[0]["messages"][-1]["content"][-1]["text"]
    assert len(arm.task_starts) == 3


@pytest.mark.parametrize("arm_name", list(ARMS))
def test_a_whole_small_stream_runs_with_a_scripted_model(arm_name):
    """Every arm runs a whole small stream end to end when the model writes the
    correct program for each task (a scripted 'model' that uses the tests'
    reference solutions through the API in the namespace)."""
    import inspect

    import scripted

    st = make_stream("small", 1)
    arm = ARMS[arm_name]("small")
    src = inspect.getsource(scripted.Scripted)
    body = "\n".join("    " + line for line in src.splitlines())
    passed = saved = 0
    for t in st.tasks:
        code = (src + "\nops = {k: v for k, v in globals().items() if callable(v)}\n"
                f"submit(Scripted(ops).solve({t.kind!r}, {t.args!r}))")
        # the library arm's save step: the whole solution as one self-contained function
        fn = (f"def solve_{t.kind}(args):\n    \"\"\"Do a {t.kind} task.\"\"\"\n{body}\n"
              f"    ops = {{k: v for k, v in globals().items() if callable(v)}}\n"
              f"    return Scripted(ops).solve({t.kind!r}, args)\n")
        r, mem, _ = run(arm, FakeLLM([code, fn + f"save(solve_{t.kind}, {t.args!r})"]), t, size="small")
        passed += r.passed
        saved += (mem.get("save") or "").startswith("saved")
    assert passed == len(st.tasks)
    if arm_name == "library":
        assert saved == len(st.tasks) and set(arm.functions) == {f"solve_{k}" for k in {t.kind for t in st.tasks}}


def test_library_skips_saving_when_the_answer_came_from_a_library_function():
    lib = Library("realistic")
    run(lib, FakeLLM([f"submit({EMAIL_TASK.expected!r})", FIND_V1 + save_code("primary_email", name_of(EMAIL_TASK))]), EMAIL_TASK)
    # printed by a library function, then retyped: covered, no save call
    llm = FakeLLM([f"print(primary_email({name_of(EMAIL_TASK_2)!r}))", f"submit({EMAIL_TASK_2.expected!r})"])
    r, mem, _ = run(lib, llm, EMAIL_TASK_2)
    assert r.passed and mem["save"].startswith("not needed") and len(llm.requests) == 2
    # a library function ran, but the answer came from the model's own lookups: a save is asked for
    llm = FakeLLM([f"x = find_contact_id({name_of(EMAIL_TASK_2)!r})\nprint(get_contact(x)['email'])",
                   f"submit({EMAIL_TASK_2.expected!r})", save_code("primary_email", name_of(EMAIL_TASK_2))])
    r, mem, _ = run(lib, llm, EMAIL_TASK_2)
    assert r.passed and mem["save"] == "saved on attempt 1" and len(llm.requests) == 3


def test_the_model_is_told_its_remaining_budget():
    llm = FakeLLM(["print(1)"] * MAX_STEPS)
    r, _, _ = run(ARMS["none"]("realistic"), llm, EMAIL_TASK)
    tools = [m["content"] for m in r.messages if m["role"] == "tool"]
    assert tools[0].endswith(f"[{MAX_STEPS - 1} tool calls left in this task]")
    assert tools[-2].endswith("[1 tool call left in this task: submit in it]")
    assert tools[-1] == "1"  # nothing after the last call


def test_notes_rewrite_cut_at_the_output_limit_is_asked_for_again_shorter():
    arm = ARMS["notes"]("realistic")
    llm = FakeLLM([f"submit({EMAIL_TASK.expected!r})"], notes_text=[("long notes that were cu", "length"), ("short notes", "stop")])
    r, mem, _ = run(arm, llm, EMAIL_TASK)
    assert arm.notes == "short notes" and len(mem["memory_calls"]) == 2 and mem["notes_cut_at_limit"] is False
    assert "cut off at the length limit" in llm.requests[-1]["messages"][-1]["content"]
    assert llm.requests[-1]["max_tokens"] == 8192


def test_the_submit_tool_ends_the_task_in_every_arm():
    for arm_name, cls in ARMS.items():
        codes = [f"x = {EMAIL_TASK.expected!r}\nprint(x)", ("submit", EMAIL_TASK.expected)]
        if arm_name == "library":
            codes.append(FIND_V1 + save_code("primary_email", name_of(EMAIL_TASK)))
        llm = FakeLLM(codes)
        r, mem, _ = run(cls("realistic"), llm, EMAIL_TASK)
        assert r.passed and r.submitted and r.steps == 2, arm_name
        assert r.messages[-1] == {"role": "tool", "tool_call_id": "call_2", "content": "Submitted."} or r.messages[-1]["content"] == "Submitted."
    # retrieval shows a tool submission as the call it stands for
    arm = Retrieval("realistic")
    run(arm, FakeLLM(["print(1)", ("submit", EMAIL_TASK.expected)]), EMAIL_TASK)
    assert arm.store[0]["code"].endswith(f"submit({EMAIL_TASK.expected!r})  # with the submit tool")


def test_library_save_step_offers_only_run_python():
    lib = Library("realistic")
    llm = FakeLLM([("submit", EMAIL_TASK.expected), FIND_V1 + save_code("primary_email", name_of(EMAIL_TASK))])
    r, mem, _ = run(lib, llm, EMAIL_TASK)
    assert [t["function"]["name"] for t in llm.requests[-1]["tools"]] == ["run_python"]
    assert [t["function"]["name"] for t in llm.requests[0]["tools"]] == ["run_python", "submit"]
    assert mem["save"] == "saved on attempt 1"


def test_notes_far_over_the_word_limit_are_asked_for_again_shorter_not_cut():
    arm = ARMS["notes"]("realistic")
    long = " ".join(["word"] * 2500)
    llm = FakeLLM([f"submit({EMAIL_TASK.expected!r})"], notes_text=[(long, "stop"), ("short notes after merging", "stop")])
    r, mem, _ = run(arm, llm, EMAIL_TASK)
    assert arm.notes == "short notes after merging" and mem["notes_shortened"] and not mem["notes_truncated"]
    assert len(mem["memory_calls"]) == 2 and "2,500 words, over the limit" in llm.requests[-1]["messages"][-1]["content"]
    # still too long after asking: the first 2,000 words are kept
    arm2 = ARMS["notes"]("realistic")
    llm = FakeLLM([f"submit({EMAIL_TASK.expected!r})"], notes_text=[(long, "stop"), (long, "stop")])
    r, mem, _ = run(arm2, llm, EMAIL_TASK)
    assert mem["notes_truncated"] and len(arm2.notes.split()) == 2000
