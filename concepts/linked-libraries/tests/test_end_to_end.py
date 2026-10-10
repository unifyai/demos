"""The whole pipeline with a scripted model: every arm changes its library and
plans every task, and the records, the governing texts and the scores come out
as they should. An oracle model makes the right edits; variants make the
mistakes each arm is prone to (a missed copy, a duplicate entry), and the
scores must show them."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from linked_libraries.agent import parse_answer, run_loop
from linked_libraries.arms import STORE, TASK_ARMS
from linked_libraries.library import rule_body, slug
from linked_libraries.runner import Records, normalize_function, run
from linked_libraries.scoring import (classify, judgment_jobs_for_task, match_title, score_task,
                                      version_jobs_for_task, version_key)
from linked_libraries.tools import FolderTools

MARK = " NEWVERSION"


@dataclass
class FakeReply:
    message: dict
    prompt_tokens: int = 10
    completion_tokens: int = 5
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.001
    latency_s: float = 0.01
    model: str = "stub"
    raw: dict = field(default_factory=dict)

    @property
    def text(self):
        return self.message.get("content") or ""

    @property
    def tool_calls(self):
        return self.message.get("tool_calls") or []


def call(tool, **args):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": f"c{abs(hash((tool, json.dumps(args, sort_keys=True)))) % 10**8}", "type": "function",
                            "function": {"name": tool, "arguments": json.dumps(args)}}]}


def final(obj):
    return {"role": "assistant", "content": json.dumps(obj)}


def arm_of(system):
    for arm, text in STORE.items():
        if text in system:
            return arm
    raise AssertionError("unknown arm")


def sections(md):
    """### or ## title -> body, from rendered markdown."""
    out = {}
    for m in re.finditer(r"^#{2,3} (?:\[\d+\] )?(.+?)\n(.*?)(?=^#{2,3} |\Z)", md, re.M | re.S):
        out[m.group(1).strip()] = m.group(2).strip()
    return out


class Oracle:
    """A scripted model. mode: 'right', 'one-copy' (pasted edits only its first copy),
    'duplicate' (linked adds a new entry instead of updating)."""

    def __init__(self, ds, mode="right"):
        self.ds = ds
        self.mode = mode
        self.calls = 0

    async def chat(self, model, messages, *, tag=None, meta=None, **kw):
        self.calls += 1
        await asyncio.sleep(0)
        system, user = messages[0]["content"], messages[1]["content"]
        arm = arm_of(system)
        done = [m for m in messages if m["role"] == "assistant"]
        change = next((c for c in self.ds["changes"] if user in (c["direct_message"], c["passing_message"])), None)
        if change:
            plan = self.change_plan(arm, change)
            return FakeReply(plan[len(done)] if len(done) < len(plan) else final({"function": None, "rules": [], "library_changed": "done"}))
        task = next(t for t in self.ds["tasks"] if t["text"] == user)
        return FakeReply(self.task_step(arm, task, messages, done))

    def change_plan(self, arm, change):
        rid = change["rule"]
        r = self.ds["rules"][rid]
        old, new = r["text"], r["text"] + MARK
        if arm == "pasted":
            folders = sorted({p["folder"] for p in self.ds["procedures"].values() if rid in p["rules"]})
            if self.mode == "one-copy":
                folders = folders[:1]
            return [call("edit_file", path=f"{f}/SKILL.md", old_text=old, new_text=new) for f in folders]
        if arm == "rule-skills":
            return [call("edit_file", path=f"{slug(r['title'])}/SKILL.md", old_text=old, new_text=new)]
        if arm == "rules-file":
            return [call("edit_file", path="RULES.md", old_text=old, new_text=new)]
        gid = list(self.ds["rules"]).index(rid) + 1
        if self.mode == "duplicate":
            via = change["passing_procedure"]
            return [call("add_guidance", title=r["title"], content=rule_body({**r, "text": new}), functions=[via])]
        return [call("search_guidance", query=r["title"]), call("update_guidance", guidance_id=gid, content=rule_body({**r, "text": new}))]

    def task_step(self, arm, task, messages, done):
        p = self.ds["procedures"][task["procedure"]]
        titles = [self.ds["rules"][r]["title"] for r in task["rules"]]
        if arm == "linked":
            if not done:
                return call("get_function", name=task["procedure"])
            res = json.loads(messages[-1]["content"])
            return final({"function": res["name"], "rules": [{"title": g["title"], "text": g["content"]} for g in res["guidance"]], "library_changed": ""})
        if arm == "pasted":
            if not done:
                return call("read_file", path=f"{p['folder']}/SKILL.md")
            secs = sections(messages[-1]["content"])
            rules = [{"title": t, "text": b} for t, b in secs.items() if f"`{task['procedure']}`" in b and "In this skill it governs" in b]
            return final({"function": task["procedure"], "rules": rules, "library_changed": ""})
        if arm == "rule-skills":
            if len(done) < len(titles):
                return call("read_file", path=f"{slug(titles[len(done)])}/SKILL.md")
            reads = [m["content"] for m in messages if m["role"] == "tool"]
            return final({"function": task["procedure"], "rules": [{"title": t, "text": x} for t, x in zip(titles, reads)], "library_changed": ""})
        # rules-file, long-context, linked-in-context: the rules are in the prompt
        secs = sections(messages[0]["content"])
        return final({"function": task["procedure"], "rules": [{"title": t, "text": secs[t]} for t in titles], "library_changed": ""})


def fake_judgments(records, ds):
    """Judge by the marker: new if it is there, old if the rule's text stands without it."""
    out = {}
    for t in records.read("tasks"):
        for key, kind, payload in judgment_jobs_for_task(t, ds):
            out[key] = {"rule": None}
        for key, kind, payload in version_jobs_for_task(t, ds, out):
            out[key] = judge_text(ds, payload["rule"], payload["text"])
    texts = {r["hash"]: r["text"] for r in records.read("texts")}
    for s in records.read("state"):
        if s["changed"]:
            text = texts[s["hash"]]
            out[version_key(s["rule"], text)] = judge_text(ds, s["rule"], text)
    return out


def judge_text(ds, rid, text):
    rt = ds["rules"][rid]["text"]
    new = (rt + MARK) in text
    old = text.replace(rt + MARK, "").count(rt) > 0
    return {"states_new": new, "states_old": old}


def run_all(ds, tmp_path, mode, arms=TASK_ARMS, conditions=("direct", "passing")):
    records = Records(tmp_path / "run")
    asyncio.run(run(Oracle(ds, mode), records, ds, "stub/model", arms=list(arms), conditions=list(conditions),
                    seeds=[1], lib_root=tmp_path / "run" / "libs"))
    return records


def state_classes(records, ds, judgments, arm):
    texts = {r["hash"]: r["text"] for r in records.read("texts")}
    out = {}
    for s in records.read("state"):
        if s["arm"] == arm and s["changed"]:
            j = judgments[version_key(s["rule"], texts[s["hash"]])]
            out[(s["condition"], s["rule"], s["procedure"])] = classify(j["states_new"], j["states_old"])
    return out


def test_oracle_gets_everything_right_in_every_arm(ds_s, tmp_path):
    records = run_all(ds_s, tmp_path, "right")
    tasks = records.read("tasks")
    assert len(tasks) == len(TASK_ARMS) * 2 * len(ds_s["tasks"])
    assert all(t["parsed"] and t["function_found"] and not t["error"] for t in tasks)
    changes = records.read("changes")
    assert len(changes) == 4 * 2 * len(ds_s["changes"])
    j = fake_judgments(records, ds_s)
    for t in tasks:
        sc = score_task(t, ds_s, j)
        assert sc["fully_correct"] and sc["exact"], (t["arm"], t["task"], sc)
    for arm in ("pasted", "rule-skills", "rules-file", "linked"):
        assert set(state_classes(records, ds_s, j, arm).values()) == {"current"}, arm
    # writes: one per copy for pasted, one for the others
    writes = {(c["arm"], c["condition"], c["change"]): len(c["writes"]) for c in changes}
    for cond in ("direct", "passing"):
        for ch in ds_s["changes"]:
            copies = len({p["folder"] for p in ds_s["procedures"].values() if ch["rule"] in p["rules"]})
            assert writes[("pasted", cond, ch["rule"])] == copies
            for arm in ("rule-skills", "rules-file", "linked"):
                assert writes[(arm, cond, ch["rule"])] == 1
    # unchanged rules are intact everywhere
    assert all(s["original_intact"] for s in records.read("state") if not s["changed"])


def test_a_missed_copy_shows_as_stale(ds_s, tmp_path):
    records = run_all(ds_s, tmp_path, "one-copy", arms=["pasted"], conditions=["passing"])
    j = fake_judgments(records, ds_s)
    classes = state_classes(records, ds_s, j, "pasted")
    multi = [c["rule"] for c in ds_s["changes"]]
    assert "stale" in classes.values() and "current" in classes.values()
    scores = [score_task(t, ds_s, j) for t in records.read("tasks")]
    assert any("stale" in s["rules"].values() for s in scores)
    assert multi


def test_a_duplicate_entry_shows_as_conflict_and_stale(ds_s, tmp_path):
    records = run_all(ds_s, tmp_path, "duplicate", arms=["linked", "linked-in-context"], conditions=["passing"])
    j = fake_judgments(records, ds_s)
    classes = state_classes(records, ds_s, j, "linked")
    assert "conflict" in classes.values() and "stale" in classes.values()
    scores = [score_task(t, ds_s, j) for t in records.read("tasks")]
    assert any("stale" in s["rules"].values() or "conflict" in s["rules"].values() for s in scores)
    adds = [w for c in records.read("changes") for w in c["writes"] if w["op"] == "add"]
    assert len(adds) == len(ds_s["changes"])


def test_resume_skips_done_work(ds_sanity, tmp_path):
    records = Records(tmp_path / "run")
    oracle = Oracle(ds_sanity)
    asyncio.run(run(oracle, records, ds_sanity, "stub/model", arms=list(TASK_ARMS), conditions=["direct"],
                    seeds=[1], lib_root=tmp_path / "run" / "libs"))
    n = oracle.calls
    asyncio.run(run(oracle, records, ds_sanity, "stub/model", arms=list(TASK_ARMS), conditions=["direct"],
                    seeds=[1], lib_root=tmp_path / "run" / "libs"))
    assert oracle.calls == n


# ---------------------------------------------------------------------------
# The loop and the scoring on their own
# ---------------------------------------------------------------------------

class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def chat(self, model, messages, **kw):
        self.seen.append([dict(m) for m in messages])
        return FakeReply(self.replies.pop(0))


def test_loop_runs_tools_nudges_once_and_caps(ds_sanity, tmp_path):
    from linked_libraries.library import build_folders
    root = tmp_path / "lib"
    build_folders(ds_sanity, "rules-file", root)
    tools = FolderTools(root, write=False)
    llm = Scripted([call("read_file", path="RULES.md"), {"role": "assistant", "content": "I think it is X."},
                    final({"function": "x", "rules": []})])
    res = asyncio.run(run_loop(llm, "m", "sys", "task", tools, max_steps=10, tag="t", meta={}))
    assert res["parsed"] and res["nudged"] and res["answer"]["function"] == "x" and res["n_tool_calls"] == 1
    assert "# Rules" in res["tool_calls"][0]["result"]
    assert llm.seen[1][-1]["role"] == "tool"
    llm = Scripted([call("read_file", path="RULES.md")] * 5)
    res = asyncio.run(run_loop(llm, "m", "sys", "task", tools, max_steps=3, tag="t", meta={}))
    assert res["capped"] and not res["parsed"] and res["llm_calls"] == 3
    llm = Scripted([{"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{bad"}}]},
                    final({"function": "y", "rules": []})])
    res = asyncio.run(run_loop(llm, "m", "sys", "task", tools, max_steps=5, tag="t", meta={}))
    assert res["tool_calls"][0]["result"].startswith("Error") and res["answer"]["function"] == "y"


def test_parse_and_normalize():
    assert parse_answer('```json\n{"function": "a", "rules": []}\n```')["function"] == "a"
    assert parse_answer('Here: {"function": "a", "rules": []} done')["function"] == "a"
    assert parse_answer("no json") is None
    assert normalize_function("billing/scripts/create_invoice.py") == "create_invoice"
    assert normalize_function("create_invoice()") == "create_invoice"
    assert normalize_function("billing.create_invoice") == "create_invoice"
    assert normalize_function(None) is None


def test_match_title(ds_l):
    rules = ds_l["rules"]
    assert match_title("Stating amounts of money", rules) == "R04"
    assert match_title("stating-amounts-of-money", rules) == "R04"
    assert match_title("[4] Stating amounts of money (euros)", rules) == "R04"
    assert match_title("Rule on stating amounts of money", rules) == "R04"
    assert match_title("Contracts", rules) == "R08"
    assert match_title("Something else entirely", rules) is None
    assert match_title("", rules) is None


def test_score_task_classes(ds_s):
    changed = ds_s["changes"][0]["rule"]
    t = next(x for x in ds_s["tasks"] if changed in x["rules"])
    title = ds_s["rules"][changed]["title"]
    rec = {"answer": {"function": t["procedure"], "rules": [{"title": title, "text": "old words"}]},
           "rules_expected": t["rules"], "function_found": True}
    from linked_libraries.scoring import joined, reported_items
    key = version_key(changed, joined(reported_items(rec["answer"])))
    for new, old, cls in ((True, False, "current"), (False, True, "stale"), (True, True, "conflict"), (False, False, "vague")):
        sc = score_task(rec, ds_s, {key: {"states_new": new, "states_old": old}})
        assert sc["rules"][changed] == cls
    rec["answer"]["rules"] = []
    assert score_task(rec, ds_s, {})["rules"][changed] == "missing"
    with pytest.raises(StopIteration):
        next(x for x in [] if x)


def test_function_found_names_exactly_the_target(ds_s):
    from linked_libraries.scoring import function_found
    procs = ds_s["procedures"]
    assert function_found("create_quote", procs, "create_quote")
    assert function_found("sales/scripts/create_quote.py (create_quote)", procs, "create_quote")
    assert function_found("`create_quote()`", procs, "create_quote")
    assert not function_found("sales", procs, "create_quote")  # a skill, not a procedure
    assert not function_found("create_quote or book_demo", procs, "create_quote")  # two procedures
    assert not function_found(None, procs, "create_quote")


def test_reasoning_and_cap_reach_every_call(ds_sanity, tmp_path):
    seen = []

    class Spy(Oracle):
        async def chat(self, model, messages, **kw):
            seen.append((kw.get("reasoning"), kw.get("max_tokens")))
            return await super().chat(model, messages, **kw)

    records = Records(tmp_path / "run")
    asyncio.run(run(Spy(ds_sanity), records, ds_sanity, "stub/model", arms=list(TASK_ARMS), conditions=["direct"],
                    seeds=[1], lib_root=tmp_path / "run" / "libs", reasoning={"effort": "high"}, max_tokens=32000))
    assert seen and all(x == ({"effort": "high"}, 32000) for x in seen)
    assert all(r["settings"] == {"reasoning": {"effort": "high"}, "max_tokens": 32000} for r in records.read("tasks"))


def test_task_subset_runs_only_those_tasks(ds_s, tmp_path):
    records = Records(tmp_path / "run")
    asyncio.run(run(Oracle(ds_s), records, ds_s, "stub/model", arms=["linked"], conditions=["passing"], seeds=[1],
                    lib_root=tmp_path / "run" / "libs", tasks=["t001", "t002"]))
    assert sorted(r["task"] for r in records.read("tasks")) == ["t001", "t002"]
    assert len(records.read("changes")) == len(ds_s["changes"])  # the change phase is whole
    with pytest.raises(ValueError):
        asyncio.run(run(Oracle(ds_s), records, ds_s, "stub/model", arms=["linked"], conditions=["passing"], seeds=[1],
                        lib_root=tmp_path / "run" / "libs", tasks=["nope"]))
