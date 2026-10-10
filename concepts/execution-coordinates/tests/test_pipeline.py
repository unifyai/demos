"""End to end with a scripted model: data -> tools -> executor -> answers ->
scoring, for every arm, including deliberate leaks and what-if damage, so the
measurement (loads, reads of other clients' state, attribution) is checked."""

import asyncio
import json

from conftest import Analyst, FakeLLM

from execution_coordinates import runner as R


def run_arm(spec, arm, k, analyst, phase="t"):
    async def go():
        llm = FakeLLM(analyst)
        path = await R.run_one(llm, model="stub/model", arm=arm, cond="tiny", k=k, seed=1, phase=phase)
        return llm, [json.loads(line) for line in path.read_text().splitlines()]

    return asyncio.run(go())


def requests(rows):
    return [r for r in rows if r["kind"] == "request"]


def test_every_arm_answers_everything_when_nothing_leaks(tiny):
    for arm in ["fresh", "shared", "per_task", "coordinate", "long_shared", "long_coordinate", "per_task_modes"]:
        reuse = arm != "shared" and arm != "long_shared"  # a careful analyst in a shared interpreter reloads
        _, rows = run_arm(tiny, arm, 2, Analyst(tiny, reuse=reuse))
        reqs = requests(rows)
        assert len(reqs) == 12 and all(r["correct"] for r in reqs), (arm, [r["answer"] for r in reqs])
        assert rows[-1]["kind"] == "run" and rows[-1]["correct"] == 12


def test_loads_count_forgetting(tiny):
    _, fresh = run_arm(tiny, "fresh", 2, Analyst(tiny))
    _, per_task = run_arm(tiny, "per_task", 2, Analyst(tiny))
    assert sum(r["loads"] for r in requests(fresh)) == 12  # one per request: nothing survives
    assert sum(r["scores"] for r in requests(fresh)) == 4
    assert sum(r["loads"] for r in requests(per_task)) == 2  # one per client
    assert sum(r["scores"] for r in requests(per_task)) == 2


def test_a_shared_interpreter_leaks_and_the_leak_is_attributed(tiny):
    _, rows = run_arm(tiny, "shared", 2, Analyst(tiny, reuse=True))
    reqs = requests(rows)
    wrong = [r for r in reqs if not r["correct"]]
    assert wrong, "the reusing analyst should pick up the other client's df"
    assert all(r["reads_other_client"] for r in wrong)
    assert any(any(c.startswith("leak:") for c in r["causes"]) for r in wrong)
    assert sum(r["loads"] for r in reqs) == 1  # the first client's load is everyone's df


def test_coordinate_sessions_isolate_unless_names_collide(tiny):
    _, own = run_arm(tiny, "coordinate", 2, Analyst(tiny, reuse=True))
    assert all(r["correct"] for r in requests(own))
    assert {tc["session"] for r in requests(own) for tc in r["tool_calls"] if tc["name"] == "run_python"} == {
        c["id"] for c in tiny["clients"][:2]}
    _, collide = run_arm(tiny, "coordinate", 2, Analyst(tiny, reuse=True, session="analysis"), phase="t2")
    assert not all(r["correct"] for r in requests(collide))


def test_own_whatif_damage_is_attributed_and_read_only_prevents_it(tiny):
    _, damaged = run_arm(tiny, "per_task", 1, Analyst(tiny, reuse=True, mutate=True))
    reqs = {r["req_id"]: r for r in requests(damaged)}
    after = [r for rid, r in reqs.items() if rid in ("r4_avg_card", "r6_risk_revenue")]
    assert reqs[[k for k in reqs if k.startswith("r3_")][0]]["correct"]  # the what-if itself is right
    assert not all(r["correct"] for r in after)
    assert any(any(c.startswith("own_whatif") for c in r["causes"]) for r in after if not r["correct"])
    assert any("#3" in x for r in after for x in r["reads_own_earlier"])
    _, protected = run_arm(tiny, "coordinate", 1, Analyst(tiny, reuse=True, mutate=True, whatif_mode="read_only"))
    assert all(r["correct"] for r in requests(protected))
    _, protected2 = run_arm(tiny, "per_task_modes", 1, Analyst(tiny, reuse=True, mutate=True, whatif_mode="read_only"))
    assert all(r["correct"] for r in requests(protected2))


def test_threads_see_only_their_client_and_inbox_sees_everything(tiny):
    llm_t, rows_t = run_arm(tiny, "per_task", 2, Analyst(tiny))
    llm_i, rows_i = run_arm(tiny, "long_shared", 2, Analyst(tiny, reuse=False))
    order = rows_t[0]["order"]
    names = [c["name"] for c in tiny["clients"][:2]]
    for llm in (llm_t, llm_i):
        for call in llm.seen:
            assert call["messages"][0]["role"] == "system"
    # in threads, every user message in a call is from the same client
    for call in llm_t.seen:
        users = [m["content"] for m in call["messages"] if m["role"] == "user"]
        assert len({n for n in names for u in users if u.startswith(f"[{n}")}) == 1
    # in the inbox, the last request sees every earlier request
    last = llm_i.seen[-2]  # the run_python call of the last request
    users = [m for m in last["messages"] if m["role"] == "user"]
    assert len(users) == len(order)


def test_complete_runs_are_skipped_and_partial_ones_discarded(tiny):
    _, rows = run_arm(tiny, "fresh", 1, Analyst(tiny))
    path = R.RUNS / "t" / f"{R.run_id('stub/model', 'fresh', 'tiny', 1, 1)}.jsonl"
    before = path.read_text()
    run_arm(tiny, "fresh", 1, Analyst(tiny))
    assert path.read_text() == before  # skipped
    path.write_text(before.splitlines()[0] + "\n")  # make it partial
    run_arm(tiny, "fresh", 1, Analyst(tiny))
    assert R.is_complete(path)
    index = (R.RUNS / "discarded" / "index.jsonl").read_text()
    assert "incomplete" in index


def test_analysis_attributes_and_compares(tiny, tmp_path):
    from execution_coordinates import analysis as A

    run_arm(tiny, "shared", 2, Analyst(tiny, reuse=True), phase="main")
    run_arm(tiny, "per_task", 2, Analyst(tiny, reuse=True, mutate=True), phase="main")
    run_arm(tiny, "coordinate", 2, Analyst(tiny, reuse=True, mutate=True, whatif_mode="read_only"), phase="main")
    run_arm(tiny, "fresh", 2, Analyst(tiny), phase="main")
    res = A.analyse(["main"], runs_dir=R.RUNS, review_path=tmp_path / "none.json")
    g = res["groups"]
    shared, per_task = g["stub/model|shared|k2"], g["stub/model|per_task|k2"]
    coord, fresh = g["stub/model|coordinate|k2"], g["stub/model|fresh|k2"]
    assert shared["wrong_by_cause"]["leak"] >= 1 and shared["wrong_by_cause"]["other"] == 0
    assert per_task["wrong_by_cause"]["leak"] == 0 and per_task["wrong_by_cause"]["own_whatif"] >= 1
    assert coord["correct"] == coord["requests"] == 12 and coord["choices"]["whatif_calls_by_mode"] == {"read_only": 2}
    assert fresh["loads_per_client"] == 6 and fresh["reloads_per_client"] == 5 + 1
    assert per_task["loads_per_client"] == 1 and per_task["reloads_per_client"] == 0
    c = res["comparisons"]["stub/model|k2|coordinate-per_task"]
    assert c["accuracy"]["diff"] > 0 and c["state_caused_rate"]["diff"] < 0
    assert "| 2 | coordinate | 12/12 |" in A.tables(res)


def test_comparisons_pair_only_common_items_and_decisions_follow_the_plan(tiny, tmp_path):
    from execution_coordinates import analysis as A

    run_arm(tiny, "shared", 2, Analyst(tiny, reuse=True), phase="main")
    run_arm(tiny, "per_task", 2, Analyst(tiny, reuse=True), phase="main")
    run_arm(tiny, "coordinate", 2, Analyst(tiny, reuse=True), phase="main")
    run_arm(tiny, "fresh", 2, Analyst(tiny), phase="main")
    reqs, _ = A.load_runs(["main"], R.RUNS)
    by_arm = {a: [r for r in reqs if r["arm"] == a] for a in ("shared", "per_task", "coordinate", "fresh")}
    # an arm that ran an extra seed is compared on the items both ran
    extra = [dict(r, seed=2, correct=False) for r in by_arm["coordinate"]]
    a, b = A.common_items(by_arm["per_task"], by_arm["coordinate"] + extra)
    assert len(a) == len(b) == len(by_arm["per_task"]) and all(r["seed"] == 1 for r in b)
    assert A.compare(by_arm["per_task"], by_arm["coordinate"] + extra)["accuracy"]["diff"] == 0
    res = A.analyse(["main"], runs_dir=R.RUNS, review_path=tmp_path / "none.json")
    d = A.decisions(res)["stub/model"]["2"]["coordinate"]
    assert d["1_avoids_forgetting"] is True  # fresh reloads, coordinate does not
    assert d["3_holds_against_default"] is True and d["baseline"] == "per_task"
    # two clients are too few for the leak interval to exclude 0; the rules themselves, on set numbers:
    iv = lambda diff, lo, hi: {"diff": diff, "lo": lo, "hi": hi}  # noqa: E731
    cmp = lambda acc, state, t, cost: {"accuracy": acc, "state_caused_rate": state,  # noqa: E731
                                       "reloads_per_client": iv(-5, -6, -4), "agent_s_per_request": t,
                                       "cost_per_request": cost}
    fake = {"groups": {"m|shared|k4": {"state_caused_wrong": 3}, "m|long_shared|k4": {"state_caused_wrong": 0}},
            "comparisons": {
                "m|k4|coordinate-per_task": cmp(iv(0, -0.01, 0.01), iv(0, 0, 0), iv(-2, -3, -1), iv(0, -1, 1)),
                "m|k4|coordinate-shared": cmp(iv(0.05, 0.01, 0.1), iv(-0.05, -0.1, -0.01), iv(0, -1, 1), iv(0, 0, 0)),
                "m|k4|coordinate-fresh": cmp(iv(0, 0, 0), iv(0, 0, 0), iv(-9, -10, -8), iv(0, 0, 0)),
                "m|k4|long_coordinate-long_shared": cmp(iv(-0.02, -0.06, 0), iv(0, 0, 0), iv(0, -1, 1),
                                                        iv(0.1, 0.05, 0.2))}}
    d = A.decisions(fake)["m"]["4"]
    assert d["coordinate"]["2_avoids_leaking"] is True
    assert d["coordinate"]["4_pulls_ahead"] is True  # holds, and faster (interval below 0)
    assert d["long_coordinate"]["2_avoids_leaking"].startswith("not shown")
    assert d["long_coordinate"]["3_holds_against_default"] is False  # lower end below -5 points
    assert d["long_coordinate"]["4_pulls_ahead"] is False
