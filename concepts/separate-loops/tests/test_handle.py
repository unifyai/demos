"""The mechanism: the handle, the work loop behind it, and the three faults of the
old code (lost result, crash on resume, lost questions), with the model stubbed."""

from __future__ import annotations

import asyncio

from conftest import FakeLLM, last_text, make_arm, say, tc, until


def _conv_default(messages):
    last = messages[-1]
    if last["role"] == "tool":
        return say("OK.")
    text = last_text(messages)
    if "Could you invite" in text:
        return say("On it.", tc("start_task", description="Invite the book club."))
    if "[Notice from task" in text and "has finished" in text:
        return say("All done: " + text.split("Its report: ")[-1])
    if "[Notice from task" in text and "has stopped" in text:
        return say("Stopped.")
    return say("Noted.")


def _work_script(steps):
    """A work policy that plays a list of replies in order, one per call that is not discarded."""
    state = {"i": 0}

    def policy(messages):
        i = state["i"]
        state["i"] += 1
        return steps[min(i, len(steps) - 1)](messages) if callable(steps[min(i, len(steps) - 1)]) else steps[min(i, len(steps) - 1)]
    return policy, state


def test_finished_result_is_a_new_notice_at_the_end_of_the_conversation():
    work, _ = _work_script([say("", tc("messaging_list_group", group="book club")),
                            say("", tc("messaging_send_message", to="Priya", text="Hi Priya, come at 6pm")),
                            say("Sent one invite, to Priya.")])
    llm = FakeLLM(lambda loop, m, t: _conv_default(m) if loop == "conv" else work(m), delays={"work": 0.05, "conv": 0.005})
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": inst["request"]})
        await asyncio.sleep(0.03)
        arm.deliver({"id": 1, "text": "How's it going?"})
        await asyncio.sleep(0.03)
        arm.deliver({"id": 2, "text": "Also mention the dish."})
        await until(lambda: any("All done" in m["text"] for m in run.agent_msgs))
        main.cancel()
        await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

    asyncio.run(go())
    conv_calls = [c for c in llm.calls if c["loop"] == "conv"]
    final = [c for c in conv_calls if "has finished" in c["messages"][-1].get("content", "")]
    assert final, "the finished notice never reached the conversation"
    msgs = final[0]["messages"]
    assert "Sent one invite, to Priya." in msgs[-1]["content"]
    # The person's later messages sit above the notice, not after it.
    idx_person = max(i for i, m in enumerate(msgs) if m["role"] == "user" and "Also mention the dish" in m["content"])
    assert idx_person < len(msgs) - 1
    assert run.agent_msgs[-1]["text"].startswith("All done: Sent one invite")


def test_resume_on_a_running_task_is_a_message_not_a_crash():
    def conv(m):
        last = m[-1]
        if last["role"] == "tool" and last["content"].startswith("Started task"):
            return say("", tc("resume_task", task_id="t1"))
        if last["role"] == "tool":
            return say("Carrying on.")
        return _conv_default(m)
    work, _ = _work_script([say("", tc("messaging_list_group", group="book club")), say("Done.")])
    llm = FakeLLM(lambda loop, m, t: conv(m) if loop == "conv" else work(m), delays={"work": 0.03})
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": inst["request"]})
        await until(lambda: any("All done" in m["text"] for m in run.agent_msgs))
        assert not main.done()
        main.cancel()
        await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

    asyncio.run(go())
    t1 = run.tasks[0]
    assert [c["op"] for c in t1["controls"]] == ["resume"]
    assert "not paused" in t1["controls"][0]["result"]
    assert t1["state"] == "finished" and not run.errors


def test_pause_resume_and_stop_answer_in_every_state():
    llm = FakeLLM(lambda loop, m, t: say("Done."), delays={"work": 0.02})
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        await arm.dispatch(tc("start_task", description="x"))
        t = arm.tasks["t1"]
        assert "not paused" in t.resume()
        assert t.pause().startswith("Paused")
        assert "already paused" in t.pause()
        assert t.resume().startswith("Resumed")
        await t.done.wait()
        for op in (t.pause, t.resume, t.stop):
            assert "finished" in op()
        assert "cannot take messages" in t.interject("more", 0)
        assert (await arm.dispatch(tc("pause_task", task_id="t9"))).startswith("Error: no task")

    asyncio.run(go())


def test_two_questions_both_reach_the_conversation_and_one_message_answers_both():
    def work(m):
        n_ask = sum(1 for x in m if x["role"] == "assistant" and "ask_person" in str(x.get("tool_calls")))
        passed = [x for x in m if x["role"] == "user" and "passed on by the assistant" in x["content"]]
        if passed:
            return say("Invited Sam Rivera at 7pm. " + passed[-1]["content"])
        if n_ask == 0:
            return say("", tc("ask_person", question="Which Sam do you mean?"))
        if n_ask == 1:
            return say("", tc("ask_person", question="Is it 6pm or 7pm?"))
        return say("Waiting for the answers.")

    def conv(m):
        text = last_text(m)
        if m[-1]["role"] == "user" and "Is it 6pm or 7pm" in text:
            return say("", tc("send_to_task", task_id="t1", message="Sam Rivera, and 7pm."))
        return _conv_default(m)

    llm = FakeLLM(lambda loop, m, t: conv(m) if loop == "conv" else work(m), delays={"work": 0.01})
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": inst["request"]})
        await until(lambda: any("All done" in m["text"] for m in run.agent_msgs))
        main.cancel()
        await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

    asyncio.run(go())
    conv_text = "\n".join(str(x.get("content")) for x in arm.ctx)
    assert "Which Sam do you mean?" in conv_text and "Is it 6pm or 7pm?" in conv_text
    qs = run.tasks[0]["questions"]
    assert len(qs) == 2 and all(q["answered_t"] is not None for q in qs)
    assert "Sam Rivera, and 7pm." in run.tasks[0]["result"]


def test_message_reaches_the_work_before_its_next_step():
    seen = []

    def work(m):
        seen.append([x["content"] for x in m if x["role"] == "user"])
        if len(seen) == 1:
            return say("", tc("messaging_send_message", to="Priya", text="Hi Priya, 6pm"))
        return say("Done.")

    llm = FakeLLM(lambda loop, m, t: work(m), delay=0.001)
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        await arm.dispatch(tc("start_task", description="x"))
        t = arm.tasks["t1"]
        await until(lambda: any(a["t_start"] is not None for a in env.log))  # the send is under way
        t.interject("It's 7pm, not 6pm.", 1)
        await t.done.wait()

    asyncio.run(go())
    assert any("It's 7pm, not 6pm." in s for s in seen[1])
    assert len(env.log) == 1


def test_message_during_a_model_call_discards_that_decision():
    def work(m):
        if any("passed on by the assistant" in x.get("content", "") for x in m if x["role"] == "user"):
            return say("Stopped before sending, as asked.")
        return say("", tc("messaging_send_message", to="Priya", text="Hi Priya, 6pm"))

    llm = FakeLLM(lambda loop, m, t: work(m), delays={"work": 0.1})
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        await arm.dispatch(tc("start_task", description="x"))
        t = arm.tasks["t1"]
        await asyncio.sleep(0.03)  # the first model call is under way
        t.interject("Don't send anything yet.", 1)
        await t.done.wait()

    asyncio.run(go())
    assert env.log == []
    work_calls = [c for c in run.calls if c["loop"] == "work:t1"]
    assert work_calls[0]["discarded"] and not work_calls[1]["discarded"]


def test_pause_holds_new_steps_and_resume_carries_on_without_repeats():
    names = ["Priya", "Tom", "Jess", "Alex"]

    def work(m):
        sent = sum(1 for x in m if x["role"] == "tool" and '"sent": true' in x["content"])
        if sent < 4:
            return say("", tc("messaging_send_message", to=names[sent], text=f"Hi {names[sent]}"))
        return say("Sent four.")

    llm = FakeLLM(lambda loop, m, t: work(m), delay=0.002)
    arm, run, env, inst = make_arm("handle", llm)
    # Real names from the instance's contacts.
    names[:] = [c["name"] for c in inst["sites"]["messaging"]["contacts"][:4]]

    async def go():
        await arm.dispatch(tc("start_task", description="x"))
        t = arm.tasks["t1"]
        await until(lambda: len([a for a in env.log if a["t_start"] is not None]) >= 2)
        t.pause()
        n_at_pause = len(env.log)
        await asyncio.sleep(0.1)
        assert len(env.log) == n_at_pause  # nothing new started while paused
        assert "paused" in t.status()
        t.resume()
        await t.done.wait()

    asyncio.run(go())
    assert [a["args"]["to"] for a in env.log] == names
    assert run.tasks[0]["state"] == "finished"


def test_stop_ends_the_work_and_the_conversation_is_told():
    def work(m):
        return say("", tc("messaging_list_group", group="book club"))

    llm = FakeLLM(lambda loop, m, t: _conv_default(m) if loop == "conv" else work(m), delays={"work": 0.01})
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": inst["request"]})
        await until(lambda: "t1" in arm.tasks and len(env.log) >= 2)
        arm.tasks["t1"].stop()
        n = len(env.log)
        await until(lambda: any(m["text"] == "Stopped." for m in run.agent_msgs))
        await asyncio.sleep(0.05)
        assert len(env.log) <= n + 1  # at most the step already decided
        main.cancel()
        await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

    asyncio.run(go())
    assert run.tasks[0]["state"] == "stopped"


def test_stop_while_the_work_waits_for_an_answer():
    def work(m):
        if not any(x["role"] == "assistant" for x in m):
            return say("", tc("ask_person", question="Which Sam?"))
        return say("Waiting for the answer.")

    llm = FakeLLM(lambda loop, m, t: work(m), delay=0.002)
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        await arm.dispatch(tc("start_task", description="x"))
        t = arm.tasks["t1"]
        await until(lambda: t.waiting)
        assert "waiting for the person's answer" in t.status()
        t.stop()
        await asyncio.wait_for(t.done.wait(), 1)

    asyncio.run(go())
    assert run.tasks[0]["state"] == "stopped"


def test_a_bug_in_the_work_fails_the_task_not_the_conversation():
    llm = FakeLLM(lambda loop, m, t: _conv_default(m) if loop == "conv" else say("", tc("messaging_list_group", group="x")),
                  delay=0.002)
    arm, run, env, inst = make_arm("handle", llm)

    async def boom(*a, **k):
        raise RuntimeError("site exploded")
    env.call = boom

    async def go():
        main = asyncio.create_task(arm.main())
        arm.deliver({"id": 0, "text": inst["request"]})
        await until(lambda: run.tasks and run.tasks[0]["state"] == "failed")
        await until(lambda: len([c for c in llm.calls if c["loop"] == "conv"]) >= 3)
        assert not main.done()
        main.cancel()
        await asyncio.gather(main, *arm.aio_tasks(), return_exceptions=True)

    asyncio.run(go())
    assert "site exploded" in run.tasks[0]["result"]
    assert any("has failed" in str(m.get("content")) for m in arm.ctx)


def test_two_tasks_run_at_once_and_are_steered_separately():
    def work(m):
        desc = m[1]["content"]
        n = sum(1 for x in m if x["role"] == "tool")
        if n < 3:
            tool = "messaging_list_group" if "A" in desc else "shop_view_cart"
            return say("", tc(tool, **({"group": "book club"} if "A" in desc else {})))
        return say(f"Done {desc}")

    llm = FakeLLM(lambda loop, m, t: work(m), delay=0.01)
    arm, run, env, inst = make_arm("handle", llm)

    async def go():
        await arm.dispatch(tc("start_task", description="task A"))
        await arm.dispatch(tc("start_task", description="task B"))
        await until(lambda: len(env.log) >= 2)
        arm.tasks["t2"].stop()
        await arm.tasks["t1"].done.wait()
        await arm.tasks["t2"].done.wait()

    asyncio.run(go())
    workers = {a["worker"] for a in env.log}
    assert workers == {"t1", "t2"}
    assert run.tasks[0]["state"] == "finished" and run.tasks[1]["state"] == "stopped"
    # They overlapped in time.
    a1 = [a for a in env.log if a["worker"] == "t1"]
    a2 = [a for a in env.log if a["worker"] == "t2"]
    assert a2[0]["t_start"] < a1[-1]["t_end"]


def _batch_policy(contacts, marker):
    """Send to every contact in one response; once the marker text is in view, report and stop."""
    def work(m):
        if any(marker in str(x.get("content")) for x in m if x["role"] in ("user", "tool")) and m[-1]["role"] != "assistant":
            if any("not started" in str(x.get("content")) for x in m if x["role"] == "tool"):
                return say("Holding the rest.")
        if not any(x["role"] == "assistant" for x in m):
            return say("", *[tc("messaging_send_message", to=c, text="Hi") for c in contacts])
        return say("Done.")
    return work


def test_pause_mid_batch_holds_the_actions_not_yet_started():
    contacts = []
    llm = FakeLLM(lambda loop, m, t: _batch_policy(contacts, "paused")(m), delay=0.002)
    arm, run, env, inst = make_arm("handle", llm)
    contacts[:] = [c["name"] for c in inst["sites"]["messaging"]["contacts"][:4]]

    async def go():
        await arm.dispatch(tc("start_task", description="x"))
        t = arm.tasks["t1"]
        await until(lambda: any(a["t_start"] is not None for a in env.log))
        t.pause()
        await asyncio.sleep(0.1)
        started = [a for a in env.log if a["t_start"] is not None]
        assert len(started) == 1 and all(a["skipped"] == "the work was paused" for a in env.log[1:])
        assert "queued" not in t.status()
        t.resume()
        await t.done.wait()

    asyncio.run(go())
    ctx = arm.tasks["t1"].ctx
    tool_results = [x["content"] for x in ctx if x["role"] == "tool"]
    assert sum("not started" in r for r in tool_results) == 3
    # The work is told it was resumed, after the results of the actions the pause held.
    notes = [i for i, x in enumerate(ctx) if x["role"] == "user" and "paused and has now been resumed" in x["content"]]
    assert len(notes) == 1 and ctx[notes[0] - 1]["role"] == "tool"


def test_message_mid_batch_skips_the_rest_in_the_work_and_in_flat():
    for arm_name in ("handle", "flat"):
        contacts = []

        def policy(loop, m, t):
            if m[-1]["role"] == "tool" and any("Stop sending" in str(x.get("content")) for x in m if x["role"] == "user"):
                return say("Stopped sending.")
            if m[-1]["role"] == "user" and "Stop sending" in m[-1]["content"]:
                return say("Stopped sending.")
            if not any(x["role"] == "assistant" for x in m):
                return say("", *[tc("messaging_send_message", to=c, text="Hi") for c in contacts])
            return say("Done.")

        llm = FakeLLM(policy, delay=0.002)
        arm, run, env, inst = make_arm(arm_name, llm)
        contacts[:] = [c["name"] for c in inst["sites"]["messaging"]["contacts"][:4]]

        async def go():
            if arm_name == "handle":
                await arm.dispatch(tc("start_task", description="x"))
                main = None
            else:
                main = asyncio.create_task(arm.main())
                arm.deliver({"id": 0, "text": "Message four people."})
            await until(lambda: any(a["t_start"] is not None for a in env.log))
            if arm_name == "handle":
                arm.tasks["t1"].interject("Stop sending.", 1)
                await arm.tasks["t1"].done.wait()
            else:
                arm.deliver({"id": 1, "text": "Stop sending."})
                await until(lambda: any(m["text"] == "Stopped sending." for m in run.agent_msgs))
                main.cancel()
                await asyncio.gather(main, return_exceptions=True)

        asyncio.run(go())
        assert len([a for a in env.log if a["t_start"] is not None]) == 1, arm_name
        assert [a["skipped"] for a in env.log[1:]] == ["a new message from the person arrived"] * 3, arm_name
